#!/usr/bin/env python
import os
import json
import argparse
from collections import defaultdict
from typing import Any, Dict, List


def infer_team_key(team_str: Any) -> str:
    """Normalize raw team names into canonical keys."""
    if not isinstance(team_str, str):
        return ""
    s = team_str.strip()
    low = s.lower()
    if "home team" in low:
        return "Home Team"
    if "away team" in low:
        return "Away Team"
    # Fall back to the raw string for club names etc.
    return s


def empty_stats() -> Dict[str, int]:
    return {
        "Goals": 0,
        "Shots": 0,
        "Fouls": 0,
        "Yellow Cards": 0,
        "Red Cards": 0,
        "Corner Kicks": 0,
        "Free Kicks": 0,
        "Offsides": 0,
    }


def get_team_from_row(row: Dict[str, Any]) -> str:
    """
    Try to find the team field in a row by looking for column names that
    contain 'team', 'side', or 'club'.
    """
    for key, val in row.items():
        kl = str(key).lower()
        if any(tok in kl for tok in ["team", "side", "club"]):
            return str(val)
    return ""


def get_event_text(row: Dict[str, Any]) -> str:
    """
    Build a lowercase text blob describing the event from columns that look
    like event / description / text, plus the 'type' column if present.
    """
    parts: List[str] = []
    for key, val in row.items():
        if not isinstance(val, str):
            continue
        kl = str(key).lower()
        if any(tok in kl for tok in ["event", "description", "text", "comment", "play"]):
            parts.append(val)
    # Fallback: also consider 'result', 'summary'
    for key in ["result", "summary", "detail"]:
        v = row.get(key)
        if isinstance(v, str):
            parts.append(v)
    # Include type as well
    t = row.get("type")
    if isinstance(t, str):
        parts.append(t)
    return " ".join(parts).lower()


def numeric_from_keys(row: Dict[str, Any], key_variants) -> int:
    """
    Look for numeric counts under any of the given key variants.
    Returns a non-negative integer; 0 if absent or not numeric.
    """
    for k, v in row.items():
        kl = str(k).lower()
        if kl in key_variants and isinstance(v, (int, float)):
            try:
                n = int(v)
            except Exception:
                continue
            if n > 0:
                return n
    return 0


def aggregate_match(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """
    Aggregate event rows for a single match into team-level Livesum/T3 stats.
    """
    stats: Dict[str, Dict[str, int]] = defaultdict(empty_stats)

    for r in rows:
        # Team
        team_raw = get_team_from_row(r)
        team = infer_team_key(team_raw)
        if not team:
            continue
        s = stats[team]

        ev = get_event_text(r)

        # ---- Shots (including headers / generic attempts) ----
        if any(p in ev for p in ["shot", "header", "attempt", "effort"]):
            s["Shots"] += 1

        # ---- Corner kicks ----
        if "corner" in ev:
            s["Corner Kicks"] += 1

        # ---- Free kicks (including penalties) ----
        if "free kick" in ev or "penalty" in ev:
            s["Free Kicks"] += 1

        # ---- Offsides ----
        if "offside" in ev:
            s["Offsides"] += 1

        # ---- Goals ----
        goal_n = numeric_from_keys(r, {"goal", "goals"})
        # Fallback to text if no numeric field
        if goal_n == 0:
            if " scores" in ev or "scores for" in ev or "scored for" in ev or "goal for" in ev:
                goal_n = 1
            elif "makes it" in ev and "goal" in ev:
                goal_n = 1
        s["Goals"] += goal_n

        # ---- Fouls ----
        foul_n = numeric_from_keys(r, {"foul", "fouls"})
        if foul_n == 0:
            if "foul" in ev or "dangerous play" in ev:
                foul_n = 1
        s["Fouls"] += foul_n

        # ---- Yellow / Red Cards ----
        yc_n = numeric_from_keys(r, {"yellow_card", "yellow_cards", "yc"})
        rc_n = numeric_from_keys(r, {"red_card", "red_cards", "rc"})

        # Text fallback
        if yc_n == 0 and "yellow card" in ev:
            # Second yellow = two yellows
            if "second yellow" in ev:
                yc_n += 2
            else:
                yc_n += 1

        # Red cards from text
        if rc_n == 0 and ("red card" in ev or "sent off" in ev):
            rc_n = 1

        # Second yellow implies a red as well
        if "second yellow" in ev and ("sent off" in ev or "has been sent off" in ev):
            if rc_n == 0:
                rc_n = 1

        s["Yellow Cards"] += yc_n
        s["Red Cards"] += rc_n

    # Ensure both teams are present with zero rows if missing
    for key in ["Away Team", "Home Team"]:
        _ = stats[key]  # will trigger empty_stats via defaultdict

    return stats


def write_csv_for_match(doc_id: int, stats: Dict[str, Dict[str, int]], out_dir: str) -> None:
    header = [
        "Team",
        "Goals",
        "Shots",
        "Fouls",
        "Yellow Cards",
        "Red Cards",
        "Corner Kicks",
        "Free Kicks",
        "Offsides",
    ]
    team_order = ["Away Team", "Home Team"]  # T3 convention

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{doc_id}.csv")
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for team in team_order:
            s = stats.get(team, empty_stats())
            row = [team] + [str(s[h]) for h in header[1:]]
            f.write(",".join(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_test", required=True, help="data/livesum/test.json")
    ap.add_argument("--pred_jsonl", required=True, help="tables/livesum_sft2_vllm_pi2.jsonl")
    ap.add_argument("--csv_dir", required=True, help="Output dir for T3-style CSVs")
    args = ap.parse_args()

    # Load predictions grouped by doc_id
    by_doc: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            doc_id = rec.get("doc_id")
            if doc_id is None:
                continue
            rows = rec.get("rows") or []
            rows = [r for r in rows if isinstance(r, dict)]
            by_doc[int(doc_id)] = rows

    # Use gold_test to define ordering / doc_ids
    with open(args.gold_test, "r", encoding="utf-8") as f:
        gold = json.load(f)

    for ex in gold:
        doc_id = int(ex["id"])
        rows = by_doc.get(doc_id, [])
        stats = aggregate_match(rows)
        write_csv_for_match(doc_id, stats, args.csv_dir)

    print(f"[info] Wrote CSVs to {args.csv_dir}")


if __name__ == "__main__":
    main()
