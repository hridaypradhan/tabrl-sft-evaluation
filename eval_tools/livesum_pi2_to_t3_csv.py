#!/usr/bin/env python
import os
import json
import argparse
from collections import defaultdict

def infer_team_key(team_str: str) -> str:
    """Normalize raw team names into canonical keys."""
    if not isinstance(team_str, str):
        return ""
    s = team_str.strip()
    low = s.lower()
    if "home team" in low:
        return "Home Team"
    if "away team" in low:
        return "Away Team"
    return s or ""

def aggregate_match(rows):
    """
    Aggregate event-level rows into team-level stats for:
    Goals, Shots, Fouls, Yellow Cards, Red Cards, Corner Kicks, Free Kicks, Offsides.
    """
    stats = defaultdict(lambda: defaultdict(int))

    for r in rows:
        team_raw = r.get("team", "")
        team = infer_team_key(team_raw)
        if not team:
            continue
        s = stats[team]

        # Direct numeric fields IF present in schema
        for field, canonical in [
            ("goal", "Goals"),
            ("foul", "Fouls"),
            ("yellow_card", "Yellow Cards"),
            ("red_card", "Red Cards"),   # may or may not exist
        ]:
            v = r.get(field)
            if isinstance(v, (int, float)):
                s[canonical] += int(v)

        # Text-based cues (using both event + type)
        ev = ((r.get("event") or "") + " " + (r.get("type") or "")).lower()

        # Shots (any shot-like thing)
        if "shot" in ev:
            s["Shots"] += 1

        # Corner kicks
        if "corner kick" in ev or "wins a corner" in ev or "earns a corner" in ev:
            s["Corner Kicks"] += 1

        # Free kicks (only when awarded)
        if "earns a free kick" in ev or "wins a free kick" in ev:
            s["Free Kicks"] += 1

        # Offsides
        if "offside" in ev:
            s["Offsides"] += 1

    return stats

def write_csv_for_match(doc_id, stats, out_dir):
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
    # Always write two rows: Away then Home (T3 convention)
    team_order = ["Away Team", "Home Team"]

    path = os.path.join(out_dir, f"{doc_id}.csv")
    with open(path, "w", encoding="utf-8") as w:
        w.write(",".join(header) + "\n")
        for team in team_order:
            s = stats.get(team, {})
            row = [
                team,
                str(s.get("Goals", 0)),
                str(s.get("Shots", 0)),
                str(s.get("Fouls", 0)),
                str(s.get("Yellow Cards", 0)),
                str(s.get("Red Cards", 0)),
                str(s.get("Corner Kicks", 0)),
                str(s.get("Free Kicks", 0)),
                str(s.get("Offsides", 0)),
            ]
            w.write(",".join(row) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_test", required=True, help="data/livesum/test.json")
    ap.add_argument("--pred_jsonl", required=True, help="tables/livesum_sft2_vllm_pi2.jsonl")
    ap.add_argument("--csv_dir", required=True, help="output dir for per-match CSVs")
    args = ap.parse_args()

    os.makedirs(args.csv_dir, exist_ok=True)

    # Load SFT2 predictions (one record per doc_id, as in your sample)
    by_doc = {}
    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            doc_id = rec.get("doc_id")
            if doc_id is None:
                continue
            rows = rec.get("rows") or []
            by_doc[doc_id] = rows

    # Use gold_test to define ordering / doc_ids
    with open(args.gold_test, "r", encoding="utf-8") as f:
        gold = json.load(f)

    for ex in gold:
        doc_id = ex["id"]
        rows = by_doc.get(doc_id, [])
        stats = aggregate_match(rows)
        write_csv_for_match(doc_id, stats, args.csv_dir)

    print(f"Wrote CSVs to {args.csv_dir}")

if __name__ == "__main__":
    main()
