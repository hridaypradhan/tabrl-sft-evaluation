#!/usr/bin/env python
import argparse
import json
from collections import Counter, defaultdict
import math
from statistics import mean, median


TYPICAL_NUMERIC_KEYS = {
    "goal", "goals",
    "shot", "shots",
    "foul", "fouls",
    "yellow_card", "yellow_cards",
    "red_card", "red_cards",
    "corner", "corners", "corner_kick", "corner_kicks",
    "free_kick", "free_kicks",
    "offside", "offsides",
}


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pred_jsonl",
        required=True,
        help="tables/livesum_sft2_vllm_pi2.jsonl",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional: only process first N records",
    )
    args = ap.parse_args()

    row_counts = []
    numeric_counts = Counter()       # how many rows have >0 for each key variant
    numeric_totals = Counter()       # total sum over all rows for each key
    docs_seen = 0

    with open(args.pred_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows = rec.get("rows") or []
            rows = [r for r in rows if isinstance(r, dict)]
            row_counts.append(len(rows))
            docs_seen += 1

            for r in rows:
                for k, v in r.items():
                    kl = str(k).lower()
                    if kl not in TYPICAL_NUMERIC_KEYS:
                        continue
                    if not is_number(v):
                        continue
                    v_int = int(v)
                    if v_int > 0:
                        numeric_counts[kl] += 1
                        numeric_totals[kl] += v_int

            if args.limit and docs_seen >= args.limit:
                break

    if not row_counts:
        print("No rows found in predictions.")
        return

    row_counts_sorted = sorted(row_counts)
    n = len(row_counts_sorted)

    def pct(p):
        idx = int(math.floor(p * (n - 1)))
        return row_counts_sorted[idx]

    print("=== Row counts per match ===")
    print(f"Matches processed: {n}")
    print(f"min rows/match: {row_counts_sorted[0]}")
    print(f"max rows/match: {row_counts_sorted[-1]}")
    print(f"mean rows/match: {mean(row_counts_sorted):.2f}")
    print(f"median rows/match: {median(row_counts_sorted):.2f}")
    print(f"25th percentile: {pct(0.25)}")
    print(f"75th percentile: {pct(0.75)}")
    print()

    print("=== Numeric flag stats (over ALL rows) ===")
    if not numeric_counts:
        print("No typical numeric keys (goal/foul/card/etc.) found.")
    else:
        for k in sorted(numeric_counts.keys()):
            print(
                f"{k}: rows_with>0={numeric_counts[k]}, "
                f"total_sum={numeric_totals[k]}"
            )


if __name__ == "__main__":
    main()
