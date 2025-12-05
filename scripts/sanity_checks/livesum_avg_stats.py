#!/usr/bin/env python
import argparse
import csv
import json
import os
from collections import defaultdict, Counter
from statistics import mean


STATS = [
    "Goals",
    "Shots",
    "Fouls",
    "Yellow Cards",
    "Red Cards",
    "Corner Kicks",
    "Free Kicks",
    "Offsides",
]


def parse_gold_table(table_str):
    """
    Gold 'table' field in test.json is a CSV-like string
    with '<NEWLINE>' between rows.
    Returns (header, rows) where rows is list of dicts.
    """
    lines = table_str.split("<NEWLINE>")
    lines = [ln.strip() for ln in lines if ln.strip()]
    reader = csv.reader(lines)
    rows = list(reader)
    if not rows:
        return [], []
    header = rows[0]
    data_rows = rows[1:]
    dict_rows = []
    for r in data_rows:
        d = {}
        for h, v in zip(header, r):
            d[h] = v
        dict_rows.append(d)
    return header, dict_rows


def parse_pred_csv(path):
    if not os.path.exists(path):
        return [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    header = reader.fieldnames or []
    return header, rows


def to_int(x):
    try:
        return int(x)
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gold_test",
        required=True,
        help="data/livesum/test.json",
    )
    ap.add_argument(
        "--csv_dir",
        required=True,
        help="outputs/livesum_t3_pi2_csv",
    )
    args = ap.parse_args()

    with open(args.gold_test, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    gold_sums = Counter()
    pred_sums = Counter()
    gold_cells = Counter()
    pred_cells = Counter()
    gold_zero_cells = Counter()
    pred_zero_cells = Counter()

    missing_pred = 0

    for ex in gold_data:
        doc_id = ex["id"]
        gold_header, gold_rows = parse_gold_table(ex["table"])

        # Gold stats
        for row in gold_rows:
            for stat in STATS:
                if stat not in gold_header:
                    continue
                v = to_int(row.get(stat, 0))
                gold_sums[stat] += v
                gold_cells[stat] += 1
                if v == 0:
                    gold_zero_cells[stat] += 1

        # Pred stats
        pred_path = os.path.join(args.csv_dir, f"{doc_id}.csv")
        if not os.path.exists(pred_path):
            missing_pred += 1
            continue

        pred_header, pred_rows = parse_pred_csv(pred_path)
        for row in pred_rows:
            for stat in STATS:
                if stat not in pred_header:
                    continue
                v = to_int(row.get(stat, 0))
                pred_sums[stat] += v
                pred_cells[stat] += 1
                if v == 0:
                    pred_zero_cells[stat] += 1

    print(f"Matches in gold: {len(gold_data)}")
    print(f"Pred CSVs missing: {missing_pred}")
    print()

    print("=== Gold vs Predicted averages per TEAM cell ===")
    print("Stat\tGold_avg\tPred_avg\tPred/Gold\tGold_zero%\tPred_zero%")
    for stat in STATS:
        g_cells = gold_cells[stat] or 1
        p_cells = pred_cells[stat] or 1
        g_avg = gold_sums[stat] / g_cells
        p_avg = pred_sums[stat] / p_cells
        ratio = p_avg / g_avg if g_avg > 1e-6 else 0.0
        g_zero = gold_zero_cells[stat] / g_cells * 100.0
        p_zero = pred_zero_cells[stat] / p_cells * 100.0
        print(
            f"{stat}\t"
            f"{g_avg:.2f}\t"
            f"{p_avg:.2f}\t"
            f"{ratio:.2f}\t"
            f"{g_zero:.1f}\t"
            f"{p_zero:.1f}"
        )


if __name__ == "__main__":
    main()
