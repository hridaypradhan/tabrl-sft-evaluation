# $PROJ/eval_tools/livesum_t3_eval.py
import os, json, csv, argparse
from pathlib import Path
from collections import defaultdict

COLUMNS = [
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

def load_livesum(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_pred_jsonl(path):
    preds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))
    return preds

def group_rows_by_match(pred_rows):
    by_match = defaultdict(list)
    for obj in pred_rows:
        doc_id = obj.get("__doc_id__")
        if doc_id is None:
            continue
        by_match[doc_id].append(obj)
    return by_match

def write_csvs(gold_data, grouped_rows, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for ex in gold_data:
        doc_id = ex["id"]
        rows = grouped_rows.get(doc_id, [])
        rows = [r for r in rows if isinstance(r, dict)]
        rows.sort(key=lambda r: str(r.get("Team", "")))
        rows = rows[:2]

        while len(rows) < 2:
            rows.append({})

        out_path = os.path.join(out_dir, f"{doc_id}.csv")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(COLUMNS)
            for r in rows:
                team = r.get("Team", "")
                vals = [team]
                for col in COLUMNS[1:]:
                    vals.append(r.get(col, ""))
                writer.writerow(vals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_test", required=True)   # data/livesum/test.json
    ap.add_argument("--pred_jsonl", required=True)  # tables/livesum_sft2_vllm.jsonl
    ap.add_argument("--csv_dir", required=True)     # outputs/livesum_t3_csv
    args = ap.parse_args()

    gold_data = load_livesum(args.gold_test)
    pred_rows = load_pred_jsonl(args.pred_jsonl)
    grouped = group_rows_by_match(pred_rows)
    write_csvs(gold_data, grouped, args.csv_dir)

if __name__ == "__main__":
    main()
