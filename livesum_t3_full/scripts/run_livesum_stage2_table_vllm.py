#!/usr/bin/env python
import os
import sys
import json
import argparse
import pathlib
from typing import Any, Dict, List

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from vllm import LLM, SamplingParams

HF_MODEL_SFT2 = "mohdusman001/pi2-table-llama3-8b-sft_final"
BASE_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Livesum-specific but still general soccer policy.
STRICT_POLICY = (
    "[POLICY]\n"
    "- Extract only facts explicitly supported by the document. No guessing.\n"
    "- Each row must describe ONE on-pitch soccer event for ONE team "
    "(shot, goal, foul, card, corner, free kick, penalty, offside, substitution, etc.).\n"
    "- Read the document in order and aim for HIGH COVERAGE: if the text "
    "describes an on-pitch action, create a row for it. It is OK to output dozens "
    "of rows for one match.\n"
    "- Never merge multiple events into a single row.\n"
    "- If the schema has a 'team' column, fill it with the team involved in the event.\n"
    "- If the schema has an 'event' or 'description' column, write a short natural "
    "language description of the event.\n"
    "- If the schema has a 'type' column, use a short label such as:\n"
    "  'shot', 'saved_shot', 'blocked_shot', 'missed_shot', 'header',\n"
    "  'goal',\n"
    "  'foul', 'handball', 'dangerous_play',\n"
    "  'yellow_card', 'red_card', 'second_yellow',\n"
    "  'corner_kick',\n"
    "  'free_kick', 'penalty',\n"
    "  'offside',\n"
    "  'substitution', 'injury', 'other'.\n"
    "  Pick the closest label; avoid inventing new ones unless none fit.\n"
    "- For integer/number columns whose names look like event counters, treat them "
    "as 0/1 FLAGS per event:\n"
    "  * Names containing 'goal'          -> 1 if this row is a goal by the team, else 0.\n"
    "  * Names containing 'shot'/'attempt'-> 1 if this row is a shot/attempt, else 0.\n"
    "  * Names containing 'foul'/'handball'/'dangerous' -> 1 if this row is a foul/handball "
    "     or dangerous play, else 0.\n"
    "  * Names containing 'yellow'        -> 1 if this row gives a yellow card, else 0.\n"
    "  * Names containing 'red'           -> 1 if this row gives a red card (incl. second "
    "     yellow), else 0.\n"
    "  * Names containing 'corner'        -> 1 if this row is a corner kick for the team, else 0.\n"
    "  * Names containing 'free_kick', 'free kick', 'fk', or 'penalty' -> 1 if this row is a free "
    "     kick or penalty for the team, else 0.\n"
    "  * Names containing 'offside'       -> 1 if this row is an offside offence by the team, else 0.\n"
    "- For other integer/number columns, use 1 if the event clearly happens, else 0; "
    "leave as 0 if uncertain.\n"
    "- Use empty string \"\" for unknown string values.\n"
    "- Use 0 for unknown integer values.\n"
    "- Never invent columns. Use exactly the columns listed in [SCHEMA], in the same order.\n"
    "- Emit ONLY JSON Lines (JSONL): one JSON object per row. No arrays, comments, or markdown.\n"
)


def parse_jsonl_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Strict JSONL parser: one JSON object per line.
    Anything else is ignored.
    """
    rows: List[Dict[str, Any]] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        # Strip accidental code fences
        if ln.startswith("```"):
            ln = ln.strip("`")
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def build_table_from_rows(schema_obj: Dict[str, Any], rows: List[Dict[str, Any]]):
    if not rows:
        return None
    columns = schema_obj.get("columns", [])
    col_names = [c.get("name") for c in columns]
    data: List[List[Any]] = []
    for r in rows:
        row_vals = [r.get(name, "") for name in col_names]
        data.append(row_vals)
    table = {
        "table_id": schema_obj.get("table_id"),
        "columns": columns,
        "data": data,
    }
    return table


def load_livesum_with_schemas(test_file: str, schema_dir: str):
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for ex in data:
        if "id" not in ex or "text" not in ex:
            continue
        doc_id = ex["id"]
        schema_path = os.path.join(schema_dir, f"{doc_id}.schema.json")
        if not os.path.exists(schema_path):
            print(
                f"[warn] missing schema for doc_id={doc_id} at {schema_path}",
                file=sys.stderr,
            )
            continue
        with open(schema_path, "r", encoding="utf-8") as sf:
            schema_str = sf.read().strip()
        samples.append(
            {"id": doc_id, "text": ex["text"], "schema_str": schema_str}
        )
    if not samples:
        raise ValueError("No samples with schemas found.")
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", required=True, help="../data/livesum/test.json")
    ap.add_argument("--schema_dir", required=True, help="schemas_sft1")
    ap.add_argument(
        "--out_jsonl",
        required=True,
        help="jsonl/sft_pipeline_zero_shot_pi2.jsonl",
    )
    ap.add_argument("--model_id", default=HF_MODEL_SFT2)
    ap.add_argument("--base_tokenizer", default=BASE_TOKENIZER)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tp_size", type=int, default=1)
    args = ap.parse_args()

    out_path = pathlib.Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=args.model_id,
        tokenizer=args.base_tokenizer,
        dtype=args.dtype,
        tensor_parallel_size=args.tp_size,
        tokenizer_mode="auto",
    )

    samples = load_livesum_with_schemas(args.test_file, args.schema_dir)

    prompts: List[str] = []
    ids: List[Any] = []
    texts: List[str] = []
    schema_strs: List[str] = []

    for s in samples:
        doc_id = s["id"]
        text = str(s["text"])
        schema_str = s["schema_str"]

        prompt = (
            f"<|policy|>\n{STRICT_POLICY}\n"
            f"<|schema|>\n{schema_str}\n"
            f"<|document|>\n{text}\n"
            f"<|output_format|>\nEmit ONLY JSON Lines (JSONL), one JSON object per row.\n"
            f"<|task|>\nFill the table strictly under [SCHEMA]."
        )

        prompts.append(prompt)
        ids.append(doc_id)
        texts.append(text)
        schema_strs.append(schema_str)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    print(
        f"[info] running SFT2 on {len(prompts)} LiveSum examples",
        file=sys.stderr,
    )
    outputs = llm.generate(prompts, sampling_params)

    results: List[Dict[str, Any]] = []
    for idx, (doc_id, text, schema_str, out) in enumerate(
        zip(ids, texts, schema_strs, outputs)
    ):
        gen_text = out.outputs[0].text.strip()
        rows = parse_jsonl_from_text(gen_text)

        try:
            schema_obj = json.loads(schema_str)
        except Exception:
            schema_obj = {"table_id": None, "columns": []}

        table_obj = build_table_from_rows(schema_obj, rows)

        rec = {
            "id": idx,
            "doc_id": doc_id,
            "text": text,
            "schema": schema_obj,
            "model_output": gen_text,
            "rows": rows,
            "table": table_obj,
        }
        results.append(rec)

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"[info] wrote {len(results)} records to {out_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
