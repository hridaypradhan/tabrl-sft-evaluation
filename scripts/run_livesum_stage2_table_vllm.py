#!/usr/bin/env python
import os
import sys
import json
import argparse
import pathlib
from typing import Any, Dict, List

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# --------- CONFIG ---------
HF_MODEL_SFT2 = "mohdusman001/pi2-table-llama3-8b-sft_final"
BASE_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# --------- Prompt building (from infer_and_upload_pi2) --------- :contentReference[oaicite:3]{index=3}

def build_user_prompt(text: str, schema: Any) -> str:
    """
    Build the user message for pi2-style table extraction.

    We instruct:
    - Given a JSON schema and a document,
      extract rows matching the schema.
    - Output JSON Lines: one JSON object per row.
    """
    schema_str = json.dumps(schema, ensure_ascii=False, indent=2)
    instruction = (
        "You are a table extraction model. Given a JSON schema and a document, "
        "you must extract rows that match the schema.\n"
        "Return the table as JSON Lines (JSONL): one valid JSON object per line, "
        "with keys exactly equal to the column names in the schema.\n"
        "Do NOT output any explanations, natural language, markdown, or code — "
        "only the JSONL table rows.\n\n"
    )
    prompt = (
        instruction
        + "[SCHEMA]\n"
        + schema_str
        + "\n<|document|>\n"
        + str(text)
    )
    return prompt


def make_messages_for_chat(text: str, schema: Any) -> List[Dict[str, str]]:
    user_prompt = build_user_prompt(text, schema)
    return [{"role": "user", "content": user_prompt}]


# --------- Output parsing (from infer_and_upload_pi2) --------- :contentReference[oaicite:4]{index=4}

def parse_jsonl_from_text(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except Exception:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def build_table_from_rows(schema: Dict[str, Any], rows: List[Dict[str, Any]]):
    if not rows:
        return None
    columns = schema.get("columns", [])
    col_names = [c.get("name") for c in columns]
    data: List[List[Any]] = []
    for r in rows:
        row_vals = [r.get(name, "") for name in col_names]
        data.append(row_vals)
    table = {
        "table_id": schema.get("table_id"),
        "columns": columns,
        "data": data,
    }
    return table


# --------- Data loading ---------

def load_livesum_with_schemas(test_file: str, schema_dir: str) -> List[Dict[str, Any]]:
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples: List[Dict[str, Any]] = []
    for ex in data:
        if "id" not in ex or "text" not in ex:
            continue
        doc_id = ex["id"]
        text = ex["text"]
        schema_path = os.path.join(schema_dir, f"{doc_id}.schema.json")
        if not os.path.exists(schema_path):
            print(f"[warn] missing schema for doc_id={doc_id} at {schema_path}", file=sys.stderr)
            continue
        with open(schema_path, "r", encoding="utf-8") as sf:
            schema = json.load(sf)
        samples.append({"id": doc_id, "text": text, "schema": schema})
    if not samples:
        raise ValueError("No samples with schemas found.")
    return samples


# --------- Inference ---------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", required=True, help="Livesum test.json")
    ap.add_argument("--schema_dir", required=True, help="Directory with <id>.schema.json from SFT1")
    ap.add_argument("--out_jsonl", required=True, help="Output JSONL file for SFT2 results")
    ap.add_argument("--model_id", default=HF_MODEL_SFT2)
    ap.add_argument("--base_tokenizer", default=BASE_TOKENIZER)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--tp_size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_path = pathlib.Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # tokenizer for chat template only
    tok = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # vLLM model
    llm = LLM(
        model=args.model_id,
        tokenizer=args.base_tokenizer,
        dtype=args.dtype,
        tensor_parallel_size=args.tp_size,
        tokenizer_mode="auto",
    )

    samples = load_livesum_with_schemas(args.test_file, args.schema_dir)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    prompts: List[str] = []
    ids: List[Any] = []
    texts: List[str] = []
    schemas: List[Dict[str, Any]] = []

    for ex in samples:
        doc_id = ex["id"]
        text = str(ex["text"])
        schema = ex["schema"]

        messages = make_messages_for_chat(text, schema)
        try:
            chat_str = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            user_prompt = messages[0]["content"]
            chat_str = f"user: {user_prompt}\nassistant: "
        prompts.append(chat_str)
        ids.append(doc_id)
        texts.append(text)
        schemas.append(schema)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    print(f"[info] running SFT2 table generation on {len(prompts)} examples", file=sys.stderr)
    outputs = llm.generate(prompts, sampling_params)

    # Results in the same rich format as infer_and_upload_pi2 :contentReference[oaicite:5]{index=5}
    results: List[Dict[str, Any]] = []
    for idx, (doc_id, text, schema, out) in enumerate(zip(ids, texts, schemas, outputs)):
        gen_text = out.outputs[0].text.strip()
        rows = parse_jsonl_from_text(gen_text)
        table_obj = build_table_from_rows(schema, rows)

        rec = {
            "id": idx,
            "doc_id": doc_id,
            "text": text,
            "schema": schema,
            "model_output": gen_text,
            "rows": rows,
            "table": table_obj,
        }
        results.append(rec)

    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[info] wrote {len(results)} records to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
