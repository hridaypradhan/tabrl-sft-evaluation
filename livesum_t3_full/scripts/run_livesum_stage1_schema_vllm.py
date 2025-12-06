#!/usr/bin/env python
import os
import sys
import json
import argparse
import pathlib
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

HF_MODEL_SFT1 = "mohdusman001/schema-gen-llama3-8b-stage1-merged"
BASE_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """
You design JSON table schemas for structured tables extracted from documents.

Return EXACTLY ONE JSON object with the following keys:
- "table_id": a short string identifying the table.
- "columns": a list of column objects.

Each column object MUST have:
- "name": the column name as a string.
- "type": one of ["string", "integer", "number"].

Do NOT include any other top-level keys.
Do NOT include markdown, code fences, comments, or explanations.
Output must be valid JSON.
Stop immediately after the final closing '}' of the JSON object.

The document is live text commentary for a single soccer match.
Choose columns that would allow summarizing important team- or match-level statistics
(e.g., team names, goals, cards, shots, fouls, corners, offsides, etc.).
Use at most 12 columns.
Only include columns that could reasonably be filled from the document.
"""

# ---------- JSON extraction helpers ----------

def first_balanced_json_span(s: str) -> Optional[Tuple[int, int]]:
    """Find first {...} span with balanced braces (ignores braces inside strings)."""
    in_str = False
    esc = False
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    return start, i + 1
    return None


def extract_json_object(generated: str) -> Optional[Dict[str, Any]]:
    """Try to extract the first top-level JSON object from the text."""
    t = generated.strip()

    # Strip code fences if the model ignores instructions
    if t.startswith("```"):
        t = t.split("```", 1)[-1]
    if "```" in t:
        t = t.split("```", 1)[0]

    span = first_balanced_json_span(t)
    if not span:
        return None
    a, b = span
    cand = t[a:b]
    try:
        return json.loads(cand)
    except Exception:
        return None

# ---------- data loading ----------

def load_livesum(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for ex in data:
        if "id" not in ex or "text" not in ex:
            continue
        rows.append(ex)
    if not rows:
        raise ValueError(f"No usable examples found in {path}")
    return rows

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_id", default=HF_MODEL_SFT1)
    ap.add_argument("--base_tokenizer", default=BASE_TOKENIZER)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--tp_size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token

    llm = LLM(
        model=args.model_id,
        tokenizer=args.base_tokenizer,
        dtype=args.dtype,
        tensor_parallel_size=args.tp_size,
        tokenizer_mode="auto",
    )

    data = load_livesum(args.test_file)
    if args.limit and args.limit > 0:
        data = data[: args.limit]

    prompts: List[str] = []
    ids: List[Any] = []

    for ex in data:
        doc_id = ex["id"]
        text = str(ex["text"]).strip()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        try:
            chat_str = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            chat_str = (
                f"<s>[SYSTEM]\n{SYSTEM_PROMPT}\n[/SYSTEM]\n"
                f"[USER]\n{text}\n[/USER]\n[ASSISTANT]"
            )

        prompts.append(chat_str)
        ids.append(doc_id)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    print(
        f"[info] running SFT1 schema induction on {len(prompts)} examples",
        file=sys.stderr,
    )
    outputs = llm.generate(prompts, sampling_params)

    n_ok = 0
    n_fail = 0

    for doc_id, out in zip(ids, outputs):
        gen = out.outputs[0].text
        schema_obj = extract_json_object(gen)

        if schema_obj is None or not isinstance(schema_obj, dict):
            n_fail += 1
            raw_path = out_dir / f"{doc_id}.raw_schema.txt"
            raw_path.write_text(gen, encoding="utf-8")
            print(
                f"[warn] doc {doc_id}: failed to parse JSON; saving raw only",
                file=sys.stderr,
            )
            continue

        # light validation + normalization (no hardcoded columns)
        table_id = schema_obj.get("table_id", str(doc_id))
        cols_in = schema_obj.get("columns", [])
        columns = []
        if isinstance(cols_in, list):
            for c in cols_in:
                if not isinstance(c, dict):
                    continue
                name = c.get("name")
                ctype = c.get("type", "string")
                if not name or not isinstance(name, str):
                    continue
                if ctype not in ("string", "integer", "number"):
                    ctype = "string"
                columns.append({"name": name, "type": ctype})

        if not columns:
            n_fail += 1
            raw_path = out_dir / f"{doc_id}.raw_schema.txt"
            raw_path.write_text(gen, encoding="utf-8")
            print(
                f"[warn] doc {doc_id}: no usable columns; saving raw only",
                file=sys.stderr,
            )
            continue

        final_schema = {
            "table_id": table_id,
            "columns": columns,
        }

        out_path = out_dir / f"{doc_id}.schema.json"
        out_path.write_text(
            json.dumps(final_schema, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        n_ok += 1

    print(
        f"[info] SFT1 done. Parsed {n_ok} schemas, {n_fail} failures.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
