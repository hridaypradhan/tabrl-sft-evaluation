import os, json, argparse, pathlib
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from vllm import LLM, SamplingParams

HF_MODEL_SFT2 = "mohdusman001/pi2-table-llama3-8b-sft_final"
BASE_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B-Instruct"

STRICT_POLICY = (
    "[POLICY]\n"
    "- Extract only facts explicitly supported by the document. No guessing.\n"
    "- Never invent rows or columns. If a value is not present, output \"\".\n"
    "- Output exactly the columns listed in [SCHEMA], in the same order, as raw JSON Lines.\n"
    "- Emit ONLY JSONL (one JSON object per row). No arrays, comments, or markdown.\n"
)

def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--schema_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--tp_size", type=int, default=1)
    args = ap.parse_args()

    pathlib.Path(os.path.dirname(args.out_jsonl)).mkdir(parents=True, exist_ok=True)

    # Use SFT2 weights + base tokenizer
    llm = LLM(
        model=HF_MODEL_SFT2,
        tokenizer=BASE_TOKENIZER,
        dtype=args.dtype,
        tensor_parallel_size=args.tp_size,
        tokenizer_mode="auto",
    )

    data = list(stream_jsonl(args.test_file))
    prompts = []
    ids = []

    for ex in data:
        doc_id = ex["id"]
        text = ex["document"]
        schema_path = os.path.join(args.schema_dir, f"{doc_id}.schema.json")
        if not os.path.exists(schema_path):
            continue
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = f.read().strip()

        prompt = (
            f"<|policy|>\n{STRICT_POLICY}\n"
            f"<|schema|>\n{schema}\n"
            f"<|document|>\n{text}\n"
            f"<|output_format|>\nEmit ONLY JSON Lines (JSONL), one JSON object per row.\n"
            f"<|task|>\nFill the table strictly under [SCHEMA]."
        )

        prompts.append(prompt)
        ids.append(doc_id)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    outputs = llm.generate(prompts, sampling_params)

    with open(args.out_jsonl, "w", encoding="utf-8") as w:
        for doc_id, out in zip(ids, outputs):
            gen = out.outputs[0].text.strip()
            for line in gen.splitlines():
                if line.strip():
                    w.write(line.rstrip() + "\n")

if __name__ == "__main__":
    main()
