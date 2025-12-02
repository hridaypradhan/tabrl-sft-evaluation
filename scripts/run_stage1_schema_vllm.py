import os, json, argparse, pathlib
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from vllm import LLM, SamplingParams

HF_MODEL_SFT1 = "mohdusman001/schema-gen-llama3-8b-stage1-merged"
BASE_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def stream_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--tp_size", type=int, default=1, help="tensor_parallel_size for vLLM")
    args = ap.parse_args()

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Use SFT1 weights but BASE_TOKENIZER for tokenization.
    llm = LLM(
        model=HF_MODEL_SFT1,
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
        prompt = f"{text}\n\nReturn only the JSON schema for the tables."
        prompts.append(prompt)
        ids.append(doc_id)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    outputs = llm.generate(prompts, sampling_params)

    for doc_id, out in zip(ids, outputs):
        gen = out.outputs[0].text
        # crude: take JSON from last '{' onwards if present
        start = gen.rfind("{")
        schema_json = gen[start:] if start != -1 else gen
        out_path = os.path.join(args.out_dir, f"{doc_id}.schema.json")
        with open(out_path, "w", encoding="utf-8") as w:
            w.write(schema_json.strip())

if __name__ == "__main__":
    main()
