# LiveSum T3 Experiments

This README describes how to run a 4-experiment benchmark on the LiveSum dataset using:

1. **Meta-Llama-3.1-8B-Instruct (zero-shot)**
2. **Meta-Llama-3.1-8B-Instruct with T3-style prompting**
3. **Two-step SFT pipeline (SFT1 + SFT2) in zero-shot mode**
4. **Two-step SFT pipeline with T3-style prompting in SFT2**

All experiments use the **same evaluation script** (`evaluate.py` from the LiveSum repo) and **the same test set** (`test.json`).

---

## 1. Directory Layout

Assume the project root looks like:

```text
TABLE_GENERATION_BENCH/
  data/
    livesum/
      test.json                # LiveSum test file (gold tables + commentary)
  livesum_t3_full/
    eval/
      evaluate.py              # LiveSum evaluate.py
    jsonl/
      sft_pipeline_zero_shot_pi2.jsonl   # (created by SFT2 zero-shot)
      sft_pipeline_t3_pi2.jsonl          # (created by SFT2 T3)
    outputs/
      llama31_zero_shot/       # CSVs for Exp.1
      llama31_t3/              # CSVs for Exp.2
      sft_pipeline_zero_shot/  # CSVs for Exp.3
      sft_pipeline_t3/         # CSVs for Exp.4
    schemas_sft1/
      <id>.schema.json         # SFT1 schemas (created once, reused)
    scripts/
      run_livesum_llama31_zero_shot.py
      run_livesum_llama31_t3.py
      run_livesum_stage1_schema_vllm.py
      run_livesum_stage2_table_vllm.py
      run_livesum_stage2_table_vllm_t3.py
      livesum_pi2_to_t3_csv.py
