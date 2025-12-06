# LiveSum T3 Experiments – Full README

This README describes how to run a **4-experiment benchmark** on the LiveSum dataset, combining:

- A **base instruction-tuned model** (Meta-Llama-3.1-8B-Instruct),
- Your **two-step SFT pipeline** (SFT1 schema + SFT2 table rows),
- And **T3-style prompting** inspired by the T3 paper (text → tuples → integration → table).

All experiments:

- Use the **same gold data**: `test.json` from the LiveSum dataset.
- Produce **per-match CSV tables** with the same columns.
- Are evaluated with the **same LiveSum `evaluate.py` script**.

---

## 0. The Four Experiments (High-Level)

We run four experiment conditions:

1. **Exp 1 – Llama-3.1-8B zero-shot (direct table prediction)**  
   - Base model only, no extra fine-tuning.  
   - A single instruction describes the 2-row (Away/Home) table and the 8 columns.  
   - Model directly outputs the final table; we convert to CSV and evaluate.

2. **Exp 2 – Llama-3.8-8B with T3-style prompting**  
   - Same base model, but the prompt follows **T3-MERGED** style:  
     “Extract events → integrate them → output table” in a single call.  
   - Still outputs a 2-row CSV table, but with more explicit reasoning guidance.

3. **Exp 3 – Two-step SFT pipeline, zero-shot**  
   - **SFT1** infers a PI2-style **schema** per match from the commentary.  
   - **SFT2** fills that schema with **event-level rows** (JSONL) using its original prompt.  
   - A Python script aggregates the event rows into final 2-row LiveSum tables (CSV).  
   - No T3-specific reasoning in the prompt.

4. **Exp 4 – Two-step SFT pipeline with T3-style prompting**  
   - Same SFT1 schemas as Exp 3.  
   - SFT2 receives a **T3-style task description** (LiveSum instruction + “identify tuples, integrate, then output rows”).  
   - Output is still PI2-style JSONL rows, aggregated by the same Python script into CSV tables.

All four end in **identical CSV format**, making the metrics directly comparable.

---

## 1. Directory Layout

Assume the project root looks like:

TABLE_GENERATION_BENCH/
  data/
    livesum/
      test.json                # LiveSum test file (gold tables + commentary)
  livesum_t3_full/
    eval/
      evaluate.py              # LiveSum evaluate.py
    jsonl/
      sft_pipeline_zero_shot_pi2.jsonl   # (created by SFT2 zero-shot)
      sft_pipeline_t3_pi2.jsonl          # (created by SFT2 T3-style)
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

### Example Inputs and Outputs

- Input: `data/livesum/test.json`  
  Contains a list of objects like:

  {
    "id": 34160708,
    "text": "Full live commentary for the match...",
    "table": "Team,Goals,Shots,...<NEWLINE>Away Team,1,8,...<NEWLINE>Home Team,0,6,..."
  }

- Per-match CSV output (for any experiment):  
  `livesum_t3_full/outputs/<experiment_name>/<id>.csv`

  Team,Goals,Shots,Fouls,Yellow Cards,Red Cards,Corner Kicks,Free Kicks,Offsides
  Away Team,1,8,10,2,0,4,3,1
  Home Team,0,6,12,3,1,5,4,0

Every experiment ultimately produces one such CSV per match; `evaluate.py` uses these to compute scores.

---

## 2. Environment Setup

From the project root:

cd TABLE_GENERATION_BENCH
# Activate your Python environment (example)
source .venv/bin/activate   # or: conda activate <env>

All experiment commands below are run from inside `livesum_t3_full`:

cd livesum_t3_full

Hardware assumption: an A100 GPU with vLLM installed.  
The scripts assume `--tp_size 1` and `--dtype bfloat16`, which is appropriate for a single A100.

---

## 3. Experiment 1 – Llama-3.1-8B Zero-Shot (Direct Table Prediction)

Goal: Use base Meta-Llama-3.1-8B-Instruct with a straightforward instruction to directly predict the final two-row (Away/Home) table, without explicit T3 reasoning or fine-tuning.

### 3.1 Generate Predictions (CSV per Match)

python scripts/run_livesum_llama31_zero_shot.py \
  --test_file ../data/livesum/test.json \
  --output_dir outputs/llama31_zero_shot \
  --tp_size 1 \
  --dtype bfloat16

What this does:  
Runs the base model in zero-shot mode, with a single task instruction that describes the desired table.  
It parses the model’s JSON output and writes one `<id>.csv` per match to `outputs/llama31_zero_shot`.

### 3.2 Evaluate

python eval/evaluate.py \
  --data   ../data/livesum \
  --output outputs/llama31_zero_shot

What this does:  
Reads `test.json` as gold, reads all CSVs from `outputs/llama31_zero_shot`, and prints LiveSum metrics (Easy/Medium/Hard + AVG).

---

## 4. Experiment 2 – Llama-3.1-8B with T3-Style Prompting

Goal: Use the same base model, but now with a T3-MERGED style prompt: we explicitly tell the model to (internally) extract event tuples, integrate them according to the LiveSum rules, and then output the final table.

### 4.1 Generate T3-Style Predictions

python scripts/run_livesum_llama31_t3.py \
  --test_file ../data/livesum/test.json \
  --output_dir outputs/llama31_t3 \
  --tp_size 1 \
  --dtype bfloat16

What this does:  
Adds the T3 LiveSum instruction and the “Text–Tuple–Table” steps into the prompt.  
The model still outputs a 2-row table, which is saved as `<id>.csv` in `outputs/llama31_t3`.

### 4.2 Evaluate

python eval/evaluate.py \
  --data   ../data/livesum \
  --output outputs/llama31_t3

What this does:  
Evaluates the T3-prompted base model’s tables against the same gold tables as Experiment 1.

---

## 5. Experiment 3 – Two-Step SFT Pipeline, Zero-Shot

Goal: Use your SFT1 + SFT2 pipeline with its original prompts (no explicit T3 reasoning). This pipeline is trained on PI2-style supervision but used here in a zero-shot way (no in-context examples or T3 instructions on the test set).

Models:

- SFT1: `mohdusman001/schema-gen-llama3-8b-stage1-merged`  
  (learned to propose JSON table schemas).
- SFT2: `mohdusman001/pi2-table-llama3-8b-sft_final`  
  (learned to fill event-level rows under a given schema).

### 5.1 Stage 1 – Schema Induction (SFT1)

python scripts/run_livesum_stage1_schema_vllm.py \
  --test_file ../data/livesum/test.json \
  --out_dir  schemas_sft1 \
  --dtype bfloat16 \
  --tp_size 1

What this does:  
For each match in `test.json`, runs SFT1 to infer a PI2-style JSON schema (team, event, flags, etc.).  
Stores them as `schemas_sft1/<id>.schema.json`. These schemas are reused by Experiments 3 and 4.

### 5.2 Stage 2 – Event Rows (PI2) with SFT2 (Zero-Shot Prompt)

python scripts/run_livesum_stage2_table_vllm.py \
  --test_file  ../data/livesum/test.json \
  --schema_dir schemas_sft1 \
  --out_jsonl  jsonl/sft_pipeline_zero_shot_pi2.jsonl \
  --dtype bfloat16 \
  --tp_size 1

What this does:  
For each match, uses SFT2 with the original “policy + schema + document” prompt to generate per-event JSONL rows.  
All records (one per match) are written to `jsonl/sft_pipeline_zero_shot_pi2.jsonl` (each with a `rows` list and a reconstructed `table` object).

### 5.3 Aggregate PI2 Rows → LiveSum Tables (CSV)

python scripts/livesum_pi2_to_t3_csv.py \
  --gold_test ../data/livesum/test.json \
  --pred_jsonl jsonl/sft_pipeline_zero_shot_pi2.jsonl \
  --csv_dir   outputs/sft_pipeline_zero_shot

What this does:  
Reads the PI2 rows for each match, aggregates them into per-team counts for:  
goals, shots, fouls, yellow cards, red cards, corner kicks, free kicks, offsides.  
Writes one `<id>.csv` per match into `outputs/sft_pipeline_zero_shot`, in the exact format expected by `evaluate.py`.

### 5.4 Evaluate

python eval/evaluate.py \
  --data   ../data/livesum \
  --output outputs/sft_pipeline_zero_shot

What this does:  
Evaluates the zero-shot SFT pipeline against gold tables, using the same metrics as in Experiments 1 and 2.

---

## 6. Experiment 4 – Two-Step SFT Pipeline with T3-Style Prompting

Goal: Use the same SFT1 schemas as Experiment 3, but run SFT2 with a T3-style task description:

- Insert the LiveSum T3 instruction (how to count goals/shots/etc.).
- Explicitly tell SFT2 to internally:
  1. Identify event tuples `(player, team, event)` or `(team, event)`.
  2. Integrate them using the LiveSum rules.
  3. Output PI2 rows under the given schema.

The output format (JSONL rows) is unchanged, so aggregation and evaluation remain identical.

### 6.1 (Optional) Regenerate Schemas with SFT1

If you already ran Experiment 3, `schemas_sft1/` is ready and can be reused.  
If needed, regenerate with the same command as in 5.1.

### 6.2 Stage 2 – Event Rows (PI2) with SFT2 + T3 Prompt

python scripts/run_livesum_stage2_table_vllm_t3.py \
  --test_file  ../data/livesum/test.json \
  --schema_dir schemas_sft1 \
  --out_jsonl  jsonl/sft_pipeline_t3_pi2.jsonl \
  --dtype bfloat16 \
  --tp_size 1

What this does:  
Uses SFT2 with a T3-style prompt: same policy and schema, but with the LiveSum T3 instruction and text→tuple→integration guidance in the `<|task|>` segment.  
Outputs PI2 JSONL rows for each match into `jsonl/sft_pipeline_t3_pi2.jsonl`.

### 6.3 Aggregate PI2 Rows → LiveSum Tables (CSV)

python scripts/livesum_pi2_to_t3_csv.py \
  --gold_test ../data/livesum/test.json \
  --pred_jsonl jsonl/sft_pipeline_t3_pi2.jsonl \
  --csv_dir   outputs/sft_pipeline_t3

What this does:  
Aggregates the T3-prompted PI2 rows into final `<id>.csv` tables in `outputs/sft_pipeline_t3`, identical in format to other experiments.

### 6.4 Evaluate

python eval/evaluate.py \
  --data   ../data/livesum \
  --output outputs/sft_pipeline_t3

What this does:  
Evaluates the T3-prompted SFT pipeline tables against the gold tables.

---

## 7. Comparing and Reporting Results

For each experiment, `evaluate.py` prints a single summary row (the mean over all matches), including:

- `Easy-RMSE`, `Easy-MAE`, `Easy-EM`
- `Medium-RMSE`, `Medium-MAE`, `Medium-EM`
- `Hard-RMSE`, `Hard-MAE`, `Hard-EM`
- `AVG-RMSE`, `AVG-MAE`, `AVG-EM`

A suggested tracking table:

Experiment ID | Description                             | Output Folder
------------- | --------------------------------------- | -------------------------------
1             | Llama-3.1-8B zero-shot                  | outputs/llama31_zero_shot
2             | Llama-3.1-8B + T3 prompting             | outputs/llama31_t3
3             | SFT pipeline (SFT1+SFT2) zero-shot      | outputs/sft_pipeline_zero_shot
4             | SFT pipeline (SFT1+SFT2) + T3 prompting | outputs/sft_pipeline_t3

How to Interpret the Comparisons:

- 1 vs 2: Effect of T3 prompting on the base model.  
- 3 vs 4: Effect of T3 prompting on the SFT pipeline, holding the SFT models and schema fixed.  
- 1 vs 3: Effect of fine-tuning (SFT) vs a purely base model, under similar non-T3 prompting.  
- 2 vs 4: Interaction of SFT + T3 prompting vs base + T3 prompting.

Running all four gives you a clean grid of results to discuss both model training (SFT vs base) and prompting strategy (direct vs T3-style) on the LiveSum task.
