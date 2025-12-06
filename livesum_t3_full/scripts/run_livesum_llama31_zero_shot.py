import argparse
import json
import os
import re
from typing import Any, Dict, List

from vllm import LLM, SamplingParams

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """You are a precise soccer statistics assistant.

You are given the full live text commentary of ONE soccer match.
At the end of the match, you must fill a FINAL team statistics table.

The table has exactly two rows:
- Away Team
- Home Team

Columns (all non-negative integers):
- Goals
- Shots
- Fouls
- Yellow Cards
- Red Cards
- Corner Kicks
- Free Kicks
- Offsides

Rules:
- Count only what is implied by the commentary.
- If something is ambiguous, make your best single guess (no ranges).
- Always produce integers, no decimals.
"""

USER_TEMPLATE = """Commentary:
---------
{commentary}

Return ONLY a single JSON object with this exact structure:

{{
  "rows": [
    {{
      "Team": "Away Team",
      "Goals": <int>,
      "Shots": <int>,
      "Fouls": <int>,
      "Yellow Cards": <int>,
      "Red Cards": <int>,
      "Corner Kicks": <int>,
      "Free Kicks": <int>,
      "Offsides": <int>
    }},
    {{
      "Team": "Home Team",
      "Goals": <int>,
      "Shots": <int>,
      "Fouls": <int>,
      "Yellow Cards": <int>,
      "Red Cards": <int>,
      "Corner Kicks": <int>,
      "Free Kicks": <int>,
      "Offsides": <int>
    }}
  ]
}}

Do NOT include explanations, comments, or markdown, only valid JSON.
"""


def load_livesum_test(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(commentary: str) -> str:
    user = USER_TEMPLATE.format(commentary=commentary)
    # Chat template for Llama 3.1 Instruct
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}\n<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user}\n<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def extract_json(text: str) -> Dict[str, Any]:
    """Heuristic JSON extractor that tolerates code fences / extra text."""
    text = text.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        # Remove ``` and ```json style wrappers
        parts = re.split(r"```(?:json|JSON)?", text)
        text = "".join(parts).strip()

    # Grab first {...} block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"Could not find JSON object in: {text[:200]}...")
    obj_str = m.group(0)
    return json.loads(obj_str)


def rows_to_stats(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Convert model 'rows' JSON into a dict keyed by team, with ints."""
    stats: Dict[str, Dict[str, int]] = {}

    def to_int(v: Any) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    for row in rows:
        if not isinstance(row, dict):
            continue
        team = str(row.get("Team", "")).strip()
        if not team:
            continue
        stats[team] = {
            "Goals": to_int(row.get("Goals", 0)),
            "Shots": to_int(row.get("Shots", 0)),
            "Fouls": to_int(row.get("Fouls", 0)),
            "Yellow Cards": to_int(row.get("Yellow Cards", 0)),
            "Red Cards": to_int(row.get("Red Cards", 0)),
            "Corner Kicks": to_int(row.get("Corner Kicks", 0)),
            "Free Kicks": to_int(row.get("Free Kicks", 0)),
            "Offsides": to_int(row.get("Offsides", 0)),
        }
    return stats


def write_csv_for_match(doc_id: int, stats: Dict[str, Dict[str, int]], out_dir: str):
    """Write one CSV per match in the exact format expected by evaluate.py."""
    os.makedirs(out_dir, exist_ok=True)

    header = [
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
    team_order = ["Away Team", "Home Team"]  # T3 / LiveSum convention

    path = os.path.join(out_dir, f"{doc_id}.csv")
    with open(path, "w", encoding="utf-8") as w:
        w.write(",".join(header) + "\n")
        for team in team_order:
            s = stats.get(team, {})
            row = [
                team,
                str(s.get("Goals", 0)),
                str(s.get("Shots", 0)),
                str(s.get("Fouls", 0)),
                str(s.get("Yellow Cards", 0)),
                str(s.get("Red Cards", 0)),
                str(s.get("Corner Kicks", 0)),
                str(s.get("Free Kicks", 0)),
                str(s.get("Offsides", 0)),
            ]
            w.write(",".join(row) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--test_file",
        required=True,
        help="Path to LiveSum test.json (e.g., ../data/livesum/test.json)",
    )
    ap.add_argument(
        "--output_dir",
        required=True,
        help="Directory to write per-match CSVs",
    )
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--tp_size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    data = load_livesum_test(args.test_file)

    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=args.tp_size,
        dtype=args.dtype,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    prompts = [build_prompt(ex["text"]) for ex in data]
    generations = llm.generate(prompts, sampling, use_tqdm=True)

    for ex, gen in zip(data, generations):
        doc_id = ex["id"]
        raw = gen.outputs[0].text

        try:
            obj = extract_json(raw)
            rows = obj.get("rows", [])
            if not isinstance(rows, list):
                rows = []
        except Exception:
            rows = []

        stats = rows_to_stats(rows)
        write_csv_for_match(doc_id, stats, args.output_dir)


if __name__ == "__main__":
    main()
