import argparse
import json
import os
import re
from typing import Any, Dict, List

from vllm import LLM, SamplingParams

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """You are a precise football (soccer) statistics assistant.
You will read the full live commentary of ONE match and summarize team statistics.
Follow the task rules exactly and be very careful with counting.
"""

# LiveSum "Instruction on LIVESUM Dataset" (T3 Appendix B.1), slightly paraphrased.
LIVESUM_INSTRUCTION = """According to the live text, please count for each team the number of:
1. goals, 2. shots, 3. fouls, 4. yellow cards, 5. red cards,
6. corner kicks, 7. free kicks, and 8. offsides.

Notes:
- Goals, saved attempts, blocked attempts, and missed attempts are all considered shots.
- Handball and dangerous play are considered fouls.
- The second yellow card is also counted as a red card.
- Penalties are considered free kicks.
"""

# T3-MERGED-style prompt specialized for LiveSum and our evaluation format.
T3_MERGED_USER_TEMPLATE = """{instruction}

Let’s do the following things:

1. Extract all the relevant events from the following live commentary in
   (player name, team name, event) or (team name, event) format.
   Restrict the event names to only these options:
   goals, shots, fouls, yellow cards, red cards, corner kicks, free kicks, offsides.

2. Integrate these tuples to count, for EACH TEAM, the total number of each event type,
   consistent with the rules above.

3. At the end, output ONLY a CSV table with EXACTLY:
   - one header row
   - two data rows, one for Away Team and one for Home Team.

The header row MUST be:
Team,Goals,Shots,Fouls,Yellow Cards,Red Cards,Corner Kicks,Free Kicks,Offsides

The two data rows MUST be:
- one row starting with "Away Team," followed by 8 integers
- one row starting with "Home Team," followed by 8 integers

Do not output anything other than this final CSV table.

Live commentary:
----------------
{commentary}
"""


def load_livesum_test(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(commentary: str) -> str:
    user = T3_MERGED_USER_TEMPLATE.format(
        instruction=LIVESUM_INSTRUCTION,
        commentary=commentary,
    )
    # Llama 3.1 chat template
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}\n<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user}\n<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # Remove markdown fences like ``` or ```csv or ```json
        text = re.sub(r"```[a-zA-Z]*", "", text)
        text = text.replace("```", "")
    return text.strip()


def parse_team_line(line: str) -> (str, List[int]):
    """
    Parse a CSV line like:
      Away Team,3,12,6,0,0,3,6,2
    Returns (team_name, [8 ints]).
    """
    parts = [p.strip() for p in line.split(",") if p.strip() != ""]
    if not parts:
        return "", []
    team = parts[0]
    nums: List[int] = []
    for p in parts[1:]:
        try:
            nums.append(int(p))
        except ValueError:
            # tolerate junk -> treat as 0
            nums.append(0)
    # pad / truncate to exactly 8 values
    if len(nums) < 8:
        nums = nums + [0] * (8 - len(nums))
    elif len(nums) > 8:
        nums = nums[:8]
    return team, nums


def extract_stats_from_output(raw: str) -> Dict[str, List[int]]:
    """
    Extract per-team stats from model's CSV output.
    Looks for lines starting with "Away Team" or "Home Team".
    Returns dict: {"Away Team": [8 ints], "Home Team": [8 ints]}.
    """
    raw = strip_code_fences(raw)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    stats: Dict[str, List[int]] = {}

    for ln in lines:
        lower = ln.lower()
        if lower.startswith("away team"):
            team, nums = parse_team_line(ln)
            stats["Away Team"] = nums
        elif lower.startswith("home team"):
            team, nums = parse_team_line(ln)
            stats["Home Team"] = nums

    # If one of them is missing, fall back to zeros
    if "Away Team" not in stats:
        stats["Away Team"] = [0] * 8
    if "Home Team" not in stats:
        stats["Home Team"] = [0] * 8

    return stats


def write_csv_for_match(doc_id: int, stats: Dict[str, List[int]], out_dir: str):
    """
    Write one CSV per match in the exact format expected by LiveSum evaluate.py:
    Columns: Team + 8 numeric stats, 2 rows (Away, Home).
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{doc_id}.csv")

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

    with open(path, "w", encoding="utf-8") as w:
        w.write(",".join(header) + "\n")

        for team in ["Away Team", "Home Team"]:
            nums = stats.get(team, [0] * 8)
            row = [team] + [str(x) for x in nums]
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
    outputs = llm.generate(prompts, sampling, use_tqdm=True)

    for ex, out in zip(data, outputs):
        doc_id = ex["id"]
        raw_text = out.outputs[0].text
        stats = extract_stats_from_output(raw_text)
        write_csv_for_match(doc_id, stats, args.output_dir)


if __name__ == "__main__":
    main()
