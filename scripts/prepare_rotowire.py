import os, json
from pathlib import Path
from huggingface_hub import snapshot_download

DS_REPO = "xqwu/text-to-table"
SPLIT_DIR = "rotowire"  # we'll use rotowire/test.text and rotowire/test.data

def read_text_docs(path: str):
    """Assume one Rotowire recap per non-empty line in test.text."""
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(line)
    return docs

def read_data_blocks(path: str):
    """
    Split test.data into blocks, one per game.
    We assume each game starts with a line that begins with 'Team:'.
    """
    blocks = []
    current = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("Team:"):
                if current:
                    blocks.append("\n".join(current))
                    current = []
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks

def parse_table_block(block_text: str):
    """
    Parse one 'test.data' block into:
      { "teams": [rows...], "players": [rows...] }
    Using '<NEWLINE>' markers and '|' separators, as in your sample.
    """
    lines = [ln.strip() for ln in block_text.split("<NEWLINE>") if ln.strip()]
    out = {"teams": [], "players": []}
    section = None  # "team" or "player"
    headers = None

    def parse_row_fields(header_cells, row_cells):
        return {
            header_cells[i].strip(): (row_cells[i].strip() if i < len(row_cells) else "")
            for i in range(len(header_cells))
        }

    i = 0    # index over "lines" expressed in <NEWLINE>-logical rows
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("Team:"):
            section = "team"
            headers = None
            i += 1
            continue
        if ln.startswith("Player:"):
            section = "player"
            headers = None
            i += 1
            continue

        cells = [c.strip() for c in ln.split("|")]
        if cells and cells[0] == "":
            cells = cells[1:]
        if cells and cells[-1] == "":
            cells = cells[:-1]

        if section == "team":
            if headers is None:
                headers = cells
                if headers and headers[0] == "":
                    headers[0] = "Team"
                else:
                    headers = ["Team"] + headers
            else:
                if cells and cells[0] != "":
                    if len(cells) < len(headers):
                        cells = cells + [""] * (len(headers) - len(cells))
                    out["teams"].append(parse_row_fields(headers, cells))
        elif section == "player":
            if headers is None:
                headers = cells
                if headers and headers[0] == "":
                    headers[0] = "Player"
                else:
                    headers = ["Player"] + headers
            else:
                if cells and cells[0] != "":
                    if len(cells) < len(headers):
                        cells = cells + [""] * (len(headers) - len(cells))
                    out["players"].append(parse_row_fields(headers, cells))

        i += 1

    return out

def main():
    data_dir = Path("data/rotowire")
    data_dir.mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id=DS_REPO,
        repo_type="dataset",
        allow_patterns=[f"{SPLIT_DIR}/test.text", f"{SPLIT_DIR}/test.data"],
    )
    split_path = Path(local_path) / SPLIT_DIR
    test_text = split_path / "test.text"
    test_data = split_path / "test.data"
    if not test_text.exists() or not test_data.exists():
        raise FileNotFoundError("Expected rotowire/test.text and rotowire/test.data in the HF dataset.")

    # 1) test.text -> JSONL with {id, document}
    text_docs = read_text_docs(str(test_text))
    with open(data_dir / "rotowire_test.jsonl", "w", encoding="utf-8") as w:
        for i, doc in enumerate(text_docs):
            w.write(json.dumps({"id": i, "document": doc}, ensure_ascii=False) + "\n")

    # 2) test.data -> gold tables
    data_blocks = read_data_blocks(str(test_data))
    if len(data_blocks) != len(text_docs):
        print(f"WARNING: #text docs ({len(text_docs)}) != #data blocks ({len(data_blocks)}).")

    gold = []
    for i, blk in enumerate(data_blocks):
        parsed = parse_table_block(blk)
        gold.append({"id": i, **parsed})

    with open(data_dir / "rotowire_gold.json", "w", encoding="utf-8") as w:
        json.dump(gold, w, ensure_ascii=False, indent=2)

    print("Wrote:")
    print("   ", data_dir / "rotowire_test.jsonl")
    print("   ", data_dir / "rotowire_gold.json")
    print("   #docs:", len(text_docs), " #gold examples:", len(gold))

if __name__ == "__main__":
    main()
