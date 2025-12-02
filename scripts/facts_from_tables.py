from typing import Dict, List, Tuple
import json

Triple = Tuple[str, str, str]

def table_rows_to_triples(rows: List[Dict], entity_key: str) -> List[Triple]:
    triples: List[Triple] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ent = row.get(entity_key, "") or row.get("name", "") or row.get("Name", "")
        ent = str(ent).strip()
        for k, v in row.items():
            if k == entity_key:
                continue
            if v is None:
                continue
            sval = str(v).strip()
            if sval == "":
                continue
            triples.append((ent, k, sval))
    return triples

def gold_example_to_triples(ex: Dict) -> List[Triple]:
    teams = ex.get("teams", [])
    players = ex.get("players", [])
    return table_rows_to_triples(teams, "Team") + table_rows_to_triples(players, "Player")

def parse_pred_jsonl_line(line: str) -> Dict[Triple, int]:
    """
    Parse a single predicted JSONL line into triples.

    Robust to:
      - empty lines
      - non-JSON junk lines
      - extra text around a JSON object (we try to extract the {...} substring)
    """
    line = line.strip()
    if not line:
        return {}

    obj = None

    # First, try to parse the whole line as JSON
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        # Try to salvage: find the first '{' and last '}' and parse that slice
        start = line.find("{")
        end = line.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = line[start : end + 1]
            try:
                obj = json.loads(candidate)
            except json.JSONDecodeError:
                return {}
        else:
            return {}

    if not isinstance(obj, dict):
        return {}

    keys = list(obj.keys())

    # slot/value style (kv)
    if set(keys) == {"slot", "value"}:
        ent = ""
        return {(ent, str(obj["slot"]).strip(), str(obj["value"]).strip()): 1}

    # otherwise treat first name-like key as entity
    cand = [k for k in keys if k.lower() in ("team", "player", "name")]
    ent_key = cand[0] if cand else None
    ent = obj.get(ent_key, "") if ent_key else ""
    ent = str(ent).strip()
    triples = {}
    for k in keys:
        if k == ent_key:
            continue
        val = obj[k]
        if val is None:
            continue
        sval = str(val).strip()
        if sval == "":
            continue
        triples[(ent, k, sval)] = 1
    return triples
