import os, json, argparse
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

from pathlib import Path
from collections import Counter, defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from facts_from_tables import gold_example_to_triples, parse_pred_jsonl_line, Triple

NLI_MODEL = "roberta-large-mnli"  # standard NLI model
LABEL_ENTAILMENT = 2  # roberta-large-mnli: 0=contradiction, 1=neutral, 2=entailment

def load_gold(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_pred(path):
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ln = line.strip()
            if ln:
                lines.append(ln)
    return lines

def triples_from_preds(pred_lines):
    triples = Counter()
    for ln in pred_lines:
        for k in parse_pred_jsonl_line(ln).keys():
            triples[k] += 1
    return list(triples.keys())

def triple_to_sentence(triple: Triple) -> str:
    ent, slot, val = triple
    ent = ent if ent else "the entity"
    return f"For {ent}, {slot} is {val}."

def batch_nli(premises, hyps, model, tok, device, batch_size=32):
    labels = []
    for i in range(0, len(premises), batch_size):
        batch_p = premises[i:i+batch_size]
        batch_h = hyps[i:i+batch_size]
        inputs = tok(
            batch_p,
            batch_h,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = logits.argmax(dim=-1).cpu().tolist()
        labels.extend(preds)
    return labels

def tabeval_nli(pred_triples, gold_triples, model, tok, device):
    gold_set = set(gold_triples)
    pred_set = set(pred_triples)

    # Exact triple EM
    inter = gold_set & pred_set
    em_prec = len(inter) / len(pred_set) if pred_set else 0.0
    em_rec  = len(inter) / len(gold_set) if gold_set else 0.0
    em_f1   = 2 * em_prec * em_rec / (em_prec + em_rec) if (em_prec + em_rec) else 0.0

    # index gold by (entity, slot)
    gold_by_es = {}
    for (e, s, v) in gold_set:
        gold_by_es[(e, s)] = (e, s, v)

    # Precision: are predicted statements entailed by gold?
    prec_correct = len(inter)
    prec_prem, prec_hyp, prec_idx = [], [], []

    for t in pred_set:
        if t in gold_set:
            continue
        e, s, v_pred = t
        g_t = gold_by_es.get((e, s))
        if g_t is None:
            continue
        g_sent = triple_to_sentence(g_t)   # premise
        p_sent = triple_to_sentence(t)     # hypothesis
        prec_prem.append(g_sent)
        prec_hyp.append(p_sent)
        prec_idx.append(t)

    if prec_prem:
        labels = batch_nli(prec_prem, prec_hyp, model, tok, device)
        for t, lab in zip(prec_idx, labels):
            if lab == LABEL_ENTAILMENT:
                prec_correct += 1

    tab_prec = prec_correct / len(pred_set) if pred_set else 0.0

    # Recall: do gold statements find entailed matches in predictions?
    gold_by_es_multi = defaultdict(list)
    for (e, s, v) in gold_set:
        gold_by_es_multi[(e, s)].append((e, s, v))

    pred_by_es_multi = defaultdict(list)
    for (e, s, v) in pred_set:
        pred_by_es_multi[(e, s)].append((e, s, v))

    rec_correct = len(inter)
    rec_prem, rec_hyp, rec_idx = [], [], []

    for g in gold_set:
        if g in pred_set:
            continue
        e, s, v_gold = g
        preds_same_es = pred_by_es_multi.get((e, s), [])
        if not preds_same_es:
            continue
        g_sent = triple_to_sentence(g)
        for p in preds_same_es:
            p_sent = triple_to_sentence(p)
            rec_prem.append(g_sent)
            rec_hyp.append(p_sent)
            rec_idx.append(g)

    if rec_prem:
        labels = batch_nli(rec_prem, rec_hyp, model, tok, device)
        seen_correct = set()
        for g, lab in zip(rec_idx, labels):
            if lab == LABEL_ENTAILMENT:
                seen_correct.add(g)
        rec_correct += len(seen_correct)

    tab_rec = rec_correct / len(gold_set) if gold_set else 0.0
    tab_f1 = 2 * tab_prec * tab_rec / (tab_prec + tab_rec) if (tab_prec + tab_rec) else 0.0

    return {
        "EM_P": em_prec, "EM_R": em_rec, "EM_F1": em_f1,
        "TabEval_P": tab_prec, "TabEval_R": tab_rec, "TabEval_F1": tab_f1,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold_file", required=True)   # data/rotowire/rotowire_gold.json
    ap.add_argument("--pred_jsonl", required=True)  # tables/rotowire_sft2_vllm.jsonl
    ap.add_argument("--out_file", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    gold_data = load_gold(args.gold_file)
    pred_lines = load_pred(args.pred_jsonl)

    # collect gold triples
    gold_triples = []
    for ex in gold_data:
        gold_triples.extend(gold_example_to_triples(ex))

    # collect pred triples
    pred_triples = triples_from_preds(pred_lines)

    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    metrics = tabeval_nli(pred_triples, gold_triples, model, tok, device)

    Path(Path(args.out_file).parent).mkdir(parents=True, exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as w:
        json.dump(metrics, w, indent=2)
    print("Saved metrics to", args.out_file)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
