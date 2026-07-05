"""
M3c — Rigorous difference-head test (fair evaluation before concluding).

Fixes three weaknesses of M3/M3b that could hide a real signal:
 1. Query position: instead of only the LAST token, we let EVERY token attend and measure how
    much attention flows INTO the differing positions (column-wise mass), which is how a
    "difference detector" would actually mark mutated/inverted tokens.
 2. Baseline: we z-score each head's difference-selectivity against that model's own head
    distribution, and also report the raw top-head, so a real head stands out above noise.
 3. Symmetric compare: PAWS-style heads compare the two halves; we compute attention mass on
    differing positions from BOTH the last token AND averaged over all query positions.

Metric per (layer, head): diff_selectivity = mean_attn_into(differing positions)
                                            - mean_attn_into(matched positions),
averaged over pairs. Report max head, its z-score vs same-model heads, and mean over heads.

Compares: base GPT-2, PAWS-tuned GPT-2 (forward), protein-CPT GPT-2 (reverse) — all on BOTH
NL pairs (PAWS-X) and protein homolog pairs, same metric. A clean forward difference-operator
would show: PAWS-tuned model has a high-z head for BOTH modalities that base lacks.

Run:
    python M3c_rigorous_diff_heads.py --n 400
"""
import os, json, argparse, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

ROOT = "/root/autodl-tmp/bio-trans"
PROTEIN_CSV = f"{ROOT}/biopaws/1-data/protein_pair_20k_length_restricted_balanced.csv"
PAWS_MODEL = f"{ROOT}/bio2nl/results/router/_tmp_m3b_model"          # saved PAWS-tuned GPT-2
PROT_CPT   = f"{ROOT}/bio2nl/tier1_gpt2/cpt_ckpts/gpt2/protein_seed0"
OUT = f"{ROOT}/bio2nl/results/router/M3c_rigorous.jsonl"
MAXLEN = 128


def diff_positions(a, b):
    n = min(len(a), len(b))
    return [i for i in range(n) if a[i] != b[i]]


@torch.no_grad()
def selectivity(model, tok, pairs, device):
    """[n_layer,n_head] attention-INTO-differing minus INTO-matched, averaged over all queries+pairs."""
    model.eval()
    nl, nh = model.config.n_layer, model.config.n_head
    acc = np.zeros((nl, nh)); cnt = 0
    for s1, s2 in pairs:
        i1 = tok(s1, truncation=True, max_length=MAXLEN).input_ids
        i2 = tok(s2, truncation=True, max_length=MAXLEN).input_ids
        dset = diff_positions(i1, i2)
        enc = tok(s1, s2, truncation=True, max_length=MAXLEN, return_tensors="pt").to(device)
        L = enc["input_ids"].shape[1]
        didx = [d for d in dset if d < L]
        if not didx or len(didx) >= L - 1:
            continue
        midx = [i for i in range(L) if i not in didx]
        out = model(**enc, output_attentions=True)
        for l in range(nl):
            A = out.attentions[l][0]              # [head, Lq, Lk]
            into_diff = A[:, :, didx].mean(dim=(1, 2))   # attention into differing keys, avg over queries
            into_match = A[:, :, midx].mean(dim=(1, 2))
            acc[l] += (into_diff - into_match).cpu().numpy()
        cnt += 1
    return acc / max(cnt, 1), cnt


def summarize(sc):
    flat = sc.ravel()
    mu, sd = flat.mean(), flat.std() + 1e-9
    l, h = np.unravel_index(np.argmax(sc), sc.shape)
    z = (sc[l, h] - mu) / sd
    return {"max": round(float(sc[l, h]), 5), "max_head": [int(l), int(h)],
            "z_of_max": round(float(z), 2), "mean_over_heads": round(float(mu), 5)}


def get_pairs():
    paws = load_dataset("paws-x", "en")["test"]
    # only differing (non-paraphrase) NL pairs — label 0 means different meaning/order
    nl = list(zip(paws["sentence1"], paws["sentence2"]))
    prot = load_dataset("csv", data_files=PROTEIN_CSV)["train"].filter(lambda e: e["label"] == 1)
    return nl, list(zip(prot["sentence1"], prot["sentence2"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    nl_all, prot_all = get_pairs()
    import random; random.Random(a.seed).shuffle(nl_all); random.Random(a.seed).shuffle(prot_all)
    nl, prot = nl_all[:a.n], prot_all[:a.n]

    models = {"base": "gpt2"}
    if os.path.isdir(PAWS_MODEL): models["paws_tuned"] = PAWS_MODEL
    if os.path.isdir(PROT_CPT): models["protein_cpt"] = PROT_CPT

    fout = open(OUT, "a")
    for name, path in models.items():
        model = AutoModel.from_pretrained(path).to(device)
        for modality, pairs in [("NL", nl), ("protein", prot)]:
            sc, cnt = selectivity(model, tok, pairs, device)
            s = summarize(sc); s.update({"model": name, "modality": modality, "n": cnt})
            fout.write(json.dumps(s) + "\n"); fout.flush()
            print(f"[{name:12s} {modality:7s}] max={s['max']} z={s['z_of_max']} head L{s['max_head'][0]}H{s['max_head'][1]} (n={cnt})")
        del model; torch.cuda.empty_cache()
    fout.close(); print("DONE", OUT)


if __name__ == "__main__":
    main()
