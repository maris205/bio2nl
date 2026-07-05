"""
M3 — Difference-operator attention heads (asymmetry of induced mechanism).

The forward paper claims PAWS training induces attention heads that act as "universal
difference operators" — attending to the tokens that DIFFER between a pair. M3 asks the
reverse-control question: does PROTEIN training induce heads that detect NL perturbations?

Metric: for each (layer, head), a "difference-attention score" = average attention mass that
the last token places on the positions that differ between sentence1 and sentence2, minus the
mass on matched positions. High score = the head localizes the structural difference.

We compute this on NL pairs (word-order/word-swap differences) for:
  - base gpt2
  - text-CPT (control)
  - protein-CPT  <- does bio training grow an NL difference detector?
Hypothesis: protein-CPT does NOT grow NL difference heads beyond base (bio training learns
narrow sequence matching, not a transferable difference operator).

Note: forward half (PAWS-tuned model attending to protein mutations) is the original paper's
Fig 3b; here we supply the REVERSE control on identical footing.

Run:
    python M3_difference_heads.py --n 300
"""
import os, json, argparse, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

ROOT = "/root/autodl-tmp/bio-trans"
CKPT = f"{ROOT}/bio2nl/tier1_gpt2/cpt_ckpts"
OUT = f"{ROOT}/bio2nl/results/router/M3_difference_heads.jsonl"
MAXLEN = 128


def diff_positions(ids1, ids2):
    """Positions (in a concatenated a</s>b view) that differ; simple aligned compare."""
    n = min(len(ids1), len(ids2))
    diff = [i for i in range(n) if ids1[i] != ids2[i]]
    return set(diff)


@torch.no_grad()
def diff_attention_scores(model, tok, pairs, device):
    """Return [n_layer, n_head] mean difference-attention score over pairs."""
    model.eval()
    nl = model.config.n_layer; nh = model.config.n_head
    acc = np.zeros((nl, nh)); cnt = 0
    for s1, s2 in pairs:
        i1 = tok(s1, truncation=True, max_length=MAXLEN).input_ids
        i2 = tok(s2, truncation=True, max_length=MAXLEN).input_ids
        diffset = diff_positions(i1, i2)
        if not diffset or len(diffset) >= min(len(i1), len(i2)):
            continue
        enc = tok(s1, s2, truncation=True, max_length=MAXLEN, return_tensors="pt").to(device)
        out = model(**enc, output_attentions=True)
        L = enc["input_ids"].shape[1]
        # map diff positions of sentence1 onto the concatenated sequence (they start at 0)
        diff_idx = [d for d in diffset if d < L]
        if not diff_idx:
            continue
        match_idx = [i for i in range(L) if i not in diff_idx]
        for l in range(nl):
            att = out.attentions[l][0]  # [head, L, L]
            last = att[:, -1, :]        # attention FROM last token, [head, L]
            dm = last[:, diff_idx].mean(1).cpu().numpy()
            mm = last[:, match_idx].mean(1).cpu().numpy()
            acc[l] += (dm - mm)
        cnt += 1
    return acc / max(cnt, 1), cnt


def make_nl_pairs(n, seed):
    ds = load_dataset("paws-x", "en")["test"].shuffle(seed=seed).select(range(n))
    return list(zip(ds["sentence1"], ds["sentence2"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    pairs = make_nl_pairs(a.n, a.seed)

    models = {"base": "gpt2",
              "text_cpt": f"{CKPT}/gpt2/text_seed0",
              "protein_cpt": f"{CKPT}/gpt2/protein_seed0"}
    fout = open(OUT, "a")
    for name, path in models.items():
        if path != "gpt2" and not os.path.isdir(path):
            continue
        model = AutoModel.from_pretrained(path).to(device)
        scores, cnt = diff_attention_scores(model, tok, pairs, device)
        top = np.dstack(np.unravel_index(np.argsort(scores.ravel())[-5:], scores.shape))[0]
        rec = {"model": name, "n_pairs": cnt,
               "max_diff_attn": round(float(scores.max()), 4),
               "mean_diff_attn": round(float(scores.mean()), 4),
               "top_heads": [[int(l), int(h), round(float(scores[l, h]), 4)] for l, h in top]}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(f"[{name}] max NL-difference-attn={rec['max_diff_attn']} (top head L{top[-1][0]}H{top[-1][1]}) n={cnt}")
        del model; torch.cuda.empty_cache()
    fout.close(); print("DONE", OUT)


if __name__ == "__main__":
    main()
