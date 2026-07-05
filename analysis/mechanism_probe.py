"""
Exp2 mechanism analysis: WHY does protein-CPT beat shuffled/random on NL structure?

Two frozen-representation probes on each CPT checkpoint (base/text/protein/shuffled/randomaa),
NO fine-tuning of the backbone:

1. Linear-probe separability: freeze the model, extract last-token hidden states for a
   structural-NL task (PAWS-X paraphrase pairs), fit a logistic-regression probe, report
   held-out accuracy. Higher => the CPT made NL structure MORE linearly separable in the
   representation (a structural prior).

2. Layer-wise probe: same, but per layer, to see WHERE (early/mid/late) the structure lives.

Prediction (matching the sample-efficiency result): protein > shuffled > randomaa in probe
accuracy; text highest. If protein > shuffled here too, the transfer is representational,
not a fine-tuning artifact.

Run:
    python mechanism_probe.py --base gpt2 --seed 0 --n 2000
"""
import os, json, argparse, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

ROOT = "/root/autodl-tmp/bio-trans/bio2nl"
CKPT_DIR = f"{ROOT}/tier1_gpt2/cpt_ckpts"
OUT = f"{ROOT}/results/router/mechanism_probe.jsonl"


def groups(base, seed):
    tag = base.replace("/", "_")
    g = {"base": base}
    for name in ["text", "protein", "shuffled", "randomaa"]:
        p = f"{CKPT_DIR}/{tag}/{name}_seed{seed}"
        if os.path.isdir(p):
            g[name] = p
    return g


@torch.no_grad()
def extract_features(model_path, base, sents1, sents2, device, layers=None, bs=32):
    """Return dict layer_idx -> [N, hidden] last-token features for concatenated pair."""
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModel.from_pretrained(model_path, output_hidden_states=True).to(device).eval()
    n_layers = model.config.n_layer + 1  # +embeddings
    if layers is None:
        layers = list(range(n_layers))
    feats = {l: [] for l in layers}
    for i in range(0, len(sents1), bs):
        b1, b2 = sents1[i:i+bs], sents2[i:i+bs]
        enc = tok(b1, b2, truncation=True, max_length=128, padding=True, return_tensors="pt").to(device)
        out = model(**enc)
        # last non-pad token index per example
        lengths = enc["attention_mask"].sum(1) - 1
        for l in layers:
            hs = out.hidden_states[l]                       # [B, T, H]
            idx = lengths.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
            last = hs.gather(1, idx).squeeze(1)             # [B, H]
            feats[l].append(last.float().cpu().numpy())
    del model; torch.cuda.empty_cache()
    return {l: np.concatenate(v, 0) for l, v in feats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gpt2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset("paws-x", "en")["train"].shuffle(seed=a.seed).select(range(a.n))
    s1, s2 = list(ds["sentence1"]), list(ds["sentence2"])
    y = np.array(ds["label"])

    G = groups(a.base, a.seed)
    print(f"base={a.base} seed={a.seed} groups:", list(G))
    fout = open(OUT, "a")

    for gname, gpath in G.items():
        feats = extract_features(gpath, a.base, s1, s2, device)
        # last-layer probe (main) + per-layer sweep
        per_layer = {}
        for l, X in feats.items():
            clf = LogisticRegression(max_iter=1000, C=1.0)
            acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()
            per_layer[l] = round(float(acc), 4)
        best_layer = max(per_layer, key=per_layer.get)
        rec = {"base": a.base, "seed": a.seed, "group": gname,
               "probe_acc_last": per_layer[max(per_layer)],
               "probe_acc_best": per_layer[best_layer], "best_layer": best_layer,
               "per_layer": per_layer}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(f"[{gname:9s}] probe best={per_layer[best_layer]:.4f}@L{best_layer}  last={per_layer[max(per_layer)]:.4f}")
    fout.close()
    print("DONE", OUT)


if __name__ == "__main__":
    main()
