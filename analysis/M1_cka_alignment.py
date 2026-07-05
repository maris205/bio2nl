"""
M1 — CKA representation alignment (shared-representation hypothesis).

Question: do language-structure and biology-structure live in aligned representational
subspaces? We compute linear CKA between a model's per-layer representations of
NL structural pairs (PAWS-X) and biological structural pairs (protein homology).

For each backbone (base gpt2 / PAWS-tuned / protein-tuned) and each layer:
  X = last-token hidden states on N PAWS pairs
  Y = last-token hidden states on N protein pairs
  CKA(X, Y) per layer  (do positive-vs-positive difference-vectors align across modalities?)

Hypothesis: representations DO align (CKA well above a shuffled-control floor), especially
in mid/high layers — so the asymmetry is NOT "representations can't be shared". Combined with
M2 (dynamics don't transfer), this isolates the asymmetry to LEARNING DYNAMICS not representability.

Run:
    python M1_cka_alignment.py --n 1000
"""
import os, json, argparse, numpy as np, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

ROOT = "/root/autodl-tmp/bio-trans"
PROTEIN_CSV = f"{ROOT}/biopaws/1-data/protein_pair_20k_length_restricted_balanced.csv"
CKPT = f"{ROOT}/bio2nl/tier1_gpt2/cpt_ckpts"
OUT = f"{ROOT}/bio2nl/results/router/M1_cka.jsonl"
MAXLEN = 256


def linear_cka(X, Y):
    """Linear CKA between two [N, d] matrices (centered)."""
    X = X - X.mean(0, keepdims=True); Y = Y - Y.mean(0, keepdims=True)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    denom = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(hsic / (denom + 1e-12))


@torch.no_grad()
def layer_feats(model, tok, s1, s2, device, bs=32):
    model.eval()
    n_layers = model.config.n_layer + 1
    feats = {l: [] for l in range(n_layers)}
    for i in range(0, len(s1), bs):
        enc = tok(s1[i:i+bs], s2[i:i+bs], truncation=True, max_length=MAXLEN,
                  padding=True, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        lengths = enc["attention_mask"].sum(1) - 1
        for l in range(n_layers):
            hs = out.hidden_states[l]
            idx = lengths.view(-1, 1, 1).expand(-1, 1, hs.size(-1))
            feats[l].append(hs.gather(1, idx).squeeze(1).float().cpu().numpy())
    return {l: np.concatenate(v, 0) for l, v in feats.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token

    paws = load_dataset("paws-x", "en")["train"].shuffle(seed=a.seed).select(range(a.n))
    prot = load_dataset("csv", data_files=PROTEIN_CSV)["train"].shuffle(seed=a.seed).select(range(a.n))
    ps1, ps2 = list(paws["sentence1"]), list(paws["sentence2"])
    qs1, qs2 = list(prot["sentence1"]), list(prot["sentence2"])

    models = {"base": "gpt2",
              "paws_tuned": None,   # forward: PAWS-tuned (from bidir? use base+FT not saved) -> use CPT text as proxy? no.
              "protein_cpt": f"{CKPT}/gpt2/protein_seed0",
              "text_cpt": f"{CKPT}/gpt2/text_seed0"}
    # Only run models that exist; paws_tuned classifier isn't saved, so compare base/text_cpt/protein_cpt.
    fout = open(OUT, "a")
    for name, path in models.items():
        if path is None: continue
        if path != "gpt2" and not os.path.isdir(path): continue
        model = AutoModel.from_pretrained(path).to(device)
        fp = layer_feats(model, tok, ps1, ps2, device)
        fq = layer_feats(model, tok, qs1, qs2, device)
        # shuffled control: shuffle rows of Y -> CKA floor
        per_layer = {}
        for l in fp:
            X, Y = fp[l], fq[l]
            m = min(len(X), len(Y)); X, Y = X[:m], Y[:m]
            cka = linear_cka(X, Y)
            Yp = Y[np.random.permutation(m)]
            cka_shuf = linear_cka(X, Yp)
            per_layer[l] = {"cka": round(cka, 4), "cka_shuffled": round(cka_shuf, 4)}
        rec = {"model": name, "path": path, "per_layer": per_layer}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        best = max(per_layer, key=lambda k: per_layer[k]["cka"])
        print(f"[{name}] max CKA={per_layer[best]['cka']}@L{best} (shuffled floor≈{per_layer[best]['cka_shuffled']})")
        del model; torch.cuda.empty_cache()
    fout.close(); print("DONE", OUT)


if __name__ == "__main__":
    main()
