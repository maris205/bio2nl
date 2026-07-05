"""
Build iso-token CPT corpora for Exp2 (bio -> NL structural transfer).

Produces plain-text corpus files, one token stream per line-group, all formatted identically
so the ONLY difference between groups is sequence CONTENT:

  G1 text     : OpenWebText plain English
  G2 protein  : real Swiss-Prot proteins   ->  <protein> M K V ... </protein>
  G3 shuffled : G2 with residues shuffled within each sequence (kills structure, keeps AA freq+len)
  G4 randomaa : residues resampled i.i.d. from G2's global AA frequencies (kills all sequence form)

All bio groups use space-separated residues so GPT-2 BPE treats each amino acid atomically.
Each corpus is truncated to ~TARGET_TOKENS GPT-2 tokens (measured with the real tokenizer) so
CPT is iso-token across groups.

Run:
    source /etc/network_turbo ; export HF_ENDPOINT=https://hf-mirror.com
    python build_cpt_corpora.py --target_tokens 50000000
"""
import os, argparse, random, json
import numpy as np
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
SPROT = f"{HERE}/sprot.fasta"
AA = "ACDEFGHIKLMNPQRSTVWY"


def read_proteins(path, max_seqs=None):
    seqs, cur = [], []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur)); cur = []
                    if max_seqs and len(seqs) >= max_seqs:
                        return seqs
            else:
                cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def fmt_protein(seq):
    return "<protein> " + " ".join(seq) + " </protein>"


def aa_frequencies(seqs, sample=20000):
    from collections import Counter
    c = Counter()
    for s in seqs[:sample]:
        c.update(s)
    keep = {a: c.get(a, 1) for a in AA}
    tot = sum(keep.values())
    letters = list(keep.keys())
    probs = np.array([keep[a] / tot for a in letters])
    return letters, probs


def write_capped(lines_iter, out_path, tokenizer, target_tokens):
    """Write formatted lines until we hit ~target_tokens (measured), return actual token count."""
    tok_count = 0
    n_lines = 0
    with open(out_path, "w") as f:
        for line in lines_iter:
            f.write(line + "\n")
            tok_count += len(tokenizer(line).input_ids)
            n_lines += 1
            if tok_count >= target_tokens:
                break
    return tok_count, n_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_tokens", type=int, default=50_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--groups", default="text,protein,shuffled,randomaa")
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed)
    groups = a.groups.split(",")

    tok = AutoTokenizer.from_pretrained("gpt2")
    manifest = {"target_tokens": a.target_tokens, "seed": a.seed, "groups": {}}

    # ---- bio source ----
    if any(g in groups for g in ["protein", "shuffled", "randomaa"]):
        print("reading Swiss-Prot...")
        seqs = read_proteins(SPROT)
        random.shuffle(seqs)
        print(f"  {len(seqs)} proteins")
        letters, probs = aa_frequencies(seqs)

    # ---- G2 protein ----
    if "protein" in groups:
        def gen():
            for s in seqs:
                yield fmt_protein(s)
        out = f"{HERE}/cpt_protein.txt"
        n, nl = write_capped(gen(), out, tok, a.target_tokens)
        manifest["groups"]["protein"] = {"file": out, "tokens": n, "lines": nl}
        print(f"protein: {n:,} tokens, {nl} seqs")

    # ---- G3 shuffled (residues shuffled within each seq) ----
    if "shuffled" in groups:
        def gen():
            for s in seqs:
                lst = list(s); random.shuffle(lst)
                yield fmt_protein("".join(lst))
        out = f"{HERE}/cpt_shuffled.txt"
        n, nl = write_capped(gen(), out, tok, a.target_tokens)
        manifest["groups"]["shuffled"] = {"file": out, "tokens": n, "lines": nl}
        print(f"shuffled: {n:,} tokens, {nl} seqs")

    # ---- G4 random-AA (i.i.d. from global AA freq, lengths matched to real seqs) ----
    if "randomaa" in groups:
        def gen():
            for s in seqs:
                r = "".join(np.random.choice(letters, size=len(s), p=probs))
                yield fmt_protein(r)
        out = f"{HERE}/cpt_randomaa.txt"
        n, nl = write_capped(gen(), out, tok, a.target_tokens)
        manifest["groups"]["randomaa"] = {"file": out, "tokens": n, "lines": nl}
        print(f"randomaa: {n:,} tokens, {nl} seqs")

    # ---- G1 text (OpenWebText) ----
    if "text" in groups:
        from datasets import load_dataset
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True,
                          trust_remote_code=True)
        def gen():
            for ex in ds:
                t = ex["text"].strip().replace("\n", " ")
                if t:
                    yield t
        out = f"{HERE}/cpt_text.txt"
        n, nl = write_capped(gen(), out, tok, a.target_tokens)
        manifest["groups"]["text"] = {"file": out, "tokens": n, "lines": nl}
        print(f"text: {n:,} tokens, {nl} docs")

    with open(f"{HERE}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("manifest written")


if __name__ == "__main__":
    main()
