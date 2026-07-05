"""
Dyck-language nested-bracket well-formedness task — a genuinely HARD long-range structural
probe (the standard test for hierarchical/long-range dependency), tunable and non-saturating.
Maps to protein nested contact topology.

Task: given a sequence of brackets of K types, is it well-formed (properly nested/matched)?
  label 1: well-formed Dyck word
  label 0: corrupted by a single edit (flip/swap one bracket) at a RANDOM position/depth,
           so local heuristics (count opens vs closes) mostly fail — only tracking the
           nesting stack across the whole string works.

Difficulty knobs: n_types (bracket types), length. Longer + more types => harder.
Brackets rendered space-separated so GPT-2 BPE treats each atomically.

Writes {tag}_train.jsonl / {tag}_val.jsonl with fields sentence, label (balanced).
"""
import os, json, argparse, random

PAIRS = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]


def gen_dyck(rng, length, n_types):
    """Generate a well-formed Dyck word of exactly `length` tokens (length even)."""
    opens = [p[0] for p in PAIRS[:n_types]]
    closes = {p[0]: p[1] for p in PAIRS[:n_types]}
    out, stack = [], []
    while len(out) < length:
        remaining = length - len(out)
        can_open = remaining > len(stack)           # must leave room to close all
        if stack and (not can_open or rng.random() < 0.5):
            out.append(closes[stack.pop()])
        else:
            o = rng.choice(opens); out.append(o); stack.append(o)
    while stack:                                     # (shouldn't trigger, safety)
        out.append(closes[stack.pop()])
    return out[:length]


def corrupt(seq, rng, n_types):
    """Single local edit that (usually) breaks well-formedness at a random position."""
    s = list(seq)
    all_tok = [t for p in PAIRS[:n_types] for t in p]
    for _ in range(10):
        i = rng.randrange(len(s))
        new = rng.choice(all_tok)
        if new != s[i]:
            cand = s[:]; cand[i] = new
            if not is_wellformed(cand, n_types):     # ensure it actually became invalid
                return cand
    # fallback: swap two
    i, j = rng.sample(range(len(s)), 2); s[i], s[j] = s[j], s[i]
    return s


def is_wellformed(seq, n_types):
    closes = {p[1]: p[0] for p in PAIRS[:n_types]}
    opens = {p[0] for p in PAIRS[:n_types]}
    stack = []
    for t in seq:
        if t in opens:
            stack.append(t)
        elif t in closes:
            if not stack or stack.pop() != closes[t]:
                return False
        else:
            return False
    return not stack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, default=40)
    ap.add_argument("--n_types", type=int, default=3)
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/autodl-tmp/bio-trans/bio2nl/data/eval_nl")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    def make(n):
        rows = []
        for _ in range(n):
            base = gen_dyck(rng, a.length, a.n_types)
            if rng.random() < 0.5:
                rows.append({"sentence": " ".join(base), "label": 1})
            else:
                rows.append({"sentence": " ".join(corrupt(base, rng, a.n_types)), "label": 0})
        return rows

    tr, va = make(a.n_train), make(a.n_val)
    tag = f"dyck_L{a.length}_t{a.n_types}"
    with open(f"{a.out}/{tag}_train.jsonl", "w") as f:
        for r in tr: f.write(json.dumps(r) + "\n")
    with open(f"{a.out}/{tag}_val.jsonl", "w") as f:
        for r in va: f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"wrote {tag}: {len(tr)} train / {len(va)} val  balance={Counter(r['label'] for r in tr)}")
    print("sample+:", [r['sentence'] for r in tr if r['label']==1][0][:80])
    print("sample-:", [r['sentence'] for r in tr if r['label']==0][0][:80])


if __name__ == "__main__":
    main()
