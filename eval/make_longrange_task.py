"""
Synthetic long-range structural dependency task (harder than PAWS-X, difficulty tunable).
Mirrors protein long-range contacts: a dependency that must be tracked across many
intervening distractor tokens.

Task: subject-verb number agreement across a variable-length distractor clause.
  label 1 (grammatical):  "The <noun_sg> [that ... <distractors> ...] <verb_sg>."
  label 0 (ungrammatical): same but verb number MISMATCHES the head subject.
The distractor clause contains nouns of the OPPOSITE number, so a model using local
heuristics (agree with nearest noun) fails; only tracking the long-range head-subject works.

Difficulty knob = distractor length (n_distract). Larger => longer dependency => harder,
so we can pick a length where 355M still has headroom (not saturated).

Writes train/val JSONL with fields sentence, label. Balanced.
"""
import os, json, argparse, random

SG_NOUNS = ["scientist", "engineer", "author", "teacher", "doctor", "artist", "pilot",
            "manager", "student", "farmer", "lawyer", "nurse", "chef", "worker", "player"]
PL_NOUNS = ["scientists", "engineers", "authors", "teachers", "doctors", "artists", "pilots",
            "managers", "students", "farmers", "lawyers", "nurses", "chefs", "workers", "players"]
SG_VERBS = ["runs", "writes", "speaks", "works", "reads", "walks", "sings", "wins", "leads"]
PL_VERBS = ["run", "write", "speak", "work", "read", "walk", "sing", "win", "lead"]
PREPS = ["near", "beside", "behind", "beyond", "under", "above", "against"]
ADJS = ["tall", "quiet", "famous", "clever", "young", "serious", "curious", "brave"]


def distractor(n, opp_nouns, rng):
    """Build a distractor clause of ~n tokens containing OPPOSITE-number nouns."""
    parts = []
    while len(" ".join(parts).split()) < n:
        parts.append(rng.choice(PREPS))
        parts.append("the")
        parts.append(rng.choice(ADJS))
        parts.append(rng.choice(opp_nouns))
    return " ".join(parts)


def make_example(rng, n_distract):
    singular = rng.random() < 0.5
    head = rng.choice(SG_NOUNS if singular else PL_NOUNS)
    opp = PL_NOUNS if singular else SG_NOUNS
    dist = distractor(n_distract, opp, rng)
    grammatical = rng.random() < 0.5
    if grammatical:
        verb = rng.choice(SG_VERBS if singular else PL_VERBS)
        label = 1
    else:
        verb = rng.choice(PL_VERBS if singular else SG_VERBS)   # mismatched
        label = 0
    sent = f"The {head} {dist} {verb} ."
    return {"sentence": sent, "label": label}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_distract", type=int, default=12, help="distractor length (difficulty)")
    ap.add_argument("--n_train", type=int, default=8000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/root/autodl-tmp/bio-trans/bio2nl/data/eval_nl")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)

    def gen(n):
        return [make_example(rng, a.n_distract) for _ in range(n)]
    tr, va = gen(a.n_train), gen(a.n_val)
    tag = f"longrange_d{a.n_distract}"
    with open(f"{a.out}/{tag}_train.jsonl", "w") as f:
        for r in tr: f.write(json.dumps(r) + "\n")
    with open(f"{a.out}/{tag}_val.jsonl", "w") as f:
        for r in va: f.write(json.dumps(r) + "\n")
    print(f"wrote {tag}: {len(tr)} train / {len(va)} val")
    print("sample:", tr[0]["sentence"][:120], "| label", tr[0]["label"])
    from collections import Counter
    print("train label balance:", Counter(r["label"] for r in tr))


if __name__ == "__main__":
    main()
