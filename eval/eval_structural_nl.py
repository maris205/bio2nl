"""
Exp2 evaluation: structural-NL sample-efficiency fine-tuning across CPT groups.

For each CPT checkpoint (base / text / protein / shuffled / randomaa) and each structural-NL
task, fine-tune a classification head on {1%,5%,10%,100%} of the task's train set and measure
val accuracy. The thesis prediction: PROTEIN-CPT wins most in the LOW-RESOURCE regime, and beats
TEXT-CPT (not just more tokens), SHUFFLED and RANDOM (structure, not token noise).

Tasks (paraphrase/equivalence, grammatical acceptability, NLI):
    paws-x (en)   : sentence-pair, label paraphrase
    glue/cola     : single-sentence, grammatical acceptability
    glue/rte      : sentence-pair, entailment

Run:
    python eval_structural_nl.py --seed 0 --fractions 0.01,0.05,0.1,1.0
"""
import os, json, argparse, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
from sklearn.metrics import accuracy_score, matthews_corrcoef

ROOT = "/root/autodl-tmp/bio-trans/bio2nl"
CKPT_DIR = f"{ROOT}/tier1_gpt2/cpt_ckpts"
OUT = f"{ROOT}/results/benchmarks/exp2_structural_nl.jsonl"
MAXLEN = 128

EVAL_NL_DIR = f"{ROOT}/data/eval_nl"
TASKS = {
    "paws-x": dict(load=("paws-x", "en"), keys=("sentence1", "sentence2"), val="validation", metric="acc"),
    "cola":   dict(load=("nyu-mll/glue", "cola"), keys=("sentence", None), val="validation", metric="mcc"),
    "rte":    dict(load=("nyu-mll/glue", "rte"), keys=("sentence1", "sentence2"), val="validation", metric="acc"),
}
# local synthetic tasks (JSONL with sentence,label). difficulty via d{N}.
for _d in [8, 16, 24]:
    TASKS[f"longrange_d{_d}"] = dict(local=f"longrange_d{_d}", keys=("sentence", None), metric="acc")
for _L in [40, 60]:
    TASKS[f"dyck_L{_L}_t3"] = dict(local=f"dyck_L{_L}_t3", keys=("sentence", None), metric="acc")


def groups(base, seed):
    """Return {group_name: checkpoint_path} for this base model + seed."""
    tag = base.replace("/", "_")
    g = {"base": base}
    for name in ["text", "protein", "shuffled", "randomaa"]:
        p = f"{CKPT_DIR}/{tag}/{name}_seed{seed}"
        if os.path.isdir(p):
            g[name] = p
    return g


def make_tok(model_path):
    t = AutoTokenizer.from_pretrained(model_path)
    if t.pad_token is None:
        t.pad_token = t.eos_token
    return t


def prep(task, tok):
    cfg = TASKS[task]
    k1, k2 = cfg["keys"]
    def f(ex):
        if k2:
            return tok(ex[k1], ex[k2], truncation=True, max_length=MAXLEN, padding="max_length")
        return tok(ex[k1], truncation=True, max_length=MAXLEN, padding="max_length")
    if "local" in cfg:
        base = cfg["local"]
        train = load_dataset("json", data_files=f"{EVAL_NL_DIR}/{base}_train.jsonl", split="train").map(f, batched=True)
        val = load_dataset("json", data_files=f"{EVAL_NL_DIR}/{base}_val.jsonl", split="train").map(f, batched=True)
    else:
        raw = load_dataset(*cfg["load"])
        train = raw["train"].map(f, batched=True)
        val = raw[cfg["val"]].map(f, batched=True)
    return train, val, cfg["metric"]


def subsample(ds, frac, seed):
    if frac >= 1.0:
        return ds
    n = max(64, int(len(ds) * frac))
    return ds.shuffle(seed=seed).select(range(n))


def run_one(model_path, tok, train, val, metric, seed, frac):
    set_seed(seed)
    sub = subsample(train, frac, seed)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id or tok.eos_token_id
    args = TrainingArguments(
        output_dir=f"{ROOT}/results/benchmarks/_tmp_e2", learning_rate=2e-5,
        per_device_train_batch_size=16, per_device_eval_batch_size=64,
        num_train_epochs=5 if frac < 0.2 else 3, weight_decay=0.01,
        seed=seed, eval_strategy="no", save_strategy="no", logging_strategy="no",
        report_to=[], disable_tqdm=True, fp16=True)
    trainer = Trainer(model=model, args=args, train_dataset=sub,
                      data_collator=DataCollatorWithPadding(tokenizer=tok), tokenizer=tok)
    trainer.train()
    pred = trainer.predict(val)
    preds = np.argmax(pred.predictions, axis=1)
    labels = np.array(pred.label_ids)
    score = matthews_corrcoef(labels, preds) if metric == "mcc" else accuracy_score(labels, preds)
    del model, trainer; torch.cuda.empty_cache()
    return round(float(score), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="gpt2")
    ap.add_argument("--fractions", default="0.01,0.05,0.1,1.0")
    ap.add_argument("--tasks", default="paws-x,cola,rte")
    a = ap.parse_args()
    fracs = [float(x) for x in a.fractions.split(",")]
    tasks = a.tasks.split(",")
    G = groups(a.base, a.seed)
    print(f"base={a.base} seed={a.seed} groups:", list(G))
    fout = open(OUT, "a")

    for task in tasks:
        # all CPT groups share the base model's vocab; use base tokenizer for data prep.
        tok = make_tok(a.base)
        train, val, metric = prep(task, tok)
        for gname, gpath in G.items():
            for frac in fracs:
                score = run_one(gpath, tok, train, val, metric, a.seed, frac)
                rec = {"task": task, "group": gname, "frac": frac, "metric": metric,
                       "score": score, "seed": a.seed, "base": a.base}
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                print(f"[{task}] {gname:9s} frac={frac:<5} {metric}={score}")
    fout.close()
    print("DONE", OUT)


if __name__ == "__main__":
    main()
