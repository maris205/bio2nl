"""
Cross-modal backbone test: can a PURE protein model (ESM-2) do natural-language structure?

ESM-2 is a SOTA masked-LM trained only on protein sequences (UniRef). Its tokenizer has
33 tokens (20 amino acids + specials) and NO English words — English is ~90% <unk>. This is
the extreme end of the "bio backbone" ablation: if evolutionary structure were a universal
prior, ESM-2 should transfer; if transfer is asymmetric, it should fail at the input layer.

We fine-tune each backbone as a sequence classifier on structural-NL tasks and compare:
  facebook/esm2_t6_8M_UR50D           : pure protein, 6L/320h, vocab 33
  google/bert_uncased_L-4_H-256_A-4   : English BERT-small, 4L/256h (size-matched control)
  prajjwal1/bert-tiny                 : English BERT-tiny, 2L/128h (lower bound)

Also logs the <unk> fraction of each task under the ESM tokenizer (quantifies the missing
language interface).

Run:
    python eval_esm_on_nl.py --tasks paws-x,cola --seeds 0,1,2
"""
import os, json, argparse, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score

ROOT = "/root/autodl-tmp/bio-trans/bio2nl"
OUT = f"{ROOT}/results/benchmarks/esm_on_nl.jsonl"
MAXLEN = 128

MODELS = {
    "esm2_8M": "facebook/esm2_t6_8M_UR50D",
    "esm2_35M": "facebook/esm2_t12_35M_UR50D",
    "esm2_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_650M": "facebook/esm2_t33_650M_UR50D",
    "bert_small": "google/bert_uncased_L-4_H-256_A-4",
    "bert_tiny": "prajjwal1/bert-tiny",
    "bert_base": "google-bert/bert-base-uncased",
    "protbert": "Rostlab/prot_bert",
}
TASKS = {
    "paws-x": dict(load=("paws-x", "en"), keys=("sentence1", "sentence2"), val="validation", metric="acc"),
    "cola":   dict(load=("nyu-mll/glue", "cola"), keys=("sentence", None), val="validation", metric="mcc"),
    "rte":    dict(load=("nyu-mll/glue", "rte"), keys=("sentence1", "sentence2"), val="validation", metric="acc"),
}


def unk_fraction(tok, texts, n=200):
    unk = tok.unk_token_id
    if unk is None:
        return 0.0
    tot = u = 0
    for t in texts[:n]:
        ids = tok(t, truncation=True, max_length=MAXLEN).input_ids
        tot += len(ids); u += sum(1 for i in ids if i == unk)
    return round(u / max(tot, 1), 4)


def prep(task, tok):
    cfg = TASKS[task]; k1, k2 = cfg["keys"]
    raw = load_dataset(*cfg["load"])
    def f(ex):
        if k2:
            return tok(ex[k1], ex[k2], truncation=True, max_length=MAXLEN, padding="max_length")
        return tok(ex[k1], truncation=True, max_length=MAXLEN, padding="max_length")
    return raw["train"].map(f, batched=True), raw[cfg["val"]].map(f, batched=True), cfg["metric"]


def run(model_name, task, seed, device):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.sep_token
    train, val, metric = prep(task, tok)
    ufrac = unk_fraction(tok, list(load_dataset(*TASKS[task]["load"])["validation"][TASKS[task]["keys"][0]]))
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    args = TrainingArguments(
        output_dir=f"{ROOT}/results/benchmarks/_tmp_esm", learning_rate=2e-5,
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        num_train_epochs=3, weight_decay=0.01, seed=seed,
        eval_strategy="no", save_strategy="no", logging_strategy="no",
        report_to=[], disable_tqdm=True, fp16=True)
    tr = Trainer(model=model, args=args, train_dataset=train,
                 data_collator=DataCollatorWithPadding(tokenizer=tok), tokenizer=tok)
    tr.train()
    pred = tr.predict(val)
    p = np.argmax(pred.predictions, 1); y = np.array(pred.label_ids)
    score = matthews_corrcoef(y, p) if metric == "mcc" else accuracy_score(y, p)
    acc = accuracy_score(y, p)
    del model, tr; torch.cuda.empty_cache()
    return round(float(score), 4), round(float(acc), 4), ufrac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="paws-x,cola,rte")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--models", default="esm2_8M,bert_small,bert_tiny")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fout = open(OUT, "a")
    for task in a.tasks.split(","):
        for mkey in a.models.split(","):
            for seed in [int(s) for s in a.seeds.split(",")]:
                score, acc, uf = run(MODELS[mkey], task, seed, device)
                rec = {"task": task, "model": mkey, "hf": MODELS[mkey], "seed": seed,
                       "metric": TASKS[task]["metric"], "score": score, "acc": acc, "unk_frac": uf}
                fout.write(json.dumps(rec) + "\n"); fout.flush()
                print(f"[{task}] {mkey:11s} seed{seed} {rec['metric']}={score} acc={acc} unk_frac={uf}")
    fout.close(); print("DONE", OUT)


if __name__ == "__main__":
    main()
