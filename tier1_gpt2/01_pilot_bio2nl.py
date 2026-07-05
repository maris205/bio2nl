"""
Tier-1 reverse-transfer pilot: BIO -> NL  (mirror of the forward GPT-2 experiment).

Forward (prior paper): train GPT-2 on PAWS-X (NL paraphrase) -> test on protein pairs.
Reverse (this pilot):  train GPT-2 on protein homology pairs -> test zero-shot on PAWS-X English.

The two tasks share an identical schema: (sentence1, sentence2, label in {0,1}).
"Difference detection" is the shared structural skill. If bio training transfers to NL,
zero-shot PAWS-X accuracy of the bio-trained head should beat the untrained baseline.

Run:
    source /etc/network_turbo ; export HF_ENDPOINT=https://hf-mirror.com
    python tier1_gpt2/01_pilot_bio2nl.py --epochs 4 --seed 56
"""
import os, json, argparse, numpy as np, torch
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

ROOT = "/root/autodl-tmp/bio-trans"
PROTEIN_CSV = f"{ROOT}/biopaws/1-data/protein_pair_20k_length_restricted_balanced.csv"
OUTDIR = f"{ROOT}/bio2nl/results/benchmarks"
MODEL = "gpt2"


def load_protein(tokenizer, max_length, seed, n_val=2000):
    """Protein homology pairs -> train/val. Columns: sentence1, sentence2, label."""
    ds = load_dataset("csv", data_files=PROTEIN_CSV)["train"]
    ds = ds.shuffle(seed=seed)
    split = ds.train_test_split(test_size=n_val, seed=seed)
    def tok(ex):
        return tokenizer(ex["sentence1"], ex["sentence2"],
                         truncation=True, max_length=max_length, padding="max_length")
    return split["train"].map(tok, batched=True), split["test"].map(tok, batched=True)


def load_pawsx_test(tokenizer, max_length):
    """PAWS-X English test+validation as the held-out NL transfer target."""
    raw = load_dataset("paws-x", "en")
    def tok(ex):
        return tokenizer(ex["sentence1"], ex["sentence2"],
                         truncation=True, max_length=max_length, padding="max_length")
    return raw["test"].map(tok, batched=True), raw["validation"].map(tok, batched=True)


def predict(trainer, dataset):
    out = trainer.predict(dataset)
    preds = np.argmax(out.predictions, axis=1)
    labels = np.array(out.label_ids)
    return preds, labels


def score(preds, labels, tag):
    """Accuracy with label-flip guard (inherited from biopaws eval convention)."""
    acc = accuracy_score(labels, preds)
    flipped = False
    if acc < 0.5:                      # binary head may invert; report flipped but log it
        preds = 1 - preds
        acc = accuracy_score(labels, preds)
        flipped = True
    return {
        "tag": tag,
        "accuracy": round(float(acc), 4),
        "f1": round(float(f1_score(labels, preds)), 4),
        "label_flipped": flipped,
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "n": int(len(labels)),
    }


def build_trainer(model, tokenizer, args, train_ds=None, eval_ds=None):
    def metrics(p):
        return {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))}
    return Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                   data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                   tokenizer=tokenizer, compute_metrics=metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=56)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()
    set_seed(a.seed)
    os.makedirs(OUTDIR, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    paws_test, paws_val = load_pawsx_test(tokenizer, a.max_length)
    results = {"config": vars(a), "model": MODEL}

    # --- Baseline: untrained GPT-2 head, zero-shot on PAWS-X (chance reference) ---
    base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    base.config.pad_token_id = tokenizer.eos_token_id
    base_args = TrainingArguments(output_dir=f"{OUTDIR}/_tmp_base",
                                  per_device_eval_batch_size=a.bs, report_to=[])
    bt = build_trainer(base, tokenizer, base_args)
    p, l = predict(bt, paws_test)
    results["baseline_untrained_pawsx"] = score(p, l, "baseline_untrained->pawsx_test")
    print("[baseline] PAWS-X test:", results["baseline_untrained_pawsx"])

    # --- Treatment: train on PROTEIN homology, then zero-shot PAWS-X ---
    pro_train, pro_val = load_protein(tokenizer, a.max_length, a.seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    model.config.pad_token_id = tokenizer.eos_token_id
    targs = TrainingArguments(
        output_dir=f"{OUTDIR}/_tmp_bio", learning_rate=a.lr,
        lr_scheduler_type="constant_with_warmup", warmup_ratio=0.1,
        optim="adamw_torch", weight_decay=0.0, seed=a.seed,
        per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
        num_train_epochs=a.epochs, eval_strategy="epoch", save_strategy="no",
        logging_strategy="epoch", report_to=[])
    trainer = build_trainer(model, tokenizer, targs, pro_train, pro_val)
    trainer.train()

    # in-domain check (did it learn protein homology at all?)
    p, l = predict(trainer, pro_val)
    results["bio_indomain_protein_val"] = score(p, l, "bio_trained->protein_val")
    print("[bio in-domain] protein val:", results["bio_indomain_protein_val"])

    # the headline: zero-shot transfer to NL
    p, l = predict(trainer, paws_test)
    results["transfer_pawsx_test"] = score(p, l, "bio_trained->pawsx_test")
    p, l = predict(trainer, paws_val)
    results["transfer_pawsx_val"] = score(p, l, "bio_trained->pawsx_val")
    print("[TRANSFER] bio-trained -> PAWS-X test:", results["transfer_pawsx_test"])
    print("[TRANSFER] bio-trained -> PAWS-X val :", results["transfer_pawsx_val"])

    out = f"{OUTDIR}/tier1_pilot_seed{a.seed}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
