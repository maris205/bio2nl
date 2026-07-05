"""
Tier-1 reverse-transfer DISTRIBUTION experiment (mirror of forward 100-seed batch).

Forward (prior paper, batch_run/gpt2_ft_en_test_protein.py): for seed in 0..99,
    train GPT-2 on PAWS-X -> test on protein pairs -> collect accuracy distribution.
Reverse (this script): for seed in 0..N-1,
    train GPT-2 on PROTEIN homology pairs -> test zero-shot on PAWS-X English.

Why a distribution: a single seed is one draw. The forward result is reported as a
distribution with a tail of high-transfer runs. We mirror that exactly so the
comparison is apples-to-apples.

Hyperparameters copied verbatim from the forward batch script:
    lr=1e-5, constant_with_warmup, warmup_ratio=0.1, bs=64, 4 epochs, max_length=256.

Run:
    source /etc/network_turbo ; export HF_ENDPOINT=https://hf-mirror.com
    python bio2nl/tier1_gpt2/02_batch_distribution.py --n_seeds 100
"""
import os, sys, json, argparse, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
from sklearn.metrics import accuracy_score, f1_score

ROOT = "/root/autodl-tmp/bio-trans"
PROTEIN_CSV = f"{ROOT}/biopaws/1-data/protein_pair_20k_length_restricted_balanced.csv"
OUTDIR = f"{ROOT}/bio2nl/results/benchmarks"
MODEL = "gpt2"
MAXLEN = 256


def tok_fn(tokenizer):
    def f(ex):
        return tokenizer(ex["sentence1"], ex["sentence2"],
                         truncation=True, max_length=MAXLEN, padding="max_length")
    return f


def evaluate_split(model, dataset, device, bs=64):
    model.eval()
    preds, labels = [], []
    for i in range(0, len(dataset), bs):
        b = dataset[i:i+bs]
        inp = {"input_ids": torch.tensor(b["input_ids"]).to(device),
               "attention_mask": torch.tensor(b["attention_mask"]).to(device)}
        with torch.no_grad():
            out = model(**inp)
        preds.extend(torch.argmax(out.logits, -1).cpu().numpy())
        labels.extend(b["label"])
    preds, labels = np.array(preds), np.array(labels)
    acc = accuracy_score(labels, preds)
    # report BOTH raw and flipped: flip-best is the forward convention, raw is the honest floor
    acc_flip = max(acc, accuracy_score(labels, 1 - preds))
    return {"acc_raw": round(float(acc), 4),
            "acc_flipbest": round(float(acc_flip), 4),
            "f1": round(float(f1_score(labels, preds, zero_division=0)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tk = tok_fn(tokenizer)

    # NL test target loaded once (PAWS-X English test+val)
    paws = load_dataset("paws-x", "en")
    paws_test = paws["test"].map(tk, batched=True)
    paws_val = paws["validation"].map(tk, batched=True)

    # protein corpus loaded once; per-seed shuffle/split for variety (mirrors forward seed-split)
    protein = load_dataset("csv", data_files=PROTEIN_CSV)["train"]

    out_path = f"{OUTDIR}/tier1_distribution.jsonl"
    fout = open(out_path, "a")
    print(f"writing per-seed records -> {out_path}")

    for seed in range(a.start, a.start + a.n_seeds):
        set_seed(seed)
        rec = {"seed": seed}

        # --- untrained baseline on PAWS-X (per-seed: random head init varies) ---
        base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
        base.config.pad_token_id = tokenizer.eos_token_id
        base.to(device)
        rec["baseline_pawsx_test"] = evaluate_split(base, paws_test, device)
        del base; torch.cuda.empty_cache()

        # --- treatment: train on protein, eval transfer to PAWS-X ---
        split = protein.shuffle(seed=seed).train_test_split(test_size=2000, seed=seed)
        pro_train = split["train"].map(tk, batched=True)
        pro_val = split["test"].map(tk, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
        model.config.pad_token_id = tokenizer.eos_token_id
        targs = TrainingArguments(
            output_dir=f"{OUTDIR}/_tmp_batch", learning_rate=a.lr,
            lr_scheduler_type="constant_with_warmup", warmup_ratio=0.1,
            optim="adamw_torch", weight_decay=0.0, seed=seed,
            per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
            num_train_epochs=a.epochs, eval_strategy="no", save_strategy="no",
            logging_strategy="no", report_to=[], disable_tqdm=True)
        trainer = Trainer(model=model, args=targs, train_dataset=pro_train,
                          data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                          tokenizer=tokenizer)
        trainer.train()
        model.to(device)

        rec["protein_indomain"] = evaluate_split(model, pro_val, device)
        rec["transfer_pawsx_test"] = evaluate_split(model, paws_test, device)
        rec["transfer_pawsx_val"] = evaluate_split(model, paws_val, device)

        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(f"[seed {seed}] indomain={rec['protein_indomain']['acc_raw']} "
              f"transfer_raw={rec['transfer_pawsx_test']['acc_raw']} "
              f"transfer_flip={rec['transfer_pawsx_test']['acc_flipbest']} "
              f"baseline_flip={rec['baseline_pawsx_test']['acc_flipbest']}")
        del model, trainer; torch.cuda.empty_cache()

    fout.close()
    print("DONE", out_path)


if __name__ == "__main__":
    main()
