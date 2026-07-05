"""
Tier-1 reverse transfer, BACKBONE-ABLATION version.
Same pipeline as 02_batch_distribution.py (train GPT-2 classifier on protein homology,
test zero-shot on PAWS-X English), but the pretrained BACKBONE is a CLI arg so we can
compare what the backbone was pretrained on:

    gpt2                      -> English only        (baseline, already run in 02)
    dnagpt/gpt2_gene_v1       -> DNA+protein only     (bio-only backbone)
    dnagpt/gene_eng_gpt2_v2   -> DNA+protein+English  (bio+English backbone)  <-- key

Holds fine-tune task + eval fixed; varies only the backbone corpus. Tests whether a
backbone that RETAINS English unlocks reverse transfer (note #2/#3), and whether a
bio-only backbone behaves like the from-scratch bio-only classifier.

Run:
    source /etc/network_turbo ; export HF_ENDPOINT=https://hf-mirror.com HF_TOKEN=...
    python bio2nl/tier1_gpt2/04_batch_backbone.py --model dnagpt/gene_eng_gpt2_v2 --n_seeds 10
"""
import os, json, argparse, re, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
from sklearn.metrics import accuracy_score, f1_score

ROOT = "/root/autodl-tmp/bio-trans"
PROTEIN_CSV = f"{ROOT}/biopaws/1-data/protein_pair_20k_length_restricted_balanced.csv"
OUTDIR = f"{ROOT}/bio2nl/results/benchmarks"
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
    acc_flip = max(acc, accuracy_score(labels, 1 - preds))
    return {"acc_raw": round(float(acc), 4), "acc_flipbest": round(float(acc_flip), 4),
            "f1": round(float(f1_score(labels, preds, zero_division=0)), 4)}


def load_clf(model_name, tokenizer):
    m = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    if m.config.pad_token_id is None:
        m.config.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tk = tok_fn(tokenizer)

    paws = load_dataset("paws-x", "en")
    paws_test = paws["test"].map(tk, batched=True)
    paws_val = paws["validation"].map(tk, batched=True)
    protein = load_dataset("csv", data_files=PROTEIN_CSV)["train"]

    slug = re.sub(r"[^a-zA-Z0-9]+", "_", a.model)
    out_path = f"{OUTDIR}/tier1_backbone_{slug}.jsonl"
    fout = open(out_path, "a")
    print(f"backbone={a.model}  writing -> {out_path}")

    for seed in range(a.start, a.start + a.n_seeds):
        set_seed(seed)
        rec = {"seed": seed, "model": a.model}

        base = load_clf(a.model, tokenizer); base.to(device)
        rec["baseline_pawsx_test"] = evaluate_split(base, paws_test, device)
        del base; torch.cuda.empty_cache()

        split = protein.shuffle(seed=seed).train_test_split(test_size=2000, seed=seed)
        pro_train = split["train"].map(tk, batched=True)
        pro_val = split["test"].map(tk, batched=True)

        model = load_clf(a.model, tokenizer)
        targs = TrainingArguments(
            output_dir=f"{OUTDIR}/_tmp_bb", learning_rate=a.lr,
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
        print(f"[seed {seed}] indom={rec['protein_indomain']['acc_raw']} "
              f"transfer_raw={rec['transfer_pawsx_test']['acc_raw']} "
              f"transfer_flip={rec['transfer_pawsx_test']['acc_flipbest']} "
              f"baseline_flip={rec['baseline_pawsx_test']['acc_flipbest']}")
        del model, trainer; torch.cuda.empty_cache()

    fout.close()
    print("DONE", out_path)


if __name__ == "__main__":
    main()
