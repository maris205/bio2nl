"""
Tier-1 reverse transfer with a HARDER, structurally-richer source: remote homology
(+ a slice of standard homology). Tests the hypothesis that transferability depends on
source-task structural complexity: remote homology (<25% seq identity) demands abstract
structure-level reasoning, not shallow sequence overlap, so it may produce a transfer tail
where standard homology did not.

Source = mix_remote * remote_pairs + (1-mix_remote) * standard_pairs, subsampled to n_source.
Everything else mirrors 02_batch_distribution.py (same HPs, same PAWS-X target, per-seed baseline).

Run:
    source /etc/network_turbo ; export HF_ENDPOINT=https://hf-mirror.com
    python bio2nl/tier1_gpt2/03_batch_remote_source.py --n_seeds 10 --mix_remote 0.7 --n_source 20000
"""
import os, json, argparse, numpy as np, torch, pandas as pd
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
from sklearn.metrics import accuracy_score, f1_score

ROOT = "/root/autodl-tmp/bio-trans"
REMOTE_CSV = f"{ROOT}/biopaws/1-data/protein_pair_remote.csv"
STD_CSV = f"{ROOT}/biopaws/1-data/protein_pair_20k_length_restricted_balanced.csv"
OUTDIR = f"{ROOT}/bio2nl/results/benchmarks"
MODEL = "gpt2"
MAXLEN = 256


def tok_fn(tokenizer):
    def f(ex):
        return tokenizer(ex["sentence1"], ex["sentence2"],
                         truncation=True, max_length=MAXLEN, padding="max_length")
    return f


def build_source(mix_remote, n_source, seed):
    """Balanced mix of remote + standard homology, subsampled to n_source, label-balanced."""
    rem = pd.read_csv(REMOTE_CSV)
    std = pd.read_csv(STD_CSV)
    n_rem = int(round(n_source * mix_remote))
    n_std = n_source - n_rem
    def bal_sample(df, n, sd):
        half = n // 2
        pos = df[df.label == 1].sample(half, random_state=sd)
        neg = df[df.label == 0].sample(n - half, random_state=sd)
        return pd.concat([pos, neg])
    parts = []
    if n_rem > 0: parts.append(bal_sample(rem, n_rem, seed))
    if n_std > 0: parts.append(bal_sample(std, n_std, seed + 1))
    mix = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    return Dataset.from_pandas(mix[["sentence1", "sentence2", "label"]], preserve_index=False)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--mix_remote", type=float, default=0.7)
    ap.add_argument("--n_source", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tk = tok_fn(tokenizer)

    paws = load_dataset("paws-x", "en")
    paws_test = paws["test"].map(tk, batched=True)
    paws_val = paws["validation"].map(tk, batched=True)

    tag = f"remote{int(a.mix_remote*100)}"
    out_path = f"{OUTDIR}/tier1_dist_{tag}.jsonl"
    fout = open(out_path, "a")
    print(f"source: {a.mix_remote:.0%} remote + {1-a.mix_remote:.0%} standard, n={a.n_source}")
    print(f"writing -> {out_path}")

    for seed in range(a.start, a.start + a.n_seeds):
        set_seed(seed)
        rec = {"seed": seed, "mix_remote": a.mix_remote, "n_source": a.n_source}

        base = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
        base.config.pad_token_id = tokenizer.eos_token_id
        base.to(device)
        rec["baseline_pawsx_test"] = evaluate_split(base, paws_test, device)
        del base; torch.cuda.empty_cache()

        src = build_source(a.mix_remote, a.n_source, seed)
        split = src.train_test_split(test_size=2000, seed=seed)
        src_train = split["train"].map(tk, batched=True)
        src_val = split["test"].map(tk, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
        model.config.pad_token_id = tokenizer.eos_token_id
        targs = TrainingArguments(
            output_dir=f"{OUTDIR}/_tmp_remote", learning_rate=a.lr,
            lr_scheduler_type="constant_with_warmup", warmup_ratio=0.1,
            optim="adamw_torch", weight_decay=0.0, seed=seed,
            per_device_train_batch_size=a.bs, per_device_eval_batch_size=a.bs,
            num_train_epochs=a.epochs, eval_strategy="no", save_strategy="no",
            logging_strategy="no", report_to=[], disable_tqdm=True)
        trainer = Trainer(model=model, args=targs, train_dataset=src_train,
                          data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                          tokenizer=tokenizer)
        trainer.train()
        model.to(device)

        rec["source_indomain"] = evaluate_split(model, src_val, device)
        rec["transfer_pawsx_test"] = evaluate_split(model, paws_test, device)
        rec["transfer_pawsx_val"] = evaluate_split(model, paws_val, device)
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(f"[seed {seed}] src_indom={rec['source_indomain']['acc_raw']} "
              f"transfer_raw={rec['transfer_pawsx_test']['acc_raw']} "
              f"transfer_flip={rec['transfer_pawsx_test']['acc_flipbest']} "
              f"baseline_flip={rec['baseline_pawsx_test']['acc_flipbest']}")
        del model, trainer; torch.cuda.empty_cache()

    fout.close()
    print("DONE", out_path)


if __name__ == "__main__":
    main()
