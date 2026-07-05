"""
THE symmetry experiment: run BOTH transfer directions under one identical protocol.

Reframed paper question: "Is structural transfer between natural language and biological
sequences symmetric?" To answer cleanly we must measure both directions with the SAME model,
SAME code, SAME eval — not compare our reverse against the forward paper's reported number.

Direction NL->Bio : train GPT-2 classifier on PAWS-X (paraphrase) -> zero-shot protein homology
Direction Bio->NL : train GPT-2 classifier on protein homology   -> zero-shot PAWS-X
(each also reports in-domain acc as a sanity check that training worked)

Shared schema: (sentence1, sentence2, label in {0,1}); "structural difference detection".
Label-flip guard applied to zero-shot transfer (binary head may invert). Multiple seeds.

Run:
    source /etc/network_turbo ; export HF_ENDPOINT=https://hf-mirror.com
    python 05_bidirectional_symmetry.py --n_seeds 15
"""
import os, json, argparse, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
from sklearn.metrics import accuracy_score, f1_score

ROOT = "/root/autodl-tmp/bio-trans"
PROTEIN_CSV = f"{ROOT}/biopaws/1-data/protein_pair_20k_length_restricted_balanced.csv"
OUT = f"{ROOT}/bio2nl/results/benchmarks/bidirectional_symmetry.jsonl"
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
    acc_flip = max(acc, accuracy_score(labels, 1 - preds))
    return {"acc_raw": round(float(acc), 4), "acc_flipbest": round(float(acc_flip), 4)}


def train_classifier(train_ds, seed, epochs, bs, lr):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id
    args = TrainingArguments(
        output_dir=f"{ROOT}/bio2nl/results/benchmarks/_tmp_sym", learning_rate=lr,
        lr_scheduler_type="constant_with_warmup", warmup_ratio=0.1, optim="adamw_torch",
        weight_decay=0.0, seed=seed, per_device_train_batch_size=bs,
        num_train_epochs=epochs, eval_strategy="no", save_strategy="no",
        logging_strategy="no", report_to=[], disable_tqdm=True, fp16=True)
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    tr = Trainer(model=model, args=args, train_dataset=train_ds,
                 data_collator=DataCollatorWithPadding(tokenizer=tok), tokenizer=tok)
    tr.train()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    tk = tok_fn(tok)

    # datasets (tokenized once)
    paws = load_dataset("paws-x", "en")
    paws_train = paws["train"].map(tk, batched=True)
    paws_test = paws["test"].map(tk, batched=True)
    protein = load_dataset("csv", data_files=PROTEIN_CSV)["train"]

    fout = open(OUT, "a")
    for seed in range(a.n_seeds):
        set_seed(seed)
        psplit = protein.shuffle(seed=seed).train_test_split(test_size=2000, seed=seed)
        pro_train = psplit["train"].map(tk, batched=True)
        pro_test = psplit["test"].map(tk, batched=True)

        # --- NL -> Bio: train on PAWS-X, zero-shot protein ---
        m = train_classifier(paws_train, seed, a.epochs, a.bs, a.lr); m.to(device)
        rec = {"seed": seed, "direction": "NL->Bio",
               "indomain": evaluate_split(m, paws_test, device),
               "transfer": evaluate_split(m, pro_test, device)}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(f"[seed {seed}] NL->Bio  indom(paws)={rec['indomain']['acc_raw']} "
              f"transfer(protein) raw={rec['transfer']['acc_raw']} flip={rec['transfer']['acc_flipbest']}")
        del m; torch.cuda.empty_cache()

        # --- Bio -> NL: train on protein, zero-shot PAWS-X ---
        m = train_classifier(pro_train, seed, a.epochs, a.bs, a.lr); m.to(device)
        rec = {"seed": seed, "direction": "Bio->NL",
               "indomain": evaluate_split(m, pro_test, device),
               "transfer": evaluate_split(m, paws_test, device)}
        fout.write(json.dumps(rec) + "\n"); fout.flush()
        print(f"[seed {seed}] Bio->NL  indom(protein)={rec['indomain']['acc_raw']} "
              f"transfer(paws) raw={rec['transfer']['acc_raw']} flip={rec['transfer']['acc_flipbest']}")
        del m; torch.cuda.empty_cache()

    fout.close()
    print("DONE", OUT)


if __name__ == "__main__":
    main()
