"""
2x2 cross-modal transfer matrix on architecture-matched backbones.

Fill in the missing half of the ESM-2 story: fine-tune each backbone on BOTH
  - a natural-language structural task (PAWS-X paraphrase)   [done in eval_esm_on_nl.py]
  - a biological structural task (protein homology pairs)    [THIS script]
so we get, on the SAME encoders, one number for each cell:

              | eval: protein homology | eval: PAWS-X (NL) |
  ESM-2 (bio) |   home turf            |  0.64 (from other) |
  BERT  (NL)  |   NL->bio transfer?    |  0.73 (from other) |

Quantifies the asymmetry directly: how much biology does a language model retain vs
how much language does a biology model retain, on identical architecture.

Protein pairs: dnagpt/biopaws-style CSV (sentence1,sentence2,label). Balanced 20k.

Run:
    python eval_crossmodal_2x2.py --seeds 0,1,2
"""
import os, json, argparse, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
from sklearn.metrics import accuracy_score, f1_score

ROOT = "/root/autodl-tmp/bio-trans"
PROTEIN_CSV = f"{ROOT}/biopaws/1-data/protein_pair_20k_length_restricted_balanced.csv"
OUT = f"{ROOT}/bio2nl/results/benchmarks/crossmodal_2x2.jsonl"
MAXLEN = 256

MODELS = {
    "esm2_8M": "facebook/esm2_t6_8M_UR50D",
    "bert_small": "google/bert_uncased_L-4_H-256_A-4",
    "bert_tiny": "prajjwal1/bert-tiny",
}


def run_protein(model_name, seed, device):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.sep_token
    ds = load_dataset("csv", data_files=PROTEIN_CSV)["train"].shuffle(seed=seed)
    split = ds.train_test_split(test_size=2000, seed=seed)
    def f(ex):
        return tok(ex["sentence1"], ex["sentence2"], truncation=True, max_length=MAXLEN, padding="max_length")
    train = split["train"].map(f, batched=True)
    val = split["test"].map(f, batched=True)

    # unk fraction on protein (should be ~0 for ESM, high for BERT on rare AA strings? actually BERT has letters)
    unk = tok.unk_token_id
    uf = 0.0
    if unk is not None:
        s = val[:200]["sentence1"]; tot=u=0
        for t in s:
            ids = tok(t, truncation=True, max_length=MAXLEN).input_ids
            tot += len(ids); u += sum(1 for i in ids if i == unk)
        uf = round(u/max(tot,1), 4)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    args = TrainingArguments(
        output_dir=f"{ROOT}/bio2nl/results/benchmarks/_tmp_2x2", learning_rate=2e-5,
        per_device_train_batch_size=32, per_device_eval_batch_size=64,
        num_train_epochs=3, weight_decay=0.01, seed=seed,
        eval_strategy="no", save_strategy="no", logging_strategy="no",
        report_to=[], disable_tqdm=True, fp16=True)
    tr = Trainer(model=model, args=args, train_dataset=train,
                 data_collator=DataCollatorWithPadding(tokenizer=tok), tokenizer=tok)
    tr.train()
    pred = tr.predict(val)
    p = np.argmax(pred.predictions, 1); y = np.array(pred.label_ids)
    acc = accuracy_score(y, p)
    del model, tr; torch.cuda.empty_cache()
    return round(float(acc), 4), round(float(f1_score(y, p)), 4), uf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--models", default="esm2_8M,bert_small,bert_tiny")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fout = open(OUT, "a")
    for mkey in a.models.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            acc, f1, uf = run_protein(MODELS[mkey], seed, device)
            rec = {"task": "protein_homology", "model": mkey, "hf": MODELS[mkey],
                   "seed": seed, "acc": acc, "f1": f1, "unk_frac": uf}
            fout.write(json.dumps(rec) + "\n"); fout.flush()
            print(f"[protein] {mkey:11s} seed{seed} acc={acc} f1={f1} unk_frac={uf}")
    fout.close(); print("DONE", OUT)


if __name__ == "__main__":
    main()
