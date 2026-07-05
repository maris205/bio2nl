"""
M2 — Learning dynamics of transfer (THE core mechanism figure).

Question: is the asymmetry a matter of *what can be represented* or *what can be learned*?
We track, DURING training, how transfer performance evolves step-by-step for both directions:
  NL->Bio : train GPT-2 classifier on PAWS-X, every K steps eval zero-shot on protein
  Bio->NL : train GPT-2 classifier on protein, every K steps eval zero-shot on PAWS-X

Hypothesis: forward transfer EMERGES early and rises; reverse transfer stays flat at chance
throughout — i.e. representations may align (see M1) but the *learning dynamics* are asymmetric.

Uses a Trainer callback to snapshot transfer accuracy on a fixed held-out target set.

Run:
    python M2_learning_dynamics.py --seeds 0,1,2 --eval_every 60
"""
import os, json, argparse, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          TrainerCallback, set_seed)
from sklearn.metrics import accuracy_score

ROOT = "/root/autodl-tmp/bio-trans"
PROTEIN_CSV = f"{ROOT}/biopaws/1-data/protein_pair_20k_length_restricted_balanced.csv"
OUT = f"{ROOT}/bio2nl/results/router/M2_learning_dynamics.jsonl"
MODEL = "gpt2"; MAXLEN = 256


def tokz(tok):
    def f(ex): return tok(ex["sentence1"], ex["sentence2"], truncation=True,
                          max_length=MAXLEN, padding="max_length")
    return f


@torch.no_grad()
def transfer_acc(model, ds, device, bs=64):
    model.eval()
    preds, labels = [], []
    for i in range(0, len(ds), bs):
        b = ds[i:i+bs]
        inp = {"input_ids": torch.tensor(b["input_ids"]).to(device),
               "attention_mask": torch.tensor(b["attention_mask"]).to(device)}
        preds.extend(torch.argmax(model(**inp).logits, -1).cpu().numpy())
        labels.extend(b["label"])
    preds, labels = np.array(preds), np.array(labels)
    a = accuracy_score(labels, preds)
    return max(a, accuracy_score(labels, 1 - preds))  # flip-best (same as main eval)


class TransferProbe(TrainerCallback):
    def __init__(self, target_ds, device, every, direction, seed, trajectory):
        self.t = target_ds; self.dev = device; self.every = every
        self.direction = direction; self.seed = seed; self.traj = trajectory
    def on_step_end(self, args, state, control, model=None, **kw):
        if state.global_step % self.every == 0:
            acc = transfer_acc(model, self.t, self.dev)
            self.traj.append({"step": state.global_step, "transfer_flip": round(float(acc), 4)})


def run_direction(direction, train_ds, target_ds, seed, device, epochs, bs, lr, every):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    traj = []
    args = TrainingArguments(
        output_dir=f"{ROOT}/bio2nl/results/router/_tmp_m2", learning_rate=lr,
        lr_scheduler_type="constant_with_warmup", warmup_ratio=0.1,
        per_device_train_batch_size=bs, num_train_epochs=epochs, seed=seed,
        eval_strategy="no", save_strategy="no", logging_strategy="no",
        report_to=[], disable_tqdm=True, fp16=True)
    cb = TransferProbe(target_ds, device, every, direction, seed, traj)
    tr = Trainer(model=model, args=args, train_dataset=train_ds,
                 data_collator=DataCollatorWithPadding(tokenizer=tok), tokenizer=tok,
                 callbacks=[cb])
    # step 0 baseline
    model.to(device); traj.append({"step": 0, "transfer_flip": round(float(transfer_acc(model, target_ds, device)), 4)})
    tr.train()
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--eval_every", type=int, default=60)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL); tok.pad_token = tok.eos_token
    tk = tokz(tok)

    paws = load_dataset("paws-x", "en")
    paws_train = paws["train"].map(tk, batched=True)
    paws_test = paws["test"].map(tk, batched=True)
    protein = load_dataset("csv", data_files=PROTEIN_CSV)["train"]

    fout = open(OUT, "a")
    for seed in [int(s) for s in a.seeds.split(",")]:
        psplit = protein.shuffle(seed=seed).train_test_split(test_size=2000, seed=seed)
        pro_train = psplit["train"].map(tk, batched=True)
        pro_test = psplit["test"].map(tk, batched=True)
        # NL->Bio: train paws, target protein
        traj = run_direction("NL->Bio", paws_train, pro_test, seed, device, a.epochs, a.bs, a.lr, a.eval_every)
        fout.write(json.dumps({"direction": "NL->Bio", "seed": seed, "trajectory": traj}) + "\n"); fout.flush()
        print(f"[seed {seed}] NL->Bio final={traj[-1]}")
        # Bio->NL: train protein, target paws
        traj = run_direction("Bio->NL", pro_train, paws_test, seed, device, a.epochs, a.bs, a.lr, a.eval_every)
        fout.write(json.dumps({"direction": "Bio->NL", "seed": seed, "trajectory": traj}) + "\n"); fout.flush()
        print(f"[seed {seed}] Bio->NL final={traj[-1]}")
    fout.close(); print("DONE", OUT)


if __name__ == "__main__":
    main()
