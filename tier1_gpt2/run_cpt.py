"""
Iso-token continued pretraining (CPT) for Exp2.
Continue-pretrain GPT-2 (causal LM, keeps the LM head) on one corpus file, matched steps/optimizer.
Produces a checkpoint per group (G1 text / G2 protein / G3 shuffled / G4 randomaa).
G0 base = stock gpt2, no CPT (handled at eval time).

Run:
    python run_cpt.py --group protein  --seed 0
    python run_cpt.py --group text     --seed 0
    ...
"""
import os, argparse, math
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments, set_seed)

ROOT = "/root/autodl-tmp/bio-trans/bio2nl"
CORPUS_DIR = f"{ROOT}/data/cpt_corpus"
CKPT_DIR = f"{ROOT}/tier1_gpt2/cpt_ckpts"
BLOCK = 512


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["text", "protein", "shuffled", "randomaa"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base", default="gpt2")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_steps", type=int, default=-1)  # set to force iso-step across groups
    a = ap.parse_args()
    set_seed(a.seed)

    corpus = f"{CORPUS_DIR}/cpt_{a.group}.txt"
    tag = a.base.replace("/", "_")          # size-namespaced: gpt2 / gpt2-medium
    out = f"{CKPT_DIR}/{tag}/{a.group}_seed{a.seed}"
    os.makedirs(out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    tok.pad_token = tok.eos_token

    ds = load_dataset("text", data_files=corpus, split="train")

    def tok_fn(ex):
        return tok(ex["text"])
    ds = ds.map(tok_fn, batched=True, remove_columns=["text"], num_proc=4)

    def group_texts(ex):
        concat = sum(ex["input_ids"], [])
        n = (len(concat) // BLOCK) * BLOCK
        ids = [concat[i:i+BLOCK] for i in range(0, n, BLOCK)]
        return {"input_ids": ids, "attention_mask": [[1]*BLOCK for _ in ids]}
    ds = ds.map(group_texts, batched=True, num_proc=4)
    print(f"[{a.group}] {len(ds)} blocks of {BLOCK} = {len(ds)*BLOCK:,} tokens")

    model = AutoModelForCausalLM.from_pretrained(a.base)
    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    targs = TrainingArguments(
        output_dir=out, overwrite_output_dir=True,
        num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        weight_decay=0.01, optim="adamw_torch", seed=a.seed,
        logging_steps=100, save_strategy="no", report_to=[], bf16=False, fp16=True)
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.save_model(out)
    tok.save_pretrained(out)
    print(f"[{a.group}] saved -> {out}  (steps={trainer.state.global_step})")


if __name__ == "__main__":
    main()
