"""
Structural-source suite: is forward Language->Biology transfer a property of STRUCTURAL training,
or specific to PAWS-X paraphrase? We fine-tune GPT-2 on several DISTINCT natural-language
structural tasks and evaluate zero-shot on protein homology. If multiple structural sources
transfer, the effect is structural, not paraphrase-specific (addresses the "paraphrase not
language ability" reviewer concern).

Sources (all natural-language, all structural-discrimination):
  paws-x  : paraphrase / word-order (original)
  cola    : grammatical acceptability (single sentence)
  dyck    : nested-bracket well-formedness (synthetic, pure structure)

Target: protein homology (standard, length-matched). Metric: flip-best acc + AUC, 5 seeds.

Run:
    python forward_source_suite.py --n_seeds 5
"""
import os, json, argparse, numpy as np, torch
from datasets import load_dataset, Dataset
import pandas as pd
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = "/root/autodl-tmp/bio-trans/bio2nl"
PROTEIN = "/root/autodl-tmp/bio-trans/biopaws/1-data/protein_pair_20k_length_restricted_balanced.csv"
EVAL_NL = f"{ROOT}/data/eval_nl"
OUT = f"{ROOT}/results/benchmarks/forward_source_suite.jsonl"
MAXLEN = 256


def load_source(name, tok, seed):
    def tok1(ex, k1, k2):
        return tok(ex[k1], ex[k2], truncation=True, max_length=MAXLEN, padding="max_length") if k2 \
            else tok(ex[k1], truncation=True, max_length=MAXLEN, padding="max_length")
    if name == "paws-x":
        raw = load_dataset("paws-x", "en")["train"]
        return raw.map(lambda e: tok1(e, "sentence1", "sentence2"), batched=True)
    if name == "cola":
        raw = load_dataset("nyu-mll/glue", "cola")["train"]
        return raw.map(lambda e: tok1(e, "sentence", None), batched=True)
    if name == "dyck":
        raw = load_dataset("json", data_files=f"{EVAL_NL}/dyck_L40_t3_train.jsonl", split="train")
        return raw.map(lambda e: tok1(e, "sentence", None), batched=True)
    raise ValueError(name)


def load_protein_test(tok, seed):
    df = pd.read_csv(PROTEIN).sample(3000, random_state=seed)
    ds = Dataset.from_pandas(df[["sentence1", "sentence2", "label"]], preserve_index=False)
    return ds.map(lambda e: tok(e["sentence1"], e["sentence2"], truncation=True,
                                max_length=MAXLEN, padding="max_length"), batched=True)


def evaluate(model, ds, device, bs=64):
    model.eval(); logits=[]; labels=[]
    for i in range(0, len(ds), bs):
        b = ds[i:i+bs]
        inp = {"input_ids": torch.tensor(b["input_ids"]).to(device),
               "attention_mask": torch.tensor(b["attention_mask"]).to(device)}
        with torch.no_grad(): out = model(**inp).logits.cpu().numpy()
        logits.append(out); labels.extend(b["label"])
    logits=np.concatenate(logits); labels=np.array(labels)
    p=np.argmax(logits,1); acc=accuracy_score(labels,p)
    auc=roc_auc_score(labels, logits[:,1]-logits[:,0]); auc=max(auc,1-auc)
    return {"acc_flipbest": round(float(max(acc,1-acc)),4), "auc": round(float(auc),4)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n_seeds",type=int,default=5); a=ap.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok=AutoTokenizer.from_pretrained("gpt2"); tok.pad_token=tok.eos_token
    fout=open(OUT,"a")
    for source in ["paws-x","cola","dyck"]:
        for seed in range(a.n_seeds):
            set_seed(seed)
            src=load_source(source,tok,seed)
            ptest=load_protein_test(tok,seed)
            m=AutoModelForSequenceClassification.from_pretrained("gpt2",num_labels=2)
            m.config.pad_token_id=tok.eos_token_id
            args=TrainingArguments(output_dir=f"{ROOT}/results/benchmarks/_tmp_fss",learning_rate=1e-5,
                lr_scheduler_type="constant_with_warmup",warmup_ratio=0.1,per_device_train_batch_size=32,
                num_train_epochs=4,seed=seed,eval_strategy="no",save_strategy="no",logging_strategy="no",
                report_to=[],disable_tqdm=True,fp16=True)
            Trainer(model=m,args=args,train_dataset=src,
                    data_collator=DataCollatorWithPadding(tokenizer=tok),tokenizer=tok).train()
            m.to(device)
            rec={"source":source,"seed":seed,"target":"protein_standard"}; rec.update(evaluate(m,ptest,device))
            fout.write(json.dumps(rec)+"\n"); fout.flush()
            print(f"[{source} seed{seed}] protein AUC={rec['auc']} flip={rec['acc_flipbest']}")
            del m; torch.cuda.empty_cache()
    fout.close(); print("DONE")


if __name__=="__main__":
    main()
