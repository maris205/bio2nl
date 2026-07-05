"""
M3b — Forward difference-operator heads (the other half of M3, same metric).

M3 showed: PROTEIN training does NOT induce heads that localize NL differences.
M3b (this) tests the forward claim on identical footing: does LANGUAGE (PAWS) training induce
heads that localize BIOLOGICAL differences (mutated residues between homologous proteins)?

Procedure:
  1. Fine-tune GPT-2 on PAWS-X (classification) — the forward model.
  2. On homologous protein pairs (which differ only at mutated residues), measure per-head
     difference-attention: excess attention the last token puts on differing vs matched positions.
  3. Compare to base GPT-2 (control). If PAWS training GROWS bio-difference heads (score rises
     above base), that is the forward difference-operator — the mechanistic partner to M3's null.

Same difference-attention metric as M3_difference_heads.py, so forward/reverse are comparable.

Run:
    python M3b_forward_difference_heads.py --n 300
"""
import os, json, argparse, numpy as np, torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

ROOT = "/root/autodl-tmp/bio-trans"
PROTEIN_CSV = f"{ROOT}/biopaws/1-data/protein_pair_20k_length_restricted_balanced.csv"
OUT = f"{ROOT}/bio2nl/results/router/M3b_forward_difference_heads.jsonl"
MAXLEN = 128


def diff_positions(a, b):
    n = min(len(a), len(b))
    return set(i for i in range(n) if a[i] != b[i])


@torch.no_grad()
def diff_attention(model, tok, pairs, device):
    """[n_layer, n_head] mean (attn on differing positions - attn on matched), last-token query."""
    model.eval()
    nl, nh = model.config.n_layer, model.config.n_head
    acc = np.zeros((nl, nh)); cnt = 0
    for s1, s2 in pairs:
        i1 = tok(s1, truncation=True, max_length=MAXLEN).input_ids
        i2 = tok(s2, truncation=True, max_length=MAXLEN).input_ids
        dset = diff_positions(i1, i2)
        enc = tok(s1, s2, truncation=True, max_length=MAXLEN, return_tensors="pt").to(device)
        L = enc["input_ids"].shape[1]
        didx = [d for d in dset if d < L]
        if not didx or len(didx) >= L:
            continue
        midx = [i for i in range(L) if i not in didx]
        out = model(**enc, output_attentions=True)
        for l in range(nl):
            last = out.attentions[l][0][:, -1, :]  # [head, L]
            acc[l] += (last[:, didx].mean(1) - last[:, midx].mean(1)).cpu().numpy()
        cnt += 1
    return acc / max(cnt, 1), cnt


def train_paws_gpt2(seed, device, epochs=4, bs=32, lr=1e-5):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    def f(ex): return tok(ex["sentence1"], ex["sentence2"], truncation=True, max_length=256, padding="max_length")
    paws = load_dataset("paws-x", "en")["train"].map(f, batched=True)
    model = AutoModelForSequenceClassification.from_pretrained("gpt2", num_labels=2)
    model.config.pad_token_id = model.config.eos_token_id
    args = TrainingArguments(output_dir=f"{ROOT}/bio2nl/results/router/_tmp_m3b", learning_rate=lr,
        lr_scheduler_type="constant_with_warmup", warmup_ratio=0.1, per_device_train_batch_size=bs,
        num_train_epochs=epochs, seed=seed, eval_strategy="no", save_strategy="no",
        logging_strategy="no", report_to=[], disable_tqdm=True, fp16=True)
    Trainer(model=model, args=args, train_dataset=paws,
            data_collator=DataCollatorWithPadding(tokenizer=tok), tokenizer=tok).train()
    # return the base transformer (AutoModel view) with the fine-tuned weights
    save = f"{ROOT}/bio2nl/results/router/_tmp_m3b_model"
    model.save_pretrained(save); tok.save_pretrained(save)
    return save


def homolog_pairs(n, seed):
    ds = load_dataset("csv", data_files=PROTEIN_CSV)["train"]
    ds = ds.filter(lambda e: e["label"] == 1).shuffle(seed=seed).select(range(n))
    return list(zip(ds["sentence1"], ds["sentence2"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    pairs = homolog_pairs(a.n, a.seed)   # homologs differ by mutations
    fout = open(OUT, "a")

    # base GPT-2 (control)
    base = AutoModel.from_pretrained("gpt2").to(device)
    sc, cnt = diff_attention(base, tok, pairs, device)
    top = np.dstack(np.unravel_index(np.argsort(sc.ravel())[-5:], sc.shape))[0]
    fout.write(json.dumps({"model": "base_gpt2", "n_pairs": cnt, "max_bio_diff_attn": round(float(sc.max()),4),
        "top_heads": [[int(l),int(h),round(float(sc[l,h]),4)] for l,h in top]}) + "\n"); fout.flush()
    print(f"[base_gpt2] max bio-difference-attn={sc.max():.4f} top head L{top[-1][0]}H{top[-1][1]}")
    del base; torch.cuda.empty_cache()

    # PAWS-tuned GPT-2 (forward model)
    path = train_paws_gpt2(a.seed, device)
    fwd = AutoModel.from_pretrained(path).to(device)
    sc, cnt = diff_attention(fwd, tok, pairs, device)
    top = np.dstack(np.unravel_index(np.argsort(sc.ravel())[-5:], sc.shape))[0]
    fout.write(json.dumps({"model": "paws_tuned_gpt2", "n_pairs": cnt, "max_bio_diff_attn": round(float(sc.max()),4),
        "top_heads": [[int(l),int(h),round(float(sc[l,h]),4)] for l,h in top]}) + "\n"); fout.flush()
    print(f"[paws_tuned] max bio-difference-attn={sc.max():.4f} top head L{top[-1][0]}H{top[-1][1]}")
    fout.close(); print("DONE", OUT)


if __name__ == "__main__":
    main()
