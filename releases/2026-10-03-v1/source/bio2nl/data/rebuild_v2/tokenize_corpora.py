"""Shared pure-byte/BPE tokenizers and exact-size, record-indexed token files.

This prepares data only. It never starts model training.
"""
import argparse
import gzip
import itertools
import json
from pathlib import Path
import numpy as np
from common import digest, write_json

SPECIAL = ["<|pad_v2|>", "<|eos_v2|>", "<|pair_v2|>", "<|unk_v2|>"]
CONDITIONS = ("english", "protein", "shuffled", "randomaa", "dna")
# One shared budget that fits every eligible corpus/tokenizer pair without
# cycling. The protein mixed-BPE train corpus contains ~19.9M usable tokens.
TARGETS = {"train":16_777_216, "validation":262_144, "test":262_144}


def records(root, condition, split):
    completed = root / "metadata" / f"corpora_{condition}.json"
    if not completed.exists():
        raise RuntimeError(f"Corpus build has not completed: {condition}")
    with gzip.open(root / "data/corpora" / condition / (split + ".jsonl.gz"), "rt") as stream:
        for line in stream:
            yield json.loads(line)


def build_byte(root):
    from tokenizers import Tokenizer, pre_tokenizers, decoders
    from tokenizers.models import BPE
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(BPE(vocab={v:i for i,v in enumerate(SPECIAL + alphabet)}, merges=[], unk_token=SPECIAL[-1]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    folder = root / "tokenizers/pure_byte"; folder.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(folder / "tokenizer.json"))
    config = dict(family="pure_byte", vocabulary_size=260, learned_merges=0,
                  special_tokens={s:tokenizer.token_to_id(s) for s in SPECIAL},
                  eos_token_id=1, pad_token_id=0, pair_separator_id=2,
                  normalization=None, ordinary_token_semantics="Exactly one UTF-8 byte per token",
                  tokenizer_sha256=digest(folder / "tokenizer.json"))
    write_json(folder / "metadata.json", config)
    probes = ["English words\nACDEFGHIKLMNPQRSTVWY", "中文 Δ🙂 ACGT", "\t spaces ", "()[]{}"]
    assert all(len(tokenizer.encode(t).ids) == len(t.encode()) and tokenizer.decode(tokenizer.encode(t).ids) == t for t in probes)
    return tokenizer


def prefix_texts(root, condition, byte_budget):
    selected, nbytes, used = [], 0, []
    for row in records(root, condition, "train"):
        remaining = byte_budget - nbytes
        if remaining <= 0: break
        data = row["text"].encode()[:remaining]
        text = data.decode("utf-8", errors="ignore")
        if text:
            selected.append(text); nbytes += len(text.encode()); used.append(row["record_id"])
    return selected, nbytes, used


def build_bpe(root, sample_bytes):
    from tokenizers import Tokenizer, trainers, pre_tokenizers, decoders
    from tokenizers.models import BPE
    english, ne, eids = prefix_texts(root, "english", sample_bytes)
    protein, np_, pids = prefix_texts(root, "protein", ne)
    if np_ < ne:
        english, ne, eids = prefix_texts(root, "english", np_)
        protein, np_, pids = prefix_texts(root, "protein", ne)
    assert ne == np_ and ne > 0
    tokenizer = Tokenizer(BPE(unk_token=SPECIAL[-1]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=32000, min_frequency=2,
                                   special_tokens=SPECIAL,
                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    # Equal UTF-8 bytes, not an equal number of unequal-length documents.
    iterator = (x for pair in itertools.zip_longest(english, protein) for x in pair if x is not None)
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    folder = root / "tokenizers/mixed_bpe"; folder.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(folder / "tokenizer.json"))
    write_json(folder / "training_records.json", {"english":eids, "protein":pids})
    write_json(folder / "metadata.json", dict(family="mixed_bpe", vocabulary_size=tokenizer.get_vocab_size(),
               normalization=None, initial_alphabet_size=256, special_tokens={s:tokenizer.token_to_id(s) for s in SPECIAL},
               eos_token_id=1, pad_token_id=0, pair_separator_id=2,
               training_utf8_bytes={"english":ne,"protein":np_}, training_split="train only",
               tokenizer_sha256=digest(folder / "tokenizer.json"),
               training_corpus_sha256={c:digest(root / "data/corpora" / c / "train.jsonl.gz") for c in ("english", "protein")}))
    return tokenizer


def encode_one(root, tokenizer, family, condition, split, target):
    folder = root / "tokenized" / family / condition; folder.mkdir(parents=True, exist_ok=True)
    binary = folder / (split + ".bin")
    index = folder / (split + ".index.jsonl")
    partial = binary.with_suffix(".bin.partial"); index_tmp=index.with_suffix(".jsonl.partial")
    eos = tokenizer.token_to_id(SPECIAL[1]); unk = tokenizer.token_to_id(SPECIAL[-1])
    count = nrecords = clipped = 0; observed=set()
    with partial.open("wb") as output, index_tmp.open("w") as metadata:
        for row in records(root, condition, split):
            ids = tokenizer.encode(row["text"]).ids
            if unk in ids: raise RuntimeError(f"Unexpected unknown token: {row['record_id']}")
            if any(tokenizer.token_to_id(s) in ids for s in SPECIAL):
                raise RuntimeError(f"Input contains an interpreted special token: {row['record_id']}")
            remaining = target - count
            if remaining <= 0: raise RuntimeError("Unreachable token boundary accounting")
            truncated = len(ids)+1 > remaining
            if truncated: ids = ids[:remaining-1]; clipped += 1
            ids.append(eos)
            np.asarray(ids, dtype="<u2").tofile(output)
            metadata.write(json.dumps(dict(record_id=row["record_id"], offset=count, length=len(ids),
                                          eos_offset=count+len(ids)-1, truncated=truncated)) + "\n")
            count += len(ids); nrecords += 1; observed.update(ids)
            if count == target: break
    if count != target:
        raise RuntimeError(f"Insufficient UNIQUE source tokens: {family}/{condition}/{split}: {count} < {target}; no repetition or silent budget reduction allowed")
    partial.replace(binary); index_tmp.replace(index)
    result = dict(family=family, condition=condition, split=split, tokens=count, records=nrecords,
                  truncated_last_records=clipped, dtype="little-endian uint16", eos_token_id=eos,
                  distinct_token_ids=len(observed), max_token_id=max(observed),
                  file=str(binary.relative_to(root)), sha256=digest(binary),
                  index=str(index.relative_to(root)), index_sha256=digest(index),
                  source_sha256=digest(root / "data/corpora" / condition / (split + ".jsonl.gz")),
                  tokenizer_sha256=digest(root / "tokenizers" / family / "tokenizer.json"))
    write_json(folder / (split + ".json"), result)
    print(f"{family}/{condition}/{split}: {count:,} tokens, {nrecords:,} records", flush=True)
    return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=("build-tokenizers", "encode", "all"), default="all")
    p.add_argument("--family", choices=("pure_byte", "mixed_bpe", "all"), default="all")
    p.add_argument("--sample-bytes", type=int, default=16_777_216)
    p.add_argument("--conditions", default=",".join(CONDITIONS)); a=p.parse_args()
    from tokenizers import Tokenizer
    for family in ("pure_byte", "mixed_bpe"):
        if a.family not in (family, "all"): continue
        if a.mode in ("build-tokenizers", "all"):
            build_byte(a.root) if family == "pure_byte" else build_bpe(a.root, a.sample_bytes)
        if a.mode in ("encode", "all"):
            tokenizer=Tokenizer.from_file(str(a.root / "tokenizers" / family / "tokenizer.json"))
            for condition in a.conditions.split(","):
                for split, target in TARGETS.items():
                    encode_one(a.root, tokenizer, family, condition, split, target)
    write_json(a.root / "configs/tokenization.json", dict(targets=TARGETS, block_size=512,
               training_blocks=TARGETS["train"]//512, conditions=list(CONDITIONS),
               serialization="text tokens then one EOS per record; no protein tags/spaces added",
               pair_serialization="tokens(A) + explicit pair separator + tokens(B) + EOS",
               padding="none in binary; all 32,768 training blocks have exactly 512 tokens",
               boundaries="index.jsonl records every source record boundary; packed training may attend to prior records unless trainer uses segment masking",
               held_out="separate validation/test bins; competence must never read train.bin",
               repetition="no cycling if any source falls short; command fails"))


if __name__ == "__main__": main()
