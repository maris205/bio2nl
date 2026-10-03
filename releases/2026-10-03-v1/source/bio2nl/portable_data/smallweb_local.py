# Adapted from the accepted v2 builder: same token-budget and EOS algorithm.
# Original source SHA-256 is retained in adapter_provenance.json.
# Only acquisition/loading and provenance helper paths are made explicitly local.
"""Prepare exact English-only GPT-2-tokenized SmallWeb data from the pinned OWT shard.

This preserves the historical GPT-2 vocabulary/model starting architecture while
rebuilding the text source and fixed held-out partitions. It does not train.
"""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
import hashlib

def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4<<20),b""):h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")

MODEL_ID = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
TARGETS = {"train":49_999_872, "validation":262_144, "test":262_144}
TOKENIZER_FILES = {"merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"}


def checked(path, root=None):
    path=Path(path).absolute()
    for part in (path,*path.parents):
        if part.is_symlink():
            raise ValueError("Symlink local path is forbidden: "+str(path))
    resolved=path.resolve()
    if root is not None and not resolved.is_relative_to(root):
        raise ValueError("Local input/output leaves reconstruction root")
    return resolved


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); a=p.parse_args(); root=checked(a.root)
    # Check every destination and all source paths before loading or writing.
    output_dir=checked(root/"tokenized/gpt2_smallweb",root)
    config_path=checked(root/"configs/gpt2_smallweb_data.json",root)
    if output_dir.exists() or config_path.exists():
        raise FileExistsError("Refuse existing SmallWeb encoding or config")
    for split in TARGETS:
        if not checked(root/"data/corpora/english"/(split+".jsonl.gz"),root).is_file():
            raise FileNotFoundError("Missing local English corpus split: "+split)
    tokenizer_dir=checked(root/"tokenizers/gpt2_pinned",root)
    recorded=json.loads(checked(tokenizer_dir/"metadata.json",root).read_text())
    if recorded["model_id"]!=MODEL_ID or recorded["revision"]!=REVISION:
        raise ValueError("Pinned external GPT-2 tokenizer identity differs")
    tokenizer_hashes=recorded["tokenizer_files_sha256"]
    if set(tokenizer_hashes)!=TOKENIZER_FILES:
        raise ValueError("Pinned tokenizer must bind exactly its five expected files")
    if {p.name for p in tokenizer_dir.iterdir()} != TOKENIZER_FILES | {"metadata.json"}:
        raise ValueError("Unexpected or missing local tokenizer member")
    for name,expected in tokenizer_hashes.items():
        path=checked(tokenizer_dir/name,root)
        if not path.is_file() or digest(path)!=expected:
            raise ValueError("Pinned tokenizer member identity differs: "+name)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True, local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.vocab_size!=50257 or tokenizer.eos_token_id!=50256:
        raise ValueError("Pinned GPT-2 vocabulary or EOS differs")
    # Preserve authenticated external assets; do not save a path-dependent tokenizer copy.
    source=root/"data/corpora/english"
    for split,target in TARGETS.items():
        input_path=source/(split+".jsonl.gz")
        output_dir=root/"tokenized/gpt2_smallweb"; output_dir.mkdir(parents=True,exist_ok=True)
        binary=output_dir/(split+".bin"); partial=binary.with_suffix(".bin.partial")
        index=output_dir/(split+".index.jsonl"); index_partial=index.with_suffix(".jsonl.partial")
        count=nrecords=truncated=0
        with gzip.open(input_path,"rt") as stream, partial.open("wb") as out, index_partial.open("w") as idx:
            for line in stream:
                row=json.loads(line)
                ids=tokenizer.encode(row["text"], add_special_tokens=False)
                if tokenizer.eos_token_id in ids:
                    raise RuntimeError("Source text unexpectedly encoded as GPT-2 EOS")
                remain=target-count
                if remain<=0: break
                did_truncate=len(ids)+1>remain
                if did_truncate: ids=ids[:remain-1];truncated+=1
                ids.append(tokenizer.eos_token_id)
                np.asarray(ids,dtype="<u2").tofile(out)
                idx.write(json.dumps(dict(record_id=row["record_id"],offset=count,length=len(ids),
                                          eos_offset=count+len(ids)-1,truncated=did_truncate))+"\n")
                count+=len(ids);nrecords+=1
                if count==target: break
        if count!=target:
            partial.unlink(missing_ok=True);index_partial.unlink(missing_ok=True)
            raise RuntimeError(f"Insufficient unique OpenWebText tokens for {split}: {count} < {target}")
        partial.replace(binary); index_partial.replace(index)
        write_json(output_dir/(split+".json"),dict(tokens=count,records=nrecords,truncated_last_records=truncated,
            split=split,dtype="little-endian uint16",eos_token_id=tokenizer.eos_token_id,
            file=str(binary.relative_to(root)),sha256=digest(binary),index=str(index.relative_to(root)),
            index_sha256=digest(index),source_sha256=digest(input_path),tokenizer_files_sha256=tokenizer_hashes))
        print(split,count,nrecords,flush=True)
    write_json(root/"configs/gpt2_smallweb_data.json",dict(source="OpenWebText one pinned shard, train split only for training",
        tokenizer={"repo":MODEL_ID,"revision":REVISION},targets=TARGETS,block_size=512,
        blocks={s:n//512 for s,n in TARGETS.items()},epochs=3,
        note="Historical entrypoint help claimed 200M tokens, but --tokens was not an argument and the local corpus was much smaller. New input is fixed to 50M tokens per epoch, with separate held-out files and per-record EOS."))


if __name__=="__main__":main()
