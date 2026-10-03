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
from common import digest, write_json

MODEL_ID = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
TARGETS = {"train":49_999_872, "validation":262_144, "test":262_144}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); a=p.parse_args(); root=a.root
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer_dir=root/"tokenizers/gpt2_pinned"; tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    tokenizer_hashes={p.name:digest(p) for p in sorted(tokenizer_dir.iterdir()) if p.is_file()}
    write_json(tokenizer_dir/"metadata.json",dict(model_id=MODEL_ID,revision=REVISION,
        tokenizer_files_sha256=tokenizer_hashes, vocab_size=tokenizer.vocab_size,
        bos_token=tokenizer.bos_token,eos_token=tokenizer.eos_token,eos_token_id=tokenizer.eos_token_id,
        normalization="Pinned GPT-2 tokenizer defaults", added_tokens=[]))
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
