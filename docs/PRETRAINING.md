# Pretraining data: EP, ES and EE

The dedicated release is [dnagpt/bio2nl-pretraining-2026-10](https://huggingface.co/datasets/dnagpt/bio2nl-pretraining-2026-10/tree/003e291abac2ba411b4054a7f60d50567e7f1dfa), revision `003e291abac2ba411b4054a7f60d50567e7f1dfa`. It covers the October 8 from-scratch experiment; the older October 3 experiment has a different budget and pretraining seed hierarchy.

| Pool | Records | Tokens in entire pool (including EOS) | Tokens consumed per stream |
|---|---:|---:|---:|
| English common | 361,356 | 446,730,942 | 134,217,728 |
| English extra | 361,643 | 447,059,806 | 134,217,728 |
| Natural protein | 137,736 | 19,918,409 | 134,217,728 |
| Shuffled protein | 137,736 | 20,374,008 | 134,217,728 |

**EP = common English + natural protein; ES = common English + shuffled protein; EE = common English + extra English.** Each has 268,435,456 input tokens, 16,384 optimizer updates, one pretraining seed (0), 12 layers/12 heads/768 hidden dimensions/512 context, and 110,025,216 language-model parameters. Each then receives the same high-identity V2 protein-pair SFT with 20 seeds. EE has twice the English token exposure; repeated protein sequences and new English documents are not equal information budgets.

The shared 32,000-entry byte-level BPE tokenizer uses 16 MiB English + 16 MiB natural protein for tokenizer training. That is separate from model pretraining. Token IDs are PAD=0, EOS=1, SEP=2, with no UNK. Protein streams repeat their original pools to the fixed budget; English streams are non-repeated prefixes. Binaries are little-endian uint16, with EOS appended per record. Training separately shuffles 512-token blocks.

## What can be downloaded directly

- Natural and composition-preserving shuffled protein JSONL, 137,736 records each. These were already in the task-data repository; the new companion makes them separately loadable.
- Their complete token pools and exact consumed streams.
- The shared tokenizer, original preparation code and source/split/filter receipts.
- All four pools' text-free row indices, hashes and per-record exposure ledgers.

Protein pretraining uses the accepted canonical Swiss-Prot 2026_03 **training** corpus with its recorded component split and SCOPe domain exclusion. It is not the length-restricted 5,000-query BLAST task bank. The source receipts describe fragment/noncanonical filtering, exact deduplication and operational homology screening. They do not establish universal absence of biological overlap.

## English reconstruction

English data come from exactly OpenWebText shards 0–7 at revision `79d93d786212f7344586290adb811d4ae6a1762c`, with recorded document/sentence exclusions and deduplication. The source's CC0 packaging does not transfer ownership of the extracted web text. We therefore publish exact acquisition and reconstruction instead of copying English documents or reversible English token bins into this repository. See the [upstream source statement](https://skylion007.github.io/OpenWebTextCorpus/) and source-specific licenses.

After downloading the pretraining snapshot:

```bash
python code/reconstruct_pretraining.py --pool english_common --upstream-cache ./owt-cache --output ./restored-common --download
python code/reconstruct_pretraining.py --pool english_extra --upstream-cache ./owt-cache --output ./restored-extra
```

Dependencies are recorded in `reconstruction_manifest.json`. The helper restores frozen selected rows, verifies exact text identities, reencodes with the original tokenizer and checks full corpus/pool/stream SHA256. It does not resample the corpus or rerun a modified filtering policy. Use a fresh output directory per pool; allow roughly 10 GiB working storage and sufficient CPU time. The English text acquisition remains subject to the original source's availability and terms.

Local release validation regenerated both full protein token pools and streams, and exactly restored 160 sampled English source records. Anonymous downloaded corpora and remote file identities were also checked. Full English reconstruction and fresh end-to-end model pretraining were not rerun for this documentation release.
