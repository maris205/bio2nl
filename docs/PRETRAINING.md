# Pretraining data: EP, ES and EE

The dedicated release is [dnagpt/bio2nl-pretraining-2026-10](https://huggingface.co/datasets/dnagpt/bio2nl-pretraining-2026-10/tree/de767573ba2ec741af5240b39ad16104fdf087bd), revision `de767573ba2ec741af5240b39ad16104fdf087bd`. It covers the October 8 from-scratch experiment; the older October 3 experiment has a different budget and pretraining seed hierarchy.

| Pool | Records | Tokens in entire pool (including EOS) | Tokens consumed per stream |
|---|---:|---:|---:|
| English common | 361,356 | 446,730,942 | 134,217,728 |
| English extra | 361,643 | 447,059,806 | 134,217,728 |
| Natural protein | 137,736 | 19,918,409 | 134,217,728 |
| Shuffled protein | 137,736 | 20,374,008 | 134,217,728 |

**EP = common English + natural protein; ES = common English + shuffled protein; EE = common English + extra English.** Each has 268,435,456 input tokens, 16,384 optimizer updates, one pretraining seed (0), 12 layers/12 heads/768 hidden dimensions/512 context, and 110,025,216 language-model parameters. Each then receives the same high-identity V2 protein-pair SFT with 20 seeds. EE has twice the English token exposure; repeated protein sequences and new English documents are not equal information budgets.

The shared 32,000-entry byte-level BPE tokenizer uses 16 MiB English + 16 MiB natural protein for tokenizer training. That is separate from model pretraining. Token IDs are PAD=0, EOS=1, SEP=2, with no UNK. Protein streams repeat their original pools to the fixed budget; English streams are non-repeated prefixes. Binaries are little-endian uint16, with EOS appended per record. Training separately shuffles 512-token blocks.

## What can be downloaded directly

- Selected English common and extra JSONL, 361,356 and 361,643 documents, retaining their original corpus bytes.
- Natural and composition-preserving shuffled protein JSONL, 137,736 records each.
- Complete protein token pools and exact consumed streams; English pools and streams can be regenerated from the downloaded text with the supplied helper.
- The shared tokenizer, original preparation code and source/split/filter receipts.
- All four pools' text-free row indices, hashes and per-record exposure ledgers.

Protein pretraining uses the accepted canonical Swiss-Prot 2026_03 **training** corpus with its recorded component split and SCOPe domain exclusion. It is not the length-restricted 5,000-query BLAST task bank. The source receipts describe fragment/noncanonical filtering, exact deduplication and operational homology screening. They do not establish universal absence of biological overlap.

## English pretraining text and token reconstruction

The exact selected pretraining documents are directly downloadable as `corpora/english_common.jsonl.gz` (361,356 rows) and `corpora/english_extra.jsonl.gz` (361,643 rows), also exposed through dataset configurations of the same names. They preserve the original texts, document order, source identifiers and corpus SHA256 values; no fresh sampling or filtering is applied for publication. English data originate from OpenWebText shards 0–7 at revision `79d93d786212f7344586290adb811d4ae6a1762c`, with the recorded exclusions and deduplication. Source attribution and redistribution terms are given in the pretraining repository's `LICENSES.md`.

After downloading the pinned pretraining snapshot, rebuild the full English token pools and exact consumed streams directly from these corpora:

```bash
python code/verify_english_text.py --root .
python code/reconstruct_pretraining.py --pool english_common --output ../rebuilt-common
python code/reconstruct_pretraining.py --pool english_extra --output ../rebuilt-extra
```

The helper verifies the original corpus hashes before tokenization, then checks the complete token-pool and fixed-stream hashes. It also verifies and copies the existing row indices and exposure ledgers unchanged. The English binary pools and streams are regenerated locally, rather than added as duplicate downloads. Use a fresh output directory per pool and allow sufficient CPU time and storage for the full token pools. No GPU or upstream shard download is needed for this direct route.

For an independent source-reconstruction route, `--from-upstream` retains the earlier frozen-row restoration and all eight pinned shard checks:

```bash
python code/reconstruct_pretraining.py --pool english_common --from-upstream --upstream-cache ./owt-cache --output ../restored-common --download
python code/reconstruct_pretraining.py --pool english_extra --from-upstream --upstream-cache ./owt-cache --output ../restored-extra
```

Without the published English corpus, earlier snapshots also support the original `--upstream-cache` invocation. A present but corrupted corpus fails its hash check; it does not silently fall back to another source. Dependencies are recorded in `reconstruction_manifest.json`. The optional upstream route needs additional shard storage and remains subject to upstream availability.

This publication concerns pretraining documents. The five English **relatedness-task** configurations in the task-data repository still contain reconstruction metadata; their evaluation sentences must be restored with the separate task scripts. The pretraining corpora are not substitutes for those held-out pairs.

The released text verifier checks retained documents against original source positions, normalized and exact text identities, and row indices. Bounded reconstruction-route tests use original records and retained token-bin prefixes; the original protein tokenization/stream algorithms remain unchanged. The full English pools were encoded in the original experiment, but this publication does not claim a fresh full-pool reencoding or end-to-end model pretraining run.
