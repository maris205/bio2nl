# Current data and model guide

The October rebuilt experiments use three new Hugging Face repositories. They do not overwrite `dnagpt/bio2nl` or `dnagpt/bio2nl-models`.

## Pinned releases

| Resource | Exact revision |
|---|---|
| [Task data](https://huggingface.co/datasets/dnagpt/bio2nl-data-2026-10/tree/df6f04914fd3f1cd79abc14e157d65a188df89b0) | `df6f04914fd3f1cd79abc14e157d65a188df89b0` |
| [Model bank](https://huggingface.co/dnagpt/bio2nl-models-2026-10/tree/62a70cc109af90d1cbe673beac4cffabdc0a4886) | `62a70cc109af90d1cbe673beac4cffabdc0a4886` |
| [Pretraining companion](https://huggingface.co/datasets/dnagpt/bio2nl-pretraining-2026-10) | 003e291abac2ba411b4054a7f60d50567e7f1dfa |

The release manifest `releases/2026-10/resources.json` repeats these pins for scripts. A revision identifies the bytes underlying the documentation; changing a dataset card does not change old pinned artifacts.

## Dataset definitions

| Family | Construction and size | Role |
|---|---|---|
| Local BLAST | Swiss-Prot 2026_03; 4 HSP identity bands × 3 screened negative strategies; 5,000 positive + 5,000 negative per combination; V1/V2 views | Forward targets, with fixed query-hash validation/test splits |
| Ordinary protein SFT | V2 + length-constrained negatives; four bands and a balanced mixed group, each 10,000 rows | Reverse source; all rows used for SFT, no source held-out evaluation |
| Standard reference | Earlier MMseqs2 operational similarity construction; validation 20,276/test 20,854 | Separate forward reference, not another BLAST band |
| SCOPe remote | Same superfamily/different family positives; cross-superfamily negatives; both labels identity <25%, coverage ≥60% | Train 13,732/validation 3,000; separate 3,000 source-test rows retained but unscored |
| Measured functional variants | ProteinGym-linked WT/single-mutant assay outcomes; both classes homologous | Train 3,684/validation 272; CALM1 test retained but unscored |
| PAWS-X cleaned | Train 49,258/validation 543/test 1,998 | English SFT source; ordinary reverse target |
| PAWS-X original | Validation/test 2,000 each | Remote and DMS English target |
| English cosine proxy | Four MiniLM cosine bands; per band validation 136/test 544/existing confirmation 1,000 | Relatedness target; exact text reconstructed from source references |

V1 is a gap-free aligned subject fragment; V2 is the complete stored subject entry, which can itself have a Fragment annotation. They are paired views, not successive software versions. Positive BLAST E≤1e-10 with no hard coverage cutoff; negative screening rejects self/equal/known-positive inputs and reported hits at E≤0.001. The accepted length constraint is 30% relative to each row's original positive B in both views. Composition is a matching priority, not a biological nonhomology guarantee. Query membership need not be identical across final positive bands, although the same query bank supplies them. See the pinned dataset card and protocols for exact commands and overlap disclosures.

## Model-to-data map

| Family | Checkpoints | Pretraining and SFT data |
|---|---:|---|
| Forward | 100 | Official GPT-2; cleaned PAWS-X, seeds 1–100 |
| Ordinary reverse | 100 | Official GPT-2; five V2/length protein groups, 20 seeds each |
| From-scratch pretrained | 3 | EP/ES/EE; shared tokenizer, pretraining seed 0 once each |
| From-scratch SFT | 60 | Each EP/ES/EE checkpoint; same V2/80–100%/length protein group, seeds 1–20 |
| Remote | 20 | Official GPT-2; SCOPe source, 20 epochs; discovery seed 1 + fresh 2–20 |
| Functional variants | 20 | Official GPT-2; measured DMS source, four epochs; discovery seed 1 + fresh 2–20 |
| Diagnostics | 6 | Alternative LR and small/full source-fitting models |
| Exact recipe replays | 4 | Two ordinary seed-19 recipes repeated twice each |

The `model_catalog.json` and per-model `record.json` link each weight to dataset, seed, exact format and SHA256. There are 313 complete checkpoint files (152,174,855,774 bytes), not 313 independent pretraining runs. Original initial classification heads are included where saved.

## Minimal use

```python
from datasets import load_dataset
source = load_dataset(
    "dnagpt/bio2nl-data-2026-10", "sft_blast_id80_100",
    revision="df6f04914fd3f1cd79abc14e157d65a188df89b0", split="train"
)
```

For one classifier, download its folder, referenced tokenizer, `model_catalog.json` and `code/` from the pinned model bank. Its README gives the exact `load_model`, `encode_pairs` and `logits` example. Use those format-specific functions: native256 has no inserted separator, native1024 uses attention-mask final-EOS pooling, and joint shared-BPE512 uses its own tokenizer and complete-state loader. Joint `.pt` files are not ordinary `AutoModel.from_pretrained` directories. All three formats were technically checked; the unmodified joint loader is not certified to fit in 2 GiB RAM.

Dataset-side commands, from a downloaded task-data snapshot:

```bash
python code/verify_data.py
python code/restore_english_proxy.py --shard-dir /path/to/pinned/shards --output /new/english_pairs
```

Original scripts in `code/original/` retain historical paths; they are exact algorithm snapshots, not a promise that every original absolute-path entrypoint runs unchanged elsewhere. Portable adapters cover materialized data verification, proxy reconstruction and model loading. [Pretraining reconstruction](PRETRAINING.md) is separate. No GPU is needed to download data, inspect metadata or reconstruct corpora/token streams.

## Results and interpretation

`results/` in the task repository contains source/target metrics, all seeds and experiment budgets. Report discovery-selected maxima together with all-seed distributions, raw class direction, matched initial and same-seed confirmation when available. Later experiment families reused the original confirmation sets. Forward aggregate comparisons lack a matched-initial target column; do not infer a net SFT improvement from those aggregates alone. PAWS-X is distinct from teacher-defined cosine relatedness, and source training fit is distinct from held-out biological competence.
