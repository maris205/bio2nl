# Current data and model guide

The October rebuilt experiments use three Hugging Face repositories. They do not overwrite `dnagpt/bio2nl` or `dnagpt/bio2nl-models`. The task-data release now exposes **46 configurations: 41 actual-pair configurations and 5 preserved English reconstruction-metadata configurations**. The [FT and test data guide](FT_AND_TEST_DATA.md) maps each experiment to exact inputs and explains the October 17 additions.

## Pinned releases

| Resource | Exact revision |
|---|---|
| [Task data](https://huggingface.co/datasets/dnagpt/bio2nl-data-2026-10/tree/36ab95168aeb717bded5f09aa915f9aacab4e71d) | `36ab95168aeb717bded5f09aa915f9aacab4e71d` |
| [Model bank](https://huggingface.co/dnagpt/bio2nl-models-2026-10/tree/62a70cc109af90d1cbe673beac4cffabdc0a4886) | `62a70cc109af90d1cbe673beac4cffabdc0a4886` |
| [Pretraining companion](https://huggingface.co/datasets/dnagpt/bio2nl-pretraining-2026-10) | de767573ba2ec741af5240b39ad16104fdf087bd |

The release manifest `releases/2026-10/resources.json` repeats these pins for scripts. A revision identifies the bytes underlying the documentation; changing a dataset card does not change old pinned artifacts.

## Dataset definitions

| Family | Construction and size | Role |
|---|---|---|
| Local BLAST | Swiss-Prot 2026_03; 4 HSP identity bands × 3 screened negative strategies; 5,000 positive + 5,000 negative per combination; V1/V2 views | Forward targets, with fixed query-hash validation/test splits |
| Ordinary protein FT | V2 + length-constrained negatives; four bands and a balanced mixed group, each 10,000 rows | Reverse source; all rows used for FT, no source held-out split |
| Standard reference | Earlier MMseqs2 operational similarity construction; train 99,818/validation 20,276/test 20,854 | Separate forward target; newly archived train is for the historical target-supervised feature baseline |
| Protein-PAWS Reference | Exact gap-cleaned historical matched CSV: train 14,000/validation 3,000/test 3,000 | Separate forward target; its protein train partition is archived, not FT input for the English-to-protein condition |
| SCOPe remote | Same superfamily/different family positives; cross-superfamily negatives; both labels identity <25%, coverage ≥60% | Reverse FT train 13,732/validation 3,000; 3,000 source-test rows archived without a reverse source-generalization score in this follow-up; the forward comparison scores the protein target separately |
| Remote identity-matched subset | Exact retained selection from SCOPe parent pools; validation 1,696/test 1,832 | Forward sensitivity analysis; retain original full-validation polarity, not a newly chosen subset polarity |
| Measured functional variants | ProteinGym-linked WT/single-mutant assay outcomes; both classes homologous | Train 3,684/validation 272; CALM1 test retained but unscored |
| PAWS-X cleaned | Train 49,258/validation 543/test 1,998 | English FT source; ordinary reverse target |
| PAWS-X original | Validation/test 2,000 each | Remote and DMS English target |
| English cosine proxy | Four MiniLM cosine bands; per band validation 136/test 544/existing confirmation 1,000 | Relatedness target; actual text downloadable, original reconstruction metadata retained |
| Expanded English relatedness | Highest cosine band: validation 136; unchanged original confirmation 1,000 + additional 1,000 = test 2,000 | Seven official-model source groups × 20 FT seeds; stored original validation directions; not an entirely new blind test |

V1 is a gap-free aligned subject fragment; V2 is the complete stored subject entry, which can itself have a Fragment annotation. They are paired views, not successive software versions. Positive BLAST E≤1e-10 with no hard coverage cutoff; negative screening rejects self/equal/known-positive inputs and reported hits at E≤0.001. The accepted length constraint is 30% relative to each row's original positive B in both views. Composition is a matching priority, not a biological nonhomology guarantee. Query membership need not be identical across final positive bands, although the same query bank supplies them. See the pinned dataset card and protocols for exact commands and overlap disclosures.

The legacy Reference filename `protein_pairs_matched_v2.csv` uses a different historical meaning of “v2”: deletion of literal ASCII `-` only. It retains fragments, self-pairs, duplicates, noncanonical symbols and documented cross-split overlap. It is neither the complete-entry BLAST V2 view nor `standard_reference`. Full data, provenance and portable checks are in the [task-data completion extension](https://huggingface.co/datasets/dnagpt/bio2nl-data-2026-10/tree/36ab95168aeb717bded5f09aa915f9aacab4e71d/extensions/paper-task-data-2026-10-17).

## Model-to-data map

| Family | Checkpoints | Pretraining and FT data |
|---|---:|---|
| Forward | 100 | Official GPT-2; cleaned PAWS-X, seeds 1–100 |
| Ordinary reverse | 100 | Official GPT-2; five V2/length protein groups, 20 seeds each |
| From-scratch pretrained | 3 | EP/ES/EE; shared tokenizer, pretraining seed 0 once each |
| From-scratch FT | 60 | Each EP/ES/EE checkpoint; same V2/80–100%/length protein group, seeds 1–20 |
| Remote | 20 | Official GPT-2; SCOPe source, 20 epochs; discovery seed 1 + fresh 2–20 |
| Functional variants | 20 | Official GPT-2; measured DMS source, four epochs; discovery seed 1 + fresh 2–20 |
| Diagnostics | 6 | Alternative LR and small/full source-fitting models |
| Exact recipe replays | 4 | Two ordinary seed-19 recipes repeated twice each |

The `model_catalog.json` and per-model `record.json` link each weight to dataset, seed, exact format and SHA256. There are 313 complete checkpoint files (152,174,855,774 bytes), not 313 independent pretraining runs. Original initial classification heads are included where saved.

This is the published bank, not a claim that every later experiment's weights are public. The PP pretrained checkpoint plus its 20 FT checkpoints and the 20 whole-network random controls remain local. The task-data completion does not upload them. A random classification head on an already pretrained backbone is also different from a random whole network. FT updates the backbone and classification head; it is supervised pair classification, not instruction tuning.

## Minimal use

```python
from datasets import load_dataset
source = load_dataset(
    "dnagpt/bio2nl-data-2026-10", "sft_blast_id80_100",
    revision="36ab95168aeb717bded5f09aa915f9aacab4e71d", split="train"
)
```

For one classifier, download its folder, referenced tokenizer, `model_catalog.json` and `code/` from the pinned model bank. Its README gives the exact `load_model`, `encode_pairs` and `logits` example. Use those format-specific functions: native256 has no inserted separator, native1024 uses attention-mask final-EOS pooling, and joint shared-BPE512 uses its own tokenizer and complete-state loader. Joint `.pt` files are not ordinary `AutoModel.from_pretrained` directories. All three formats were technically checked; the unmodified joint loader is not certified to fit in 2 GiB RAM.

Dataset-side commands, from a downloaded task-data snapshot:

```bash
python code/verify_data.py
python code/restore_english_proxy.py --shard-dir /path/to/pinned/shards --output /new/english_pairs
python extensions/paper-task-data-2026-10-17/reference/code/verify_reference.py
python extensions/paper-task-data-2026-10-17/code/export_pairs.py \
  --config protein_paws_reference_gap_cleaned --split test --output /new/reference_test.csv
```

Original scripts in `code/original/` retain historical paths; they are exact algorithm snapshots, not a promise that every original absolute-path entrypoint runs unchanged elsewhere. Portable adapters cover materialized data verification, proxy reconstruction and model loading. The [task guide](FT_AND_TEST_DATA.md) includes direct English-target loading examples, the optional reconstruction commands and actual-pair versus metadata schemas. [Pretraining materials](PRETRAINING.md) are separate: the selected common/extra English documents can be downloaded directly, and their full token pools and consumed streams can be regenerated locally. The separate English relatedness targets now have five directly loadable configurations; their five original metadata configurations remain unchanged. No GPU is needed to download data, inspect metadata or reconstruct corpora/token streams.

## Results and interpretation

`results/` in the task repository contains source/target metrics, all seeds and experiment budgets. The expanded-English extension adds frozen-direction old/new/combined per-model metrics and paired summaries for seven official-model groups. Report descriptive maxima together with all-seed distributions, raw class direction, matched initial and the originally selected seed's confirmation when available. Later experiment families reused the original confirmation sets. Historical forward aggregate files without matched-initial target columns do not establish a net FT improvement on their own. PAWS-X is distinct from teacher-defined cosine relatedness, and source training fit is distinct from held-out biological competence. The expanded 2,000-pair target does not replace the original 544/1,000 targets used in the EP/ES/EE/PP comparison.
