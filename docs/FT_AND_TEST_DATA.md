# Exact fine-tuning and test data

The task dataset is [dnagpt/bio2nl-data-2026-10 at revision 36ab95168aeb717bded5f09aa915f9aacab4e71d](https://huggingface.co/datasets/dnagpt/bio2nl-data-2026-10/tree/36ab95168aeb717bded5f09aa915f9aacab4e71d). It exposes **46 loadable configurations**: **41 with actual protein/English pairs and labels**, plus **5 preserving exact English reconstruction metadata**. The five new English-relatedness configurations include the sentence text directly; the original `_metadata` views remain available for provenance and optional reconstruction.

The complete [task index, catalog and portable tools](https://huggingface.co/datasets/dnagpt/bio2nl-data-2026-10/tree/36ab95168aeb717bded5f09aa915f9aacab4e71d/extensions/paper-task-data-2026-10-17) record file hashes, row counts, schemas and roles. This page identifies which data are used for which experiment. FT means supervised classification fine-tuning of both the backbone and its classification head. Historical `sft_` configuration names remain unchanged.

## Fine-tuning sources

| Experiment | Configuration and split | Exact rows | Role |
|---|---|---:|---|
| English → protein | `pawsx_clean/train` | 49,258 | Actual English sentence pairs used to fine-tune the classifiers |
| Ordinary protein → English | `sft_blast_id30_40/train`, `sft_blast_id40_60/train`, `sft_blast_id60_80/train`, `sft_blast_id80_100/train`, `sft_blast_mixed/train` | 10,000 each | Five complete FT inputs; BLAST complete-entry V2 pairs with length-constrained negatives |
| Remote protein → English | `scope_remote/train` | 13,732 | SCOPe remote-homology-labelled pairs |
| Functional variant → English | `dms_function/train` | 3,684 | Measured functional outcomes for protein WT/single-mutant pairs; both classes are homologous |
| EP / ES / EE / PP after pretraining | `sft_blast_id80_100/train` | 10,000 | Same highest-identity protein FT input for the pretraining comparison |
| Historical target-supervised feature baseline | `standard_reference/train` | 99,818 | Protein feature-baseline fitting; not an English-to-protein transfer FT input |

The five ordinary groups use all their 10,000 rows for FT and have no source held-out split. `scope_remote` has 3,000 validation and 3,000 test rows; its source-test partition is archived but was not scored as a reverse source-generalization result in the follow-up. `dms_function` has 272 validation and 388 CALM1 test rows, with the latter archived but unscored in that follow-up. Availability does not imply every archived split was evaluated. The ordinary FT groups share their construction with the forward BLAST pair views and are not independent resamplings.

## Protein targets

| Target | Configuration | Validation / test rows | Distinction |
|---|---|---|---|
| Four main local-BLAST bands | `blast_id30_40_length_v2`, `blast_id40_60_length_v2`, `blast_id60_80_length_v2`, `blast_id80_100_length_v2` | 1,992/8,008; 2,066/7,934; 1,990/8,010; 2,008/7,992 | Each totals 10,000 pairs; the corresponding English-trained classifier is scored directly |
| Expanded BLAST grid | `blast_id{30_40,40_60,60_80,80_100}_{random,length,composition}_{v1,v2}` | 10,000 total per view | 24 paired views: four positive bands × three negative strategies × two sequence representations |
| Protein-PAWS Reference | `protein_paws_reference_gap_cleaned` | 3,000 / 3,000 | Historical matched pairs with literal ASCII hyphens removed; complete 20,000-row CSV also preserved |
| Standard reference | `standard_reference` | 20,276 / 20,854 | Earlier MMseqs2 operational-similarity target; distinct from Protein-PAWS Reference |
| Remote comparison | `scope_remote` | 3,000 / 3,000 | The English-to-protein comparison uses this target separately from reverse source fitting |
| Exact identity-matched remote sensitivity | `scope_remote_identity_matched` | 1,696 / 1,832 | Frozen subset of the remote pools; original full-validation directions are retained |

For Protein-PAWS Reference, the 14,000 protein `train` rows are archived to preserve the original split, **not** used to fine-tune the English-to-protein classifiers. This condition also uses the 49,258 English PAWS-X source rows. Its seeds 0–19 overlap the main forward seeds 1–100 in seeds 1–19; it is not an independent set of 20 new training runs.

The exact [Reference files and limitations](https://huggingface.co/datasets/dnagpt/bio2nl-data-2026-10/tree/36ab95168aeb717bded5f09aa915f9aacab4e71d/extensions/paper-task-data-2026-10-17/reference) preserve all rows, order, labels, fragments, self-pairs, duplicates and original split overlap. The legacy filename's “v2” means hyphen removal, not the complete-entry local-BLAST V2 representation. The new [standard-reference train archive](https://huggingface.co/datasets/dnagpt/bio2nl-data-2026-10/tree/36ab95168aeb717bded5f09aa915f9aacab4e71d/extensions/paper-task-data-2026-10-17/standard_reference) preserves the exact 99,818-row feature-baseline input without changing existing validation/test data.

The [remote sensitivity subset](https://huggingface.co/datasets/dnagpt/bio2nl-data-2026-10/tree/36ab95168aeb717bded5f09aa915f9aacab4e71d/extensions/paper-task-data-2026-10-17/remote_identity) includes actual sequences, selected row IDs, exact identity matching rules and reconstruction code. Its validation subset must not be used to choose a new polarity: the analysis keeps the original full-validation direction. Subsetting changes endpoint-degree balance, as documented in its audit.

## English targets

The constructed cosine-relatedness pairs below are target validation, test and confirmation data for classifiers fine-tuned on protein pairs. They are not the English FT training corpus: the English-to-protein direction uses `pawsx_clean/train`, as listed above.

| Task | Configuration | Splits and exact rows |
|---|---|---|
| Clean human paraphrase | `pawsx_clean` | validation 543; test 1,998; actual sentences included |
| Original human paraphrase | `pawsx_original` | validation 2,000; test 2,000; actual sentences included |
| Four teacher-defined relatedness bands | `english_cos60_70`, `english_cos70_80`, `english_cos80_90`, `english_cos90_100` | Each: validation 136; test 544; existing confirmation 1,000; actual sentences and labels |
| Expanded highest-cosine target | `english_cos90_100_expanded` | validation 136; original1000 1,000; new1000 1,000; test 2,000; actual sentences and labels |

The expanded target concatenates the unchanged original 1,000-pair confirmation set and 1,000 additional pairs, in that order. It is **not an entirely new blind test**. The 136 validation pairs remain unchanged. Seven official GPT-2 source groups (five ordinary, remote and functional) use the same expanded target across all 20 prescribed FT seeds per group, without refitting or choosing new label directions. The accompanying result files retain old/new/combined scores, matched initial controls, descriptive maxima and the originally selected seed's results.

The original 544-pair test and 1,000-pair confirmation remain the targets for the EP/ES/EE/PP pretraining comparison; they are not replaced with the expanded 2,000. Original-band comparisons remain tied to their own protocols; the presence of all configurations does not imply every possible source × target combination was run. PAWS-X human paraphrase is distinct from embedding-cosine relatedness.

## Direct loading and optional reconstruction

```python
from datasets import load_dataset

repo = "dnagpt/bio2nl-data-2026-10"
revision = "36ab95168aeb717bded5f09aa915f9aacab4e71d"
english_ft = load_dataset(repo, "pawsx_clean", revision=revision, split="train")
protein_ft = load_dataset(repo, "sft_blast_id80_100", revision=revision, split="train")
protein_test = load_dataset(repo, "protein_paws_reference_gap_cleaned",
                            revision=revision, split="test")
english_test = load_dataset(repo, "english_cos90_100_expanded",
                            revision=revision, split="test")  # 2,000 actual pairs
original_english_test = load_dataset(repo, "english_cos90_100",
                                     revision=revision, split="test")  # original 544
print(english_test[0]["sequence_a"], english_test[0]["sequence_b"],
      english_test[0]["label"])
```

BLAST, ordinary protein FT and the new English-relatedness files use `sequence_a`/`sequence_b`; the remaining actual-pair configurations use `sentence1`/`sentence2`. The new English files have integer `label` and original `row_id` fields. Their rows, order, labels and text identities match the existing evaluated data; publication does not draw new pairs. The original task catalog and the English-text extension describe their respective schemas.

For the command-line tools, download the pinned snapshot into an ordinary local directory and run the scripts from there. Using `local_dir` preserves the repository layout expected by the scripts; executing a script directly through a Hugging Face cache symlink can resolve it into the cache's blob directory instead.

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dnagpt/bio2nl-data-2026-10",
    repo_type="dataset",
    revision="36ab95168aeb717bded5f09aa915f9aacab4e71d",
    local_dir="bio2nl_data",
)
```

```bash
cd bio2nl_data
```

From this directory, export a uniform ordered CSV without tokenization, filtering, truncation, shuffling or label changes:

```bash
python extensions/paper-task-data-2026-10-17/code/export_pairs.py \
  --config sft_blast_id80_100 --split train --output /new/protein_ft.csv
python extensions/paper-task-data-2026-10-17/reference/code/verify_reference.py
python extensions/english-relatedness-text-2026-10-17/code/verify_english_relatedness.py --root .
```

Direct loading above needs no upstream shard. For optional independent reconstruction, the five original `_metadata` configurations still point to pinned OpenWebText shard 00000 from `Skylion007/openwebtext` revision `79d93d786212f7344586290adb811d4ae6a1762c`. Obtain that shard as described in `pretraining/source_lock.json`, then use the retained scripts. Do not run a new similarity search or draw new random pairs:

```bash
python code/restore_english_proxy.py \
  --shard-dir /path/to/pinned/shards --output /new/original_english
python extensions/english-relatedness-2000-2026-10-17/code/reconstruct_english.py \
  --shard-dir /path/to/pinned/shards --output /new/expanded_english
```

The original four metadata configurations retain string `"0"`/`"1"` labels; the expanded metadata and actual-pair configurations use integer labels. The export/reconstruction tools handle the documented schemas and verify row identities. The [English extension](https://huggingface.co/datasets/dnagpt/bio2nl-data-2026-10/tree/36ab95168aeb717bded5f09aa915f9aacab4e71d/extensions/english-relatedness-2000-2026-10-17) includes portable numerical result checks; those checks do not claim a full fresh neural-inference replay.

Pretraining materials and the 313-file published model bank remain separate and retain their existing pins in [resources.json](../releases/2026-10/resources.json). Model-specific tokenization, truncation, pooling and retained classification-head identities still matter. The PP pretrained model and its 20 FT checkpoints, and 20 whole-network random controls, remain local; this task-data update does not publish those weights. A complete fresh public-only rerun of every historical experiment is not claimed.
