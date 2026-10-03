# October 2026 reproducibility release

This directory preserves the complete 190-file combined source snapshot for the completed deterministic protein-data reconstruction and 9-pretraining / 27-source-SFT fixed transfer study. The snapshot is under [`source/`](source/), with its original `bio2nl/` and `biopaws/` layout retained. It contains code, aggregate reports, SourceData and figure scripts, CPU inspection tools, and source-specific cards; it contains no model weights or protein-data payloads.

The exact local source capture is `b8d50d0ae61c8407ea8dcf62e47edef5969fc56c`. Its [`MANIFEST.json`](source/MANIFEST.json) SHA256 is `4ac59ec3847ba406ba1f0c812550e263dd6ea10a0bd7cc046e948ba1b438c832`. This later capture is not a retroactive training revision. The GitHub publication commit is distinct from that local capture and from the Hugging Face dataset commit.

## Published data and immutable links

The Hugging Face **dataset** repository is `dnagpt/bio2nl`, pinned to commit `62ced1ec44444b1879f2c473c553e6b5a9c6767f`:

- [Version README, attribution and safe extraction instructions](https://huggingface.co/datasets/dnagpt/bio2nl/blob/62ced1ec44444b1879f2c473c553e6b5a9c6767f/releases/2026-10-03-v1/README.md)
- [Artifact manifest](https://huggingface.co/datasets/dnagpt/bio2nl/blob/62ced1ec44444b1879f2c473c553e6b5a9c6767f/releases/2026-10-03-v1/manifest.json)
- [Exact source archive](https://huggingface.co/datasets/dnagpt/bio2nl/resolve/62ced1ec44444b1879f2c473c553e6b5a9c6767f/releases/2026-10-03-v1/source.tar.gz)
- [Biological data archive](https://huggingface.co/datasets/dnagpt/bio2nl/resolve/62ced1ec44444b1879f2c473c553e6b5a9c6767f/releases/2026-10-03-v1/protein_data.tar.gz)

The biological archive contains eight Swiss-Prot-derived payloads: three corpus splits, three operational sequence-pair splits, and the exact experimental source train/validation representations. Pair split sizes are 99,818 / 20,276 / 20,854; the experimental training subset has 8,044 rows and validation has 20,276. These subsets reuse the pair splits; they are not additional independent samples. Data and weights are not copied into this GitHub snapshot.

## CPU entry points

From the GitHub checkout root, using Python 3.12:

```sh
SOURCE="$PWD/releases/2026-10-03-v1/source"
python -I -B "$SOURCE/release_cli.py" --source-root "$SOURCE" verify
python -I -B "$SOURCE/release_cli.py" --source-root "$SOURCE" report --output "$PWD/oct2_recomputed_report.json"
```

Use a fresh output path outside `source/`. `verify` checks the complete declared file set and SHA256 values; `report` checks the published aggregate statistics without loading models or rerunning predictions. For plotting, install `source/requirements-release.txt` into an isolated environment, then run:

```sh
python -I -B "$SOURCE/release_cli.py" --source-root "$SOURCE" rebuild-figures --manuscript --output "$PWD/oct2_figures"
```

After downloading, checking and safely extracting the pinned biological archive using the Hugging Face instructions, verify it with an explicit data root:

```sh
python -I -B "$SOURCE/release_cli.py" --source-root "$SOURCE" source-input --kind protein --asset-root /absolute/path/to/protein_data
```

These CPU tools passed 16 tests and seven isolated integrations before publication. They are the independently checked portable entry points. The preserved historical builders/workers still have reference-archive and execution-evidence dependencies; this snapshot does not claim a complete public-download-only raw-to-training pipeline. There is no placeholder training command.

## Scope and notices

The study's English-target results are exploratory after prior target use. The source-trained models retain weak source-task and fixed-decision target performance; the release does not establish protein relation semantics or a new blind confirmation. The surface reference uses shared-BPE content-token statistics. Full scientific scope and limitations are described in the captured reports and cards.

Existing source-specific notices are retained. [`source/bio2nl/LICENSE`](source/bio2nl/LICENSE) is the existing bio2nl license; inclusion does not assign a new blanket license to historical biopaws code. Swiss-Prot-derived data attribution and terms are recorded in the [protein attribution](source/cards/PROTEIN_ATTRIBUTION.md). Model-weight terms are separate; the 36-checkpoint inventory includes metadata only.

Private archives, reviewer/submission materials, imported sessions, credentials, logs, tokenizer payloads, per-row predictions, and restricted natural-language text/reversible inputs are excluded. The version directory adds the explicit snapshot and this README without replacing existing repository contents.

Some documents inside the immutable `source/` snapshot describe the earlier local-only, unpublished state. Those statements remain historical capture records; the fixed Hugging Face links above identify the completed publication. Do not edit the snapshot merely to update those historical statements.
