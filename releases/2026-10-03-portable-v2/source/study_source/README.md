# Bio2NL October 2026 reproducibility materials

This local release captures the completed deterministic protein-data reconstruction and the 9-pretraining / 27-source-SFT / 54-neural-cell + 6-reference-cell fixed transfer study. It contains original scientific code, aggregate results, independent CPU inspection tools, and manuscript source-data/figure scripts. Model weights and restricted natural-language text are not embedded.

The historical executable sources are retained byte-for-byte in `bio2nl/` and `biopaws/`; their dated protocols and local executions remain the evidence for completed experiments. Some original scripts depend on absolute paths and private reference archives. **This release does not claim a fully portable public-download-only raw-to-training pipeline.** The standalone commands below are the scope actually checked from an empty working directory. They do not alter or relax old gates. There is no placeholder training command.

## Standalone CPU commands

Use an absolute `SOURCE` path to this folder and choose fresh output paths outside it:

```sh
SOURCE=/path/to/source
python -I -B "$SOURCE/release_cli.py" --source-root "$SOURCE" verify
python -I -B "$SOURCE/release_cli.py" --source-root "$SOURCE" report --output /tmp/oct2-report.json
python -I -B "$SOURCE/release_cli.py" --source-root "$SOURCE" rebuild-figures --output /tmp/oct2-diagnostic-figures
python -I -B "$SOURCE/release_cli.py" --source-root "$SOURCE" rebuild-figures --manuscript --output /tmp/oct2-paper-figures
python -I -B "$SOURCE/release_cli.py" --source-root "$SOURCE" source-input --kind protein --asset-root /path/to/protein_data
python -I -B "$SOURCE/tests/test_release_cli.py"
```

`verify` hashes every explicit source-package member. `report` recomputes 36 nested statistics and 36 paired contrasts from the 54 per-job aggregate rows, checks every mean/sample-SD against frozen tables, and checks six reference cells and the primary direction flag. It does not recompute predictions or load models. Each PT group's three FT seeds are averaged first; mean and sample SD are then over three PT seeds.

`rebuild-figures` creates a compact transfer diagnostic. `--manuscript` rebuilds all three manuscript figures from the included source data with isolated Python workers, without importing historical workspace modules or editing bundled sources. Plot dependencies are pinned in `requirements-release.txt`. Historical raw/training numerical environments are separately recorded in `metadata/environment_lock.json`; do not replace their torch 2.9.1 requirement with older workspace notes.

`source-input --kind protein` hashes the separate eight-file biological data candidate. `--kind raw_inputs` checks four fixed upstream input descriptors when supplied locally; this is explicitly not the full raw-rebuild dependency closure. `--kind models` checks the 36 metadata-indexed checkpoints under the original workspace-relative layout if supplied locally. `--size-only` skips content hashes and says so in its result; it must not be reported as model identity verification. None of these checks authorizes training.

## Data and model boundaries

`../protein_data/` is a separate actual-data candidate: six Swiss-Prot-derived corpus/pair split files plus the exact 8,044-row source training and 20,276-row validation text subsets. It has its own manifest and attribution. See `cards/DATA_CARD.md` and `cards/PROTEIN_ATTRIBUTION.md`. The all-pairs duplicate and raw 94 MB Swiss-Prot download are not duplicated. Protein examples are operational sequence-similarity examples, not certified remote structural homology labels.

`metadata/models_assets.json` lists exactly nine fresh pretraining checkpoints and 27 complete source-validation-selected classifiers, with byte sizes and SHA256. No model payload is copied. See `cards/MODEL_CARD.md`; no new license is assigned to weights. `metadata/acquisition_recipe.json` and `fetch_verified.py` provide fixed-file download verification for declared upstream dependencies. Mutable provider URLs must return the locked bytes or fail; there is no silent revision substitution. QQP membership/exclusion closure and private archive bootstrap remain a stated portability gap. No private archive or NLP text is supplied to conceal that gap.

QQP, OpenWebText, CoLA and RTE texts, reversible NLP tokens, tokenizers, per-row predictions, private reviewer/submission correspondence, imported sessions, credentials, logs and external binaries are excluded. The source-code license does not grant rights in these assets. Historical `biopaws` code has no declared repository-wide license; this records its status and does not imply that its owner cannot store their own code. No new blanket license is imposed. Existing `bio2nl/LICENSE` is retained, with per-source notices taking precedence.

## Scientific scope

EP and ES use English plus natural/shuffled protein streams; EE uses two English streams with the same total token budget. There are three independent PT seeds and three FT seeds nested within each. PT sees 16,777,216 input tokens; each source fit uses 8,044 training rows, 20,276 validation rows and five epochs, selected only by source validation CE. All 27 classifiers are retained regardless of source score, then score protein test (20,854 rows) and historically observed QQP (39,893 rows) with the same complete head and fixed label-1 direction.

EP minus ES target AUC is +0.029560 with PT sample SD 0.018655; all three PT differences are positive. This is descriptive, exploratory ranking evidence, not a significance claim or proof of usable semantic transfer. EP source-test accuracy is about 50.76% and target balanced accuracy about 50.18%. The nine-dimensional surface comparator uses shared-BPE content-token statistics, not raw amino-acid composition. The raw residue baselines use 99,818 training rows and must be kept separate from the 8,044-row supervised comparison. Target history, weak source competence, representation exposure and the EE information-budget distinction limit conclusions.

## Versioning and completeness

The isolated source Git commit, explicit tar membership and hash-readback live in the sibling `release_identity.json` once captured. That commit is a new exact-subset capture, not a retroactive execution revision or an update to either working repository. `metadata/source_origins.json` records 142 original-source hashes, 12 aggregate origins, all 54 executed-code mappings and the 28 builder overlays / six declared transforms. The original project HEADs and indexes remain untouched.

SourceData.xlsx, the aggregate source-data exports and four figure scripts are included under `paper_assets/`; no manuscript or peer-review text is included. Remote GitHub/Hugging Face revision fields remain empty until an actual authenticated upload and readback. Repository type matters: a model repository named `dnagpt/biopaws` is not `datasets/dnagpt/biopaws`. This release performs no upload.
