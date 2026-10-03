# Bio2NL: portable current-study reproduction

This package adds public input acquisition, local preparation, guarded source
training and model-loading tools to the exact 190-file source release of the
October 2 EP/ES/EE study. The historical package is retained under
`study_source/`, including its source code, aggregate SourceData, provenance and
original limitations. The new tools use explicit locations and new output
directories; they do not change the historical experiments or copy their
training authorization.

## Pinned resources and scope

- [Current input bundle, immutable dataset commit](https://huggingface.co/datasets/dnagpt/bio2nl/resolve/5c5692107582866a3984973f83a2ac27d9a1ea89/releases/2026-10-03-portable-v2/input_bundle.tar.gz),
  SHA-256 `e9d6e7260879a9132c3cfee41b0e4b6efe8c0f13b8602e5095daf903ce08d7b9`.
- [Model repository, immutable revision](https://huggingface.co/dnagpt/bio2nl-models/tree/9fdd0794820ef4534b7f97fcd2c7a2893aba7a9d).
  The 36-entry model catalog SHA-256 is
  `5f6694f206084ad51dcc27c1485c4b4d000ffe5023310482aab367d0815df067`.
  The weights are separate downloads, approximately 15.85 GB in total; they are
  absent from this source archive. `model/` includes the exact shared tokenizer,
  complete-checkpoint loader, catalog and support evidence. See its README for
  loading only a selected model and retaining the trained classifier head.
- [Previous source release](https://huggingface.co/datasets/dnagpt/bio2nl/resolve/62ced1ec44444b1879f2c473c553e6b5a9c6767f/releases/2026-10-03-v1/source.tar.gz),
  SHA-256 `35434a25ceb330def3902d7d031ea76dfe4de052ec7a85388fc1fe9fa39839af`.
  Its 190 files are unchanged under `study_source/`. Its source-capture commit
  `b8d50d0ae61c8407ea8dcf62e47edef5969fc56c` is a later capture, not proof of the
  Git state used during historical training.

The input bundle contains the current-study protein inputs, exact tokenizer,
fixed metadata and a nontext English-exclusion hash index. English documents,
QQP text, locally prepared token arrays and row predictions are not bundled in
this source archive. `resources/ATTRIBUTION.md` describes the upstream sources,
terms and the limits of the historical exclusion projection. Third-party terms
remain applicable; there is no blanket license for all upstream material.

This route reconstructs the fixed inputs needed for the current study. It does
not reconstruct all 361 members of the historical raw release, independently
requalify all NLP benchmarks, or demonstrate a new complete GPU training run.
The historically scored QQP cohort remains exploratory. Source selection uses
only protein train/validation; the training adapter cannot score held-out data.

## 1. Check this source package and reproduce the reported aggregates

Use absolute paths and a fresh work directory outside this source tree. The
standard-library commands below do not require a GPU or model weights.

```bash
SOURCE=/absolute/path/to/unpacked_source_package
WORK=/absolute/path/to/new_reproduction_work
mkdir -p "$WORK"
python -B "$SOURCE/build_archive.py" verify --root "$SOURCE"
CUDA_VISIBLE_DEVICES='' python -I -B "$SOURCE/study_source/release_cli.py" \
  --source-root "$SOURCE/study_source" verify
CUDA_VISIBLE_DEVICES='' python -I -B "$SOURCE/study_source/release_cli.py" \
  --source-root "$SOURCE/study_source" report --output "$WORK/aggregate_report.json"
```

The report recomputes 54 neural evaluation cells, six fixed-reference cells,
nested means/SD across three pretraining seeds and paired contrasts from
released per-model aggregate metrics. It does not replay row predictions,
retrain the models, conduct hypothesis tests, or create a new blind evaluation.
`study_source/release_cli.py rebuild-figures --help` describes optional figure
regeneration from the same released SourceData.

## 2. Acquire fixed public inputs and rebuild English locally

Install the preparation dependencies in
`training/requirements-source-prepare.txt` in an isolated Python 3.12.3
environment. These are direct scientific requirements, not a complete OS or
transitive package lock. Acquisition uses the exact commits, sizes and hashes
in `resources/resources.json`; interrupted downloads resume only after checks.

```bash
python -B "$SOURCE/resources/fetch.py" fetch --id input_bundle \
  --output "$WORK/input_bundle.tar.gz"
python -B "$SOURCE/resources/fetch.py" unpack \
  --archive "$WORK/input_bundle.tar.gz" --root "$WORK/inputs"
python -B "$SOURCE/resources/fetch.py" fetch --id openwebtext \
  --output "$WORK/openwebtext-00000.parquet"
CUDA_VISIBLE_DEVICES='' python -B "$SOURCE/resources/fetch.py" english \
  --raw "$WORK/openwebtext-00000.parquet" --root "$WORK/inputs"
```

The English recipe processes one explicitly fixed OpenWebText shard and checks
all five rebuilt file identities. It uses the released nontext projection of
the historical exclusion decision; it does not rerun the original NLP benchmark
qualification. Keep the English corpus and derived token streams local.

For the separately fixed, historically evaluated QQP cohort,
`resources/rebuild_target.py --help` documents optional local reconstruction
from the pinned `qqp` upstream resource and membership indices. That tool does
not tune, train or score any model. It is not part of the source-training gate.

## 3. Prepare and independently verify source training inputs

```bash
CUDA_VISIBLE_DEVICES='' python -B "$SOURCE/prepare/portable_prepare.py" \
  verify-inputs --raw-root "$WORK/inputs"
CUDA_VISIBLE_DEVICES='' python -B "$SOURCE/prepare/portable_prepare.py" \
  build --raw-root "$WORK/inputs" --output "$WORK/prepared"
CUDA_VISIBLE_DEVICES='' python -B "$SOURCE/prepare/portable_prepare.py" \
  verify --output "$WORK/prepared"
```

This writes new protocol, manifest and acceptance identities. Twenty-two
scientific payloads must match the original hashes exactly; two provenance
documents are intentionally new, and the exact tokenizer is copied alongside
them. The source set is 8,044 train / 20,276 validation rows, with four
8,388,608-token streams and three paired pretraining schedules. Preparation
acceptance is not permission to run GPU training. Outputs must be fresh and
must not be published as part of this source archive.

## 4. Compile a permanently CPU-only verification run

The distribution metadata on the validation host reports PyTorch 2.9.1 while
the imported binary reports 2.3.0+cu121. Normal GPU compilation rejects this
mismatch. `training/ENVIRONMENT_NOTE.json` records the limited evidence; the
actual October 2 imported binary is not established. A clean consistent
PyTorch 2.9.1 training installation remains unvalidated. Do not recreate
inconsistent package metadata or modify the historical environment.

The following invocation intentionally creates a permanently GPU-disabled
protocol. All roots are explicit, source code identities are checked, and no
historical acceptance is reused. The SHA values come from your own completed
preparation.

```bash
MANIFEST_SHA=$(python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$WORK/prepared/manifest.json")
ACCEPTANCE_SHA=$(python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$WORK/prepared/acceptance.json")
CUDA_VISIBLE_DEVICES='' python -B "$SOURCE/training/compile.py" \
  --prepared-root "$WORK/prepared" --output-root "$WORK/cpu_validation" \
  --source-code-root "$SOURCE/study_source/bio2nl/new_data_full_training_v1" \
  --prepared-manifest-sha256 "$MANIFEST_SHA" \
  --prepared-acceptance-sha256 "$ACCEPTANCE_SHA" \
  --gpu-lock "$WORK/.bio2nl_gpu_training_queue.lock" --cpu-validation-only
PROTOCOL_SHA=$(python -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$WORK/cpu_validation/protocol.json")
CUDA_VISIBLE_DEVICES='' python -B "$WORK/cpu_validation/code/cpu_check.py" \
  --protocol "$WORK/cpu_validation/protocol.json" --protocol-sha256 "$PROTOCOL_SHA"
CUDA_VISIBLE_DEVICES='' python -B "$WORK/cpu_validation/code/fit_surface.py" \
  --protocol "$WORK/cpu_validation/protocol.json" --protocol-sha256 "$PROTOCOL_SHA"
CUDA_VISIBLE_DEVICES='' python -B "$WORK/cpu_validation/code/audit_surface.py" \
  --protocol "$WORK/cpu_validation/protocol.json" --protocol-sha256 "$PROTOCOL_SHA"
```

Use one shared GPU lock path for all queues targeting the same device. CPU
compilation/checking and five CPU surface fits do not mint training acceptance.
Their protocol cannot be upgraded into a GPU run.

## 5. Explicit future GPU smoke and full source training

These operations have not been performed by the portable adapter. On a clean,
consistent and validated environment satisfying
`training/requirements-gpu-train.txt`, compile a **separate fresh run** with
the same explicit roots and hashes, omitting `--cpu-validation-only`. Follow
`training/README.md` for the exact technical and full queue commands:

```bash
python -B "$GPU_RUN/code/run_queue.py" --protocol "$GPU_RUN/protocol.json" \
  --protocol-sha256 "$GPU_PROTOCOL_SHA" --phase technical --execute-gpu-smoke
python -B "$GPU_RUN/code/run_queue.py" --protocol "$GPU_RUN/protocol.json" \
  --protocol-sha256 "$GPU_PROTOCOL_SHA" --phase full --execute-full-training
```

`GPU_RUN` is that separately compiled run and `GPU_PROTOCOL_SHA` its compiler
output. The first command performs fresh bounded GPU work and, only after all
checks pass, produces that run's acceptance. The second command is a separately
requested full run: nine PT jobs (EP/ES/EE × three PT seeds), 27 source SFT jobs
(three FT seeds per matching new parent), then independent source selection.
Each PT consumes 16,777,216 tokens / 1,024 updates; each SFT has five epochs /
1,260 updates / 40,220 presentations. Source-validation CE selects epochs with
earlier ties. Numerical algorithms and matched initialization are locked.

The queue holds a shared process lock, checks fresh `nvidia-smi` and disk state
before every GPU stage, runs GPU work serially and stops on the first failure.
There is no automatic target evaluation, resume, budget extension or full run
triggered by passing CPU tests.

## Tests and release capture

```bash
CUDA_VISIBLE_DEVICES='' python -B -m unittest discover \
  -s "$SOURCE/resources" -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES='' python -B -m unittest discover \
  -s "$SOURCE/prepare" -p 'test_*.py' -v
CUDA_VISIBLE_DEVICES='' BIO2NL_ALGORITHM_ROOT="$SOURCE/study_source/bio2nl/new_data_full_training_v1" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B "$SOURCE/training/test_training.py"
```

Tests use temporary synthetic fixtures. They do not constitute full GPU or
cross-machine numerical reproduction. `build_archive.py` verifies the explicit
file allowlist, all historical source identities, model support hashes and the
source manifest. A maintainer can regenerate the manifest and deterministic
tar archive after inserting the actual model revision in this README:

```bash
python -B "$SOURCE/build_archive.py" manifest --root "$SOURCE" --model-revision MODEL_COMMIT
python -B "$SOURCE/build_archive.py" archive --root "$SOURCE" \
  --model-revision MODEL_COMMIT --output /fresh/path/source_portable.tar.gz
```

The manifest excludes its own hash, which is covered by the archive hash. The
builder rejects extra files, symlinks, a changed historical package and final
capture with a missing model revision. It performs no upload. No DOI is
assigned by these scripts; public repository receipts establish publication
separately from a local capture.

## Verified publication checks

Fresh anonymous public downloads reconstructed all five English outputs and all 22 scientific training-input payloads exactly. The 25-payload portable preparation passed independent acceptance; the exported training entrypoint compiled it and passed 13 CPU checks. This did not run a new full GPU training queue. Model readback checked 36 remote LFS identities, downloaded 18 support files and two prespecified weights, and passed standalone CPU loading and a two-row unlabeled toy prediction. The other 34 weights were not physically downloaded in that check.
