# Portable source-training adapter, version 1

This is a new execution adapter for the completed October 2 EP/ES/EE design. It compiles explicit prepared/output/source-code roots into a fresh run with new code, data, environment and execution identities. The original protocols, gates, source code, results and checkpoints remain untouched. No old acceptance is copied or relabelled. Compiling and passing CPU checks do **not** mean that GPU smoke or full training has run.

The compiler consumes the independently accepted output of `../prepare/portable_prepare.py`: `manifest.json` (`bio2nl_portable_prepared_inputs_v1`, status `prepared_inputs_built`) and `acceptance.json` (`bio2nl_portable_preparation_acceptance_v1`, status `passed`). The 22 original scientific payload identities must match `scientific_payload_lock.json`, along with the fixed tokenizer identity. Only source train 8,044 and validation 20,276 are readable; target/source-test examples, old model weights and held-out scoring are forbidden. The new input-resource lock replaces the historical raw-manifest provenance field. No historical raw gate is loosened or impersonated.

## Dependencies and source identity

Use Python **3.12.3** and the package versions in `historical_environment.json`. The compiler checks the seven scientific runtime package versions and freezes the actual Python build and installed versions for the new run. `requirements-source-prepare.txt` describes source-preparation scientific dependencies; `requirements-cpu-report.txt` distinguishes standard-library aggregate reporting from the model checkpoint audit; `requirements-gpu-train.txt` adds training dependencies. These are direct package requirements, not a claim of a full transitive binary/OS lock. Raw construction also needs the separately pinned MMseqs2 tool and raw-build dependencies; this training adapter does not download or invoke that builder.

The new host must provide a CUDA-capable PyTorch build, a compatible NVIDIA driver, `nvidia-smi`, and BF16/FP16 support. Installing these requirements is not a GPU acceptance. The historical distribution record says PyTorch 2.9.1. A new read-only check found this machine currently imports PyTorch 2.3.0+cu121 while its installed distribution metadata reports 2.9.1. The compiler records both; it does not infer which binary ran historically. Normal training compilation and GPU smoke/full entrypoints now refuse this mismatch. The explicit compiler option `--cpu-validation-only` can create a permanently GPU-disabled protocol for source-data checks and CPU surface validation; that protocol cannot later mint training acceptance or enter either GPU queue. Resolve it in an isolated environment with a consistent package/binary installation before executing GPU work; do not repair the historical environment or treat a metadata-only lock as binary identity. The execution preserves FP32 parameters, FP16 pretraining, BF16 SFT, TF32 disabled, 1e-5 GPU reload tolerance, and the corrected FP32 channel-layout LM cross-entropy. The CPU synthetic formula check allows 1e-6 FP32 rounding between two algebraically equivalent expressions; it does not change the GPU reload tolerance.

`--source-code-root` is the directory containing the seven published October 2 algorithm files (`runtime.py`, `model.py`, `audit_runtime.py`, `fit_surface.py`, `surface_features.py`, `audit_surface.py`, `audit_source.py`). Their exact identities are listed in `algorithm_lock.json`. They can come from the published source archive's `bio2nl/new_data_full_training_v1/` directory. At compilation they are copied into the new run's `code/` directory. The only source substitutions are the explicitly recorded provenance field `raw_manifest_sha256` → `input_resource_lock_sha256`, and the immutable-output guard's `data.raw_root` → `data.source_code_root`; numerical code is unchanged. The old `runtime_data.py` gate is not imported. A new reviewed `runtime_data.py` enforces the portable contract.

## Compile without GPU use

Set these variables to your own explicit locations. `RUN_ROOT` must not exist or overlap the prepared/source-code/adapter trees. `GPU_LOCK` must be the single shared lock used by all queues targeting this GPU, outside this run root. The compiler refuses symlinked roots. Supply SHA-256 values from your completed preparation, not the historical prepared manifest.

```bash
PREPARED_ROOT=/absolute/path/to/new_prepared
ALGORITHM_ROOT=/absolute/path/to/source/bio2nl/new_data_full_training_v1
RUN_ROOT=/absolute/path/to/fresh_training_run
GPU_LOCK=/absolute/path/to/workspace/.bio2nl_gpu_training_queue.lock
MANIFEST_SHA=your_new_prepared_manifest_sha256
ACCEPTANCE_SHA=your_new_preparation_acceptance_sha256
CUDA_VISIBLE_DEVICES='' python -B compile.py \
  --prepared-root "$PREPARED_ROOT" --output-root "$RUN_ROOT" \
  --source-code-root "$ALGORITHM_ROOT" \
  --prepared-manifest-sha256 "$MANIFEST_SHA" \
  --prepared-acceptance-sha256 "$ACCEPTANCE_SHA" --gpu-lock "$GPU_LOCK"
```

The compiler prints the new protocol SHA. Keep that independent value for every command below. A compiled run is immutable and path-bound; recompiling from explicit roots is the supported way to use another machine or location, not editing paths inside a frozen protocol.

CPU-only checking and surface reference fitting can be invoked explicitly before any GPU work:

```bash
PROTOCOL_SHA=the_sha_printed_by_the_compiler
CUDA_VISIBLE_DEVICES='' python -B "$RUN_ROOT/code/cpu_check.py" --protocol "$RUN_ROOT/protocol.json" --protocol-sha256 "$PROTOCOL_SHA"
CUDA_VISIBLE_DEVICES='' python -B "$RUN_ROOT/code/fit_surface.py" --protocol "$RUN_ROOT/protocol.json" --protocol-sha256 "$PROTOCOL_SHA"
CUDA_VISIBLE_DEVICES='' python -B "$RUN_ROOT/code/audit_surface.py" --protocol "$RUN_ROOT/protocol.json" --protocol-sha256 "$PROTOCOL_SHA"
```

These manual CPU commands deliberately do not create `training_acceptance.json`. Do not then rerun the same stages through the technical queue: it refuses existing artifacts. Use a separately compiled fresh root when invoking the complete technical queue.

## Explicit technical GPU verification; no automatic full training

The following command **runs new bounded GPU work when a user invokes it**. This implementation task has not run it. It serializes three two-update pretraining smokes (EP/ES/EE seed 0), one two-update EP source smoke on the fixed first 64 rows, and independent complete-checkpoint replay of all four. The technical queue also runs current CPU checks, five CPU surface fits and their independent audit, then creates this run's acceptance only if all evidence closes and the projected storage reserve passes.

```bash
python -B "$RUN_ROOT/code/run_queue.py" --protocol "$RUN_ROOT/protocol.json" \
  --protocol-sha256 "$PROTOCOL_SHA" --phase technical --execute-gpu-smoke
```

Each GPU stage checks `nvidia-smi` immediately before launch, rejects active compute processes/non-idle utilization, checks disk after that check, sets GPU 0, and holds the shared OS lock throughout the queue. The first failed stage stops execution and preserves its log. There is no automatic resume, overwrite, skipped condition, changed learning rate or target evaluation. Smoke failure requires diagnosis and a new compiled run; it never grants full training acceptance.

## Separately invoked full source-training budget

Only after that new technical acceptance exists can a user invoke:

```bash
python -B "$RUN_ROOT/code/run_queue.py" --protocol "$RUN_ROOT/protocol.json" \
  --protocol-sha256 "$PROTOCOL_SHA" --phase full --execute-full-training
```

The roster is exactly 9 PT jobs (EP/ES/EE × seeds 0/1/2), then 27 source fits (each PT parent × FT seeds 0/1/2), followed by the independent CPU source-selection audit. Each PT uses 16,777,216 input tokens, 1,024 successful updates and the original paired schedules. Each SFT uses five epochs, 1,260 updates and 40,220 presentations, preserves the last partial batch, and selects minimum source-validation CE with earlier-epoch ties. ES/EE must match the same-seed EP initialization; every SFT loads only its corresponding newly trained PT parent. The exact complete selected head/backbone is retained. The surface penalty grid, source-only selection and independent audit remain unchanged.

Acceptance requires fresh CPU, surface, four-model GPU replay and storage evidence bound to this protocol/prepared manifest. Every evidence file is rehashed; full runs cannot enter using only a `passed` status or an old acceptance. Storage reserves 24 GiB for remaining artifacts plus 8 GiB free; each queue launch also checks a 10 GiB immediate reserve. These are conservative disk requirements, not estimated GPU memory use. The full queue stops at source selection and never opens or scores source test/English targets.

## CPU tests and limits

```bash
CUDA_VISIBLE_DEVICES='' BIO2NL_ALGORITHM_ROOT="$ALGORITHM_ROOT" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B test_training.py
```

Tests use temporary synthetic fixtures and a tiny randomly initialized model, including one synthetic CPU optimizer update. They test source/target boundaries, scientific budgets, deterministic seed pairing, corrected CE, pooling, mutation/symlink rejection and queue ordering. They do not claim real model GPU smoke, training reproduction, raw acquisition completion, held-out inference, public deployment or cross-machine numerical equivalence. The current implemented scope and actual integration artifacts are recorded separately; future users must generate their own acceptance.
