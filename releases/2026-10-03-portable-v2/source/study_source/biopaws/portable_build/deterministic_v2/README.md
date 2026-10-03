# Deterministic protein raw construction, version 2

This is a new CPU construction path for `2026-10-01-portable-v2`. The accepted
September release, rejected portable-v1 build, deterministic intermediate-input
replays, and their original builders are preserved. These files prepare a new
build; this implementation alone is neither a data release gate nor a training
authorization.

`patches.py` exposes two pure functions, each returning `(bytes, provenance)`:

```python
patched_pair, pair_receipt = patch_pair_source(original_pair_bytes)
patched_remote, remote_receipt = patch_remote_source(original_remote_bytes)
```

Both functions require the fixed original SHA and exactly one occurrence of each
edit anchor. Changed input, repeat patching and unexpected anchors fail. They do
not write or execute a builder. The complete snapshot is prepared separately:
the 28 original bootstrap source files stay in `RELEASE/code/`, while the new
executed tree is `RELEASE/execution_code/`. The existing
`bio2nl/portable_data/deterministic_v1/patches.py` supplies the canonical protein
TSV column fix. No original module is changed or monkeypatched.

The pair edit validates and sorts the complete already-loaded candidates by
ascending ASCII `(a,b)` before the existing seeded shuffle. It rejects duplicate
edges and malformed/invalid candidates without dropping records. It preserves
all values, positive/negative definitions, alignment thresholds, RNG seed and
calls, endpoint matching, and final output ordering algorithm. The original
builder already loads all candidates; sorting retains that memory regime.

Before shuffling, the new builder writes
`work/protein/positive_candidates_sorted.jsonl` using the same compact,
sorted-key ASCII JSON serialization as deterministic-v1. It keeps the original
candidate file unchanged. `metadata/protein_candidate_order.json` records the
actual input SHA, sorted file SHA/bytes/count, ordering definition, fixed seed,
builder SHA and worker Python/hash-seed information. Sorting is a declared new
construction rule; it is not a claim that historical pair rows are reproduced.

Both newly emitted pair and remote manifests use `2026-10-01-portable-v2` and
label `2026-09-25-v2` only as `original_algorithm_release_id`. Pair metadata binds
the ordering record and its actual four output gzip hashes. Remote descriptors
are generated from current physical files and checked again by the runner;
there is no exception for the old stale source descriptor.

The runner accepts the common 28-file manifest produced by this build's
`prepare_builders.py`. It verifies the manifest SHA and root, bootstrap SHA/raw
inputs, all eight original/executed protein source descriptors, exact expected
patched code identities and transform names. Unknown source members, symlinks,
preexisting protein outputs/logs, and a previous start marker fail. It hashes
code before and after each stage and records actual stage input/output/log
identities. Only remote exclusion may rewrite its three declared current-run
canonical tables. Other stage inputs must remain unchanged.

```bash
CUDA_VISIBLE_DEVICES='' /root/miniconda3/bin/python -B \
  biopaws/portable_build/deterministic_v2/protein_phase.py \
  --release-root /absolute/new/release \
  --builder-root /absolute/new/release/execution_code/biopaws \
  --builder-manifest /absolute/new/release/portable_validation/builder_manifest.json \
  --builder-manifest-sha256 ACTUAL_MANIFEST_SHA256 \
  --log-root /absolute/new/protein_logs
```

Execution is serial and CPU-only: normalize, cluster, search, graph/split, remote
prepare, remote exclusion, protein pairs, remote pairs, cross-dataset audit.
MMseqs/remote workers use 16 threads/processes; the original pair builder retains
its single-thread numerical-library settings. Python child workers receive hash
seed 0 and `-s -B`, which preserve that seed. Failures stop the queue and retain
logs and partial outputs. A failed build cannot resume in the same destination.

`protein_phase_status.json` binds release ID/root, bootstrap and builder-manifest
SHA, runner identity and nine per-stage records. Completed execution still has
`complete_release_gate_passed=false`, `training_enabled=false` and
`target_scoring_enabled=false`. The separate new full-release verifier must
compare deterministic pair outputs to the successful intermediate replay and
check all unchanged scientific payloads against the old accepted reference.

Tests use synthetic fixtures, never the full raw build:

```bash
CUDA_VISIBLE_DEVICES='' /root/miniconda3/bin/python -B -m pytest -q \
  biopaws/portable_build/deterministic_v2/tests/test_protein_phase.py
```

The actual patched original pair algorithm runs on overlapping candidate graphs
under two candidate permutations/hash seeds, and matches all four decompressed
pair files from the original builder fed deterministic-v1 canonical ordering.
Additional tests reject malformed/duplicate candidates, changed source and raw
identities, stale remote descriptors, reused outputs and injected failures.
Those fixtures do not establish successful full raw reconstruction.
