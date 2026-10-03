# Deterministic intermediate-input pair replay, version 1

This adapter externally orders the complete positive-candidate collection by
`(a, b)` before executing the unchanged frozen `build_pairs.py` with seed
`20260925`. It is an **intermediate-input determinism replay**, not a raw-data
rebuild, accepted data release, model-training gate or replacement for old data.

The original positive/negative definitions, thresholds, orientation draws,
greedy disjoint-edge selection, matching, filtering, final shuffle and baseline
validation execute inside the original builder. The only candidate input change
is explicit ordering and JSON key/whitespace serialization. Score values are
neither rounded nor recomputed. Duplicate candidate edges (including conflicting
duplicates), duplicate JSON keys, malformed or nonfinite values, unknown endpoint
IDs and inconsistent split/component assignments fail instead of being dropped.

`replay_pairs.py` uses bounded chunk sorting and multi-pass heap merging. The
default chunk budget is 64 MiB of encoded records with 32-way merging; Python
object overhead and the canonical accession index are additional memory. It
does not load a second full candidate collection into RAM. The unchanged builder
subsequently retains its original whole-file loading behavior.

## Frozen protocol and command

The protocol must provide `schema_version: 1`, a new `release_id`, `seed: 20260925`
and a `runs` mapping. Each run contains an absolute fresh `root`, a fixed
`python_hash_seed`, an `inputs` mapping and a `code` mapping:

```json
{
  "inputs": {
    "positive_candidates": {"file": "/absolute/candidates.jsonl", "sha256": "...", "bytes": 0},
    "canonical_sequences": {"file": "/absolute/canonical.tsv.gz", "sha256": "...", "bytes": 0}
  },
  "code": {
    "/absolute/replay_pairs.py": {"sha256": "...", "bytes": 0},
    "/absolute/build_pairs.py": {"sha256": "...", "bytes": 0},
    "/absolute/build_protein.py": {"sha256": "...", "bytes": 0}
  }
}
```

The zero sizes and ellipses above are schema placeholders, not usable identities.
The adapter matches the output root to exactly one run and requires all paths,
hashes, sizes, seed and code entries to agree with actual inputs. It hard-pins the
original builder and its `build_protein.py` dependency. Output roots must not
exist; symlinked input/output paths are refused. No resume or overwrite exists.

```bash
PYTHONHASHSEED=0 /root/miniconda3/bin/python -B replay_pairs.py \
  --builder /absolute/frozen/data_v2/build_pairs.py \
  --positive-candidates /absolute/verified-cache/positive_candidates.jsonl \
  --candidates-sha256 ACTUAL_FROZEN_INPUT_SHA \
  --canonical-sequences /absolute/archived/protein_canonical_sequences.tsv.gz \
  --canonical-sha256 ACTUAL_FROZEN_TABLE_SHA \
  --output-root /absolute/fresh/replay-root \
  --release-id deterministic-pair-replay-v1-2026-10-01 \
  --protocol /absolute/frozen/protocol.json \
  --protocol-sha256 ACTUAL_PROTOCOL_SHA \
  --sort-buffer-mib 64
```

The callable API is `replay_pairs(builder, positive_candidates,
candidates_sha256, canonical_sequences, canonical_sha256, output_root,
release_id, protocol, protocol_sha256, sort_buffer_bytes=64<<20,
merge_fanin=32)`. It runs the original main in a CPU-only child and stops on any
execution, integrity or acceptance failure. The child uses `-P -s -B`, explicit
builder import location and a cleaned Python environment. `-I` is intentionally
not used because it would ignore the required `PYTHONHASHSEED`.

## Completion records

`replay_status.json` retains success/failure and the command. On success only,
`replay_manifest.json` records actual source hashes/bytes/counts, the canonical
sorted-content hash, fixed seed, protocol and code identities, worker Python and
dependency versions, and nine builder output identities. Four pair gzip files
also carry SHA-256 of their **complete decoded TSV bytes in row order**, plus
record counts. Gzip header times and measured runtimes can differ between runs;
ordered scientific payload equality must not be confused with those differences.

The actual child writes `worker_runtime.json` before executing the frozen main.
It records its hash seed, `sys.flags.hash_randomization` and a fixed-string hash
probe; its bytes are independently bound by the top-level manifest. Adapter
environment metadata alone is not evidence of the child's hashing regime.

The original builder hardcodes `release_id=2026-09-25-v2`. Its emitted manifest is
preserved byte-for-byte as `metadata/original_builder_manifest.json`; the old
`metadata/protein_pair_manifest.json` path no longer exists after a successful
builder run. The top-level replay manifest has the new identity and explicitly
sets `training_enabled`, `data_release_gate_passed` and
`complete_raw_reconstruction` to false. The embedded old ID is labelled as a
historical builder constant, not acceptance of these new pairs as old v2 data.

## Verification performed before any full replay

```bash
/root/miniconda3/bin/python -m pytest -q tests/test_replay_pairs.py
```

All **21 CPU tests** passed, including an independent rerun. Three candidate
permutations, actual worker hash seeds 0/12345/1259 and different hash probes
produce exactly equal ordered pair TSVs through the real frozen algorithm. A
direct original-builder invocation on the sorted fixture is identical. Tiny
chunk budgets exercise multi-pass external merge; tests cover numeric-value
preservation, duplicate/content failures, protocol bindings, fresh-root/symlink
guards, legacy identity separation and stop-on-builder-failure behavior.

Those tests use a synthetic small fixture. They do not run full candidate caches
or establish that two external caches contain identical complete multisets;
that precondition and subsequent full-run audits belong to the separate frozen
replay protocol.
