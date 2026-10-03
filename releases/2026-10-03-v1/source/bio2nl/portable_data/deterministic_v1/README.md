# Deterministic serialization support, version 1

This module supplies two minimal source edits for a **new snapshot** and a strict
remote-manifest reference validator. It never edits existing builders/data,
executes construction, writes an acceptance gate, trains a model or publishes.
It does not implement the separate candidate-order/pair-sampling revision.

The rejected `2026-10-01-portable-v1` run and its original verifier remain frozen.
Diagnostic evidence is in
`review/portable_data_rebuild_v1_2026-10-01/failure_review_2026-10-01/metadata_diff_review.json`.
The new implementation has new source identities; its outputs must not be
described as an exact replay of every historical serialized file.

## Minimal API

```python
from portable_data.deterministic_v1 import (
    patch_source, write_patched_snapshot,
    historical_remote_erratum, verify_current_remote_manifest,
)

patched_bytes, provenance = patch_source(relative_name, original_bytes)
```

`relative_name` is exactly one of:

- `code/bio2nl/data/rebuild_v2/nlp_synthetic/build.py`
- `code/biopaws/data_v2/build_protein.py`

Both accepted original SHA-256 values are pinned in `patches.py`. Unknown or
changed inputs, repeated patching and missing/duplicate edit anchors fail.
Combining a further adapter requires passing the returned bytes explicitly and
recording its additional before/after identities. No original module imports or
monkeypatching are used.

`write_patched_snapshot(source_root, fresh_output_root)` writes **only the two
patched builders plus `patch_manifest.json`**, after validating both inputs. It
is a source overlay, not a complete runnable builder tree. Copy unchanged helpers
under a separately recorded complete snapshot; do not overwrite a historical
tree or assume this module launched anything. The source and output roots must
be disjoint. The optional CLI performs only this overlay operation:

```bash
PYTHONPATH=/path/to/bio2nl python -B -m portable_data.deterministic_v1 \
  --source-root /path/to/unmodified/raw-build-root \
  --output /fresh/path/serialization-overlay
```

## Exact changes

The NLP edit changes `for key in bad` to `for key in sorted(bad)` for the conflict
exclusion ledger. It does not sort training examples or change exclusion
membership, retained examples or the order within each conflict group. The old
ledger had 208 identical full records in another order; its original ordered
comparison remains failed. The new deterministic log is a declared new output.

The canonical protein table uses the verified accepted header:

```text
accession entry_name taxon_id sequence_version sequence_sha256 sequence length pretrain_eligible pretrain_exclusion_reason cluster_id split
```

The write point checks **every row's complete named-field set** before writing.
Missing or additional fields fail; rows, values, sequence strings, labels and
split assignments are not changed. Other table formats are untouched.

## Historical remote metadata discrepancy

`historical_remote_erratum(old_manifest_bytes, old_source_bytes)` accepts only the
two exact pinned archived byte strings. It returns an explanatory record, never
a corrected old manifest. The old nested descriptor says 1,825 bytes and SHA
`513f7120...`; the actual archived source is 1,663 bytes and SHA `16535a38...`.
The top-level old release manifest correctly binds that actual source. The time
and contents of the prior source-record revision are not established here.

`verify_current_remote_manifest(release_root, *, allowed_code_roots=())` reads the
new manifest and rehashes every listed artifact against its **own** SHA and byte
count, validates its explicit source record and code reference, then repeats the
identity checks before returning. Symlinks, escapes, duplicate descriptors,
wrong sizes and stale/copied hashes fail. External builder snapshots require an
explicit allowed code root. The function neither normalizes away mismatches nor
asserts equivalence to old scientific outputs. Its success is reference-integrity
evidence only, with training and scoring disabled.

## Validation

```bash
CUDA_VISIBLE_DEVICES='' python -B -m unittest discover \
  -s /path/to/portable_data/deterministic_v1 -p 'test_*.py' -v
```

Tests execute the actual inserted source snippets on tiny fixtures, vary group
and dict insertion order, launch independent Python processes with four hash
seeds, and reject changed source/schema/current artifact identities. They never
run historical builders or produce training data. Physical header/content checks
against the archived tables are separate read-only integration evidence.
