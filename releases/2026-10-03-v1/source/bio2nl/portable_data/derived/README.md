# Portable reconstruction of M1–M3 data

This CPU-only entry point rebuilds fixed derived inputs from a separately verified v2 release and a pinned historical private archive. It never imports historical executable modules, loads model weights, launches training, or scores a target.

```bash
CUDA_VISIBLE_DEVICES='' TOKENIZERS_PARALLELISM=false \
  python -I -B /path/to/portable_data/derived/run.py \
  --archive /path/to/m1_m3_2026-09-30_v1 \
  --release /path/to/rebuilt_v2 \
  --output /path/to/new_derived_output --stage all
```

`--stage all` requires a nonexistent output root and executes `source`, `m1`, `m2`, `m3` serially. Individual stages require their own fresh stage directory; dependencies must already have accepted manifests under the same output root. A failed directory is preserved. A new attempt needs a new directory; there is no silent resume or overwrite.

The release must provide `portable_validation/acceptance.json`:

```json
{
  "status": "accepted_for_derived_reconstruction",
  "training_enabled": false,
  "target_scoring_enabled": false,
  "files": {
    "data/corpora/english/train.jsonl.gz": {
      "sha256": "actual new physical hash",
      "bytes": 123,
      "reference_sha256": "accepted archived file hash",
      "comparison_mode": "explicit upstream comparison rule",
      "equivalent": true
    }
  }
}
```

The upstream gate must bind every file actually read. The builder verifies containment, rejects member symlinks, rechecks bytes and SHA, and checks the archived reference identity. It does not create that upstream gate or infer equivalence from filenames. The archive manifest itself is pinned by SHA `31db85eb9e88e0b12d5ed68cd0ea08e276bd509d2565fd11723ba192007aa44f`.

- `source` replays the 8,002-row whole-block source training selection and retains all 20,338 source validation rows. It never opens source test or QQP examples.
- `m1` revisits the pinned QQP raw bytes offline and reconstructs all historical lexical/graph/encoding exclusions. The resulting 39,893 examples are already observed local data; no new blind or global-unseen claim is made.
- `m2` builds the four exact token streams, three schedules, all parent controls and source train/validation encodings. It reads only M1 aggregate metadata, not M1 example/exclusion files. Canonical/remote identity tables include held-out identities for collision screening; held-out pair examples remain unopened.
- `m3` first checks historical source-selection/barrier metadata, then reconstructs source-test rows and both held-out encodings. This only reproduces input tensors; it performs no inference, new selection, calibration or prediction.

Every data comparison includes the full ordered records or all named tensor arrays with exact dtype and shape. New files have new physical hashes even when gzip headers or JSON spacing differ. The M1 aggregate comparison excludes only execution timestamp and wall time. Token bins and schedule arrays must agree exactly. Stage manifests record actual input paths/hashes, historical reference bindings, semantic comparison details, implementation hashes and new output hashes. Previously generated stage manifests are also bound as input files and rechecked.

`algorithm_sources.json` records the exact upstream file hashes and copied symbols. The copied algorithms have explicit output roots and use new I/O code; frozen originals are unchanged. Requirements: Python 3.12, NumPy 2.2.6, tokenizers 0.20.0, pyarrow 17.0.0, pyahocorasick 2.1.0; the validated environment is recorded by the enclosing execution workflow. Run without `-O`, because historical qualification assertions are required. `python -m unittest discover -s /path/to/derived -p 'test_*.py'` runs the local CPU tests.

A successful test using a byte-copy of accepted v2 inputs is an integration test, not raw-data reconstruction. Raw Swiss-Prot clustering/pairs and upstream corpus/tokenizer construction require their separate successful upstream gate. Raw/reversibly encoded QQP text remains private; this workflow does not authorize uploading it under a project license.
