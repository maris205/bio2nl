# Corpus phases for portable raw rebuild v2

`corpus_phase.py` retains the original seven initial and three dependent stages.
It executes a separate, complete `RELEASE/execution_code/` tree. The historical
`RELEASE/code/` and all 110 bootstrap members remain unchanged and are rehashed.
No training, target inference, release acceptance or upload is performed here.

The parent preparation creates the 28-file execution tree. This module never
copies bootstrap assets or creates that tree. `corpus_source.py` exposes:

```python
new_bytes, provenance = patch_corpus_source(name, original_bytes, release_id)
```

Exactly three source files are allowed, each with a fixed original SHA and exact
replacement count. The NLP conflict ledger edit equals the existing
`deterministic_v1` source transformation. Two `data_release_id` literals in
`write_experiment_configs.py` and one `release_id` literal in `build_longrange.py`
become `2026-10-01-portable-v2`. No other config values, tokenizer algorithm,
sampling parameters or source-acquisition provenance change.

The affected output identity fields are:

- `configs/training_matrix_v2.json`: `data_release_id`.
- `configs/smallweb_gpt2_v2.json`: `data_release_id`.
- `metadata/synthetic_longrange_manifest.json`: `release_id`; the recorded
  `code_sha256` also correctly changes to the actual executed source.

Other historical training/finalization sources in the complete snapshot remain
identity copies and are not invoked. Their old literals do not authorize use of
those entrypoints with the new data.

The builder manifest must be outside the 28-file execution tree. Its top level
contains `schema_version=1`,
`status="execution_code_overlay_not_data_acceptance"`, `release_id`,
`release_root`, `builder_root`, `bootstrap_sha256`, `training_enabled=false`,
`target_scoring_enabled=false`, and `files`. File keys are the original
`code/...` bootstrap names. Each descriptor has exactly:

```json
{
  "source": "/absolute/release/code/relative/file.py",
  "source_sha256": "original physical SHA",
  "source_bytes": 123,
  "snapshot": "/absolute/release/execution_code/relative/file.py",
  "sha256": "executed physical SHA",
  "bytes": 123,
  "transform": "identity or the exact declared patch name"
}
```

Copy both `corpus_phase.py` and `corpus_source.py` into the frozen runner
directory. The runner verifies its sibling helper SHA before loading it.

```bash
CUDA_VISIBLE_DEVICES='' python -B corpus_phase.py \
  --release-root /fresh/release \
  --release-id 2026-10-01-portable-v2 \
  --phase initial --log-root /fresh/review/corpus_initial_execution \
  --bootstrap-sha256 BOOTSTRAP_SHA \
  --builder-root /fresh/release/execution_code \
  --builder-manifest /fresh/release/portable_validation/builder_manifest.json \
  --builder-manifest-sha256 MANIFEST_SHA \
  --smallweb-script /frozen/smallweb_local.py --smallweb-sha256 SMALLWEB_SHA
```

The dependent phase additionally requires `--protein-status` and its own fresh
log directory. The protein status must name all nine stages in exact order and
bind this release, bootstrap and builder manifest. Current protein outputs are
checked using the last writer for each path. Initial corpus completion binds its
final status and all generated outputs; altered dependencies fail before launch.

Each stage checks the complete executed tree before and after its subprocess,
records actual source/output hashes and stops on failure. Final completion also
rehashes all bootstrap members and protein dependencies. Existing outputs,
markers, log directories and symlinked paths are refused. The encoding stage
must preserve the tokenizer specification written by its predecessor.

CPU tests use tiny temporary files and real small workers. The source integration
executes only the original/new config writers and compares all scientific config
fields. It does not run a full corpus build or train a tokenizer/model.
