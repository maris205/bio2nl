# Offline raw-data and derived-input reconstruction

These entrypoints rebuild the accepted v2 data from checksum-pinned cached raw inputs, then rebuild M1–M3 inputs. The output is a new reconstruction with its own physical hashes. The historical release, results and model weights stay unchanged. All stages use CPU only. A successful data gate authorizes derived-data construction, not training or target scoring.

This currently requires the private archive `m1_m3_2026-09-30_v1` (manifest SHA-256 `31db85eb9e88e0b12d5ed68cd0ea08e276bd509d2565fd11723ba192007aa44f`). It is an offline reproducibility entrypoint, not a claim that a public Hugging Face download alone can reproduce everything. Raw/reversibly encoded QQP and other text distribution restrictions remain in force. Read `../review/portable_data_rebuild_v1_2026-10-01/release_plan/SOURCE_DISTRIBUTION.md` before assembling public assets.

## Inputs and environment

Use Python 3.12.3 and the pinned packages in `requirements.txt`. MMseqs2 is supplied as a hash-verified historical binary/tool archive, not downloaded from a mutable latest URL. Protein work uses 16 CPU threads. Reserve at least 20 GiB free before bootstrap; more space is useful for the derived reconstruction and source exports. The full protein phase can take about an hour; other CPU work adds time. Frozen source snapshots and per-stage logs identify the actual code and inputs used.

Set `CUDA_VISIBLE_DEVICES=''`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and `PYTHONDONTWRITEBYTECODE=1`. Supply absolute paths and use a fresh output root and fresh log directories. Failed outputs are preserved, and these runners do not silently resume them. Do not run Python with `-O`, which disables required historical assertions.

## Execution order

1. `bootstrap.py --archive ARCHIVE --output NEW_RELEASE` verifies the private archive and accepted-v2 manifests, then copies exactly 77 raw/tool/external-tokenizer assets, 28 builder code files, three source-provenance records and two environment records. It copies no prepared rows, work products, token bins or shared tokenizers. The external GPT-2 tokenizer is an upstream input; shared English/protein BPE is retrained later.
2. Run `../../biopaws/portable_build/protein_phase.py --release-root NEW_RELEASE --builder-root NEW_RELEASE/code/biopaws --log-root NEW_PROTEIN_LOGS`. This executes the original nine-stage protein/SCOPe construction with explicit roots. It stops at the first failure.
3. Independently of step 2, run `corpus_phase.py --phase initial --release-root NEW_RELEASE --log-root NEW_INITIAL_LOGS --bootstrap-sha256 BOOTSTRAP_SHA --smallweb-script /absolute/path/smallweb_local.py --smallweb-sha256 ADAPTER_SHA`. This rebuilds official/clean NLP, typed Dyck and longrange examples, English/DNA corpora, configuration files and GPT-2-tokenized SmallWeb. `smallweb_local.py` preserves the original token-budget/EOS algorithm while loading only verified local tokenizer files; provenance is in `adapter_provenance.json`.
4. After both phases complete, run the same `corpus_phase.py` with `--phase dependent`, a new log root and `--protein-status NEW_PROTEIN_LOGS/protein_phase_status.json`. This rebuilds protein/shuffled/random-AA corpora, both shared tokenizers and all 30 matrix token streams.
5. `verify_release.py --archive ARCHIVE --release-root NEW_RELEASE --verification-protocol /absolute/path/VERIFICATION_PROTOCOL.json` compares the complete declared scientific payload against the fixed accepted reference. The frozen verification protocol is shipped as `VERIFICATION_PROTOCOL.json`. Gzip and relocation differences are explicitly accounted for; ordered rows, labels, splits, tokenizer vocabulary/merge order, token IDs, budgets, EOS and index boundaries must match. It writes a new `portable_validation/acceptance.json`, never an old integrated training gate.
6. Run `derived/run.py --archive ARCHIVE --release NEW_RELEASE --output NEW_DERIVED --stage all` with a fresh output root. See `derived/README.md` for the four stages and their data-access boundaries. M3 reconstructs already-observed held-out input tensors only. No checkpoint is loaded and no prediction is computed.

For each `*_SHA`, compute SHA-256 of the indicated file before launch and record it with the executed source hashes. The phase runners use those exact bindings, not a directory name as evidence of identity. Review the actual status/logs and acceptance report; the existence of an output directory does not establish success.

## Source versions and exports

The release-plan generator expands an explicit source allowlist and records candidate bytes/hashes. `source_release.py` consumes that inventory with its SHA, creates real local Git commits on a new branch through isolated temporary indexes, and exports only the allowlisted commit blobs. It does not touch the original checkout/index, perform a whole-tree export or contact a remote. Historical experiments retain their original per-file code identities; a later commit is a source capture, not a retroactive experiment revision. HF dataset/model revisions remain unset until real uploads and read-back checks exist.

The original repository history contains older data. Exporting the complete Git tree would include assets beyond the reviewed source allowlist. The local source export is not automatically public-release ready, and source repository licensing does not relicense underlying datasets.

## Validation

`python -B -m unittest discover -s /path/to/portable_data/tests -p 'test_*.py'` runs corpus adapter, acceptance and source-export checks. Run `python -B -m pytest -q -p no:cacheprovider /path/to/portable_data/tests/test_bootstrap.py` for the 15 parametrized bootstrap checks. `python -B -m unittest discover -s /path/to/portable_data/derived -p 'test_*.py'` runs the derived-builder tests. The separate full reconstruction and execution reports are under `../review/portable_data_rebuild_v1_2026-10-01/`; an accepted-input copy fixture test is explicitly distinct from an actual raw reconstruction. Full-budget model retraining and fresh online source acquisition are outside these validations.
