# Portable current-study input preparation

This adapter accepts an explicit raw-input root and creates a fresh output
directory. It needs the 18 relative members in `resource_lock.json`, with exact
SHA256 and sizes. It does not need the historical private archive, its original
absolute paths, original acceptance files, prior prepared examples or weights.

The original source-selection, tokenization, shuffling, exclusion and scheduling
algorithms are vendored unchanged under `vendor/`; their identities are locked.
The 5 former historical metadata dependencies have been replaced by a nontext
projection preserving their hashes and limitations. That projection records an
attributed historical qualification; it does not requalify the QQP cohort or
grant permission to train. QQP text, source-test pairs and predictions are not
read by this preparation. Canonical and remote identity tables are read for
exclusion, including their nontraining identities.

From any working directory, use the actual path to this script:

```bash
CUDA_VISIBLE_DEVICES='' python /path/to/prepare/portable_prepare.py verify-inputs --raw-root /path/to/inputs
CUDA_VISIBLE_DEVICES='' python /path/to/prepare/portable_prepare.py build --raw-root /path/to/inputs --output /path/to/new-prepared
CUDA_VISIBLE_DEVICES='' python /path/to/prepare/portable_prepare.py verify --output /path/to/new-prepared
```

`build` produces 25 payloads: the 22 byte-identical scientific outputs of the
completed Oct 2 study, 2 new provenance documents, and the exact shared-BPE
tokenizer. A new immutable protocol and build manifest identify this execution.
`verify` independently reopens every payload, checks the 22 original hashes,
replays tensor/row identities, source separation, stream indexes and complete
schedules, then writes a new `acceptance.json`. Acceptance is exclusive and
cannot overwrite an earlier run. The acceptance enables only downstream
preparation use; GPU training needs a separate protocol and its own gates.

The source pair subset remains 8,044 train / 20,276 validation. Four streams each
contain 8,388,608 tokens. EP, ES and EE retain their original stream pairings,
with three schedules for pretraining seeds 0, 1 and 2. No scientific budget or
endpoint has changed. Model training and inference are absent from this adapter.

This step begins from locked current-study inputs. It is not a claim that the
entire historical 361-member raw release or original benchmark preparation has
been independently reconstructed. `../resources/` supplies public acquisition
and local English reconstruction separately. The English documents and prepared
English token streams are local outputs, not approved publication payloads.

Run boundary checks with:

```bash
python -m unittest discover -s /path/to/prepare -p 'test_*.py' -v
```

Publish only this README, the two lock/projection JSON files,
`portable_prepare.py`, `test_portable_prepare.py`, and the four `vendor/*.py`
files. `local_integration_*` directories are local scientific outputs and must
not be included in a code upload.
