# New-data technical smoke runtime

The executable accepts exactly five jobs:

```sh
CUDA_VISIBLE_DEVICES=0 python -B runtime.py \
  --protocol /absolute/execution_protocol.json --protocol-sha256 SHA256 \
  --job pretrain_EP --output-root /absolute/results
```

Jobs run serially as `pretrain_EP`, `pretrain_ES`, `pretrain_EE`, `sft_EP`,
`audit_runtime`. Output directories are `results/pretrain/{EP,ES,EE}`,
`results/sft/EP`, and `results/audit_runtime`. Each job requires a fresh directory.
The launcher checks GPU0 immediately before every launch; this worker does not
schedule jobs or change budgets. No full-budget or historical replay mode exists.

Required execution protocol structure:

```text
schema_version: 1
status: frozen_for_new_data_smoke
scope: source_only_new_data_smoke
bounded_smoke_training_enabled: true
full_budget_training_enabled: false
target_scoring_enabled: false
source_test_scoring_enabled: false
output_root: absolute results directory
environment:
  python: exact sys.version
  packages: name -> exact installed version
runtime_code: absolute runtime file -> {sha256, bytes}
raw_release:
  root: absolute new raw-v2 root
  manifest: {file, sha256, bytes}
  acceptance: {file, sha256, bytes}
prepared:
  root: absolute new prepared directory
  manifest: {file, sha256, bytes}
  acceptance: {file, sha256, bytes}
design:
  pretraining: architecture, context_length, batch_size, gradient_accumulation,
               updates_per_run, optimizer, grad_scaler
  source_sft: epochs, batch_size, learning_rate, weight_decay, betas, eps,
              max_grad_norm
  tokenizer: sha256, vocab_size, pad_token_id, eos_token_id,
             pair_separator_id, unk_token_id
```

`runtime_code` must contain the actual copied `runtime.py`, `runtime_data.py`,
`model.py`, and `audit_runtime.py`. Required package identities include NumPy,
PyTorch, Transformers, tokenizers and scikit-learn. Extra frozen code/package
identities are permitted and also checked. Every consumed input/code/control
file is rehashed before a job can complete.

The raw acceptance and manifest retain
`accepted_new_raw_release_training_disabled`, `training_enabled=false`,
`training_gate_pass=false`, and `target_scoring_enabled=false`. New derived
acceptance must provide `status=source_only_smoke_inputs_accepted`,
`prepared_manifest_sha256`, `data_protocol_sha256`,
`raw_release_manifest_sha256`, `raw_release_acceptance_sha256`, `source_counts`,
`all_checks_passed=true`, and false `full_training_enabled`,
`target_scoring_enabled`, `source_test_scoring_enabled`. Thus data construction
and this narrowly scoped execution use separate protocol hashes without a
circular gate.

The prepared manifest follows `data_build.py`: source descriptors and output
hashes, four fixed pretraining streams, seed schedules, tokenizer identity,
raw lineage, and matching `source_prefix64`/`smoke_source` descriptors. Source
counts are read from the new manifest and gate. The first 64 rows per role are
frozen technical prefixes; they need not form complete balanced blocks and are
not an alternative scientific sampling protocol.

Three pretraining jobs start from fresh seed-0 initialization and consume two
updates each. Their initial states must match. The full 1,024-step cosine/21-step
warmup schedule is retained; actual smoke learning rates are 0 and 3e-4/21. A
zero-rate first update is recorded; parameters must change after the second.
Source SFT inherits only this smoke's new EP backbone, initializes a fresh
bias-free head, and performs two updates over 64 new source training rows at
2e-5 then 1e-5. Its sole retained epoch is a technical checkpoint, not evidence
of validation-selected generalization.

Independent replay reloads all four complete new checkpoints with exact
file/state/protocol/data identities. Its forward implementations differ from
the training scoring helpers. Every LM scores its two scheduled probe rows;
the classifier scores 64 training and 64 validation rows. Absolute tolerance
is fixed at 1e-5, classification decisions must agree exactly, and model states
must remain unchanged. This validates the technical path only.

`model.py` is a byte-exact copy of
`bio2nl/portable_runtime/biotrans_portable/model.py` (SHA-256
`b2362edf8850fe20fbd23cc958a71f92ac2a187de4316f53f8750d3bca857592`).
Training code derives from its companion `training.py`, with historical reads
removed, only bounded jobs retained, and new data/protocol/parent bindings.
Neither module imports old project code or reads old model artifacts.
