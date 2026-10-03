# New-data full-training runtime API

This package trains only from the independently accepted 2026-10-01 prepared release. The raw release and original derived-data gate remain unchanged. No previous model is an initialization source.

`ReleaseData(protocol_path, protocol_sha256, mode="prepare" | "smoke" | "full")` exposes `.design`, `.execution_protocol`, `.output_root`, `.root`, `.raw_root`, `.source_counts`, `.manifest`, `.manifest_sha256`, `.protocol_sha256`, `.accessed`, and `.full_training_accepted`. `load_source("train" | "validation", smoke=False)` returns int64 arrays and source rows. `load_pretrain(condition, seed)` returns the two memory-mapped streams and one schedule. `verify_current_inputs()` rehashes all accessed inputs.

The protocol uses schema 1, status `frozen_for_new_data_full_training`, scope `source_only_new_data_full_training`, `source_training_authorized=true`, `bounded_smoke_training_enabled=true`, `full_budget_training_enabled=true`, and false `old_weights_allowed`, `target_scoring_enabled`, and `source_test_scoring_enabled`. `raw_release`, `prepared`, `environment`, `runtime_code`, and `design` have the earlier smoke schema. Required current runtime files are runtime.py, runtime_data.py and model.py; other pinned audit/queue/surface files are also verified. The resolved source design must include `steps_per_epoch=252`, `updates_per_fit=1260`, and `presentations_per_fit=40220` for accepted train 8044 / validation 20276.

Full mode additionally reads protocol `training_acceptance_file`, which must be `training_acceptance.json` beside the protocol. This file requires status `accepted_for_source_training`, `all_checks_passed=true`, `full_budget_training_enabled=true`, target/source-test flags false, the exact `protocol_sha256` and `prepared_manifest_sha256`, and a nonempty `verified_files` absolute-path mapping to `{sha256, bytes}`. That mapping must include the protocol. The mandatory current-code evidence paths are `output_root/smoke/audit_runtime/metadata.json` (completed, 4 successful complete-model replays), `protocol_parent/verification/surface_audit.json` (passed, 5 candidates / 10 predictions), and `protocol_parent/verification/storage_acceptance.json` (passed, finite projected_peak_free_gib >= 8). All must bind this protocol; smoke and surface must also bind this prepared manifest. All evidence and the acceptance file itself are pinned on entry and rehashed at completion. The protocol also pins exactly six historical `prior_smoke_evidence` JSON metadata descriptors: original execution protocol/status, corrected replay protocol/result, replay metadata, and independent completion review. The reader preserves the original failed status, requires the corrected four-checkpoint replay to have passed with zero additional updates, and reads no historical checkpoint bytes. This historical evidence does not replace the new current-code smoke gate. Preparation cannot enter a training function. Smoke does not read this later gate.

CLI:

```text
python -B runtime.py --protocol FILE --protocol-sha256 SHA \
  --job pretrain|sft --condition EP|ES|EE --pt-seed 0|1|2 \
  [--ft-seed 0|1|2] --run-kind smoke|full
```

No output override, resume, extra epoch, precision override, or unaccepted-data flag exists. Output paths derive from protocol `output_root`:

- `{run_kind}/pretrain/{condition}/pt{pt_seed}/`
- `{run_kind}/source/{condition}/pt{pt_seed}/ft{ft_seed}/`

Smoke accepts only three pt0 two-update pretraining fits and one EP/pt0/ft0 two-update SFT on the first 64 source rows per split. Full mode supports exactly 9 fresh pretraining jobs and 27 crossed source fits. Pretraining has 1024 successful updates, 16777216 input tokens and 16744448 causal positions. SFT has five epochs, retains each 12-example final batch, 1260 successful updates and 40220 presentations. Source-validation CE selects the complete best checkpoint; exact ties retain the earlier epoch.

Each fit writes `metadata.json` with `kind`, `job={condition,pt_seed,smoke[,ft_seed]}`, `run_kind`, protocol/prepared/raw identities, status, numerical budget and trace. Pretraining writes `model.pt` and `probe.npz` (two source-stream input rows, per-token NLL). NLL uses the fixed float32 channel-layout `cross_entropy(logits[:, :-1].float().transpose(1, 2), ids[:, 1:], reduction="none")`; this preserves the audited arithmetic layout without changing the 1e-5 replay tolerance.

Source fitting retains only `best.pt` plus all `epochs/epoch01..05/{train,validation}.predictions.npz` and `metrics.json` files. History records every epoch's full tensor-state/head/backbone hash; final state hashes do not imply final weights are retained. Per-epoch fixed-checkpoint metrics remain distinct from online/dropout losses. `scoring` points to the selected epoch; `last_scoring` points to epoch 5. Full flags are `only_source_validation_selected_checkpoint=true`, `scientific_model_selection_performed=true`, `technical_smoke_single_epoch=false`. Each epoch's selection metric is `source_validation_cross_entropy`.

Checkpoint payload is `{schema_version:1,kind,config,state_dict,state_sha256,metadata}`. Checkpoint metadata contains job, protocol SHA, prepared manifest SHA and (classifier only) selected epoch. Descriptors contain file, bytes, SHA-256, tensor-state SHA-256, kind and a conservative disk reserve check. Atomic checkpoint replacement is allowed only within the same fit's candidate-best file; at least 8 GiB plus an estimated full payload must be free before each write.

`inputs` includes initial reader pins and job data files. ES/EE also pin same-seed EP metadata for initial-state equality. SFT also pins its new matching pretraining metadata and model. These parent descriptors are repeated as `paired_EP_initialization`, `new_parent_metadata`, and `pretraining_checkpoint`; all are rehashed before completion. The queue owns GPU availability checks, serialization, workspace lock and per-job logs.
