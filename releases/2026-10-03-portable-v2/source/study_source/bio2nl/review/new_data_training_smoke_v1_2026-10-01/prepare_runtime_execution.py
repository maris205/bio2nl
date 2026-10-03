"""Freeze the bounded GPU smoke only after independent new-input acceptance."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

R = Path(__file__).resolve().parent
W = R.parents[2]
D = W / 'data_rebuild/2026-10-01-portable-v2'
P = R / 'prepared'
A = W / 'release_staging/m1_m3_2026-09-30_v1'
SOURCE = W / 'bio2nl/new_data_training_v1'
SNAPSHOT = R / 'runtime_code_snapshot'


def identity(path):
    if not path.is_file() or any(x.is_symlink() for x in (path, *path.parents)):
        raise ValueError('Missing/symlinked frozen input')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'sha256': digest, 'bytes': path.stat().st_size}


def pin(path):
    return {'file': str(path), **identity(path)}


def exclusive(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    assert not SNAPSHOT.exists() and not (R / 'execution_protocol.json').exists()
    assert not (R / 'results').exists()
    data_status = json.loads((R / 'data_execution_status.json').read_text())
    assert data_status['status'] == 'completed' and len(data_status['jobs']) == 2
    assert all(x['status'] == 'completed' and x['exit_code'] == 0 for x in data_status['jobs'])
    prepared_manifest = pin(P / 'manifest.json')
    prepared_acceptance = pin(R / 'data_acceptance.json')
    data_protocol_pin = pin(R / 'data_protocol.json')
    assert {k: prepared_acceptance[k] for k in ('sha256', 'bytes')} == {k: data_status['jobs'][1]['result'][k] for k in ('sha256', 'bytes')}
    manifest = json.loads((P / 'manifest.json').read_text())
    gate = json.loads((R / 'data_acceptance.json').read_text())
    assert gate['status'] == 'source_only_smoke_inputs_accepted' and gate['all_checks_passed'] is True
    assert gate['prepared_manifest_sha256'] == prepared_manifest['sha256']
    assert gate['source_counts'] == manifest['source_counts'] == {'train': 8044, 'validation': 20276}
    assert gate['data_protocol_sha256'] == data_protocol_pin['sha256']
    assert gate['full_training_enabled'] is False and gate['target_scoring_enabled'] is False
    for relative, expected in manifest['outputs'].items():
        assert identity(P / relative) == expected
    raw_manifest, raw_acceptance = pin(D / 'manifest.json'), pin(D / 'portable_validation/acceptance.json')
    assert raw_manifest['sha256'] == gate['raw_release_manifest_sha256'] == 'c65ad951dced3285856f8edf49618e3d6b5329fc4ad006520dab748d35b407bf'
    assert raw_acceptance['sha256'] == gate['raw_release_acceptance_sha256'] == 'dc35bd6dd36566fef25120434419ab2b358f204678b5d83f7daa89903f5be3f2'
    archive_manifest = pin(A / 'archive_manifest.json')
    assert archive_manifest['sha256'] == '31db85eb9e88e0b12d5ed68cd0ea08e276bd509d2565fd11723ba192007aa44f'
    historical_member = 'bio2nl/review/joint_pretraining_v1_2026-09-29/protocol.json'
    historical_pin = pin(A / historical_member)
    archive_files = {x['path']: x for x in json.loads((A / 'archive_manifest.json').read_text())['files']}
    assert {k: historical_pin[k] for k in ('sha256', 'bytes')} == {k: archive_files[historical_member][k] for k in ('sha256', 'bytes')}
    old = json.loads((A / historical_member).read_text())
    pt_keys = ('architecture', 'context_length', 'batch_size', 'gradient_accumulation', 'updates_per_run', 'optimizer', 'grad_scaler')
    sft_keys = ('epochs', 'batch_size', 'learning_rate', 'weight_decay', 'betas', 'eps', 'max_grad_norm', 'dropout', 'endpoint_content_token_cap', 'format', 'pooling', 'head_initializer_std', 'keep_final_partial_batch', 'parameter_dtype', 'autocast_dtype')
    design = {'pretraining': {k: old['pretraining'][k] for k in pt_keys},
              'source_sft': {k: old['source_sft'][k] for k in sft_keys},
              'tokenizer': {k: old['tokenizer'][k] for k in ('vocab_size', 'pad_token_id', 'eos_token_id', 'pair_separator_id', 'unk_token_id')}}
    design['tokenizer'].update(sha256=manifest['tokenizer']['sha256'], source_release_id=D.name,
                               rebuilt_in_raw_release=True, retrained_in_this_smoke=False)
    assert design['tokenizer']['sha256'] == old['tokenizer']['sha256']
    design['source_sft']['new_full_source_counts'] = manifest['source_counts']
    design['source_sft']['full_budget_reference_only'] = {'enabled': False, 'updates_per_fit': 1260, 'sample_presentations_per_fit': 40220, 'steps_per_epoch': 252}
    sources = {name: SOURCE / name for name in ('runtime.py', 'runtime_data.py', 'model.py', 'audit_runtime.py')}
    sources['run_queue.py'] = R / 'run_queue.py'
    source_pins = {name: pin(path) for name, path in sources.items()}
    payloads = {name: path.read_bytes() for name, path in sources.items()}
    assert source_pins['model.py']['sha256'] == 'b2362edf8850fe20fbd23cc958a71f92ac2a187de4316f53f8750d3bca857592'
    for name, payload in payloads.items():
        assert hashlib.sha256(payload).hexdigest() == source_pins[name]['sha256']
        compile(payload, name, 'exec')
    SNAPSHOT.mkdir()
    for name, payload in payloads.items():
        with (SNAPSHOT / name).open('xb') as stream:
            stream.write(payload)
    code = {str(SNAPSHOT / name): identity(SNAPSHOT / name) for name in sources}
    for name, source in sources.items():
        assert identity(source) == {k: source_pins[name][k] for k in ('sha256', 'bytes')}
    gpu_policy = {'max_samples': 16, 'interval_seconds': 1, 'memory_used_mib_below': 500, 'utilization_percent_at_most': 10, 'requires_no_compute_processes': True, 'fresh_before_each_gpu_job': True}
    protocol = {
        'schema_version': 1, 'status': 'frozen_for_new_data_smoke', 'scope': 'source_only_new_data_smoke',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'bounded_smoke_training_enabled': True, 'full_budget_training_enabled': False,
        'target_scoring_enabled': False, 'source_test_scoring_enabled': False,
        'old_weights_allowed': False, 'output_root': str(R / 'results'),
        'environment': {'python': sys.version, 'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'torch', 'transformers', 'tokenizers', 'scikit-learn', 'scipy', 'pyarrow')}},
        'runtime_code': code,
        'raw_release': {'root': str(D), 'manifest': raw_manifest, 'acceptance': raw_acceptance},
        'prepared': {'root': str(P), 'manifest': prepared_manifest, 'acceptance': prepared_acceptance},
        'design': design, 'numerical_design_metadata_source': historical_pin,
        'bounded_jobs': {'pretraining': [{'condition': c, 'pt_seed': 0, 'updates': 2} for c in ('EP', 'ES', 'EE')],
                         'source_sft': {'condition': 'EP', 'pt_seed': 0, 'ft_seed': 0, 'updates': 2, 'source_prefix_rows': 64, 'parent': 'this_run_new_EP_only'},
                         'independent_reload': {'complete_models': 4, 'lm_probe_rows_per_model': 2, 'source_train_rows': 64, 'source_validation_rows': 64, 'absolute_tolerance': 1e-5, 'decisions_exact': True}},
        'numerical_contract': {'pretraining_schedule_total_steps': 1024, 'warmup_steps': 21,
                               'pretraining_learning_rates': [0.0, 3e-4 / 21], 'sft_learning_rates': [2e-5, 1e-5],
                               'first_pretraining_zero_learning_rate_is_expected': True,
                               'pretraining_final_parameters_must_change': True, 'sft_head_and_backbone_must_change': True,
                               'tf32_enabled': False, 'full_training_or_generalization_claimed': False},
        'source_prefixes': manifest['source_prefix64'],
        'gpu_availability_policy': gpu_policy, 'local_device': 0, 'serial_only': True,
        'required_free_disk_gib': 8, 'estimated_checkpoint_gib': 2,
        'new_blind_confirmation_claimed': False, 'remote_publication_enabled': False,
        'preparation_code': pin(Path(__file__).resolve()),
    }
    controls = (prepared_manifest, prepared_acceptance, raw_manifest, raw_acceptance, historical_pin, archive_manifest, data_protocol_pin)
    for descriptor in controls:
        assert identity(Path(descriptor['file'])) == {k: descriptor[k] for k in ('sha256', 'bytes')}
    exclusive(R / 'execution_protocol.json', protocol)
    protocol_pin = pin(R / 'execution_protocol.json')
    jobs = []
    for name, output in [('pretrain_EP', 'pretrain/EP'), ('pretrain_ES', 'pretrain/ES'), ('pretrain_EE', 'pretrain/EE'), ('sft_EP', 'sft/EP'), ('audit_runtime', 'audit_runtime')]:
        jobs.append({'id': name, 'command': [sys.executable, '-B', str(SNAPSHOT / 'runtime.py'), '--protocol', str(R / 'execution_protocol.json'), '--protocol-sha256', protocol_pin['sha256'], '--job', name, '--output-root', str(R / 'results')],
                     'cwd': str(R / (name + '_cwd')), 'log': str(R / (name + '.log')), 'gpu': True,
                     'result_file': str(R / 'results' / output / 'metadata.json'), 'result_status': 'completed',
                     'required_false_flags': ['old_weights_loaded'],
                     'required_true_flags': ['all_new_checkpoint_replays_passed'] if name == 'audit_runtime' else []})
    pins = {**code, str(R / 'execution_protocol.json'): identity(R / 'execution_protocol.json')}
    for descriptor in controls:
        pins[descriptor['file']] = {k: descriptor[k] for k in ('sha256', 'bytes')}
    plan = {'schema_version': 1, 'scope': 'bounded_new_data_smoke', 'review_root': str(R),
            'status_file': str(R / 'execution_status.json'), 'pins': pins, 'jobs': jobs,
            'gpu_availability_policy': gpu_policy, 'required_free_disk_gib': 8,
            'full_budget_training_enabled': False, 'target_scoring_enabled': False}
    exclusive(R / 'execution_plan.json', plan)
    exclusive(R / 'execution_freeze.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_pins': source_pins, 'protocol': protocol_pin, 'execution_plan': pin(R / 'execution_plan.json'),
        'launcher': str(SNAPSHOT / 'run_queue.py'), 'gpu_jobs': 5, 'training_updates_authorized': 8,
        'formal_budget_runs_authorized': 0})
    print(json.dumps({'status': 'bounded_runtime_frozen_not_launched', 'protocol_sha256': protocol_pin['sha256']}))


if __name__ == '__main__':
    main()
