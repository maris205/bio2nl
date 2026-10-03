"""Freeze only the new source-input builder and its independent CPU audit."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

R = Path(__file__).resolve().parent
W = R.parents[2]
D = W / 'data_rebuild/2026-10-01-portable-v2'
A = W / 'release_staging/m1_m3_2026-09-30_v1'
SOURCE = W / 'bio2nl/new_data_training_v1'
SNAPSHOT = R / 'data_code_snapshot'


def identity(path):
    if not path.is_file() or any(x.is_symlink() for x in (path, *path.parents)):
        raise ValueError('Missing/symlinked source: ' + str(path))
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'sha256': digest, 'bytes': path.stat().st_size}


def pin(path):
    return {'path': str(path), **identity(path)}


def exclusive(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def main():
    assert not SNAPSHOT.exists() and not (R / 'data_protocol.json').exists()
    assert not (R / 'prepared').exists() and not (R / 'data_acceptance.json').exists()
    raw_acceptance = pin(D / 'portable_validation/acceptance.json')
    raw_manifest = pin(D / 'manifest.json')
    assert raw_acceptance['sha256'] == 'dc35bd6dd36566fef25120434419ab2b358f204678b5d83f7daa89903f5be3f2'
    assert raw_manifest['sha256'] == 'c65ad951dced3285856f8edf49618e3d6b5329fc4ad006520dab748d35b407bf'
    archive_manifest = pin(A / 'archive_manifest.json')
    assert archive_manifest['sha256'] == '31db85eb9e88e0b12d5ed68cd0ea08e276bd509d2565fd11723ba192007aa44f'
    entries = {v['path']: v for v in json.loads((A / 'archive_manifest.json').read_text())['files']}
    spec = importlib.util.spec_from_file_location('new_data_freeze_common', SOURCE / 'common.py')
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    inputs = {'raw:' + name: pin(D / name) for name in common.RAW_MEMBERS}
    accepted = json.loads((D / 'portable_validation/acceptance.json').read_text())
    for name in common.RAW_MEMBERS:
        assert {k: inputs['raw:' + name][k] for k in ('sha256', 'bytes')} == {k: accepted['files'][name][k] for k in ('sha256', 'bytes')}
    for name, suffix in common.METADATA_NAMES.items():
        prefix = 'bio2nl/review/next_round_preparation_v1_2026-09-29/' if name == 'design' else 'bio2nl/review/confirmation_data_v1_2026-09-29/'
        member = prefix + suffix
        inputs[name] = pin(A / member)
        assert {k: inputs[name][k] for k in ('sha256', 'bytes')} == {k: entries[member][k] for k in ('sha256', 'bytes')}
    names = ('__init__.py', 'common.py', 'algorithms.py', 'algorithm_sources.json', 'data_build.py', 'audit_data.py')
    sources = {'new_data_training_v1/' + name: SOURCE / name for name in names}
    sources['run_queue.py'] = R / 'run_queue.py'
    source_pins = {name: pin(path) for name, path in sources.items()}
    payloads = {name: path.read_bytes() for name, path in sources.items()}
    for name, payload in payloads.items():
        assert hashlib.sha256(payload).hexdigest() == source_pins[name]['sha256']
        if name.endswith('.py'):
            compile(payload, name, 'exec')
    for name, payload in payloads.items():
        target = SNAPSHOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(payload)
    code = {str(SNAPSHOT / name): identity(SNAPSHOT / name) for name in sources}
    for name, source in sources.items():
        assert identity(source) == {k: source_pins[name][k] for k in ('sha256', 'bytes')}
    protocol = {
        'schema_version': 1, 'status': 'frozen_before_new_source_input_build',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'release_id': D.name, 'raw_release_root': str(D),
        'raw_acceptance': raw_acceptance, 'raw_manifest': raw_manifest,
        'metadata_archive': {'root': str(A), 'manifest': archive_manifest},
        'inputs': inputs, 'code_pins': code,
        'data': {'source': {'target_rows': 8000, 'selection_seed': 20260925,
                           'expected_train_rows': 8044, 'expected_train_groups': 63,
                           'expected_validation_rows': 20276, 'expected_raw_train_rows': 99818},
                 'token_budget_per_stream': 8388608, 'seeds': [0, 1, 2]},
        'output_root': str(R / 'prepared'),
        'new_derived_construction_authorized': True,
        'training_enabled': False, 'target_scoring_enabled': False,
        'source_test_examples_allowed': False, 'full_budget_training_enabled': False,
        'historical_qqp_has_been_scored': True, 'new_blind_confirmation_claimed': False,
        'raw_acceptance_modified': False,
        'preparation_code': pin(Path(__file__).resolve()),
    }
    for descriptor in (raw_acceptance, raw_manifest, archive_manifest, *inputs.values()):
        assert identity(Path(descriptor['path'])) == {k: descriptor[k] for k in ('sha256', 'bytes')}
    exclusive(R / 'data_protocol.json', protocol)
    protocol_pin = pin(R / 'data_protocol.json')
    package = SNAPSHOT / 'new_data_training_v1'
    base = [sys.executable, '-B']
    common_args = ['--protocol', str(R / 'data_protocol.json'), '--protocol-sha256', protocol_pin['sha256']]
    jobs = [
        {'id': 'data_build', 'command': [*base, str(package / 'data_build.py'), *common_args, '--output', str(R / 'prepared')],
         'cwd': str(R / 'data_build_cwd'), 'log': str(R / 'data_build.log'), 'gpu': False,
         'result_file': str(R / 'prepared/manifest.json'), 'result_status': 'built_pending_independent_audit',
         'required_false_flags': ['training_enabled', 'target_scoring_enabled']},
        {'id': 'data_audit', 'command': [*base, str(package / 'audit_data.py'), *common_args, '--prepared', str(R / 'prepared'), '--output', str(R / 'data_acceptance.json')],
         'cwd': str(R / 'data_audit_cwd'), 'log': str(R / 'data_audit.log'), 'gpu': False,
         'result_file': str(R / 'data_acceptance.json'), 'result_status': 'source_only_smoke_inputs_accepted',
         'required_false_flags': ['full_training_enabled', 'target_scoring_enabled', 'source_test_scoring_enabled'],
         'required_true_flags': ['all_checks_passed']},
    ]
    pins = {**code, str(R / 'data_protocol.json'): identity(R / 'data_protocol.json')}
    for descriptor in (raw_acceptance, raw_manifest, archive_manifest):
        pins[descriptor['path']] = {k: descriptor[k] for k in ('sha256', 'bytes')}
    plan = {'schema_version': 1, 'scope': 'new_data_build_and_independent_audit',
            'review_root': str(R), 'status_file': str(R / 'data_execution_status.json'),
            'pins': pins, 'jobs': jobs, 'required_free_disk_gib': 8,
            'full_budget_training_enabled': False, 'target_scoring_enabled': False}
    exclusive(R / 'data_execution_plan.json', plan)
    exclusive(R / 'data_execution_freeze.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_pins': source_pins,
        'data_protocol': protocol_pin, 'execution_plan': pin(R / 'data_execution_plan.json'),
        'launcher': str(SNAPSHOT / 'run_queue.py'), 'data_construction_launched': False,
    })
    print(json.dumps({'status': 'data_code_and_protocol_frozen', 'protocol_sha256': protocol_pin['sha256'], 'input_count': len(inputs)}))


if __name__ == '__main__':
    main()
