import copy
import csv
import gzip
import importlib.util
import json
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parents[1]
BASE = HERE.parents[2]
sys.path.insert(0, str(HERE))
try:
    import verify_release as v
finally:
    sys.path.pop(0)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


@pytest.fixture
def phase_fixture(tmp_path):
    root = tmp_path / 'release'
    root.mkdir()
    code = tmp_path / 'code.py'
    code.write_text('# pinned fixture\n')
    first = {'sha256': '1' * 64, 'bytes': 1}
    value = root / 'data/value'
    value.parent.mkdir()
    value.write_text('final')
    side = root / 'data/side'
    side.write_text('unchanged')
    protocol = {'release_id': 'new-fixture', 'bootstrap': {'sha256': 'b' * 64},
                'execution_code_manifest': {'sha256': 'c' * 64},
                'execution_code': {str(code): v.identity(code)}, 'phases': {}}
    status = {'schema_version': 1, 'status': 'completed', 'protocol_sha256': 'p' * 64,
              'release_root': str(root), 'release_id': protocol['release_id'], 'phases': {}}
    for name in ('protein', 'corpus_initial', 'corpus_dependent'):
        path = tmp_path / (name + '.json')
        stage = {'name': name + '_stage', 'status': 'completed', 'returncode': 0,
                 'command': ['python', str(code)], 'builder_sha256': v.sha(code)}
        stage['outputs'] = ({'data/value': first} if name == 'protein' else
                            {'data/side': v.identity(side)} if name == 'corpus_initial' else
                            {'data/value': v.identity(value)})
        if name == 'corpus_dependent':
            stage['inputs'] = {'data/value': first}
        phase = {'status': 'completed', 'release_root': str(root), 'release_id': protocol['release_id'],
                 'bootstrap_sha256': 'b' * 64, 'builder_manifest_sha256': 'c' * 64, 'stages': [stage]}
        dump(path, phase)
        declared = {'command': ['wrapper', name], 'status_file': str(path), 'stage_count': 1,
                    'stage_names': [stage['name']]}
        protocol['phases'][name] = declared
        status['phases'][name] = {**declared, 'status': 'completed', 'exit_code': 0,
                                 'status_sha256': v.sha(path), 'status_bytes': path.stat().st_size}
    status_path = tmp_path / 'build_status.json'
    dump(status_path, status)
    return root, protocol, status_path


def test_stage_chain_allows_only_declared_later_overwrite(phase_fixture):
    root, protocol, path = phase_fixture
    result = v.validate_build_status(protocol, 'p' * 64, root, path, v.load_old_helper(), {})
    assert result['stage_count'] == 3 and len(result['last_writer_outputs']) == 2


@pytest.mark.parametrize('fault', ['incomplete', 'wrong_phase_count', 'wrong_command', 'wrong_status_size', 'wrong_previous_input', 'changed_final'])
def test_build_lineage_faults_fail_closed(phase_fixture, fault):
    root, protocol, path = phase_fixture
    status = json.loads(path.read_text())
    if fault == 'incomplete':
        status['phases']['protein']['status'] = 'failed'
    elif fault == 'wrong_phase_count':
        del status['phases']['protein']
    elif fault == 'wrong_command':
        status['phases']['protein']['command'] = ['foreign']
    elif fault == 'wrong_status_size':
        status['phases']['protein']['status_bytes'] += 1
    elif fault == 'wrong_previous_input':
        phase_path = Path(status['phases']['corpus_dependent']['status_file'])
        phase = json.loads(phase_path.read_text())
        phase['stages'][0]['inputs']['data/value']['sha256'] = 'f' * 64
        dump(phase_path, phase)
        status['phases']['corpus_dependent'].update(status_sha256=v.sha(phase_path), status_bytes=phase_path.stat().st_size)
    else:
        (root / 'data/value').write_text('tampered')
    dump(path, status)
    with pytest.raises(ValueError):
        v.validate_build_status(protocol, 'p' * 64, root, path, v.load_old_helper(), {})


def test_closed_policy_uses_exactly_eight_deterministic_pair_references():
    archive = BASE / 'release_staging/m1_m3_2026-09-30_v1'
    old = v.load_old_helper()
    entries = old.authenticate(archive)
    review = BASE / 'bio2nl/review/deterministic_rebuild_v1_2026-10-01'
    audit = json.loads((review / 'verification/pair_replay_audit.json').read_text())
    policy = v.rules.make_policy(entries, archive / old.PREFIX, review / 'runs/order_a', audit)
    assert len(policy['files']) == 331
    assert {n for n, x in policy['files'].items() if x['reference_kind'] != 'accepted_v2'} == set(v.rules.PAIR_REFERENCE)
    del entries['data/pairs/protein_sequence_similarity_all.tsv.gz']
    with pytest.raises(ValueError):
        v.rules.make_policy(entries, archive / old.PREFIX, review / 'runs/order_a', audit)


def test_check_pin_rejects_changed_file_and_symlink(tmp_path):
    p = tmp_path / 'input'
    p.write_text('original')
    pin = v.identity(p)
    watch = {}
    v.check_pin(p, pin, watch)
    p.write_text('changed')
    with pytest.raises(ValueError):
        v.check_pin(p, pin, watch)
    link = tmp_path / 'link'
    link.symlink_to(p)
    with pytest.raises(ValueError):
        v.identity(link)


@pytest.fixture(scope='module')
def actual_pair_fixture(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('v2_gate_real_pair_inputs')
    helper = load('v2_real_build_fixture', BASE / 'biopaws/portable_build/deterministic_v2/tests/test_protein_phase.py')
    _, builders, _, _ = helper.bound.__wrapped__(tmp)
    sources = tmp / 'sources'
    sources.mkdir()
    table, source, candidates = helper.previous.inputs.__wrapped__(sources)
    with gzip.open(table, 'rt') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    for row in rows:
        row['length'] = str(len(row['sequence']))
    with gzip.open(table, 'wt', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    current, reference = tmp / 'current', tmp / 'reference'
    result = helper.run_pairs(builders, table, list(reversed(candidates)), current, 0)
    assert result.returncode == 0, result.stderr
    expected_sorted = current / v.rules.SORTED_CANDIDATES
    result = helper.run_pairs(helper.previous.BUILDER.parent.parent, table,
                              [json.loads(x) for x in expected_sorted.read_text().splitlines()], reference, 0)
    assert result.returncode == 0, result.stderr
    ref_input = reference / 'work/protein/positive_candidates.jsonl'
    ref_input.write_bytes(expected_sorted.read_bytes())
    auditor_path = BASE / 'bio2nl/review/deterministic_rebuild_v1_2026-10-01/code_snapshot/audit/verify_pair_replays.py'
    auditor = v.load_module('v2_fixture_frozen_pair_auditor', auditor_path, v.rules.PAIR_AUDITOR_SHA)
    _, pair_rows = auditor.table(reference / auditor.PAIR_FILES[0])
    result = auditor.check_row_semantics(reference, pair_rows)
    for rel in auditor.PAIR_FILES:
        assert auditor.data_bytes(current / rel) == auditor.data_bytes(reference / rel)
    replay = {'verified_files': {str(ref_input): v.identity(ref_input)}, 'scientific_audit': result}
    manifest = {'files': {'code/biopaws/data_v2/build_pairs.py': v.identity(builders / 'data_v2/build_pairs.py')}}
    protocol = {'release_id': '2026-10-01-portable-v2', 'environment': {'python_hash_seed': 0, 'python': platform.python_version()}}
    return current, protocol, manifest, reference, replay, auditor


def test_actual_v2_builder_to_frozen_pair_auditor_integration(actual_pair_fixture):
    current, protocol, builders, reference, replay, auditor = actual_pair_fixture
    result = v.verify_current_pair_inputs(current, protocol, builders, reference, replay, auditor, v.load_old_helper(), {})
    assert result['candidate_records'] == 27 and result['alignment_rows_recomputed'] == 18


def test_current_ordering_metadata_cannot_fake_passing_reference(actual_pair_fixture, monkeypatch):
    current, protocol, builders, reference, replay, auditor = actual_pair_fixture
    old = v.load_old_helper()
    original = old.loads
    def changed(text):
        obj = original(text)
        if isinstance(obj, dict) and 'canonical_sorted_jsonl_sha256' in obj:
            obj['input_candidate_sha256'] = '0' * 64
        return obj
    monkeypatch.setattr(old, 'loads', changed)
    with pytest.raises(ValueError, match='current raw candidates'):
        v.verify_current_pair_inputs(current, protocol, builders, reference, replay, auditor, old, {})


def test_scoped_cli_cannot_accept_unimplemented_or_missing_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    with pytest.raises(FileNotFoundError):
        v.verify_release(tmp_path / 'missing', '0' * 64, tmp_path / 'status', tmp_path / 'acceptance')
