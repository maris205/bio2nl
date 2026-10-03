"""Tiny mocked-process tests; never invokes a corpus/protein builder or GPU."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('portable_v2_root_queue_test', HERE / 'run_queue.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + '\n')


@pytest.fixture
def fixture(tmp_path):
    review, root = tmp_path / 'review', tmp_path / 'release'
    review.mkdir(); root.mkdir()
    controls = {}
    for name in ('bootstrap', 'execution_code_manifest'):
        path = root / 'portable_validation' / (name + '.json')
        save(path, {'tiny': name})
        controls[name] = {'file': str(path), **q.identity(path)}
    worker = review / 'fixture_worker.py'; worker.write_text('# never executed\n')
    auditor = review / 'fixture_auditor.py'; auditor.write_text('# never executed\n')
    phases = {name: {'status_file': str(review / name / 'status.json'),
                     'stage_count': 1, 'stage_names': [name + '_tiny_stage'], 'command': [name]}
              for name in ('protein', 'corpus_initial', 'corpus_dependent')}
    protocol = {'review_root': str(review), 'release_root': str(root),
                'release_id': '2026-10-01-portable-v2', 'training_enabled': False,
                'target_scoring_enabled': False, 'remote_publication_performed': False,
                'execution_code': {str(p): q.identity(p) for p in (HERE / 'run_queue.py', worker, auditor)},
                **controls, 'phases': phases, 'auditor': str(auditor)}
    path = review / 'protocol.json'; save(path, protocol)
    return SimpleNamespace(review=review, root=root, protocol=protocol, path=path, worker=worker,
                           events=[], signals=[], processes={})


def execute_fixture(f, mode='success'):
    class Process:
        def __init__(self, name, code):
            self.name, self.code = name, code
            self.pid = 90000 + len(f.processes)
            self.returncode = None
            self.mutated = False

        def poll(self):
            f.events.append(('poll', self.name))
            if self.name == 'protein' and mode in ('code_mutation', 'input_control_mutation') and not self.mutated:
                target = f.worker if mode == 'code_mutation' else Path(f.protocol['bootstrap']['file'])
                target.write_text('changed after launch')
                self.mutated = True
            self.returncode = self.code
            return self.code

        def wait(self, timeout=None):
            f.events.append(('wait', self.name))
            if self.code is None:
                raise subprocess.TimeoutExpired(self.name, timeout)
            self.returncode = self.code
            return self.code

    def popen(command, **kwargs):
        name = command[0] if command[0] in f.protocol['phases'] else 'independent_acceptance'
        f.events.append(('spawn', name))
        assert kwargs['start_new_session'] is True
        assert kwargs['stdin'] == subprocess.DEVNULL
        assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
        assert kwargs['env']['HF_HUB_OFFLINE'] == '1'
        assert kwargs['env']['PYTHONHASHSEED'] == '0'
        assert 'PYTHONPATH' not in kwargs['env'] and 'PYTHONSTARTUP' not in kwargs['env']
        if mode == 'second_popen_exception' and name == 'corpus_initial':
            raise OSError('injected Popen failure')
        code = 0
        if name == 'protein' and mode in ('first_failure', 'kill_escalation'):
            code = 7
        if ((name == 'corpus_initial' and mode in ('first_failure', 'kill_escalation')) or
                (name == 'protein' and mode == 'second_popen_exception')):
            code = None
        if name == 'independent_acceptance' and mode == 'audit_nonzero':
            code = 8
        process = Process(name, code); f.processes[name] = process
        if name in f.protocol['phases']:
            spec = f.protocol['phases'][name]
            value = {'status': 'completed', 'release_root': str(f.root),
                     'stages': [{'name': spec['stage_names'][0], 'status': 'completed', 'exit_code': 0}]}
            if mode == 'invalid_phase_state' and name == 'protein':
                value['stages'][0]['name'] = 'wrong_stage'
            save(Path(spec['status_file']), value)
        else:
            value = {'status': 'accepted_new_raw_release_training_disabled', 'training_enabled': False,
                     'target_scoring_enabled': False, 'release_id': f.protocol['release_id'], 'training_gate_pass': False}
            if mode == 'audit_zero_rejected':value['status'] = 'rejected'
            if mode == 'audit_wrong_flag':value['training_enabled'] = True
            save(f.root / 'portable_validation/acceptance.json', value)
            save(f.root / 'manifest.json', value)
        return process

    def killpg(pid, sig):
        f.signals.append((pid, sig))
        owned = [p for p in f.processes.values() if p.pid == pid]
        assert len(owned) == 1 and owned[0].returncode is None, 'Only live queue-owned process groups may be signalled'
        if mode != 'kill_escalation' or sig == signal.SIGKILL:
            owned[0].code = -int(sig)

    with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': '/untrusted', 'PYTHONSTARTUP': '/untrusted'}), \
            patch.object(q.subprocess, 'Popen', side_effect=popen), patch.object(q.os, 'killpg', side_effect=killpg), \
            patch.object(q.shutil, 'disk_usage', return_value=SimpleNamespace(free=100 << 30)), \
            patch.object(q.time, 'sleep', side_effect=lambda _: None):
        q.execute(f.path, q.identity(f.path)['sha256'])


def states(f):
    return (json.loads((f.review / 'execution_status.json').read_text()),
            json.loads((f.review / 'build_status.json').read_text()))


def test_initial_workers_both_start_before_wait_and_success_flags(fixture):
    f = fixture
    execute_fixture(f)
    assert f.events[:2] == [('spawn', 'protein'), ('spawn', 'corpus_initial')]
    assert [name for event, name in f.events if event == 'spawn'] == ['protein', 'corpus_initial', 'corpus_dependent', 'independent_acceptance']
    execution, build = states(f)
    assert execution['status'] == build['status'] == 'completed'
    assert all(s['status'] == 'completed' and s['exit_code'] == 0 for s in execution['stages'].values())
    manifest = json.loads((f.root / 'manifest.json').read_text())
    assert not manifest['training_enabled'] and not manifest['target_scoring_enabled'] and not manifest['training_gate_pass']
    assert execution['manifest'] == q.identity(f.root / 'manifest.json')
    assert execution['acceptance'] == q.identity(f.root / 'portable_validation/acceptance.json')
    assert (f.review / 'EXECUTION_COMPLETION.md').is_file() and f.signals == []


@pytest.mark.parametrize('mode', ['first_failure', 'kill_escalation'])
def test_first_failure_stops_only_owned_other_group(fixture, mode):
    f = fixture
    with pytest.raises(ValueError, match='Raw phase failed'):execute_fixture(f, mode)
    execution, build = states(f)
    assert execution['status'] == build['status'] == 'failed'
    assert execution['stages']['protein']['status'] == 'failed'
    assert execution['stages']['corpus_initial']['status'] == 'stopped_after_queue_failure'
    assert all(pid == f.processes['corpus_initial'].pid for pid, _ in f.signals)
    assert [sig for _, sig in f.signals] == ([signal.SIGTERM, signal.SIGKILL] if mode == 'kill_escalation' else [signal.SIGTERM])
    assert 'corpus_dependent' not in f.processes and 'independent_acceptance' not in f.processes


def test_second_popen_exception_stops_started_first_group(fixture):
    f = fixture
    with pytest.raises(OSError, match='Popen'):execute_fixture(f, 'second_popen_exception')
    execution, build = states(f)
    assert execution['status'] == build['status'] == 'failed'
    assert execution['stages']['corpus_initial']['status'] == 'failed'
    assert f.signals == [(f.processes['protein'].pid, signal.SIGTERM)]


@pytest.mark.parametrize('mode', ['code_mutation', 'input_control_mutation', 'invalid_phase_state'])
def test_integrity_or_phase_identity_failure_prevents_successors(fixture, mode):
    f = fixture
    with pytest.raises(ValueError):execute_fixture(f, mode)
    execution, build = states(f)
    assert execution['status'] == build['status'] == 'failed'
    assert execution['stages']['protein']['status'] == 'failed'
    assert 'corpus_dependent' not in f.processes
    assert not (f.review / 'EXECUTION_COMPLETION.md').exists()


@pytest.mark.parametrize('mode', ['audit_nonzero', 'audit_zero_rejected', 'audit_wrong_flag'])
def test_audit_failure_preserves_completed_build_and_fails_global(fixture, mode):
    f = fixture
    with pytest.raises(ValueError):execute_fixture(f, mode)
    execution, build = states(f)
    assert execution['status'] == 'failed' and build['status'] == 'completed'
    assert all(s['status'] == 'completed' for s in build['phases'].values())
    assert execution['stages']['independent_acceptance']['status'] == 'failed'
    assert not (f.review / 'EXECUTION_COMPLETION.md').exists()
    assert f.signals == []
