"""Fault checks for real child execution and fail-closed queue bookkeeping."""
import importlib.util
import json
from pathlib import Path
import sys
import pytest

SPEC = importlib.util.spec_from_file_location('new_data_queue_tests', Path(__file__).with_name('run_queue.py'))
q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(q)


def setup_plan(tmp_path, first_action='pass', gpu=False):
    control = tmp_path / 'control.txt'
    control.write_text('fixed')
    jobs = []
    for index in range(2):
        artifact = tmp_path / f'result{index}.json'
        script = tmp_path / f'worker{index}.py'
        action = first_action if index == 0 else 'pass'
        extra = 'raise SystemExit(3)' if action == 'exit' else ''
        if action == 'mutate':
            extra = f'Path({str(control)!r}).write_text("changed")'
        status = 'rejected' if action == 'reject' else 'accepted'
        script.write_text(f'from pathlib import Path\nimport json\n{extra}\nPath({str(artifact)!r}).write_text(json.dumps({{"status":{status!r},"target_scoring_enabled":False}}))\n')
        jobs.append({'id': f'job{index}', 'command': [sys.executable, '-B', str(script)], 'cwd': str(tmp_path / f'cwd{index}'), 'log': str(tmp_path / f'job{index}.log'), 'gpu': gpu, 'result_file': str(artifact), 'result_status': 'accepted', 'required_false_flags': ['target_scoring_enabled']})
    pins = {str(Path(q.__file__).resolve()): q.identity(q.__file__), str(control): q.identity(control)}
    pins.update({job['command'][2]: q.identity(job['command'][2]) for job in jobs})
    plan = {'schema_version': 1, 'scope': 'source_only_new_data_full_training', 'review_root': str(tmp_path), 'status_file': str(tmp_path / 'status.json'), 'pins': pins, 'jobs': jobs, 'required_free_disk_gib': 0, 'full_budget_training_enabled': True, 'target_scoring_enabled': False, 'source_test_scoring_enabled': False, 'gpu_availability_policy': {'max_samples': 1, 'interval_seconds': 0, 'utilization_percent_at_most': 10, 'memory_used_mib_below': 500}}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    return path, q.identity(path)['sha256']


def test_serial_real_children_and_success(tmp_path):
    path, digest = setup_plan(tmp_path)
    result = q.execute(path, digest)
    assert result['status'] == 'completed'
    assert [x['status'] for x in result['jobs']] == ['completed', 'completed']
    assert result['jobs'][0]['completed_at_utc'] <= result['jobs'][1]['started_at_utc']
    assert all(x['result']['sha256'] for x in result['jobs'])


@pytest.mark.parametrize('action', ['exit', 'mutate', 'reject'])
def test_failure_stops_before_second_worker(tmp_path, action):
    path, digest = setup_plan(tmp_path, action)
    with pytest.raises(ValueError):
        q.execute(path, digest)
    state = json.loads((tmp_path / 'status.json').read_text())
    assert state['status'] == state['jobs'][0]['status'] == 'failed'
    assert not (tmp_path / 'cwd1').exists()
    assert not (tmp_path / 'result1.json').exists()


def test_spawn_failure_is_recorded(tmp_path, monkeypatch):
    path, digest = setup_plan(tmp_path)
    def fail(*args, **kwargs):
        raise OSError('synthetic spawn failure')
    monkeypatch.setattr(q.subprocess, 'Popen', fail)
    with pytest.raises(OSError):
        q.execute(path, digest)
    state = json.loads((tmp_path / 'status.json').read_text())
    assert state['status'] == state['jobs'][0]['status'] == 'failed'
    assert 'synthetic spawn failure' in state['error']


def test_gpu_rejection_prevents_launch(tmp_path, monkeypatch):
    path, digest = setup_plan(tmp_path, gpu=True)
    def occupied(record, status_path, state, policy):
        record['gpu_prelaunch_samples'] = [{'available': False}]
        raise RuntimeError('occupied')
    monkeypatch.setattr(q, 'await_gpu', occupied)
    with pytest.raises(RuntimeError):
        q.execute(path, digest)
    state = json.loads((tmp_path / 'status.json').read_text())
    assert state['jobs'][0]['gpu_prelaunch_samples'] == [{'available': False}]
    assert not (tmp_path / 'cwd0').exists()


def test_changed_plan_refused_before_status(tmp_path):
    path, digest = setup_plan(tmp_path)
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='Wrong plan SHA'):
        q.execute(path, digest)
    assert not (tmp_path / 'status.json').exists()


def test_code_change_during_gpu_wait_prevents_launch(tmp_path, monkeypatch):
    path, digest = setup_plan(tmp_path, gpu=True)
    def changed(record, status_path, state, policy):
        (tmp_path / 'control.txt').write_text('changed during availability wait')
    monkeypatch.setattr(q, 'await_gpu', changed)
    with pytest.raises(ValueError, match='Pinned code/control changed'):
        q.execute(path, digest)
    assert not (tmp_path / 'cwd0').exists()
    assert not (tmp_path / 'result0.json').exists()


def test_refuses_repeat_or_existing_result(tmp_path):
    path, digest = setup_plan(tmp_path)
    (tmp_path / 'result0.json').write_text('{}')
    with pytest.raises(ValueError, match='result already exists'):
        q.execute(path, digest)
    assert not (tmp_path / 'status.json').exists()


def test_workspace_lock_rejects_a_second_queue(tmp_path):
    path, digest = setup_plan(tmp_path)
    with (tmp_path / '.queue.lock').open('a+') as lock:
        q.fcntl.flock(lock.fileno(), q.fcntl.LOCK_EX | q.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='owns the workspace'):
            q.execute(path, digest)
        assert not (tmp_path / 'status.json').exists()


def test_completed_acceptance_cannot_change_during_gpu_wait(tmp_path, monkeypatch):
    path, digest = setup_plan(tmp_path, gpu=True)
    calls = []
    def available(record, status_path, state, policy):
        calls.append(record['id'])
        if len(calls) == 2:
            (tmp_path / 'result0.json').write_text('{}')
    monkeypatch.setattr(q, 'await_gpu', available)
    with pytest.raises(ValueError, match='acceptance gate changed'):
        q.execute(path, digest)
    assert not (tmp_path / 'cwd1').exists()
    assert json.loads((tmp_path / 'status.json').read_text())['status'] == 'failed'


def test_disk_reserve_prevents_worker_launch(tmp_path, monkeypatch):
    path, digest = setup_plan(tmp_path)
    plan = json.loads(path.read_text())
    plan['jobs'][0]['additional_peak_disk_gib'] = 2
    path.write_text(json.dumps(plan))
    digest = q.identity(path)['sha256']
    class Disk:
        free = 2**30
    monkeypatch.setattr(q.shutil, 'disk_usage', lambda _: Disk())
    with pytest.raises(ValueError, match='disk reserve'):
        q.execute(path, digest)
    assert not (tmp_path / 'cwd0').exists()


def test_heldout_scoring_scope_rejected(tmp_path):
    path, digest = setup_plan(tmp_path)
    plan = json.loads(path.read_text())
    plan['target_scoring_enabled'] = True
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='authorization differs'):
        q.execute(path, q.identity(path)['sha256'])
    assert not (tmp_path / 'status.json').exists()


def test_disk_consumption_during_gpu_wait_prevents_launch(tmp_path, monkeypatch):
    path, digest = setup_plan(tmp_path, gpu=True)
    plan = json.loads(path.read_text())
    plan['required_free_disk_gib'] = 8
    plan['jobs'][0]['additional_peak_disk_gib'] = 2
    path.write_text(json.dumps(plan))
    calls = []
    class Disk:
        def __init__(self, free):
            self.free = free
    def usage(_):
        calls.append(1)
        return Disk((12 if len(calls) == 1 else 9) * 2**30)
    monkeypatch.setattr(q.shutil, 'disk_usage', usage)
    monkeypatch.setattr(q, 'await_gpu', lambda *args: None)
    with pytest.raises(ValueError, match='Disk reserve changed'):
        q.execute(path, q.identity(path)['sha256'])
    assert not (tmp_path / 'cwd0').exists()
