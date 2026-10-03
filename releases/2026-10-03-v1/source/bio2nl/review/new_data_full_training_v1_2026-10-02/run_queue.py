"""Serial frozen source-training queue with exclusive workspace ownership."""
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Missing/symlinked pinned file')
    before = path.stat()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'Pinned file changed during hashing')
    return {'sha256': digest, 'bytes': after.st_size}


def atomic(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def check_pins(plan_path, digest, pins):
    require(identity(plan_path)['sha256'] == digest, 'Execution plan changed')
    for path, expected in pins.items():
        require(identity(path) == expected, 'Pinned code/control changed: ' + path)


def check_result(job):
    path = Path(job['result_file'])
    value = json.loads(path.read_text())
    require(value.get('status') == job['result_status'], 'Unexpected result status: ' + job['id'])
    for key in job.get('required_false_flags', []):
        require(value.get(key) is False, 'Forbidden result scope: ' + key)
    for key in job.get('required_true_flags', []):
        require(value.get(key) is True, 'Missing acceptance flag: ' + key)
    return {'file': str(path), **identity(path)}


def await_gpu(record, status_path, state, policy):
    samples = record.setdefault('gpu_prelaunch_samples', [])
    for attempt in range(policy['max_samples']):
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
        processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], capture_output=True, text=True, check=True)
        lines = [line for line in gpu.stdout.splitlines() if line.strip()]
        require(len(lines) == 1, 'This queue requires exactly one visible physical GPU')
        index, name, utilization, memory, total = [v.strip() for v in lines[0].split(',')]
        available = int(index) == 0 and int(utilization) <= policy['utilization_percent_at_most'] and int(memory) < policy['memory_used_mib_below'] and not processes.stdout.strip()
        samples.append({'at_utc': now(), 'index': int(index), 'name': name, 'utilization_percent': int(utilization), 'memory_used_mib': int(memory), 'memory_total_mib': int(total), 'compute_processes': processes.stdout.strip(), 'available': available})
        atomic(status_path, state)
        if available:
            return
        if attempt + 1 < policy['max_samples']:
            time.sleep(policy['interval_seconds'])
    raise RuntimeError('GPU remained occupied; stopped before launching the next worker')


def execute_locked(plan_path, digest):
    plan_path = Path(plan_path).absolute()
    require(identity(plan_path)['sha256'] == digest, 'Wrong plan SHA')
    plan = json.loads(plan_path.read_text())
    require(plan['schema_version'] == 1 and plan['scope'] == 'source_only_new_data_full_training', 'Wrong execution scope')
    require(plan['full_budget_training_enabled'] is True and plan['target_scoring_enabled'] is False and
            plan['source_test_scoring_enabled'] is False, 'Source-only full-training authorization differs')
    review = Path(plan['review_root'])
    require(plan_path.parent == review and review.is_absolute(), 'Wrong review root')
    status_path = Path(plan['status_file'])
    require(status_path.parent == review and not status_path.exists(), 'Fresh status file required')
    require(str(Path(__file__).resolve()) in plan['pins'], 'Runner source is not pinned')
    require(plan['jobs'] and len({x['id'] for x in plan['jobs']}) == len(plan['jobs']), 'Empty or duplicate job grid')
    check_pins(plan_path, digest, plan['pins'])
    for job in plan['jobs']:
        require(not Path(job['result_file']).exists(), 'Job result already exists')
        require(not Path(job['cwd']).exists() and not Path(job['log']).exists(), 'Fresh worker cwd/log required')
        require(all(Path(job[k]).is_relative_to(review) for k in ('cwd', 'log', 'result_file')), 'Worker paths escape review')
        require(isinstance(job['gpu'], bool), 'GPU flag must be explicit')
    marker = status_path.with_suffix('.started.json')
    with marker.open('x') as stream:
        json.dump({'pid': os.getpid(), 'plan_sha256': digest, 'started_at_utc': now()}, stream)
    state = {'schema_version': 1, 'status': 'running', 'scope': plan['scope'], 'pid': os.getpid(), 'plan_sha256': digest, 'started_at_utc': now(), 'jobs': [], 'full_budget_training_enabled': True, 'target_scoring_enabled': False, 'source_test_scoring_enabled': False}
    atomic(status_path, state)
    active = None
    current = None
    log_handle = None
    try:
        for job in plan['jobs']:
            check_pins(plan_path, digest, plan['pins'])
            for completed in state['jobs']:
                require(completed['status'] == 'completed' and identity(completed['result']['file']) ==
                        {k: completed['result'][k] for k in ('sha256', 'bytes')}, 'A completed result or acceptance gate changed')
            free = shutil.disk_usage(review).free / 2**30
            minimum_free = plan['required_free_disk_gib'] + job.get('additional_peak_disk_gib', 0)
            require(free >= minimum_free, 'Insufficient disk reserve and next-job peak storage')
            current = {'id': job['id'], 'status': 'prelaunch', 'command': job['command'], 'gpu': job['gpu'], 'log': job['log'], 'result_file': job['result_file'], 'started_at_utc': now(), 'free_disk_gib_before': free}
            state['jobs'].append(current)
            atomic(status_path, state)
            if job['gpu']:
                await_gpu(current, status_path, state, plan['gpu_availability_policy'])
            check_pins(plan_path, digest, plan['pins'])
            for completed in state['jobs'][:-1]:
                require(identity(completed['result']['file']) == {k: completed['result'][k] for k in ('sha256', 'bytes')},
                        'A completed result or acceptance gate changed during availability wait')
            immediate_free = shutil.disk_usage(review).free / 2**30
            require(immediate_free >= minimum_free, 'Disk reserve changed during availability wait')
            current['free_disk_gib_immediately_before_launch'] = immediate_free
            Path(job['cwd']).mkdir()
            env = os.environ.copy()
            env.update({'CUDA_VISIBLE_DEVICES': '0' if job['gpu'] else '', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled'})
            log_handle = Path(job['log']).open('xb')
            active = subprocess.Popen(job['command'], cwd=job['cwd'], env=env, stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
            current.update(status='running', worker_pid=active.pid)
            atomic(status_path, state)
            exit_code = active.wait()
            current.update(exit_code=exit_code, process_ended_at_utc=now())
            log_handle.close()
            log_handle = None
            active = None
            current['log_identity'] = identity(job['log'])
            require(exit_code == 0, 'Worker failed: ' + job['id'])
            check_pins(plan_path, digest, plan['pins'])
            for completed in state['jobs'][:-1]:
                require(identity(completed['result']['file']) == {k: completed['result'][k] for k in ('sha256', 'bytes')},
                        'A completed result or acceptance gate changed during execution')
            current['result'] = check_result(job)
            current.update(status='completed', completed_at_utc=now())
            atomic(status_path, state)
        state.update(status='completed', completed_at_utc=now())
        atomic(status_path, state)
        return state
    except BaseException as error:
        if active is not None and active.poll() is None:
            try:
                os.killpg(active.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                active.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait(timeout=5)
        if log_handle is not None:
            log_handle.close()
        if current is not None and current['status'] != 'completed':
            current.update(status='failed', error=f'{type(error).__name__}: {error}', completed_at_utc=now())
        state.update(status='failed', error=f'{type(error).__name__}: {error}', completed_at_utc=now())
        atomic(status_path, state)
        raise


def execute(plan_path, digest):
    plan_path = Path(plan_path).absolute()
    require(identity(plan_path)['sha256'] == digest, 'Wrong plan SHA')
    plan = json.loads(plan_path.read_text())
    review = Path(plan['review_root'])
    lock_path = Path(plan.get('workspace_lock', str(review / '.queue.lock'))).absolute()
    require(not any(p.is_symlink() for p in (lock_path, *lock_path.parents)), 'Symlinked queue lock forbidden')
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another queue owns the workspace training lock') from error
        try:
            return execute_locked(plan_path, digest)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    def terminate(signum, frame):
        raise SystemExit('Queue terminated by signal ' + str(signum))
    signal.signal(signal.SIGTERM, terminate)
    result = execute(args.plan, args.plan_sha256)
    print(json.dumps({'status': result['status'], 'jobs': len(result['jobs'])}), flush=True)


if __name__ == '__main__':
    main()
