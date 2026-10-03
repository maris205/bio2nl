"""Execute two independent CPU raw phases, dependent corpora, then acceptance."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'Missing or symlinked control file: ' + str(path))
    with path.open('rb') as stream:
        value = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'sha256': value, 'bytes': path.stat().st_size}


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temp.replace(path)


def check_pins(protocol, path, digest):
    require(identity(path)['sha256'] == digest, 'Protocol changed')
    for filename, expected in protocol['execution_code'].items():
        require(identity(filename) == expected, 'Execution code changed: ' + filename)
    for label in ('bootstrap', 'execution_code_manifest'):
        expected = protocol[label]
        require(identity(expected['file']) == {k: expected[k] for k in ('sha256', 'bytes')},
                'Control changed: ' + label)


def verify_phase(spec, root):
    path = Path(spec['status_file'])
    state = json.loads(path.read_text())
    require(state['status'] == 'completed' and Path(state['release_root']) == root,
            'Incomplete or wrong-root raw phase')
    require(len(state['stages']) == spec['stage_count'], 'Wrong phase stage count')
    require([s['name'] for s in state['stages']] == spec['stage_names'], 'Wrong stage order')
    for stage in state['stages']:
        require(stage['status'] == 'completed' and stage.get('exit_code', stage.get('returncode')) == 0,
                'Raw stage did not complete successfully')
    return identity(path)


def stop_owned(active, records):
    """Only signal process groups created by this queue and still owned/alive."""
    for name, (process, log) in active.items():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for name, (process, log) in active.items():
        try:
            code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = process.wait(timeout=5)
        log.close()
        if records[name]['status'] == 'running':
            records[name].update(status='stopped_after_queue_failure', exit_code=code,
                                 completed_at_utc=now())


def execute(protocol_path, expected_sha):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only environment required')
    require(sys.flags.optimize == 0, 'Assertions must remain enabled')
    path = Path(protocol_path).resolve()
    require(identity(path)['sha256'] == expected_sha, 'Protocol identity mismatch')
    p = json.loads(path.read_text())
    review, root = Path(p['review_root']), Path(p['release_root'])
    require(review == path.parent, 'Protocol/review root mismatch')
    require(p['release_id'] == '2026-10-01-portable-v2', 'Unexpected new release identity')
    require(all(p[k] is False for k in ('training_enabled', 'target_scoring_enabled', 'remote_publication_performed')),
            'Unexpected model/publication scope')
    require(set(p['phases']) == {'protein', 'corpus_initial', 'corpus_dependent'}, 'Unexpected raw phase grid')
    require(str(Path(__file__).resolve()) in p['execution_code'], 'Queue source not pinned')
    check_pins(p, path, expected_sha)
    require(shutil.disk_usage(root).free >= 20 * 2**30, 'Need 20GiB prelaunch planning/reserve')
    for name in ('execution_status.json', 'build_status.json', '.queue_started.json'):
        require(not (review / name).exists(), 'Refuse duplicate queue: ' + name)
    require(not (root / 'portable_validation/acceptance.json').exists() and not (root / 'manifest.json').exists(),
            'Refuse existing acceptance or release manifest')
    for spec in p['phases'].values():
        require(not Path(spec['status_file']).parent.exists(), 'Phase logs already exist')
    with (review / '.queue_started.json').open('x') as f:
        json.dump({'pid': os.getpid(), 'started_at_utc': now(), 'protocol_sha256': expected_sha}, f)
    state = {'schema_version': 1, 'status': 'running', 'pid': os.getpid(),
             'protocol_sha256': expected_sha, 'started_at_utc': now(), 'stages': {}}
    build = {'schema_version': 1, 'status': 'running', 'release_id': p['release_id'],
             'release_root': str(root), 'protocol_sha256': expected_sha,
             'started_at_utc': now(), 'phases': {}}
    write(review / 'execution_status.json', state)
    write(review / 'build_status.json', build)
    env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    env.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', PYTHONHASHSEED='0',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    active = {}
    try:
        for group in (('protein', 'corpus_initial'), ('corpus_dependent',)):
            check_pins(p, path, expected_sha)
            require(shutil.disk_usage(root).free >= 12 * 2**30, 'Insufficient phase storage reserve')
            for name in group:
                spec = p['phases'][name]
                record = {'status': 'running', 'started_at_utc': now(), 'command': spec['command'],
                          'status_file': spec['status_file'], 'stage_count': spec['stage_count']}
                state['stages'][name] = record
                build['phases'][name] = record
                cwd = review / (name + '_cwd')
                cwd.mkdir()
                log = (review / (name + '.log')).open('x')
                try:
                    process = subprocess.Popen(spec['command'], cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                record['worker_pid'] = process.pid
                active[name] = (process, log)
                write(review / 'execution_status.json', state)
                write(review / 'build_status.json', build)
            while active:
                finished = False
                for name in list(active):
                    process, log = active[name]
                    code = process.poll()
                    if code is None:
                        continue
                    finished = True
                    log.close()
                    record = state['stages'][name]
                    record.update(exit_code=code, process_ended_at_utc=now())
                    require(code == 0, 'Raw phase failed: ' + name)
                    check_pins(p, path, expected_sha)
                    result = verify_phase(p['phases'][name], root)
                    record.update(status='completed', status_sha256=result['sha256'],
                                  status_bytes=result['bytes'], completed_at_utc=now())
                    del active[name]
                    write(review / 'execution_status.json', state)
                    write(review / 'build_status.json', build)
                if active and not finished:
                    time.sleep(0.5)
        build.update(status='completed', completed_at_utc=now())
        write(review / 'build_status.json', build)
        check_pins(p, path, expected_sha)
        command = [sys.executable, '-B', p['auditor'], '--protocol', str(path),
                   '--protocol-sha256', expected_sha, '--build-status', str(review / 'build_status.json'),
                   '--output', str(root / 'portable_validation/acceptance.json')]
        record = {'status': 'running', 'command': command, 'started_at_utc': now()}
        state['stages']['independent_acceptance'] = record
        cwd = review / 'independent_acceptance_cwd'
        cwd.mkdir()
        with (review / 'independent_acceptance.log').open('x') as log:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record['worker_pid'] = process.pid
            active['independent_acceptance'] = (process, log)
            write(review / 'execution_status.json', state)
            code = process.wait()
        record.update(exit_code=code, process_ended_at_utc=now())
        require(code == 0, 'Independent new-release acceptance failed')
        check_pins(p, path, expected_sha)
        acceptance = json.loads((root / 'portable_validation/acceptance.json').read_text())
        manifest = json.loads((root / 'manifest.json').read_text())
        for value in (acceptance, manifest):
            require(value['status'] == 'accepted_new_raw_release_training_disabled' and
                    value['training_enabled'] is False and value['target_scoring_enabled'] is False,
                    'Missing scoped new-data acceptance')
        require(manifest['release_id'] == p['release_id'] and manifest['training_gate_pass'] is False,
                'Incorrect final release identity or gate')
        record.update(status='completed', completed_at_utc=now())
        active.clear()
        state.update(status='completed', acceptance=identity(root / 'portable_validation/acceptance.json'),
                     manifest=identity(root / 'manifest.json'))
        with (review / 'EXECUTION_COMPLETION.md').open('x') as f:
            f.write('# 新版本完整 raw 重建验收通过\n\n'
                    f'完成时间：{now()}。原始数据构建、语料/词表重建及独立新版本验收成功。\n\n'
                    '状态为 `accepted_new_raw_release_training_disabled`。详见执行状态、'
                    '新版 portable_validation/acceptance.json 和 manifest.json。'
                    '这批配对属于新样本版本；未启动派生训练输入构建、神经训练、目标评分或上传。\n')
    except BaseException as exc:
        for name, (process, _) in active.items():
            if process.poll() is not None and state['stages'][name]['status'] == 'running':
                state['stages'][name].update(status='failed', error=repr(exc),
                                           exit_code=process.returncode, completed_at_utc=now())
        stop_owned(active, state['stages'])
        state.update(status='failed', error=repr(exc))
        for record in state['stages'].values():
            if record['status'] == 'running':
                record.update(status='failed', error=repr(exc), completed_at_utc=now())
        if build['status'] != 'completed':
            build.update(status='failed', error=repr(exc), completed_at_utc=now())
        raise
    finally:
        state['completed_at_utc'] = now()
        write(review / 'build_status.json', build)
        write(review / 'execution_status.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    execute(args.protocol, args.protocol_sha256)
