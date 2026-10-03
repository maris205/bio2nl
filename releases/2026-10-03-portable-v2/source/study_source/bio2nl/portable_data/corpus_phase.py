"""CPU-only, offline corpus phases using explicit roots and pinned builder bytes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            value.update(block)
    return value.hexdigest()

def require(ok, message):
    if not ok:
        raise ValueError(message)

def now():
    return datetime.now(timezone.utc).isoformat()

def atomic(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)

def checked(path, *, root=None):
    """Reject lexical symlinks before resolving an explicit local path."""
    path = Path(path).absolute()
    for part in (path, *path.parents):
        require(not part.is_symlink(), 'Symlink path is forbidden: ' + str(path))
    resolved = path.resolve()
    if root is not None:
        require(resolved.is_relative_to(root), 'Path leaves reconstruction root')
    return resolved

def member(root, relative):
    name = PurePosixPath(relative)
    require(relative and not name.is_absolute() and '..' not in name.parts and
            '\\' not in relative and '\x00' not in relative and str(name) == relative,
            'Unsafe reconstruction-relative path')
    return checked(root / relative, root=root)

def tree_files(root, relative):
    directory = member(root, relative)
    require(directory.is_dir(), 'Expected output directory: ' + relative)
    result = []
    for path in sorted(directory.rglob('*')):
        path = checked(path, root=root)
        if path.is_file():
            result.append(path.relative_to(root).as_posix())
    return result

def require_new_outputs(root, stages):
    for stage in stages:
        for relative in stage['outputs']:
            require(not member(root, relative).exists(), 'Refuse existing stage output: ' + relative)
        for relative in stage['output_dirs']:
            require(not member(root, relative).exists(), 'Refuse existing stage output directory: ' + relative)

def verify_initial(root, bootstrap_sha):
    marker = json.loads(member(root, '.corpus_initial_completed.json').read_text())
    require(marker['status'] == 'completed' and marker['bootstrap_sha256'] == bootstrap_sha,
            'Initial corpus marker is not bound to this bootstrap')
    path = checked(marker['status_file'])
    require(sha(path) == marker['status_sha256'], 'Initial corpus status changed')
    status = json.loads(path.read_text())
    expected_names = [s['name'] for s in plan(root, 'initial', Path('unused'))]
    require(status['status'] == 'completed' and status['phase'] == 'initial' and
            status['bootstrap_sha256'] == bootstrap_sha and
            checked(status['release_root']) == root and
            [s['name'] for s in status['stages']] == expected_names,
            'Initial corpus phase identity or stage sequence differs')
    for stage in status['stages']:
        require(stage['status'] == 'completed' and stage['exit_code'] == 0 and
                stage['output_sha256'], 'Initial corpus phase was not fully completed')
        for name, expected in stage['output_sha256'].items():
            require(sha(member(root, name)) == expected, 'Initial corpus output changed: ' + name)
    return marker

def plan(root, phase, smallweb):
    base = root / 'code/bio2nl/data/rebuild_v2'
    def stage(name, code, arguments, outputs, output_dirs=()):
        return {'name': name, 'script': str(code), 'command': [sys.executable, '-B', str(code), *arguments],
                'outputs': outputs, 'output_dirs': list(output_dirs)}
    if phase == 'initial':
        return [
            stage('nlp_and_dyck', base / 'nlp_synthetic/build.py', ['--release-root', str(root), '--mode', 'all', '--offline'],
                  ['validation/nlp_acceptance_report.json', 'validation/nlp_clean_acceptance_report.json', 'validation/synthetic_dyck_acceptance_report.json',
                   'validation/nlp_official_overlap_details.json', 'validation/nlp_clean_label_conflicts.json', 'validation/nlp_clean_exclusions.jsonl',
                   'metadata/nlp_sources_manifest.json', 'metadata/nlp_clean_manifest.json', 'metadata/synthetic_dyck_manifest.json',
                   'metadata/nlp_synthetic_README.md', 'metadata/nlp_sources_lock.json'] +
                  [f'metadata/nlp_source_card_{task}.md' for task in ('pawsx_en', 'cola', 'rte')],
                  ['data/nlp', 'data/nlp_clean', 'data/synthetic']),
            stage('nlp_independent_verify', base / 'nlp_synthetic/verify.py', ['--release-root', str(root)], ['validation/nlp_synthetic_independent_verification.json']),
            stage('longrange', base / 'nlp_synthetic/build_longrange.py', ['--root', str(root)], ['metadata/synthetic_longrange_manifest.json'] +
                  [f'validation/synthetic_longrange_d{d}_acceptance.json' for d in (8, 16, 24)],
                  [f'data/synthetic/longrange_d{d}' for d in (8, 16, 24)]),
            stage('english', base / 'build_corpora.py', ['--root', str(root), '--condition', 'english'], ['metadata/corpora_english.json', 'validation/corpora_english.json'], ['data/corpora/english']),
            stage('dna', base / 'build_corpora.py', ['--root', str(root), '--condition', 'dna'], ['metadata/corpora_dna.json', 'validation/corpora_dna.json'], ['data/corpora/dna']),
            stage('configs', base / 'write_experiment_configs.py', ['--root', str(root)], ['configs/training_matrix_v2.json', 'configs/smallweb_gpt2_v2.json']),
            stage('smallweb_local', smallweb, ['--root', str(root)], ['configs/gpt2_smallweb_data.json', 'tokenized/gpt2_smallweb/train.bin', 'tokenized/gpt2_smallweb/validation.bin', 'tokenized/gpt2_smallweb/test.bin'], ['tokenized/gpt2_smallweb']),
        ]
    return [
        stage('protein_and_controls', base / 'build_corpora.py', ['--root', str(root), '--condition', 'protein'],
              [f'{folder}/corpora_{c}.json' for folder in ('metadata', 'validation') for c in ('protein', 'shuffled', 'randomaa')],
              ['data/corpora/' + c for c in ('protein', 'shuffled', 'randomaa')]),
        stage('shared_tokenizers', base / 'tokenize_corpora.py', ['--root', str(root), '--mode', 'build-tokenizers'], ['tokenizers/pure_byte/tokenizer.json', 'tokenizers/mixed_bpe/tokenizer.json', 'configs/tokenization.json'], ['tokenizers/pure_byte', 'tokenizers/mixed_bpe']),
        stage('encode_all_conditions', base / 'tokenize_corpora.py', ['--root', str(root), '--mode', 'encode'], ['configs/tokenization.json'] +
              [f'tokenized/{family}/{condition}/{split}.bin' for family in ('pure_byte', 'mixed_bpe') for condition in ('english', 'protein', 'shuffled', 'randomaa', 'dna') for split in ('train', 'validation', 'test')], ['tokenized/pure_byte', 'tokenized/mixed_bpe']),
    ]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--phase', choices=['initial', 'dependent'], required=True)
    parser.add_argument('--log-root', type=Path, required=True)
    parser.add_argument('--bootstrap-sha256', required=True)
    parser.add_argument('--smallweb-script', type=Path, required=True)
    parser.add_argument('--smallweb-sha256', required=True)
    parser.add_argument('--protein-status', type=Path)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Corpus construction must be CPU-only')
    root, logs, smallweb = (checked(p) for p in (args.release_root, args.log_root, args.smallweb_script))
    require(not logs.exists(), 'Refuse existing phase log directory')
    require(not member(root, '.corpus_' + args.phase + '_completed.json').exists(),
            'Refuse existing completion marker')
    require(not logs.is_relative_to(root) or logs.is_relative_to(root / 'portable_validation'),
            'Logs inside reconstruction must use portable_validation')
    bootstrap_path = member(root, 'portable_validation/bootstrap.json')
    require(sha(bootstrap_path) == args.bootstrap_sha256, 'Bootstrap identity changed')
    bootstrap = json.loads(bootstrap_path.read_text())
    require(bootstrap['status'] == 'raw_assets_seeded_not_a_data_acceptance' and
            checked(bootstrap['output_root']) == root, 'Bootstrap status or output root differs')
    require(sha(smallweb) == args.smallweb_sha256, 'Local SmallWeb adapter changed')
    if args.phase == 'dependent':
        require(args.protein_status is not None, 'Completed raw protein phase status is required')
        status = json.loads(checked(args.protein_status).read_text())
        require(status['status'] == 'completed' and len(status['stages']) == 9 and all(x['status'] == 'completed' and x['returncode'] == 0 for x in status['stages']), 'Raw protein phase incomplete')
        require(checked(status['release_root']) == root, 'Protein phase used a different output root')
        verify_initial(root, args.bootstrap_sha256)
    stages = plan(root, args.phase, smallweb)
    for name, item in bootstrap['files'].items():
        if name.startswith(('code/', 'raw/', 'metadata/nlp_raw/', 'tokenizers/gpt2_pinned/')):
            require(sha(member(root, name)) == item['sha256'], 'Immutable bootstrap member changed: ' + name)
    require_new_outputs(root, stages)
    require(shutil.disk_usage(root).free >= 8 * (1 << 30), 'Need at least 8 GiB free reserve')
    with member(root, '.corpus_' + args.phase + '_started.json').open('x') as file:
        json.dump({'pid': os.getpid(), 'started_at_utc': now()}, file)
    logs.mkdir()
    for name in ('hf', 'tmp'):
        (logs / name).mkdir()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HOME=str(logs / 'hf'), TMPDIR=str(logs / 'tmp'), PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONSTARTUP', None)
    state = {'status': 'running', 'phase': args.phase, 'started_at_utc': now(), 'pid': os.getpid(), 'release_root': str(root),
             'bootstrap_sha256': args.bootstrap_sha256, 'smallweb_sha256': args.smallweb_sha256, 'stages': []}
    atomic(logs / 'status.json', state)
    try:
        for stage in stages:
            code = checked(stage['script'])
            expected = args.smallweb_sha256 if code == smallweb else bootstrap['files'][code.relative_to(root).as_posix()]['sha256']
            require(sha(code) == expected, 'Builder code changed')
            item = {**stage, 'code_sha256': expected, 'status': 'running', 'started_at_utc': now()}
            state['stages'].append(item)
            atomic(logs / 'status.json', state)
            start = time.monotonic()
            prior_tokenization = (sha(member(root, 'configs/tokenization.json'))
                                  if stage['name'] == 'encode_all_conditions' else None)
            with checked(logs / (stage['name'] + '.log')).open('x') as log:
                process = subprocess.Popen(stage['command'], cwd=logs, env=env, stdout=log, stderr=subprocess.STDOUT)
                item['pid'] = process.pid
                atomic(logs / 'status.json', state)
                result = process.wait()
            item.update(exit_code=result, wall_seconds=time.monotonic() - start, completed_at_utc=now())
            require(result == 0, 'Stage failed: ' + stage['name'])
            outputs = set(stage['outputs'])
            for directory in stage['output_dirs']:
                outputs.update(tree_files(root, directory))
            item['output_sha256'] = {name: sha(member(root, name)) for name in sorted(outputs)}
            if prior_tokenization is not None:
                require(item['output_sha256']['configs/tokenization.json'] == prior_tokenization,
                        'Encoding changed the tokenizer specification')
            item['status'] = 'completed'
            atomic(logs / 'status.json', state)
        state['status'] = 'completed'
    except BaseException as exc:
        state.update(status='failed', error=repr(exc))
        if state['stages'] and state['stages'][-1]['status'] == 'running':
            state['stages'][-1]['status'] = 'failed'
        raise
    finally:
        state['completed_at_utc'] = now()
        atomic(logs / 'status.json', state)
    # The completion marker binds the final status, after its last write.
    marker = member(root, '.corpus_' + args.phase + '_completed.json')
    require(not marker.exists(), 'Refuse existing completion marker')
    atomic(marker, {'status': 'completed', 'status_file': str(logs / 'status.json'),
                    'status_sha256': sha(logs / 'status.json'), 'bootstrap_sha256': args.bootstrap_sha256,
                    'completed_at_utc': now()})

if __name__ == '__main__':
    main()
