"""Offline corpus phases with a separate, hash-bound executed source tree.

Adapted from portable_data/corpus_phase.py, source SHA
f2a1cc1ea0bca378d559ed9b91c75c69e35ad2f8ab79bf8860d627cb1d28d017.
The seven initial and three dependent scientific stages are unchanged.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time

RELEASE_ID = '2026-10-01-portable-v2'
SOURCE_HELPER_SHA = '19d00191235be486a8b7e4c969a853ec89ba423a2ff036decf3be947e0c3aad6'
DESCRIPTOR_FIELDS = {'source', 'source_sha256', 'source_bytes', 'snapshot', 'sha256', 'bytes', 'transform'}
PROTEIN_STAGE_NAMES = ['normalize', 'cluster', 'search', 'graph_split', 'remote_prepare',
                       'remote_exclusion', 'protein_pairs', 'remote_pairs', 'cross_dataset_audit']

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
    with temporary.open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True)
        stream.write('\n')
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

def verify_initial(root, bootstrap_sha, builder_manifest_sha, release_id, builder_root):
    marker = json.loads(member(root, '.corpus_initial_completed.json').read_text())
    require(marker['status'] == 'completed' and marker['bootstrap_sha256'] == bootstrap_sha and
            marker['builder_manifest_sha256'] == builder_manifest_sha and marker['release_id'] == release_id,
            'Initial corpus marker is not bound to this bootstrap')
    path = checked(marker['status_file'])
    require(sha(path) == marker['status_sha256'], 'Initial corpus status changed')
    status = json.loads(path.read_text())
    expected_names = [s['name'] for s in plan(root, 'initial', Path('unused'), builder_root)]
    require(status['status'] == 'completed' and status['phase'] == 'initial' and
            status['bootstrap_sha256'] == bootstrap_sha and
            status['builder_manifest_sha256'] == builder_manifest_sha and status['release_id'] == release_id and
            checked(status['builder_root']) == builder_root and
            checked(status['release_root']) == root and
            [s['name'] for s in status['stages']] == expected_names,
            'Initial corpus phase identity or stage sequence differs')
    for stage in status['stages']:
        require(stage['status'] == 'completed' and stage['exit_code'] == 0 and
                stage['output_sha256'], 'Initial corpus phase was not fully completed')
        for name, expected in stage['output_sha256'].items():
            require(sha(member(root, name)) == expected, 'Initial corpus output changed: ' + name)
    return marker

def plan(root, phase, smallweb, builder_root):
    base = builder_root / 'bio2nl/data/rebuild_v2'
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


def identity(path):
    path = checked(path)
    require(path.is_file(), 'Missing regular file: ' + str(path))
    return {'sha256': sha(path), 'bytes': path.stat().st_size}


def load_source_helper():
    path = checked(Path(__file__).with_name('corpus_source.py'))
    require(sha(path) == SOURCE_HELPER_SHA, 'Corpus patch helper changed')
    spec = importlib.util.spec_from_file_location('pinned_corpus_source_v2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_bootstrap(root, bootstrap_path, bootstrap_sha):
    require(sha(bootstrap_path) == bootstrap_sha, 'Bootstrap identity changed')
    bootstrap = json.loads(bootstrap_path.read_text())
    require(bootstrap['status'] == 'raw_assets_seeded_not_a_data_acceptance' and
            checked(bootstrap['output_root']) == root, 'Bootstrap identity differs')
    kinds = [item['kind'] for item in bootstrap['files'].values()]
    require(len(kinds) == 110 and kinds.count('raw_or_external_tool_asset') == 77 and
            kinds.count('historical_builder_source') == 28 and
            kinds.count('historical_source_provenance') == 5, 'Bootstrap member coverage differs')
    for name, item in bootstrap['files'].items():
        require(identity(member(root, name)) == {k: item[k] for k in ('sha256', 'bytes')},
                'Immutable bootstrap member changed: ' + name)
    return bootstrap


def verify_builder_manifest(root, builder_root, manifest_path, manifest_sha, bootstrap, bootstrap_sha, release_id):
    require(sha(manifest_path) == manifest_sha, 'Builder manifest changed')
    m = json.loads(manifest_path.read_text())
    require(m['schema_version'] == 1 and m['status'] == 'execution_code_overlay_not_data_acceptance' and
            m['release_id'] == release_id and m['bootstrap_sha256'] == bootstrap_sha and
            checked(m['release_root']) == root and checked(m['builder_root']) == builder_root,
            'Builder manifest scope/root binding differs')
    require(m['training_enabled'] is False and m['target_scoring_enabled'] is False, 'Unexpected execution scope')
    expected = {name for name, item in bootstrap['files'].items() if item['kind'] == 'historical_builder_source'}
    require(len(expected) == 28 and set(m['files']) == expected and all(n.startswith('code/') for n in expected),
            'Executed code must close the 28 original code members')
    actual = {'code/' + p.relative_to(builder_root).as_posix()
              for p in builder_root.rglob('*') if checked(p, root=builder_root).is_file()}
    require(actual == expected, 'Unexpected/missing execution tree member')
    helper = load_source_helper()
    for name, record in m['files'].items():
        require(set(record) == DESCRIPTOR_FIELDS, 'Unexpected executed source descriptor fields')
        source, snapshot = member(root, name), member(builder_root, name[5:])
        require(record['source'] == str(source) and record['snapshot'] == str(snapshot), 'Executed source path differs')
        original = identity(source)
        require(original == {k: bootstrap['files'][name][k] for k in ('sha256', 'bytes')} and
                original == {'sha256': record['source_sha256'], 'bytes': record['source_bytes']},
                'Original executed-source binding differs')
        require(identity(snapshot) == {k: record[k] for k in ('sha256', 'bytes')}, 'Executed source identity changed')
        if name.startswith('code/bio2nl/'):
            if name in helper.PATCHES:
                desired, provenance = helper.patch_corpus_source(name, source.read_bytes(), release_id)
                require(record['transform'] == provenance['transform'] and snapshot.read_bytes() == desired,
                        'Executed corpus patch differs from fixed transformation')
            else:
                require(record['transform'] == 'identity' and identity(snapshot) == original,
                        'Unexpected corpus algorithm/source change')
    return m


def verify_protein_status(path, root, bootstrap_sha, manifest_sha, release_id):
    path = checked(path)
    before = identity(path)
    state = json.loads(path.read_text())
    require(state['status'] == 'completed' and [s['name'] for s in state['stages']] == PROTEIN_STAGE_NAMES and
            all(s['status'] == 'completed' and s['returncode'] == 0 for s in state['stages']),
            'Raw protein phase incomplete')
    require(state['release_id'] == release_id and checked(state['release_root']) == root and
            state['bootstrap_sha256'] == bootstrap_sha and state['builder_manifest_sha256'] == manifest_sha,
            'Protein phase provenance differs')
    require(checked(state['builder_root']) == root / 'execution_code/biopaws' and
            sha(checked(state['builder_manifest_file'])) == manifest_sha, 'Protein executed source binding differs')
    require(state['training_enabled'] is False and state['target_scoring_enabled'] is False and
            state['complete_release_gate_passed'] is False, 'Protein phase scope differs')
    # Some files are intentionally rewritten by later protein stages. Verify the
    # final writer, not an earlier hash recorded before those legitimate writes.
    final = {}
    for stage in state['stages']:
        require(stage['outputs'], 'Protein stage omitted output bindings')
        final.update(stage['outputs'])
    for name, expected in final.items():
        require(identity(member(root, name)) == expected, 'Protein final output changed: ' + name)
    require(identity(path) == before, 'Protein status changed while checking')
    return {'path': str(path), **before, 'last_writer_outputs': final}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--phase', choices=['initial', 'dependent'], required=True)
    parser.add_argument('--log-root', type=Path, required=True)
    parser.add_argument('--bootstrap-sha256', required=True)
    parser.add_argument('--smallweb-script', type=Path, required=True)
    parser.add_argument('--smallweb-sha256', required=True)
    parser.add_argument('--builder-root', type=Path, required=True,
                        help='Complete execution_code root containing bio2nl and biopaws')
    parser.add_argument('--builder-manifest', type=Path, required=True)
    parser.add_argument('--builder-manifest-sha256', required=True)
    parser.add_argument('--release-id', required=True)
    parser.add_argument('--protein-status', type=Path)
    args = parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Corpus construction must be CPU-only')
    root, logs, smallweb = (checked(p) for p in (args.release_root, args.log_root, args.smallweb_script))
    builder_root, builder_manifest = checked(args.builder_root), checked(args.builder_manifest)
    require(args.release_id == RELEASE_ID, 'Unexpected new release identity')
    require(builder_root == member(root, 'execution_code'), 'Execution tree must be separate release/execution_code')
    require(not logs.exists(), 'Refuse existing phase log directory')
    require(not member(root, '.corpus_' + args.phase + '_completed.json').exists(),
            'Refuse existing completion marker')
    require(not logs.is_relative_to(root) or logs.is_relative_to(root / 'portable_validation'),
            'Logs inside reconstruction must use portable_validation')
    bootstrap_path = member(root, 'portable_validation/bootstrap.json')
    bootstrap = verify_bootstrap(root, bootstrap_path, args.bootstrap_sha256)
    overlay = verify_builder_manifest(root, builder_root, builder_manifest, args.builder_manifest_sha256,
                                     bootstrap, args.bootstrap_sha256, args.release_id)
    require(sha(smallweb) == args.smallweb_sha256, 'Local SmallWeb adapter changed')
    protein_binding = None
    if args.phase == 'dependent':
        require(args.protein_status is not None, 'Completed raw protein phase status is required')
        protein_binding = verify_protein_status(args.protein_status, root, args.bootstrap_sha256,
                                               args.builder_manifest_sha256, args.release_id)
        verify_initial(root, args.bootstrap_sha256, args.builder_manifest_sha256, args.release_id, builder_root)
    stages = plan(root, args.phase, smallweb, builder_root)
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
    env['PYTHONHASHSEED'] = '0'
    state = {'status': 'running', 'phase': args.phase, 'started_at_utc': now(), 'pid': os.getpid(), 'release_root': str(root),
             'release_id': args.release_id, 'builder_root': str(builder_root),
             'builder_manifest_file': str(builder_manifest), 'builder_manifest_sha256': args.builder_manifest_sha256,
             'bootstrap_sha256': args.bootstrap_sha256, 'smallweb_sha256': args.smallweb_sha256,
             'runner_sha256': sha(Path(__file__)), 'corpus_source_sha256': SOURCE_HELPER_SHA,
             'protein_dependency': protein_binding, 'training_enabled': False, 'target_scoring_enabled': False,
             'complete_release_gate_passed': False, 'stages': []}
    atomic(logs / 'status.json', state)
    try:
        for stage in stages:
            verify_builder_manifest(root, builder_root, builder_manifest, args.builder_manifest_sha256,
                                    bootstrap, args.bootstrap_sha256, args.release_id)
            require(sha(bootstrap_path) == args.bootstrap_sha256, 'Bootstrap marker changed')
            require(sha(smallweb) == args.smallweb_sha256 and sha(Path(__file__)) == state['runner_sha256'],
                    'Corpus execution support code changed')
            code = checked(stage['script'])
            entry = None if code == smallweb else overlay['files']['code/' + code.relative_to(builder_root).as_posix()]
            expected = args.smallweb_sha256 if entry is None else entry['sha256']
            require(sha(code) == expected, 'Builder code changed')
            item = {**stage, 'code_sha256': expected, 'source_descriptor': entry,
                    'status': 'running', 'started_at_utc': now()}
            state['stages'].append(item)
            atomic(logs / 'status.json', state)
            start = time.monotonic()
            prior_tokenization = (sha(member(root, 'configs/tokenization.json'))
                                  if stage['name'] == 'encode_all_conditions' else None)
            with checked(logs / (stage['name'] + '.log')).open('x') as log:
                process = subprocess.Popen(stage['command'], cwd=logs, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
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
            verify_builder_manifest(root, builder_root, builder_manifest, args.builder_manifest_sha256,
                                    bootstrap, args.bootstrap_sha256, args.release_id)
            item['status'] = 'completed'
            atomic(logs / 'status.json', state)
        verify_bootstrap(root, bootstrap_path, args.bootstrap_sha256)
        require(sha(smallweb) == args.smallweb_sha256 and sha(Path(__file__)) == state['runner_sha256'],
                'Corpus execution support changed during phase')
        if protein_binding is not None:
            require(verify_protein_status(args.protein_status, root, args.bootstrap_sha256,
                                         args.builder_manifest_sha256, args.release_id) == protein_binding,
                    'Protein dependency changed during phase')
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
                    'builder_manifest_sha256': args.builder_manifest_sha256, 'release_id': args.release_id,
                    'completed_at_utc': now()})

if __name__ == '__main__':
    main()
