"""Verify an explicit public-source allowlist and create a deterministic archive.

No acquisition, model inference, training or network operations are performed.
The manifest deliberately excludes its own hash; the archive hash covers it.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile

GROUPS = {
    'resources': (
        'fetch.py', 'test_fetch.py', 'english_recipe.json',
        'input_bundle_manifest.json', 'resources.json', 'rebuild_target.py',
        'qqp_membership.json', 'ATTRIBUTION.md', 'vendor/common.py',
        'vendor/fetch_corpora.py', 'vendor/build_corpora.py'),
    'prepare': (
        'README.md', 'historical_qualification_projection.json',
        'portable_prepare.py', 'resource_lock.json', 'test_portable_prepare.py',
        'vendor/__init__.py', 'vendor/algorithms.py', 'vendor/common.py',
        'vendor/data_build.py'),
    'training': (
        'README.md', 'accept_training.py', 'algorithm_lock.json', 'compile.py',
        'cpu_check.py', 'design.json', 'historical_environment.json',
        'requirements-cpu-report.txt', 'requirements-gpu-train.txt',
        'requirements-source-prepare.txt', 'run_queue.py', 'runtime_data.py',
        'scientific_payload_lock.json', 'support.py', 'test_training.py',
        'ENVIRONMENT_NOTE.json'),
}
MODEL_FILES = (
    'README.md', 'inference.py', 'portable_core.py', 'model_catalog.json',
    'config.json', 'tokenizer.json', 'example_pairs.json',
    'requirements-inference.txt', 'test_inference.py', 'test_audit.json',
    'test_output.txt', 'catalog_build_audit.json', 'cpu_replay_audit.json',
    'weight_payload_metadata_audit.json', 'build_catalog.py',
    'validate_cpu_replay.py', 'upload_mapping.json', 'ENVIRONMENT_NOTE.json',
)
OLD_ARCHIVE_SHA = '35434a25ceb330def3902d7d031ea76dfe4de052ec7a85388fc1fe9fa39839af'
OLD_MANIFEST_SHA = '4ac59ec3847ba406ba1f0c812550e263dd6ea10a0bd7cc046e948ba1b438c832'
INPUT_REVISION = '5c5692107582866a3984973f83a2ac27d9a1ea89'
CATALOG_SHA = '5f6694f206084ad51dcc27c1485c4b4d000ffe5023310482aab367d0815df067'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def descriptor(root, name):
    path = root / name
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'Regular non-symlink file required: {name}')
    return {'path': name, 'bytes': path.stat().st_size, 'sha256': digest(path)}


def check(root, item, prefix=''):
    name = prefix + item.get('path', item.get('path_in_repo', ''))
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Unsafe manifest path')
    actual = descriptor(root, name)
    if any(actual[key] != item[key] for key in ('sha256', 'bytes')):
        raise ValueError(f'File identity changed: {name}')


def inspect(root):
    # The old package is retained byte-for-byte, including its own manifest.
    if digest(root / 'study_source/MANIFEST.json') != OLD_MANIFEST_SHA:
        raise ValueError('Historical source manifest changed')
    historical = read(root / 'study_source/MANIFEST.json')
    old_files = historical['files']
    if len(old_files) != 189:
        raise ValueError('Expected 189 historical payloads plus their manifest')
    expected = {'study_source/MANIFEST.json', 'README.md', 'build_archive.py'}
    for item in old_files:
        check(root, item, 'study_source/')
        expected.add('study_source/' + item['path'])
    for group, names in GROUPS.items():
        expected.update(group + '/' + name for name in names)
    support = read(root / 'model/publication_support.json')
    if {x['path_in_repo'] for x in support['support_files']} != set(MODEL_FILES):
        raise ValueError('Public model support allowlist changed')
    for item in support['support_files']:
        check(root, item, 'model/')
    expected.update('model/' + name for name in MODEL_FILES)
    expected.add('model/publication_support.json')
    if digest(root / 'model/model_catalog.json') != CATALOG_SHA:
        raise ValueError('Model catalog changed')
    if read(root / 'resources/resources.json')['resources']['input_bundle']['revision'] != INPUT_REVISION:
        raise ValueError('Input-bundle revision changed')
    entries = list(root.rglob('*'))
    if any(p.is_symlink() for p in entries):
        raise ValueError('Symlinks are forbidden in the release tree')
    actual = {str(p.relative_to(root)) for p in entries if p.is_file()}
    if actual - {'MANIFEST.json'} != expected:
        raise ValueError(f'Public allowlist differs: missing={sorted(expected-actual)}, extra={sorted(actual-expected-{"MANIFEST.json"})}')
    return [descriptor(root, name) for name in sorted(expected)]


def manifest(root, model_revision=None):
    if model_revision is not None and not re.fullmatch('[0-9a-f]{40}', model_revision):
        raise ValueError('Model revision must be an immutable 40-character commit')
    if model_revision:
        required = f'https://huggingface.co/dnagpt/bio2nl-models/tree/{model_revision}'
        readme = (root / 'README.md').read_text()
        if required not in readme or '__MODEL_REVISION__' in readme:
            raise ValueError('Finalize the root README with the actual model revision first')
    value = {
        'schema_version': 1,
        'artifact_type': 'bio2nl_public_portable_source_export_v1',
        'status': 'finalized_local_capture' if model_revision else 'local_draft_model_revision_pending',
        'scope': 'Published historical sources and aggregate results, public acquisition and preparation adapters, guarded source-training adapter, and model loader support. No weight, English corpus, prepared token stream, row predictions or private correspondence payloads.',
        'historical_source_archive_sha256': OLD_ARCHIVE_SHA,
        'historical_source_files_preserved': 190,
        'input_dataset_repository': 'dnagpt/bio2nl',
        'input_dataset_revision': INPUT_REVISION,
        'model_repository': 'dnagpt/bio2nl-models',
        'model_revision': model_revision,
        'model_catalog_sha256': CATALOG_SHA,
        'full_historical_raw_release_rebuilt': False,
        'full_gpu_training_reproduced_by_this_export': False,
        'self_included': False,
        'files': inspect(root),
    }
    value['payload_file_count'] = len(value['files'])
    (root / 'MANIFEST.json').write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    return value


def archive(root, output, model_revision):
    output = output.absolute()
    if output.exists() or output.is_relative_to(root):
        raise ValueError('Use a fresh archive path outside the source tree')
    value = manifest(root, model_revision)
    if not model_revision:
        raise ValueError('A final archive needs the real model-repository commit')
    names = [r['path'] for r in value['files']] + ['MANIFEST.json']
    with output.open('xb') as stream:
        with gzip.GzipFile(filename='', fileobj=stream, mode='wb', mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as bundle:
                for name in sorted(names):
                    path = root / name
                    info = tarfile.TarInfo(name)
                    info.size = path.stat().st_size
                    info.mode = 0o644
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    with path.open('rb') as source:
                        bundle.addfile(info, source)
    return {'status': 'built_local_archive', 'members': len(names),
            'bytes': output.stat().st_size, 'sha256': digest(output),
            'model_revision': model_revision, 'remote_publication_performed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('manifest', 'verify', 'archive'))
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--model-revision')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.absolute()
    if root.is_symlink():
        raise ValueError('Source root must not be a symlink')
    if args.command == 'manifest':
        value = manifest(root, args.model_revision)
        result = {'status': value['status'], 'payload_files': value['payload_file_count'],
                  'manifest_sha256': digest(root / 'MANIFEST.json')}
    elif args.command == 'verify':
        value = read(root / 'MANIFEST.json')
        if value['files'] != inspect(root):
            raise ValueError('Public source manifest differs from current bytes')
        result = {'status': 'passed', 'files': len(value['files']),
                  'manifest_sha256': digest(root / 'MANIFEST.json')}
    else:
        if args.output is None or args.model_revision is None:
            parser.error('archive requires --output and --model-revision')
        result = archive(root, args.output, args.model_revision)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
