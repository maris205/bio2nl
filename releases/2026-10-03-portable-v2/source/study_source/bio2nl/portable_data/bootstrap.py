"""Seed a new offline raw-data reconstruction from authenticated upstream assets.

No prepared data, token bins, mixed tokenizers or MMseqs work products are copied.
Historical code is copied byte-for-byte and later executed with explicit roots.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat

ARCHIVE_SHA = '31db85eb9e88e0b12d5ed68cd0ea08e276bd509d2565fd11723ba192007aa44f'
V2_SHA = '34f3bd78b6a4d923fafe3d76a53d0741886e39bc0af61245ff5bd6b34c511317'
PREFIX = 'data_rebuild/2026-09-25-v2/'
SOURCE_PREFIXES = ('raw/', 'metadata/nlp_raw/', 'tokenizers/gpt2_pinned/', 'tools/protein/')
PROVENANCE_FILES = ('metadata/remote_sources.json', 'metadata/corpora_english_source.json',
                    'metadata/corpora_dna_source.json', 'environment/data_build_requirements.txt',
                    'environment/data_build_environment.json')

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            result.update(chunk)
    return result.hexdigest()

def relative(name):
    path = PurePosixPath(name)
    require(isinstance(name, str) and name and not path.is_absolute() and
            '..' not in path.parts and '\\' not in name and '\x00' not in name and path.as_posix() == name,
            'Unsafe relative path')
    return name

def member(root, name):
    path = root / relative(name)
    for part in (path, *path.parents):
        require(not part.is_symlink(), 'Symlinked inputs are forbidden')
        if part == root:
            break
    require(path.resolve().is_relative_to(root.resolve()), 'Archive path escaped root')
    return path

def authenticate(root, archive_sha=ARCHIVE_SHA, release_sha=V2_SHA):
    manifest_path = member(root, 'archive_manifest.json')
    require(sha(manifest_path) == archive_sha, 'Archive manifest identity differs')
    manifest = json.loads(manifest_path.read_text())
    entries = {}
    for item in manifest['files']:
        name = relative(item['path'])
        require(name not in entries, 'Duplicate archive path')
        entries[name] = item
    require(manifest['file_count'] == len(entries), 'Archive file count differs')
    path = member(root, PREFIX + 'manifest.json')
    require(entries[PREFIX + 'manifest.json']['sha256'] == release_sha and sha(path) == release_sha,
            'Accepted v2 manifest identity differs')
    release = json.loads(path.read_text())
    require(release['training_gate_pass'] is True and release['status'] == 'accepted_for_training',
            'Historical reference release was not accepted')
    release_entries = {}
    for item in release['files']:
        name = relative(item['file'])
        require(name not in release_entries, 'Duplicate release path')
        bound = entries[PREFIX + name]
        require(bound['sha256'] == item['sha256'] and bound['bytes'] == item['bytes'], 'Nested release binding differs')
        release_entries[name] = item
    return release_entries

def selected(name):
    relative(name)
    if name.startswith(SOURCE_PREFIXES):
        return 'raw_or_external_tool_asset'
    if name.startswith('code/'):
        return 'historical_builder_source'
    if name in PROVENANCE_FILES:
        return 'historical_source_provenance'
    return None

def bootstrap(archive, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Raw reconstruction must be CPU-only')
    archive, output = Path(archive).absolute(), Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), 'Refuse existing output root')
    require(not output.resolve().is_relative_to(archive.resolve()), 'Output cannot be inside immutable archive')
    for parent in output.parents:
        require(not parent.is_symlink(), 'Symlinked output parent')
    require(shutil.disk_usage(output.parent).free >= 20 * (1 << 30), 'Need 12 GiB planning allowance plus 8 GiB reserve')
    entries = authenticate(archive)
    selection = [(name, item, selected(name)) for name, item in sorted(entries.items()) if selected(name)]
    assets = [x for x in selection if x[2] == 'raw_or_external_tool_asset']
    require(len(assets) == 77 and sum(x[1]['bytes'] for x in assets) == 955240396, 'Frozen raw/tool asset selection differs')
    for name, item, _ in selection:
        path = member(archive, PREFIX + name)
        require(path.is_file() and path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Input identity differs: ' + name)
    output.mkdir()
    copied = {}
    try:
        for name, item, kind in selection:
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(member(archive, PREFIX + name), destination)
            if name == 'tools/protein/mmseqs/bin/mmseqs':
                destination.chmod(destination.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
            require(sha(destination) == item['sha256'], 'Copied bytes differ: ' + name)
            copied[name] = {'sha256': item['sha256'], 'bytes': item['bytes'], 'kind': kind}
        for directory in ('data/sequences', 'data/pairs', 'data/corpora', 'metadata', 'validation', 'configs',
                          'work/protein', 'work/remote', 'portable_validation'):
            (output / directory).mkdir(parents=True, exist_ok=True)
        report = {'schema_version': 1, 'status': 'raw_assets_seeded_not_a_data_acceptance',
                  'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'archive_manifest_sha256': ARCHIVE_SHA, 'reference_release_manifest_sha256': V2_SHA,
                  'archive_root': str(archive), 'output_root': str(output), 'files': copied,
                  'raw_tool_assets': len(assets), 'raw_tool_bytes': sum(x[1]['bytes'] for x in assets),
                  'prepared_data_copied': False, 'work_products_copied': False,
                  'new_shared_tokenizers_copied': False, 'training_enabled': False, 'target_scoring_enabled': False,
                  'historical_provenance_note': 'Copied source evidence is historical acquisition provenance; no new acquisition or training acceptance is asserted.'}
        (output / 'portable_validation/bootstrap.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        return report
    except BaseException as exc:
        (output / 'BOOTSTRAP_FAILED.txt').write_text(repr(exc) + '\n')
        raise

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = bootstrap(args.archive, args.output)
    print(json.dumps({'status': report['status'], 'files': len(report['files']), 'output': report['output_root']}))

if __name__ == '__main__':
    main()
