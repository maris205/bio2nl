"""Document one pinned historical inconsistency; strictly validate new references."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from .patches import require, regular, sha

OLD_MANIFEST_SHA = 'f75596d56a1b8a129bb292210b7ad3a2d6b21d20f2999aae2d469d5d4d0da8ec'
SOURCE_SHA = '16535a38bfec4057cc27d6799b3fb299d6b34eac9868f18c4a4fd1d06f0b50c0'
SOURCE_SIZE = 1663
STALE_SHA = '513f71203094bbb7a0a08c7e7bfe5448f85c8d657da1535a651b120afb737232'
STALE_SIZE = 1825
SOURCE_NAME = 'metadata/remote_sources.json'


def loads(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def historical_remote_erratum(reference_manifest: bytes, reference_source: bytes):
    """Describe the exact known inconsistency without returning corrected metadata.

    Both physical inputs must be the pinned archived bytes. No generic exception
    for mismatched size/SHA is exposed, and this record is not an acceptance gate.
    """
    require(sha(reference_manifest) == OLD_MANIFEST_SHA, 'Unexpected historical remote manifest')
    require(len(reference_source) == SOURCE_SIZE and sha(reference_source) == SOURCE_SHA,
            'Unexpected historical remote source record')
    entries = [x for x in loads(reference_manifest)['files'] if x.get('file') == SOURCE_NAME]
    require(entries == [{'file': SOURCE_NAME, 'bytes': STALE_SIZE, 'sha256': STALE_SHA}],
            'Historical stale descriptor differs from the pinned erratum')
    return {
        'schema_version': 1, 'status': 'pinned_historical_metadata_erratum',
        'reference_manifest_sha256': OLD_MANIFEST_SHA,
        'field': 'files[file=metadata/remote_sources.json]',
        'historical_recorded_descriptor': entries[0],
        'actual_archived_source': {'file': SOURCE_NAME, 'bytes': SOURCE_SIZE, 'sha256': SOURCE_SHA},
        'historical_files_modified': False, 'historical_exact_reproduction_claimed': False,
        'current_manifest_policy': 'Every current artifact descriptor must match its own current physical file; no stale hash/size exception.',
        'scientific_scope': 'Metadata provenance discrepancy; no authority to change rows, splits, labels or model results.',
    }


def member(root, name):
    require(isinstance(name, str) and name and '\\' not in name and '\x00' not in name,
            'Invalid manifest member')
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and path.as_posix() == name,
            'Escaping/noncanonical manifest member')
    result = regular(root / name)
    require(result.resolve().is_relative_to(root.resolve()), 'Manifest member escaped root')
    return result


def file_identity(path):
    path = regular(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'sha256': digest, 'bytes': path.stat().st_size}


def verify_current_remote_manifest(release_root, manifest_name='metadata/remote_manifest.json', *, allowed_code_roots=()):
    """Rehash current references; return evidence, never write or enable a gate.

    Optional code roots must be explicit new snapshots. Relative data references
    always remain beneath release_root. Referenced files are rechecked at end.
    """
    root = Path(release_root).absolute()
    path = member(root, manifest_name)
    initial_manifest = file_identity(path)
    manifest = loads(path.read_bytes())
    require(manifest['source_record'] == SOURCE_NAME, 'Unexpected remote source record')
    require(isinstance(manifest['files'], list) and manifest['files'], 'Empty remote file inventory')
    identities, paths = {}, {}
    for item in manifest['files']:
        name = item['file']
        require(name not in identities, 'Duplicate remote file descriptor')
        target = member(root, name)
        require(type(item['bytes']) is int and item['bytes'] >= 0, 'Invalid descriptor byte count')
        require(isinstance(item['sha256'], str) and re.fullmatch('[0-9a-f]{64}', item['sha256']), 'Invalid descriptor SHA')
        identity = file_identity(target)
        require(identity == {'sha256': item['sha256'], 'bytes': item['bytes']},
                'Current descriptor differs from its own file: ' + name)
        identities[name], paths[name] = identity, target
    require(SOURCE_NAME in identities, 'Source descriptor missing')
    code = manifest['code']
    code_path = Path(code['file'])
    if not code_path.is_absolute():
        code_path = root / code_path
    code_path = regular(code_path)
    roots = (root, *(Path(x).absolute() for x in allowed_code_roots))
    require(any(code_path.resolve().is_relative_to(x.resolve()) for x in roots), 'Code reference outside declared roots')
    code_identity = file_identity(code_path)
    require(code_identity['sha256'] == code['sha256'], 'Current code descriptor SHA differs')
    for name, target in paths.items():
        require(file_identity(target) == identities[name], 'Current referenced file changed during validation')
    require(file_identity(code_path) == code_identity and file_identity(path) == initial_manifest,
            'Current manifest/code changed during validation')
    return {
        'status': 'current_remote_references_verified_not_data_acceptance',
        'manifest_file': manifest_name, 'manifest': initial_manifest,
        'files': identities, 'code': {'file': str(code_path), **code_identity},
        'file_count': len(identities), 'historical_exact_reproduction_claimed': False,
        'training_enabled': False, 'target_scoring_enabled': False,
    }
