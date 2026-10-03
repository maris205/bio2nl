"""Narrow, explicit metadata transitions for the new raw data release.

The caller authenticates reference bytes, builder manifest and comparison
bindings, then compares the returned values using the original strict helper.
This module never writes an artifact or authorizes training. It does not make
changed scientific fields equivalent merely because their hashes are known.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

try:
    from . import verification_protocol as rules
except ImportError:
    import verification_protocol as rules

OLD_ID = '2026-09-25-v2'
NEW_ID = '2026-10-01-portable-v2'
OLD_WORKSPACE = '/root/autodl-tmp/bio-trans/'
OLD_README_SHA = '8f4c9a7cd73de7cfc85569c8c8daea808215feb56cdbe5216c13b1d7566956c1'
OLD_SEARCH_SHA = '8dc56b3018b43f24cfb93e393f1c8312e976ae1284f6df6302224fd384974dfe'
NLP_PREFIX = 'code/bio2nl/data/rebuild_v2/nlp_synthetic/'
NLP_MANIFESTS = {'metadata/nlp_sources_manifest.json', 'metadata/nlp_clean_manifest.json',
                 'metadata/synthetic_dyck_manifest.json'}
REMOTE_SOURCE = 'metadata/remote_sources.json'
REMOTE_CODE = 'code/biopaws/data_v2_remote/build_remote.py'
PAIR_CODE = 'code/biopaws/data_v2/build_pairs.py'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def identity(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'Missing/symlinked metadata input: ' + str(path))
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'sha256': digest, 'bytes': path.stat().st_size}


def member(root, name):
    require(isinstance(name, str) and name and '\\' not in name and '\x00' not in name,
            'Invalid artifact member')
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and '..' not in rel.parts and rel.as_posix() == name,
            'Escaping/noncanonical artifact member')
    return root / name


def prepare_metadata_pair(reference, current, name, *, root, release_id, entries,
                          bindings, builder_manifest, search_sha,
                          cross_expected=None, pair_reference=None):
    """Return (reference, current, old-helper name, explicit change records).

    References and caller-owned objects are never mutated. `entries` is the
    authenticated original release inventory; `bindings` maps exact artifact
    names to their current and scientific-reference physical identities. The
    caller must recheck its complete read set before accepting the release.
    """
    root = Path(root).absolute()
    require(release_id == NEW_ID, 'Unexpected new release identity')
    require(isinstance(reference, dict) and isinstance(current, dict), 'Metadata must be JSON objects')
    a, b = copy.deepcopy(reference), copy.deepcopy(current)
    changes, effective_name = [], name
    cache = {}

    def physical(path):
        key = str(path)
        if key not in cache:
            cache[key] = identity(path)
        return cache[key]

    def code(key):
        require(key in entries and key in builder_manifest['files'], 'Unbound executed code: ' + key)
        descriptor = builder_manifest['files'][key]
        source = member(root, key)
        snapshot = root / 'execution_code' / key.removeprefix('code/')
        require(descriptor['source'] == str(source) and descriptor['snapshot'] == str(snapshot),
                'Execution code descriptor path mismatch: ' + key)
        original = {k: entries[key][k] for k in ('sha256', 'bytes')}
        require({k: descriptor['source_' + k] for k in ('sha256', 'bytes')} == original,
                'Execution code original identity mismatch: ' + key)
        require(physical(source) == original, 'Original code changed: ' + key)
        require(physical(snapshot) == {k: descriptor[k] for k in ('sha256', 'bytes')},
                'Executed code changed: ' + key)
        return descriptor

    def current_artifact(key):
        require(key in bindings, 'Current descriptor lacks exact file binding: ' + key)
        pin = bindings[key]
        actual = physical(member(root, key))
        require(actual == {k: pin[k] for k in ('sha256', 'bytes')}, 'Current file differs from binding: ' + key)
        return actual

    # Own-file checks are independent of generic hash alias normalization.
    def descriptors(node):
        if isinstance(node, dict):
            file_name = node.get('file', node.get('path'))
            if isinstance(file_name, str) and file_name in bindings:
                actual = current_artifact(file_name)
                if 'sha256' in node:
                    require(node['sha256'] == actual['sha256'], 'Descriptor SHA differs from own current file: ' + file_name)
                if 'bytes' in node:
                    require(type(node['bytes']) is int and node['bytes'] == actual['bytes'],
                            'Descriptor bytes differ from own current file: ' + file_name)
            for value in node.values():
                descriptors(value)
        elif isinstance(node, list):
            for value in node:
                descriptors(value)
    descriptors(b)

    if name in rules.IDENTITY_FIELDS:
        field = rules.IDENTITY_FIELDS[name]
        require(a.get(field) == OLD_ID and b.get(field) == release_id, 'Release identity transition mismatch: ' + name)
        b[field] = OLD_ID
        changes.append({'field': field, 'rule': 'explicit_new_release_identity', 'reference': OLD_ID, 'current': release_id})

    if name in NLP_MANIFESTS:
        selected = {key[len(NLP_PREFIX):]: key for key in entries
                    if key.startswith(NLP_PREFIX) and '/' not in key[len(NLP_PREFIX):]
                    and Path(key).suffix in ('.py', '.json', '.md')}
        require('README.md' in selected and 'build_longrange.py' in selected, 'Incomplete NLP code inventory')
        expected_old, expected_new = {}, {}
        for basename, key in selected.items():
            descriptor = code(key)
            expected_old[basename] = descriptor['source_sha256']
            expected_new[basename] = descriptor['sha256']
        historical = dict(expected_old)
        historical['README.md'] = OLD_README_SHA
        del historical['build_longrange.py']
        require(a.get('code_sha256') == historical, 'Historical NLP code inventory transition mismatch')
        require(b.get('code_sha256') == expected_new, 'NLP code inventory does not bind exact executed files')
        a['code_sha256'] = dict(expected_old)
        b['code_sha256'] = dict(expected_old)
        effective_name = 'metadata/versioned_inventory.json'
        changes.append({'field': 'code_sha256', 'rule': 'explicit_current_execution_and_later_accepted_inventory',
                        'source_files': sorted(selected.values())})

    elif name == 'metadata/synthetic_longrange_manifest.json':
        descriptor = code(NLP_PREFIX + 'build_longrange.py')
        require(a.get('code_sha256') == descriptor['source_sha256'] and b.get('code_sha256') == descriptor['sha256'],
                'Longrange code hash does not bind own executed builder')
        b['code_sha256'] = descriptor['source_sha256']
        changes.append({'field': 'code_sha256', 'rule': 'verified_execution_code_lineage'})

    elif name == 'metadata/remote_manifest.json':
        require(entries[name]['sha256'] == rules.REMOTE_MANIFEST_SHA,
                'Historical remote manifest is not the pinned erratum reference')
        require(entries[REMOTE_SOURCE]['sha256'] == rules.REMOTE_SOURCE_SHA and entries[REMOTE_SOURCE]['bytes'] == 1663,
                'Historical remote source identity differs from pinned erratum')
        require(a.get('source_record') == b.get('source_record') == REMOTE_SOURCE, 'Unexpected remote source record')
        descriptor = code(REMOTE_CODE)
        require(a['code']['file'] == OLD_WORKSPACE + REMOTE_CODE[5:] and a['code']['sha256'] == descriptor['source_sha256'],
                'Historical remote code descriptor mismatch')
        require(b['code']['file'] == descriptor['snapshot'] and b['code']['sha256'] == descriptor['sha256'],
                'Remote code must reference exact executed snapshot and SHA')
        # Use the old helper's allowed current-root/code path only after checking
        # the actual execution_code path above. This is an in-memory comparison.
        b['code']['file'] = str(root / REMOTE_CODE)
        b['code']['sha256'] = descriptor['source_sha256']
        require(b.pop('original_algorithm_release_id', None) == OLD_ID, 'Remote original algorithm identity mismatch')
        old_files, new_files = a['files'], b['files']
        require(isinstance(old_files, list) and isinstance(new_files, list), 'Remote descriptor inventory must be lists')
        names = [v['file'] for v in old_files]
        require(len(set(names)) == len(names) and [v['file'] for v in new_files] == names,
                'Remote descriptor coverage/order differs')
        stale = [v for v in old_files if v['file'] == REMOTE_SOURCE]
        require(stale == [{'file': REMOTE_SOURCE, 'bytes': 1825, 'sha256': rules.REMOTE_STALE_SHA}],
                'Historical remote stale descriptor is not the exact documented error')
        stale[0].update(bytes=1663, sha256=rules.REMOTE_SOURCE_SHA)
        for item in new_files:
            actual = current_artifact(item['file'])
            require(set(item) == {'file', 'sha256', 'bytes'} and type(item['bytes']) is int and
                    {k: item[k] for k in ('sha256', 'bytes')} == actual, 'Invalid own-file remote descriptor')
        changes.extend([
            {'field': 'files[file=metadata/remote_sources.json]', 'rule': 'pinned_reference_only_descriptor_erratum',
             'reference_recorded_sha256': rules.REMOTE_STALE_SHA, 'reference_actual_sha256': rules.REMOTE_SOURCE_SHA,
             'reference_recorded_bytes': 1825, 'reference_actual_bytes': 1663,
             'historical_exact_reproduction_claimed': False},
            {'field': 'code', 'rule': 'verified_execution_code_lineage'},
            {'field': 'original_algorithm_release_id', 'rule': 'explicit_original_algorithm_identity'}])

    elif name == 'metadata/protein_pair_manifest.json':
        require(isinstance(pair_reference, dict), 'Authenticated deterministic pair manifest required')
        pair_reference = copy.deepcopy(pair_reference)
        variable = {'acceptance', 'data_files', 'duration_seconds'}
        require({k: v for k, v in a.items() if k not in variable} ==
                {k: v for k, v in pair_reference.items() if k not in variable},
                'Pair replay changed undeclared algorithm/source fields')
        for field in variable:
            a[field] = pair_reference[field]
        descriptor = code(PAIR_CODE)
        require(a['builder_sha256'] == descriptor['source_sha256'] and b['builder_sha256'] == descriptor['sha256'],
                'Pair builder hash does not bind executed snapshot')
        b['builder_sha256'] = descriptor['source_sha256']
        require(b.pop('original_algorithm_release_id', None) == OLD_ID, 'Pair original algorithm identity mismatch')
        order = b.pop('candidate_order_record', None)
        require(isinstance(order, dict) and set(order) == {'file', 'sha256'} and
                order['file'] == 'metadata/protein_candidate_order.json', 'Pair candidate order reference mismatch')
        order_actual = current_artifact(order['file'])
        require(order['sha256'] == order_actual['sha256'], 'Pair order record SHA differs from own file')
        expected_names = {Path(n).name for n in rules.PAIR_FILES}
        require(set(a['data_files']) == set(b['data_files']) == expected_names, 'Pair data file coverage mismatch')
        for basename in expected_names:
            path = 'data/pairs/' + basename
            require(a['data_files'][basename] == bindings[path]['reference_sha256'], 'Pair replay file binding mismatch')
            require(b['data_files'][basename] == current_artifact(path)['sha256'], 'Pair output SHA differs from own file')
        acceptance_name = 'validation/protein_pair_acceptance.json'
        current_artifact(acceptance_name)
        require(canonical(b['acceptance']) == canonical(json.loads(member(root, acceptance_name).read_text())),
                'Pair embedded acceptance differs from current file')
        changes.extend([
            {'field': 'acceptance,data_files,duration_seconds', 'rule': 'authenticated_deterministic_pair_reference'},
            {'field': 'builder_sha256', 'rule': 'verified_execution_code_lineage'},
            {'field': 'candidate_order_record', 'rule': 'verified_current_order_record', 'sha256': order_actual['sha256']},
            {'field': 'original_algorithm_release_id', 'rule': 'explicit_original_algorithm_identity'}])

    if name in {'metadata/protein_remote_pretraining_exclusion.json', 'validation/cross_dataset_homology.json'}:
        require(isinstance(search_sha, str) and re.fullmatch('[0-9a-f]{64}', search_sha) is not None,
                'Current search SHA must be explicitly bound')
        require(a.get('search_output_sha256') == OLD_SEARCH_SHA and b.get('search_output_sha256') == search_sha,
                'Search output SHA does not bind the explicit old/current inputs')
        if name == 'validation/cross_dataset_homology.json':
            require(isinstance(cross_expected, dict) and canonical(current) == canonical(cross_expected),
                    'Current cross-dataset report differs from independent recomputation')
            a['pair_task_overlap']['sequence_similarity'] = copy.deepcopy(cross_expected['pair_task_overlap']['sequence_similarity'])
            changes.append({'field': 'pair_task_overlap.sequence_similarity', 'rule': 'independent_current_pair_overlap_recomputation'})
        b['search_output_sha256'] = OLD_SEARCH_SHA
        changes.append({'field': 'search_output_sha256', 'rule': 'explicit_current_search_identity', 'current': search_sha})

    # Detect mutations during local validation; the caller also rechecks its full
    # read set before writing any acceptance output.
    for path, pin in cache.items():
        require(identity(path) == pin, 'Metadata input changed during comparison: ' + path)
    return a, b, effective_name, changes
