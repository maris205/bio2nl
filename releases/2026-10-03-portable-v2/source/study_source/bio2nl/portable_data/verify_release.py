"""Independent, exact scientific-content acceptance for an offline v2 rebuild.

The accepted archive is a reference, not an acceptance certificate for new files.
This program authorizes equivalent inputs for derived-data reconstruction only;
it neither enables model training nor reads model results.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
from itertools import zip_longest
import json
import math
import os
from pathlib import Path
import re

try:
    from .bootstrap import ARCHIVE_SHA, V2_SHA, PREFIX, authenticate, member, sha, selected as bootstrap_selected
except ImportError:
    from bootstrap import ARCHIVE_SHA, V2_SHA, PREFIX, authenticate, member, sha, selected as bootstrap_selected

VERIFICATION_PROTOCOL_SHA = '5771d3635b06ff689483dbbc4cd893110b3d551747e1cea360c6ded071994b18'

SOURCE_PROVENANCE = {
    'metadata/remote_sources.json', 'metadata/corpora_english_source.json',
    'metadata/corpora_dna_source.json',
}
EXCLUDED_REFERENCE = {
    'README.md': 'Historical progress document; not a rebuilt scientific input.',
    'metadata/code_snapshot_bio2nl.json': 'Historical Git state is not asserted for the new execution.',
    'metadata/code_snapshot_biopaws.json': 'Historical Git state is not asserted for the new execution.',
    'metadata/nlp_synthetic_README.md': 'New documentation is checked against the accepted later code README separately.',
    'validation/integrated_release_acceptance.json': 'Historical training acceptance is never copied or reused.',
}
CODE_MANIFESTS = {
    'metadata/nlp_sources_manifest.json', 'metadata/nlp_clean_manifest.json',
    'metadata/synthetic_dyck_manifest.json',
}
OLD_README_SHA = '8f4c9a7cd73de7cfc85569c8c8daea808215feb56cdbe5216c13b1d7566956c1'
CODE_NLP = 'code/bio2nl/data/rebuild_v2/nlp_synthetic/'
TIME_FIELDS = {
    'metadata/protein_sources.json': {'created_utc': 'utc'},
    'metadata/protein_full_cluster_command.json': {'seconds': 'duration', 'utc_completed': 'utc'},
    'metadata/protein_short_search_command.json': {'seconds': 'duration', 'utc_completed': 'utc'},
    'metadata/protein_pair_construction.json': {'runtime_seconds': 'duration'},
    'metadata/protein_pair_manifest.json': {'duration_seconds': 'duration'},
    'metadata/protein_remote_pretraining_exclusion.json': {'completed_utc': 'utc'},
}
COMMAND_FIELDS = {
    'metadata/protein_full_cluster_command.json': 'argv',
    'metadata/protein_short_search_command.json': 'argv',
    'metadata/protein_remote_pretraining_exclusion.json': 'command',
}
ORIGINAL_WORKSPACE = '/root/autodl-tmp/bio-trans/'
ORIGINAL_RELEASE = ORIGINAL_WORKSPACE + 'data_rebuild/2026-09-25-v2/'
RAW_PREFIXES = ('raw/', 'metadata/nlp_raw/', 'tools/', 'tokenizers/gpt2_pinned/')
SHA_RE = re.compile(r'[0-9a-f]{64}\Z')


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def loads(text):
    def nonfinite(value):
        raise ValueError('Non-finite JSON value: ' + value)
    return json.loads(text, object_pairs_hook=strict_object, parse_constant=nonfinite)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def ordered_jsonl_equal(left, right, compressed=False):
    opener = gzip.open if compressed else open
    count = 0
    with opener(left, 'rt', encoding='utf-8') as a, opener(right, 'rt', encoding='utf-8') as b:
        for count, (x, y) in enumerate(zip_longest(a, b), 1):
            if x is None or y is None:
                return False, {'first_different_row': count, 'reason': 'row_count'}
            if canonical(loads(x)) != canonical(loads(y)):
                return False, {'first_different_row': count, 'reason': 'ordered_row_content'}
    return True, {'rows': count}


def decompressed_equal(left, right):
    count = 0
    with gzip.open(left, 'rb') as a, gzip.open(right, 'rb') as b:
        while True:
            x, y = a.read(4 << 20), b.read(4 << 20)
            if x != y:
                return False, {'first_different_block_offset': count}
            count += len(x)
            if not x:
                return True, {'decompressed_bytes': count}


def parquet_equal(left, right):
    import pyarrow.parquet as pq
    a, b = pq.read_table(left), pq.read_table(right)
    # Column order, Arrow type and nullability are scientific structure. Writer
    # metadata is intentionally not a substitute for checking ordered values.
    schemas_equal = a.schema.equals(b.schema, check_metadata=False)
    equal = schemas_equal and a.equals(b, check_metadata=False)
    return equal, {'reference_rows': a.num_rows, 'rows': b.num_rows,
                   'column_names': b.column_names, 'schema_equal': schemas_equal,
                   'schema': str(b.schema.remove_metadata())}


def compare_content(reference, current, name, reference_sha, current_sha):
    if name.startswith(RAW_PREFIXES) or name.startswith(('code/', 'environment/')) or name in SOURCE_PROVENANCE:
        return current_sha == reference_sha, 'bytes_exact', {}
    if name.endswith('.bin'):
        if reference.stat().st_size % 2 or current.stat().st_size % 2:
            return False, 'little_endian_uint16_exact', {'reason': 'odd_byte_count'}
        return current_sha == reference_sha, 'little_endian_uint16_exact', {'tokens': current.stat().st_size // 2}
    if name.endswith('.parquet'):
        equal, info = parquet_equal(reference, current)
        return equal, 'parquet_schema_and_ordered_rows_exact', info
    if name.endswith('.jsonl.gz'):
        equal, info = ordered_jsonl_equal(reference, current, True)
        return equal, 'gzip_ordered_json_rows_exact', info
    if name.endswith('.jsonl'):
        equal, info = ordered_jsonl_equal(reference, current)
        return equal, 'ordered_json_rows_exact', info
    if name.endswith('.gz'):
        equal, info = decompressed_equal(reference, current)
        return equal, 'gzip_decompressed_bytes_exact', info
    if name.endswith('.json'):
        equal = canonical(loads(reference.read_text())) == canonical(loads(current.read_text()))
        return equal, 'canonical_json_exact', {}
    return current_sha == reference_sha, 'bytes_exact', {}


def is_metadata_json(name):
    return name.endswith('.json') and (
        name.startswith(('metadata/', 'validation/', 'tokenized/')) or
        name in ('tokenizers/pure_byte/metadata.json', 'tokenizers/mixed_bpe/metadata.json')) and name not in SOURCE_PROVENANCE


def mapped_location(value, current_root, side):
    if not isinstance(value, str):
        return value
    current_prefix = str(current_root) + '/'
    if side == 'current' and value.startswith(current_prefix + 'code/'):
        return '$CODE/' + value[len(current_prefix + 'code/'):]
    if side == 'current' and value.startswith(current_prefix):
        return '$RELEASE/' + value[len(current_prefix):]
    if side == 'reference' and value.startswith(ORIGINAL_RELEASE):
        return '$RELEASE/' + value[len(ORIGINAL_RELEASE):]
    if side == 'reference' and value.startswith('data_rebuild/2026-09-25-v2/'):
        return '$RELEASE/' + value[len('data_rebuild/2026-09-25-v2/'):]
    if side == 'reference' and (value.startswith(ORIGINAL_WORKSPACE + 'biopaws/') or value.startswith(ORIGINAL_WORKSPACE + 'bio2nl/')):
        return '$CODE/' + value[len(ORIGINAL_WORKSPACE):]
    return value


def normalize_metadata(value, name, side, root, bindings, entries, changes):
    value = copy.deepcopy(value)
    # Explicitly permitted historical code-directory differences, pinned before
    # seeing the new build's results. The executable builder itself is unchanged.
    if name in CODE_MANIFESTS:
        codes = value['code_sha256']
        accepted_readme = entries[CODE_NLP + 'README.md']['sha256']
        accepted_longrange = entries[CODE_NLP + 'build_longrange.py']['sha256']
        if side == 'reference':
            if codes['README.md'] != OLD_README_SHA or 'build_longrange.py' in codes:
                raise ValueError('Historical NLP code inventory does not match the declared transition')
            codes['README.md'] = accepted_readme
            codes['build_longrange.py'] = accepted_longrange
            changes.append({'field': 'code_sha256', 'rule': 'later_accepted_README_and_longrange_inventory'})
        elif codes.get('README.md') != accepted_readme or codes.get('build_longrange.py') != accepted_longrange:
            raise ValueError('New NLP code inventory differs from accepted snapshot')
    for key, kind in TIME_FIELDS.get(name, {}).items():
        v = value[key]
        if kind == 'utc':
            when = datetime.fromisoformat(v)
            if when.tzinfo is None or when.utcoffset().total_seconds() != 0:
                raise ValueError('Timestamp must include UTC offset: ' + key)
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError('Invalid execution duration: ' + key)
        value[key] = '$EXECUTION_' + kind
        changes.append({'field': key, 'rule': 'execution_' + kind, 'observed': v})
    if name in COMMAND_FIELDS:
        key = COMMAND_FIELDS[name]
        value[key] = [mapped_location(x, root, side) for x in value[key]]
        changes.append({'field': key, 'rule': 'explicit_release_path_relocation'})
    if name == 'metadata/remote_manifest.json':
        value['code']['file'] = mapped_location(value['code']['file'], root, side)
        changes.append({'field': 'code.file', 'rule': 'explicit_builder_snapshot_relocation'})

    hash_map = {}
    alias_groups = {}
    for item in bindings.values():
        if item.get('equivalent'):
            alias_groups.setdefault(item['reference_sha256'], set()).add(item['sha256'])
    ambiguous_hashes = {old for old, current in alias_groups.items() if len(current) > 1}
    for old in list(ambiguous_hashes):
        ambiguous_hashes.update(alias_groups[old])
    for filename, item in bindings.items():
        if not item.get('equivalent'):
            continue
        key = item['reference_sha256'] if side == 'reference' else item['sha256']
        token = '$EQUIVALENT_CONTENT_SHA256/' + item['reference_sha256']
        previous = hash_map.get(key)
        if previous is not None and previous != token:
            # Identical current bytes cannot authorize two distinct historical
            # content identities unless they had identical physical hashes.
            raise ValueError('Ambiguous verified hash relocation: ' + filename)
        hash_map[key] = token

    def recurse(node, path=()):
        if isinstance(node, dict):
            result = {}
            file_name = node.get('file', node.get('path'))
            binding = bindings.get(file_name) if isinstance(file_name, str) else None
            for key, val in node.items():
                location = (*path, key)
                # Only artifact descriptor bytes, accompanied by a verified
                # file/path, may change. Counts, budgets and text-byte sizes may not.
                if key == 'sha256' and binding and binding.get('equivalent'):
                    expected = binding['reference_sha256'] if side == 'reference' else binding['sha256']
                    if val != expected:
                        raise ValueError('Artifact descriptor SHA-256 differs from its own file: ' + str(file_name))
                    result[key] = '$ARTIFACT_CONTENT_SHA256/' + str(file_name)
                    changes.append({'field': '/'.join(map(str, location)), 'rule': 'verified_own_file_sha256', 'file': file_name})
                elif key == 'bytes' and binding and binding.get('equivalent'):
                    expected = binding['reference_bytes'] if side == 'reference' else binding['bytes']
                    if type(val) is not int or val != expected:
                        raise ValueError('Artifact descriptor byte count differs from file: ' + str(file_name))
                    result[key] = '$PHYSICAL_BYTES/' + str(file_name)
                    changes.append({'field': '/'.join(map(str, location)), 'rule': 'verified_file_physical_bytes', 'file': file_name})
                else:
                    result[key] = recurse(val, location)
            return result
        if isinstance(node, list):
            return [recurse(x, (*path, i)) for i, x in enumerate(node)]
        if isinstance(node, str) and SHA_RE.fullmatch(node):
            # Hash replacement is allowed only inside a hash-labelled field or
            # map. Arbitrary scientific strings containing a digest stay exact.
            if any('sha256' in str(part) for part in path) or (name == 'metadata/protein_pair_manifest.json' and path[:1] == ('data_files',)):
                if node in ambiguous_hashes:
                    raise ValueError('Hash alias has divergent current file hashes; explicit file context required: ' + '/'.join(map(str, path)))
                if node in hash_map:
                    changes.append({'field': '/'.join(map(str, path)), 'rule': 'verified_equivalent_file_sha256'})
                    return hash_map[node]
        return node
    return recurse(value)


def compare_metadata(reference, current, name, root, bindings, entries):
    changes = {'reference': [], 'current': []}
    a = normalize_metadata(loads(reference.read_text()), name, 'reference', root, bindings, entries, changes['reference'])
    b = normalize_metadata(loads(current.read_text()), name, 'current', root, bindings, entries, changes['current'])
    return canonical(a) == canonical(b), changes


def current_gates(root):
    gates, issues = {}, []
    def read(name):
        return loads(member(root, name).read_text())
    checks = (
        ('official_nlp', 'validation/nlp_acceptance_report.json', 'construction_acceptance_pass', True),
        ('clean_nlp', 'validation/nlp_clean_acceptance_report.json', 'acceptance_pass', True),
        ('typed_dyck', 'validation/synthetic_dyck_acceptance_report.json', 'acceptance_pass', True),
        ('independent_nlp_synthetic', 'validation/nlp_synthetic_independent_verification.json', 'passed', True),
        ('protein_pairs', 'validation/protein_pair_acceptance.json', 'acceptance_pass', True),
        ('remote_pairs', 'validation/remote_validation.json', 'status', 'passed_local_data_checks'),
        ('remote_pretraining_exclusion', 'metadata/protein_remote_pretraining_exclusion.json', 'status', 'completed'),
        ('cross_dataset_audit', 'validation/cross_dataset_homology.json', 'status', 'completed'),
    )
    for label, name, key, expected in checks:
        try:
            data = read(name)
            gates[label] = type(data.get(key)) is type(expected) and data.get(key) == expected
            if 'failures' in data:
                gates[label] &= data['failures'] == []
            if label == 'cross_dataset_audit':
                gates[label] &= data['remaining_eligible_swissprot_accessions_with_qualifying_hits'] == 0
            if label == 'protein_pairs':
                gates[label] &= data['sequence_cross_split_overlap'] == 0 and data['endpoint_role_degree_balance'] is True
                gates[label] &= data['gap_or_noncanonical_input_count'] == 0 and data['length_difference_matched_per_A'] is True
            if label == 'remote_pairs':
                gates[label] &= data['endpoint_label_degree_balance'] is True and data['zero_gap_input_sequences'] is True
        except Exception as exc:
            gates[label] = False; issues.append(f'{name}: {type(exc).__name__}: {exc}')
    for condition in ('english', 'protein', 'shuffled', 'randomaa', 'dna'):
        label = 'corpus_' + condition
        try:
            data = read(f'validation/corpora_{condition}.json')
            gates[label] = data['condition'] == condition and not any(data['exact_text_cross_split_overlap'].values())
            gates[label] &= all(data['splits'][s]['records'] > 0 for s in ('train', 'validation', 'test'))
        except Exception as exc:
            gates[label] = False; issues.append(f'{label}: {type(exc).__name__}: {exc}')
    for depth in (8, 16, 24):
        label = f'longrange_d{depth}'
        try:
            data = read(f'validation/synthetic_longrange_d{depth}_acceptance.json')
            gates[label] = data['acceptance_pass'] is True and not any(data['split_overlap'].values())
            gates[label] &= all(metrics == {'majority': .5, 'nearest_noun_agreement': .5, 'subject_verb_agreement_oracle': 1.0}
                                for metrics in data['baselines_accuracy'].values())
        except Exception as exc:
            gates[label] = False; issues.append(f'{label}: {type(exc).__name__}: {exc}')
    issues.extend('Failed current gate: ' + name for name, passed in gates.items() if not passed)
    return gates, issues


def check_token_cell(root, folder, split, target, eos, vocab):
    import numpy as np
    name = folder + '/' + split
    binary = member(root, name + '.bin')
    index = member(root, name + '.index.jsonl')
    metadata = loads(member(root, name + '.json').read_text())
    if binary.stat().st_size != target * 2 or metadata['tokens'] != target:
        raise ValueError('Token budget mismatch: ' + name)
    values = np.memmap(binary, dtype='<u2', mode='r')
    eos_count = 0
    for offset in range(0, target, 1 << 20):
        chunk = values[offset:offset + (1 << 20)]
        if int(chunk.max()) >= vocab:
            raise ValueError('Token ID outside vocabulary: ' + name)
        if eos == 1 and np.isin(chunk, [0, 2, 3]).any():
            raise ValueError('PAD/SEP/UNK inside packed corpus: ' + name)
        eos_count += int((chunk == eos).sum())
    offset, records, ids = 0, 0, set()
    with index.open() as stream:
        for line in stream:
            row = loads(line)
            if type(row['length']) is not int or row['length'] < 1 or row['offset'] != offset:
                raise ValueError('Invalid contiguous record index: ' + name)
            offset += row['length']
            if offset > target or row['eos_offset'] != offset - 1 or int(values[offset - 1]) != eos:
                raise ValueError('Index/EOS position mismatch: ' + name)
            if row['record_id'] in ids:
                raise ValueError('Repeated record in tokenized source: ' + name)
            ids.add(row['record_id']); records += 1
    if offset != target or records != metadata['records'] or eos_count != records:
        raise ValueError('Token/index final accounting mismatch: ' + name)
    if metadata['sha256'] != sha(binary) or metadata['index_sha256'] != sha(index):
        raise ValueError('Token metadata hashes do not bind current files: ' + name)
    if metadata['eos_token_id'] != eos:
        raise ValueError('EOS metadata differs: ' + name)
    return {'file': name + '.bin', 'tokens': target, 'records': records, 'eos_count': eos_count}


def current_token_checks(root):
    cells, issues = [], []
    for family, vocab in (('pure_byte', 260), ('mixed_bpe', 32000)):
        try:
            tokenizer = loads(member(root, f'tokenizers/{family}/tokenizer.json').read_text())
            entries = tokenizer['model']['vocab']
            expected_special = {'<|pad_v2|>': 0, '<|eos_v2|>': 1, '<|pair_v2|>': 2, '<|unk_v2|>': 3}
            if len(entries) != vocab or any(entries.get(k) != v for k, v in expected_special.items()):
                raise ValueError('Vocabulary or special IDs differ')
        except Exception as exc:
            issues.append(f'{family} tokenizer: {type(exc).__name__}: {exc}')
        for condition in ('english', 'protein', 'shuffled', 'randomaa', 'dna'):
            for split, target in (('train', 16777216), ('validation', 262144), ('test', 262144)):
                try:
                    cells.append(check_token_cell(root, f'tokenized/{family}/{condition}', split, target, 1, vocab))
                except Exception as exc:
                    issues.append(f'{family}/{condition}/{split}: {type(exc).__name__}: {exc}')
    for split, target in (('train', 49999872), ('validation', 262144), ('test', 262144)):
        try:
            cells.append(check_token_cell(root, 'tokenized/gpt2_smallweb', split, target, 50256, 50257))
        except Exception as exc:
            issues.append(f'gpt2_smallweb/{split}: {type(exc).__name__}: {exc}')
    return cells, issues


def validate_protocol(path, entries):
    path = Path(path)
    if path.is_symlink() or sha(path) != VERIFICATION_PROTOCOL_SHA:
        raise ValueError('Verification protocol identity differs')
    protocol = loads(path.read_text())
    names = [n for n in sorted(entries) if n not in EXCLUDED_REFERENCE]
    required = {
        'accepted_reference_manifest_sha256': V2_SHA,
        'accepted_archive_manifest_sha256': ARCHIVE_SHA,
        'files': names,
        'selected_reference_file_count': len(names),
        'historical_files_excluded': EXCLUDED_REFERENCE,
        'allowed_execution_fields': TIME_FIELDS,
        'allowed_command_root_fields': COMMAND_FIELDS,
        'scientific_tolerance': 0,
        'required_token_cells': 33,
        'success_status': 'accepted_for_derived_reconstruction',
        'training_enabled': False,
        'target_scoring_enabled': False,
    }
    for key, expected in required.items():
        if canonical(protocol.get(key)) != canonical(expected):
            raise ValueError('Verification protocol/code mismatch: ' + key)
    expected_inventory = {
        'files': sorted(CODE_MANIFESTS), 'README_previous_sha256': OLD_README_SHA,
        'README_accepted_later_sha256': entries[CODE_NLP + 'README.md']['sha256'],
        'added_build_longrange_py_sha256': entries[CODE_NLP + 'build_longrange.py']['sha256'],
        'other_entries': 'exact',
    }
    if protocol['allowed_code_inventory_change'] != expected_inventory:
        raise ValueError('Protocol code-inventory transition differs')
    return {'file': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size}


def validate_bootstrap(root, archive, entries):
    path = member(root, 'portable_validation/bootstrap.json')
    boot = loads(path.read_text())
    if boot.get('status') != 'raw_assets_seeded_not_a_data_acceptance':
        raise ValueError('Invalid raw bootstrap status')
    if boot['reference_release_manifest_sha256'] != V2_SHA or boot['archive_manifest_sha256'] != ARCHIVE_SHA:
        raise ValueError('Bootstrap reference identities differ')
    if Path(boot['output_root']).resolve() != root or Path(boot['archive_root']).resolve() != archive:
        raise ValueError('Bootstrap root differs from current execution')
    for key in ('prepared_data_copied', 'work_products_copied', 'new_shared_tokenizers_copied',
                'training_enabled', 'target_scoring_enabled'):
        if boot[key] is not False:
            raise ValueError('Bootstrap has an invalid execution flag: ' + key)
    expected = {name: {'sha256': item['sha256'], 'bytes': item['bytes'], 'kind': bootstrap_selected(name)}
                for name, item in entries.items() if bootstrap_selected(name)}
    if canonical(boot['files']) != canonical(expected):
        raise ValueError('Bootstrap inventory does not close against authenticated selected assets')
    assets = [item for item in expected.values() if item['kind'] == 'raw_or_external_tool_asset']
    if boot['raw_tool_assets'] != len(assets) or boot['raw_tool_bytes'] != sum(x['bytes'] for x in assets):
        raise ValueError('Bootstrap raw asset totals differ')
    return {'sha256': sha(path), 'bytes': path.stat().st_size, 'copied_asset_count': len(expected),
            'raw_asset_count': len(assets), 'inventory_matches_authenticated_selection': True}


def final_recheck(root, bindings, extra_pins):
    issues, count = [], 0
    for name, item in bindings.items():
        try:
            path = member(root, name)
            if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
                raise ValueError('File changed after initial comparison')
            count += 1
        except Exception as exc:
            item['equivalent'] = False
            item['changed_after_comparison'] = True
            issues.append({'file': name, 'reason': f'Final integrity check: {exc}'})
    for label, item in extra_pins.items():
        try:
            path = Path(item['file'])
            if path.is_symlink() or sha(path) != item['sha256']:
                raise ValueError('Pinned control file changed during verification')
            if 'bytes' in item and path.stat().st_size != item['bytes']:
                raise ValueError('Pinned control-file size changed during verification')
            count += 1
        except Exception as exc:
            issues.append({'file': label, 'reason': f'Final integrity check: {exc}'})
    return {'passed': not issues, 'hashes_checked': count, 'issues': issues}


def verify_release(archive, root, output=None, verification_protocol=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Release verification is CPU-only; set CUDA_VISIBLE_DEVICES to empty')
    archive, root = Path(archive).resolve(), Path(root).resolve()
    output = Path(output) if output else root / 'portable_validation/acceptance.json'
    if output.exists() or output.is_symlink():
        raise FileExistsError('Refusing to overwrite an existing acceptance report')
    if root == archive or root.is_relative_to(archive):
        raise ValueError('New release may not be inside the immutable reference archive')
    if not output.resolve().is_relative_to(root):
        raise ValueError('Acceptance report must be inside the new release')
    for part in (output, *output.parents):
        if part.is_symlink():
            raise ValueError('Acceptance report path must not contain symlinks')
        if part == root:
            break
    # Never create new acceptance artifacts inside an old release accidentally.
    # The raw-only bootstrap is required before any new verification output.
    if not member(root, 'portable_validation/bootstrap.json').is_file():
        raise ValueError('A new raw-only bootstrap is required before verification')
    if verification_protocol is None:
        raise ValueError('An explicit pinned verification protocol is required')
    entries = authenticate(archive)
    protocol_info = validate_protocol(verification_protocol, entries)
    initial_bootstrap_hash = sha(member(root, 'portable_validation/bootstrap.json'))
    verifier_initial_hash = sha(Path(__file__))
    bootstrap_module = Path(__file__).parent / 'bootstrap.py'
    bootstrap_module_initial_hash = sha(bootstrap_module)
    reference_root = archive / PREFIX
    files, mismatches, pending = {}, [], []
    selected = [name for name in sorted(entries) if name not in EXCLUDED_REFERENCE]
    actual_scientific = {p.relative_to(root).as_posix() for prefix in ('data', 'tokenized', 'tokenizers')
                         for p in (root / prefix).rglob('*') if p.is_file()}
    expected_scientific = {name for name in entries if name.startswith(('data/', 'tokenized/', 'tokenizers/'))}
    for name in sorted(actual_scientific - expected_scientific):
        mismatches.append({'file': name, 'reason': 'unexpected_scientific_artifact'})
    for name in selected:
        ref = entries[name]
        try:
            reference, current = member(reference_root, name), member(root, name)
            if not reference.is_file() or reference.stat().st_size != ref['bytes'] or sha(reference) != ref['sha256']:
                raise ValueError('Reference archive member differs from authenticated manifest')
            current_sha = sha(current)
            record = {'sha256': current_sha, 'bytes': current.stat().st_size,
                      'reference_sha256': ref['sha256'], 'reference_bytes': ref['bytes'],
                      'physical_bytes_equal': current_sha == ref['sha256'], 'equivalent': False}
            files[name] = record
            if is_metadata_json(name):
                pending.append(name)
            else:
                equal, mode, info = compare_content(reference, current, name, ref['sha256'], current_sha)
                record.update(equivalent=bool(equal), comparison_mode=mode, details=info)
                if not equal:
                    mismatches.append({'file': name, 'reason': mode, 'details': info})
        except Exception as exc:
            mismatches.append({'file': name, 'reason': f'{type(exc).__name__}: {exc}'})
    # A manifest may refer to another newly verified metadata file; repeated
    # passes authorize each hash substitution only after the referent passes.
    last_errors = {}
    while pending:
        next_pending, progress = [], False
        for name in pending:
            try:
                equal, changes = compare_metadata(member(reference_root, name), member(root, name), name, root, files, entries)
                if equal:
                    files[name].update(equivalent=True, comparison_mode='metadata_exact_with_declared_relocations', normalizations=changes)
                    progress = True
                else:
                    next_pending.append(name); last_errors[name] = 'Unpermitted metadata content difference'
            except Exception as exc:
                next_pending.append(name); last_errors[name] = f'{type(exc).__name__}: {exc}'
        if not progress:
            for name in next_pending:
                files[name].update(equivalent=False, comparison_mode='metadata_exact_with_declared_relocations')
                mismatches.append({'file': name, 'reason': last_errors[name]})
            break
        pending = next_pending
    # The accepted snapshot's newer README is intentional documentation history,
    # not an assertion that the old and new README contents are equivalent.
    documentation = {}
    current_search = None
    try:
        current = member(root, 'metadata/nlp_synthetic_README.md')
        expected = entries[CODE_NLP + 'README.md']['sha256']
        documentation = {'file': 'metadata/nlp_synthetic_README.md', 'sha256': sha(current),
                         'historical_document_sha256': entries['metadata/nlp_synthetic_README.md']['sha256'],
                         'accepted_later_code_document_sha256': expected,
                         'equals_later_accepted_document': sha(current) == expected,
                         'equals_historical_document': sha(current) == entries['metadata/nlp_synthetic_README.md']['sha256']}
        if not documentation['equals_later_accepted_document']:
            mismatches.append({'file': documentation['file'], 'reason': 'New README does not match accepted later code snapshot'})
    except Exception as exc:
        mismatches.append({'file': 'metadata/nlp_synthetic_README.md', 'reason': str(exc)})
    gates, gate_issues = current_gates(root)
    token_cells, token_issues = current_token_checks(root)
    gates['all_33_token_cells'] = len(token_cells) == 33 and not token_issues
    # Search files are work products, not copied archive data. Validate that the
    # freshly produced exclusion/audit reports actually bind their current TSV.
    try:
        current_search = sha(member(root, 'work/protein/remote_exclusion/remote_vs_swissprot.tsv'))
        for filename in ('metadata/protein_remote_pretraining_exclusion.json', 'validation/cross_dataset_homology.json'):
            if loads(member(root, filename).read_text())['search_output_sha256'] != current_search:
                raise ValueError(filename + ' does not bind the current search TSV')
    except Exception as exc:
        gate_issues.append('Current search provenance: ' + str(exc))
    bootstrap_info = {}
    try:
        bootstrap_info = validate_bootstrap(root, archive, entries)
    except Exception as exc:
        gate_issues.append('Bootstrap provenance: ' + str(exc))
    pins = {
        'verification_protocol': protocol_info,
        'verifier': {'file': str(Path(__file__)), 'sha256': verifier_initial_hash},
        'verifier_bootstrap_dependency': {'file': str(bootstrap_module), 'sha256': bootstrap_module_initial_hash},
        'bootstrap': {'file': str(member(root, 'portable_validation/bootstrap.json')), 'sha256': initial_bootstrap_hash},
        'archive_manifest': {'file': str(archive / 'archive_manifest.json'), 'sha256': ARCHIVE_SHA},
        'reference_release_manifest': {'file': str(archive / PREFIX / 'manifest.json'), 'sha256': V2_SHA},
    }
    if documentation.get('sha256'):
        pins['current_documentation'] = {'file': str(root / documentation['file']), 'sha256': documentation['sha256']}
    if current_search is not None:
        pins['current_search_tsv'] = {'file': str(root / 'work/protein/remote_exclusion/remote_vs_swissprot.tsv'), 'sha256': current_search}
    final_integrity = final_recheck(root, files, pins)
    mismatches.extend(final_integrity['issues'])
    final_scientific = {p.relative_to(root).as_posix() for prefix in ('data', 'tokenized', 'tokenizers')
                        for p in (root / prefix).rglob('*') if p.is_file()}
    for name in sorted(final_scientific ^ expected_scientific):
        mismatches.append({'file': name, 'reason': 'Final scientific inventory differs from authenticated inventory'})
    accepted = not mismatches and not gate_issues and not token_issues and all(gates.values())
    coverage = {
        'ordered_data_files': [n for n in selected if n.startswith('data/')],
        'integer_token_bins': [n for n in selected if n.startswith('tokenized/') and n.endswith('.bin')],
        'ordered_record_indexes': [n for n in selected if n.startswith('tokenized/') and n.endswith('.index.jsonl')],
        'all_tokenizer_files': [n for n in selected if n.startswith('tokenizers/')],
        'raw_tool_external_tokenizer_assets': [n for n in selected if n.startswith(RAW_PREFIXES)],
        'fixed_configs': [n for n in selected if n.startswith('configs/')],
    }
    report = {'schema_version': 1, 'status': 'accepted_for_derived_reconstruction' if accepted else 'reconstruction_rejected',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'training_enabled': False, 'target_scoring_enabled': False,
              'reference_archive_manifest_sha256': ARCHIVE_SHA, 'reference_release_manifest_sha256': V2_SHA,
              'release_root': str(root), 'reference_archive': str(archive), 'verifier_sha256': verifier_initial_hash,
              'verification_protocol': protocol_info, 'final_integrity_recheck': final_integrity,
              'verifier_dependency_sha256': {'bootstrap.py': bootstrap_module_initial_hash},
              'files': files, 'compared_file_count': len(files), 'equivalent_file_count': sum(v['equivalent'] for v in files.values()),
              'scientific_coverage': coverage,
              'physical_difference_files': [k for k, v in files.items() if not v['physical_bytes_equal']],
              'mismatches': mismatches, 'current_gates': gates, 'gate_issues': gate_issues, 'token_issues': token_issues,
              'token_cells': token_cells, 'bootstrap': bootstrap_info, 'documentation_history': documentation,
              'excluded_historical_files': EXCLUDED_REFERENCE,
              'comparison_policy': {'data_order': 'All data rows, columns, values and token IDs are compared in order; no sorting of rows or tolerance.',
                                    'json_objects': 'Keys canonicalized; array order, numeric types and all values remain exact.',
                                    'gzip': 'Container hashes retained; ordered decompressed content must match.',
                                    'metadata': 'Only explicit execution times, known root relocation, pinned later NLP documentation/code inventory, and hashes/physical bytes of already verified referents are mapped.',
                                    'scope': 'Exact scientific-content reconstruction; copied historical acquisition evidence stays exact. This gate does not authorize training or target scoring.'}}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also guards another verifier racing to accept this root.
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False); stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--release-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verification-protocol', type=Path, required=True)
    args = parser.parse_args()
    report = verify_release(args.archive, args.release_root, args.output, args.verification_protocol)
    print(json.dumps({key: report[key] for key in ('status', 'compared_file_count', 'equivalent_file_count', 'training_enabled', 'target_scoring_enabled')}))
    if report['status'] != 'accepted_for_derived_reconstruction':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
