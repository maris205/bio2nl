"""Relocatable CPU preparation with new provenance and an independent data gate.

The immutable study algorithms are vendored. Historical acceptance files are
neither required nor manufactured. This module never launches model training.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from vendor import algorithms as alg
from vendor import data_build as build_alg
from vendor.common import RAW_MEMBERS, inside, now, read_json, require, require_regular, rows, sha, write_json

HERE = Path(__file__).resolve().parent


def descriptor(path):
    return {'bytes': path.stat().st_size, 'sha256': sha(path)}


def validate_lock(lock):
    require(lock.get('artifact_type') == 'bio2nl_portable_preparation_resource_lock_v1', 'Wrong resource lock')
    require(lock.get('schema_version') == 1, 'Unsupported resource lock schema')
    require(set(lock['raw_members']) == set(RAW_MEMBERS), 'Raw input allowlist changed')
    require(lock['training_enabled'] is False and lock['target_scoring_enabled'] is False, 'Resource scope changed')
    require(len(lock['expected_scientific_outputs']) == 22, 'Scientific output lock changed')
    for member, value in {**lock['raw_members'], **lock['expected_scientific_outputs']}.items():
        p = Path(member)
        require(member and not p.is_absolute() and '..' not in p.parts and '.' not in p.parts, 'Unsafe resource member')
        require(type(value['bytes']) is int and value['bytes'] > 0, 'Invalid resource size')
        require(len(value['sha256']) == 64 and all(c in '0123456789abcdef' for c in value['sha256']), 'Invalid SHA256')
    expected_data = {'seeds': [0, 1, 2], 'token_budget_per_stream': 8388608,
                     'source': {'expected_raw_train_rows': 99818, 'expected_train_groups': 63,
                                'expected_train_rows': 8044, 'expected_validation_rows': 20276,
                                'selection_seed': 20260925, 'target_rows': 8000}}
    require(lock['data'] == expected_data, 'Original study design changed')
    return lock


def verify_inputs(raw_root, lock):
    root = Path(raw_root).absolute()
    require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)), 'Regular raw root required')
    verified = {}
    for member, expected in lock['raw_members'].items():
        path = require_regular(inside(root, member))
        require(descriptor(path) == expected, 'Raw input identity mismatch: ' + member)
        verified[member] = expected
    return root, verified


def load_resources(lock_path=None):
    path = require_regular(Path(lock_path or HERE / 'resource_lock.json').absolute())
    lock = validate_lock(read_json(path))
    for member, expected in lock['source_provenance']['vendor'].items():
        current = require_regular(inside(HERE / 'vendor', member))
        require(descriptor(current) == {k: expected[k] for k in ('bytes', 'sha256')}, 'Vendored algorithm changed: ' + member)
    projection_path = require_regular(HERE / 'historical_qualification_projection.json')
    projection = read_json(projection_path)
    require(projection['artifact_type'] == 'historical_qualification_projection_not_current_acceptance', 'Wrong historical projection')
    require(projection['projection_is_training_gate'] is False and projection['target_scoring_enabled'] is False,
            'Historical projection cannot grant execution')
    require(projection['historical_evidence_hashes'] == lock['source_provenance']['historical_metadata'], 'Historical metadata hashes differ')
    return path, lock, projection_path, projection


class PortableStore:
    """Only verified raw-root members may be consumed by vendored algorithms."""
    def __init__(self, root, output, lock):
        self.root, self.output, self.lock = root, output, lock
        self.protocol = {'data': lock['data']}
        self.read_members = set()

    def raw(self, member):
        require(member in self.lock['raw_members'], 'Input outside fixed readset')
        self.read_members.add(member)
        return inside(self.root, member)


def build(raw_root, output, lock_path=None):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Set CUDA_VISIBLE_DEVICES="" for CPU-only preparation')
    from tokenizers import Tokenizer
    import tokenizers
    started = time.monotonic()
    lock_path, lock, projection_path, projection = load_resources(lock_path)
    root, verified = verify_inputs(raw_root, lock)
    out = Path(output).absolute()
    require(not out.exists() and not any(p.is_symlink() for p in (out, *out.parents)), 'Fresh regular output required')
    require(not out.is_relative_to(root) and not root.is_relative_to(out), 'Input/output roots must be disjoint')
    require(not HERE.is_relative_to(out), 'Output may not contain the implementation')
    out.mkdir(parents=True)
    protocol = {'schema_version': 1, 'artifact_type': 'bio2nl_portable_preparation_protocol_v1',
                'created_at_utc': now(), 'raw_root': str(root), 'output_root': str(out),
                'resource_lock_sha256': sha(lock_path), 'historical_projection_sha256': sha(projection_path),
                'data': lock['data'], 'source_test_examples_allowed': False,
                'training_enabled': False, 'target_scoring_enabled': False,
                'code_pins': {str(p.relative_to(HERE)): descriptor(p) for p in sorted(HERE.rglob('*.py')) if '__pycache__' not in p.parts},
                'input_member_hashes': verified}
    write_json(out / 'protocol.json', protocol)
    store = PortableStore(root, out, lock)
    # Explicitly account for all immutable metadata members as well as payloads.
    for member in RAW_MEMBERS:
        store.raw(member)
    tokenizer_path = store.raw('tokenizers/mixed_bpe/tokenizer.json')
    tokenizer = Tokenizer.from_file(str(tokenizer_path)); tokenizer.no_padding(); tokenizer.no_truncation()
    require(tokenizer.get_vocab_size() == 32000, 'Wrong vocabulary size')
    require([tokenizer.token_to_id(s) for s in ('<|pad_v2|>', '<|eos_v2|>', '<|pair_v2|>', '<|unk_v2|>')] == [0, 1, 2, 3], 'Wrong special IDs')
    canonical, by_sequence, clusters = alg.read_canonical_metadata(store.raw('data/sequences/protein_canonical_sequences.tsv.gz'))
    remote_hashes, remote_count = alg.remote_hashes(store.raw('data/sequences/remote_sequences.tsv.gz'))
    canonical_forbidden = {k for k, v in by_sequence.items() if v['split'] != 'train'}
    forbidden = canonical_forbidden | remote_hashes
    split_policy = read_json(store.raw('metadata/protein_split_policy.json'))
    exclusion = read_json(store.raw('metadata/protein_remote_pretraining_exclusion.json'))
    pair_gate = read_json(store.raw('validation/protein_pair_acceptance.json'))
    require(pair_gate['acceptance_pass'] and exclusion['status'] == 'completed', 'Pinned raw source/exclusion evidence failed')
    require(len(clusters) == split_policy['cluster_count'], 'Cluster count changed')
    training_records = read_json(store.raw('tokenizers/mixed_bpe/training_records.json'))
    require(all(p in canonical and canonical[p]['split'] == 'train' and canonical[p]['eligible'] for p in training_records['protein']), 'Tokenizer protein membership differs')
    for member, binding in projection['reference_bindings'].items():
        require(verified[member]['sha256'] == binding['current_sha256'], 'Historical exposure binding mismatch')
    current_projection = {**projection, 'new_preparation_projection_sha256': sha(projection_path),
                          'current_exposure_input_hashes_verified': True, 'target_examples_opened': False,
                          'historical_metadata_files_opened': False}
    write_json(out / 'confirmation_metadata_inheritance.json', current_projection)
    write_json(out / 'build_policy.json', {'artifact_type': 'portable_source_preparation_policy_v1',
        'created_at_utc': now(), 'protocol_sha256': sha(out / 'protocol.json'),
        'source_selection': lock['data']['source'], 'immutable_algorithm_provenance': lock['source_provenance']['vendor'],
        'source_test_pair_examples_read': False, 'target_examples_read': False,
        'identity_table_bytes_read_for_exclusion': True, 'model_training_or_inference': False,
        'historical_acceptance_copied_or_manufactured': False,
        'equal_residue_exposure_claimed': False, 'public_raw_rebuild_claimed': False})
    source, selection, source_overlap, prefix = build_alg.build_source(store, tokenizer, by_sequence)
    print('Source prepared: 8044 train / 20276 validation', flush=True)
    inheritance = {'source_train_validation_reencoded': True, 'source_train_validation_overlap': source_overlap,
        'source_test_pair_examples_or_labels_read': False, 'operational_cluster_definition': split_policy['definition'],
        'near_homology_exclusion': exclusion['criterion'], 'near_homology_search_rerun_in_this_stage': False,
        'search_limitations': split_policy['search_limitations'], 'structural_family_overlap': None,
        'structural_family_availability': 'not_provided; unmeasured, not zero',
        'canonical_nontrain_hashes': len(canonical_forbidden), 'remote_domain_rows': remote_count,
        'remote_all_split_hashes': len(remote_hashes), 'combined_exact_exclusion_hashes': len(forbidden),
        'tokenizer_protein_parents_all_train_eligible': True, 'tokenizer_protein_training_records': len(training_records['protein']),
        'identity_tables_physically_read': True, 'heldout_pair_files_opened': False}
    write_json(out / 'source_identity_and_exclusions.json', inheritance)
    streams, schedules, parent_file, parent_report, coverage = build_alg.build_streams(store, tokenizer, canonical, forbidden)
    require(store.read_members == set(RAW_MEMBERS), 'Readset incomplete')
    # Close input mutation races before creating the immutable build manifest.
    verify_inputs(root, lock)
    (out / 'tokenizer').mkdir()
    with tokenizer_path.open('rb') as src, (out / 'tokenizer/tokenizer.json').open('xb') as dst:
        shutil.copyfileobj(src, dst)
    outputs = {str(p.relative_to(out)): descriptor(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'protocol.json'}
    for split, spec in source.items():
        spec['raw_input'] = 'source/' + split + '.jsonl'
    manifest = {'schema_version': 1, 'artifact_type': 'bio2nl_portable_prepared_inputs_v1',
        'status': 'prepared_inputs_built', 'created_at_utc': now(), 'elapsed_seconds': time.monotonic() - started,
        'training_enabled': False, 'target_scoring_enabled': False, 'source_test_examples_allowed': False,
        'raw_release_id': lock['release_id'], 'input_resource_lock_sha256': sha(lock_path),
        'input_member_hashes': verified, 'protocol_sha256': sha(out / 'protocol.json'), 'outputs': outputs,
        'source': source, 'source_counts': {k: v['count'] for k, v in source.items()},
        'source_selection': selection, 'source_prefix64': prefix, 'smoke_source': prefix,
        'streams': streams, 'schedules': schedules,
        'tokenizer': {'file': 'tokenizer/tokenizer.json', 'sha256': sha(tokenizer_path), 'vocab_size': 32000,
                      'pad_id': 0, 'eos_id': 1, 'sep_id': 2, 'unk_id': 3},
        'condition_streams': {'EP': ['english_common', 'protein'], 'ES': ['english_common', 'shuffled'], 'EE': ['english_common', 'english_extra']},
        'protein_parent_controls': parent_file, 'parent_control_summary': parent_report,
        'parent_prefix_coverage': coverage, 'source_provenance_and_exclusions': inheritance,
        'confirmation_metadata_inheritance': current_projection,
        'read_boundaries': {'source_test_pair_files_opened': 0, 'qqp_example_or_row_audit_files_opened': 0,
                            'old_predictions_opened': 0, 'old_prepared_source_files_opened': 0,
                            'historical_metadata_files_opened': 0, 'identity_tables_used_for_exclusion': True},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'tokenizers': tokenizers.__version__, 'cuda_used': False}}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'manifest_sha256': sha(out / 'manifest.json')}), flush=True)
    return manifest


def verify(output, lock_path=None):
    """Reopen and independently check content hashes, schedules and tensor rows."""
    lock_path, lock, _, _ = load_resources(lock_path)
    root = Path(output).absolute()
    manifest = read_json(require_regular(inside(root, 'manifest.json')))
    require(manifest['artifact_type'] == 'bio2nl_portable_prepared_inputs_v1' and manifest['status'] == 'prepared_inputs_built', 'Wrong prepared manifest')
    require(not manifest['training_enabled'] and not manifest['target_scoring_enabled'] and not manifest['source_test_examples_allowed'], 'Preparation scope changed')
    require(manifest['input_resource_lock_sha256'] == sha(lock_path), 'Resource lock binding changed')
    require(manifest['input_member_hashes'] == lock['raw_members'], 'Input provenance changed')
    protocol_path = require_regular(inside(root, 'protocol.json'))
    require(manifest['protocol_sha256'] == sha(protocol_path), 'Protocol hash changed')
    protocol = read_json(protocol_path)
    require(protocol['data'] == lock['data'] and protocol['resource_lock_sha256'] == sha(lock_path), 'Protocol design changed')
    for member, expected in protocol['code_pins'].items():
        require(descriptor(require_regular(inside(HERE, member))) == expected, 'Preparation code changed')
    expected_names = set(lock['expected_scientific_outputs']) | set(lock['replaced_provenance_outputs']) | {'tokenizer/tokenizer.json'}
    require(set(manifest['outputs']) == expected_names, 'Prepared payload list changed')
    for member, expected in manifest['outputs'].items():
        require(descriptor(require_regular(inside(root, member))) == expected, 'Prepared output changed: ' + member)
    for member, expected in lock['expected_scientific_outputs'].items():
        require(manifest['outputs'][member] == expected, 'Scientific output differs from fixed study: ' + member)
    require(manifest['outputs']['tokenizer/tokenizer.json'] == lock['raw_members']['tokenizers/mixed_bpe/tokenizer.json'], 'Tokenizer changed')
    identities = {}
    checks = {'scientific_payloads_exact': 22, 'output_payloads_rehashed': len(manifest['outputs']), 'source_roles': {}, 'streams': {}, 'schedules': {}}
    for split, n in lock['expected_source'].items():
        spec = manifest['source'][split]
        require(spec['count'] == n and spec['npz'] == f'source/{split}.npz' and spec['rows'] == f'source/{split}.rows.jsonl.gz', 'Source descriptor changed')
        values = list(rows(inside(root, spec['rows'])))
        require(len(values) == n, 'Source row count changed')
        with np.load(inside(root, spec['npz']), allow_pickle=False) as loaded:
            require(set(loaded.files) == {'input_ids', 'attention_mask', 'labels'}, 'Unexpected NPZ arrays')
            ids, masks, labels = (loaded[k] for k in ('input_ids', 'attention_mask', 'labels'))
            require(ids.shape == masks.shape == (n, 512) and labels.shape == (n,), 'Source tensor dimensions differ')
            require(all(x.dtype == np.int64 for x in (ids, masks, labels)), 'Source tensor dtype differs')
            require(set(labels.tolist()) == {0, 1} and int(labels.sum()) == n // 2, 'Source labels not balanced')
            for index, row in enumerate(values):
                sequence = row['ids_a'] + [2] + row['ids_b'] + [1]
                require(row['row_index'] == index and row['label'] == int(labels[index]), 'Source row identity differs')
                require(np.array_equal(ids[index, :len(sequence)], sequence) and not ids[index, len(sequence):].any(), 'Encoded input differs')
                require(np.array_equal(masks[index], np.arange(512) < len(sequence)), 'Padding mask differs')
        identities[split] = {'rows': {r['row_id'] for r in values},
                             'blocks': {r['metadata']['block_id'] for r in values},
                             'sequences': {r['metadata']['sequence_sha256_' + s] for r in values for s in ('a', 'b')},
                             'clusters': {r['metadata']['cluster_' + s] for r in values for s in ('a', 'b')}}
        checks['source_roles'][split] = {'rows': n, 'labels_per_class': n // 2, 'tensor_row_replay': 'exact'}
    checks['source_overlap'] = {k: len(identities['train'][k] & identities['validation'][k]) for k in identities['train']}
    require(not any(checks['source_overlap'].values()), 'Source cross-split identity overlap')
    for name, spec in manifest['streams'].items():
        require(spec['tokens'] == alg.BUDGET and spec['bin'] == f'streams/{name}.bin', 'Stream descriptor changed')
        ids = np.memmap(inside(root, spec['bin']), dtype='<u2', mode='r')
        require(len(ids) == alg.BUDGET and int(ids.min()) >= 1 and int(ids.max()) < 32000 and not np.isin(ids, [2, 3]).any(), 'Stream token range invalid')
        index = list(rows(inside(root, spec['index'])))
        cursor = 0
        for row in index:
            require(row['offset'] == cursor and row['length'] >= 1 and row['eos_offset'] == cursor + row['length'] - 1, 'Stream index discontinuity')
            require(int(ids[row['eos_offset']]) == 1, 'Stream EOS missing')
            cursor += row['length']
        require(cursor == alg.BUDGET and len({r['parent_id'] for r in index}) == len(index), 'Stream coverage or parent uniqueness differs')
        checks['streams'][name] = {'tokens': len(ids), 'records': len(index), 'complete_index': True}
    for seed, spec in manifest['schedules'].items():
        values = np.load(inside(root, spec['file']), allow_pickle=False)
        require(values.shape == (2048, 16, 2) and values.dtype == np.int64, 'Schedule dimensions differ')
        for stream_id in (0, 1):
            selected = values[values[:, :, 0] == stream_id, 1]
            require(np.array_equal(np.sort(selected), np.arange(16384)), 'Schedule is not a complete stream permutation')
        checks['schedules'][seed] = {'complete_per_stream_permutation': True}
    acceptance = {'schema_version': 1, 'artifact_type': 'bio2nl_portable_preparation_acceptance_v1', 'status': 'passed',
                  'passed_at_utc': now(), 'prepared_manifest_sha256': sha(root / 'manifest.json'),
                  'protocol_sha256': sha(protocol_path), 'resource_lock_sha256': sha(lock_path),
                  'scientific_payloads_exact': 22, 'independent_invariant_checks': checks,
                  'training_enabled': False, 'target_scoring_enabled': False,
                  'historical_gate_reused': False, 'fresh_public_raw_download_execution_claimed': False,
                  'new_preparation_only': True}
    write_json(root / 'acceptance.json', acceptance)
    print(json.dumps({'status': 'passed', 'scientific_payloads_exact': 22, 'acceptance_sha256': sha(root / 'acceptance.json')}), flush=True)
    return acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('verify-inputs', 'build', 'verify'):
        p = sub.add_parser(name)
        p.add_argument('--resource-lock', type=Path)
        if name != 'verify': p.add_argument('--raw-root', required=True, type=Path)
        if name != 'verify-inputs': p.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'verify-inputs':
        _, lock, _, _ = load_resources(args.resource_lock)
        _, verified = verify_inputs(args.raw_root, lock)
        print(json.dumps({'status': 'passed', 'verified_inputs': len(verified)}))
    elif args.command == 'build': build(args.raw_root, args.output, args.resource_lock)
    else: verify(args.output, args.resource_lock)


if __name__ == '__main__':
    main()
