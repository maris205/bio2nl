"""Build new source/mixed-stream inputs; no target, source-test pair, or model reads."""
import argparse
from collections import Counter
import csv
import gzip
import json
import os
import sys
from pathlib import Path
import platform
import time
import numpy as np
try:
    from .common import Store, RAW_MEMBERS, now, read_json, require, rows, sha, text_sha, write_json, write_rows
    from . import algorithms as alg
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from common import Store, RAW_MEMBERS, now, read_json, require, rows, sha, text_sha, write_json, write_rows
    import algorithms as alg


def check_design(design, data):
    require(design['pretraining']['seeds'] == data['seeds'] == [0, 1, 2], 'Pretraining seeds changed')
    require(design['pretraining']['input_tokens_per_run'] == 2 * alg.BUDGET and data['token_budget_per_stream'] == alg.BUDGET, 'Token budget changed')
    require(design['pretraining']['context_length'] == 512 and design['pretraining']['updates_per_run'] == 1024, 'Original pretraining design changed')
    s = data['source']
    require(s == {'target_rows': 8000, 'selection_seed': 20260925, 'expected_train_rows': 8044,
                  'expected_train_groups': 63, 'expected_validation_rows': 20276,
                  'expected_raw_train_rows': 99818}, 'Unexpected new source selection protocol')


def inherit_confirmation(store):
    acceptance = store.metadata('confirmation_acceptance')
    manifest = store.metadata('confirmation_manifest')
    references = store.metadata('confirmation_references')
    audit = store.metadata('confirmation_audit')
    require(acceptance['status'] == 'accepted_for_local_confirmation' and acceptance['confirmation_data_ready'] is True, 'Historical qualification metadata failed')
    require(audit['status'] == 'passed' and audit['confirmation_data_qualified'] is True, 'Historical independent qualification failed')
    for key, name in [('confirmation_manifest', 'manifest.json'), ('confirmation_references', 'references.json'), ('confirmation_audit', 'verification/independent_verification.json')]:
        require(store.inputs[key]['sha256'] == acceptance['files'][name], 'Qualification metadata binding mismatch')
    bindings = {}
    for member in ('data/corpora/english/train.jsonl.gz', 'tokenizers/mixed_bpe/tokenizer.json'):
        prior = references['files'][member]
        current = store.acceptance['files'][member]
        old_path = prior['path']
        require(current['reference_kind'] == 'accepted_v2' and current['equivalent_to_declared_reference'] is True, 'Exposure reference lacks accepted semantic equivalence')
        require(current['reference_sha256'] == prior['sha256'] == manifest['inputs'][old_path] == audit['file_sha256'][old_path], 'Exposure reference identity mismatch')
        require(current['sha256'] == store.inputs['raw:' + member]['sha256'], 'Current exposure input mismatch')
        bindings[member] = {'current_sha256': current['sha256'], 'historical_reference_sha256': prior['sha256'],
                            'raw_gate_comparison_mode': current['comparison_mode'], 'same_scientific_content': True}
    return {'acceptance_sha256': store.inputs['confirmation_acceptance']['sha256'],
            'same_full_english_train_pool_and_tokenizer_scientific_content': True,
            'reference_bindings': bindings, 'retained_lexical_hits_zero_inherited_from_M1': True,
            'target_raw_or_encoded_or_row_ledger_files_opened': False,
            'qualification_scope': acceptance['qualification_scope'],
            'historical_target_has_already_been_scored': True, 'new_blind_confirmation_claimed': False,
            'current_target_bytes_verification_deferred_to_separate_scoring_gate': True}


def read_raw_source(path, split):
    require(split in ('train', 'validation'), 'Source-test pair read forbidden')
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        values = list(csv.DictReader(f, delimiter='\t'))
    for row in values:
        row['label'] = int(row['label'])
        row.setdefault('row_id', row['pair_id'])
    audit = alg.audit_split(values, 'protein_sequence_similarity', split)
    return values, audit


def source_identity(values):
    return {'row_ids': {r['row_id'] for r in values},
            'groups': {r['metadata']['block_id'] for r in values},
            'sequences': {r['metadata']['sequence_sha256_' + s] for r in values for s in ('a', 'b')},
            'clusters': {r['metadata']['cluster_' + s] for r in values for s in ('a', 'b')}}


def build_source(store, tokenizer, by_sequence):
    out = store.output; prepared = {}; report = {'policy': store.protocol['data']['source'], 'roles': {}}
    for split, expected in (('train', 8044), ('validation', 20276)):
        raw_path = store.raw(f'data/pairs/protein_sequence_similarity_{split}.tsv.gz')
        values, raw_audit = read_raw_source(raw_path, split)
        selection = None
        if split == 'train':
            require(len(values) == 99818, 'New training population differs')
            values, selection = alg.select_groups(values, 'block_id', 8000, 20260925, 'protein_sequence_similarity')
            require(selection['selected_groups'] == 63, 'New selected block count differs')
        require(len(values) == expected, 'New source row count differs')
        selected_audit = alg.audit_split(values, 'protein_sequence_similarity', split)
        prepared[split] = [alg.prepared_row(row, 'protein_sequence_similarity') for row in values]
        write_rows(out / 'source' / (split + '.jsonl'), prepared[split])
        report['roles'][split] = {'raw_rows': raw_audit['rows'], 'raw_groups': raw_audit['groups'],
                                 'selected_rows': len(values), 'selected_groups': selected_audit['groups'],
                                 'raw_split_audit': raw_audit, 'selected_split_audit': selected_audit,
                                 'selection': selection,
                                 'ordered_row_ids_sha256': text_sha(json.dumps([r['row_id'] for r in prepared[split]], separators=(',', ':')))}
    identities = {split: source_identity(value) for split, value in prepared.items()}
    overlap = {key: len(identities['train'][key] & identities['validation'][key]) for key in identities['train']}
    require(not any(overlap.values()), 'Source raw cross-split overlap')
    report['cross_split_overlap'] = overlap
    report['source_test_pair_examples_read'] = False
    report['target_examples_read'] = False
    write_json(out / 'source_selection.json', report)
    encoded, source_ids, prefix = {}, {}, {}
    for split in ('train', 'validation'):
        encoded[split], source_ids[split] = alg.make_source(split, out / 'source' / (split + '.jsonl'), tokenizer, by_sequence, out / 'source', out)
        ids = [r['row_id'] for r in prepared[split][:64]]
        prefix[split] = {'indices': list(range(64)), 'row_ids': ids, 'rows': 64,
                         'row_ids_sha256': text_sha(json.dumps(ids, separators=(',', ':'))),
                         'purpose': 'bounded technical smoke only; not full-group scientific training subset'}
    encoded_overlap = {key: len(source_ids['train'][key] & source_ids['validation'][key]) for key in source_ids['train']}
    require(not any(encoded_overlap.values()), 'Source encoded cross-split overlap')
    return encoded, report, encoded_overlap, prefix


def build_streams(store, tokenizer, canonical, forbidden):
    out = store.output
    english_path = store.raw('data/corpora/english/train.jsonl.gz')
    writers = {name: alg.StreamWriter(out, out / 'streams', name, tokenizer) for name in ('english_common', 'english_extra')}
    seen_ids, seen_text = set(), set()
    for position, row in enumerate(rows(english_path)):
        parent, text = row['record_id'], row['text']
        require(row['split'] == 'train' and parent not in seen_ids and row['text_sha256'] not in seen_text, 'English repeats or wrong split')
        require(alg.text_sha(text) == row['text_sha256'], 'English text identity differs')
        seen_ids.add(parent); seen_text.add(row['text_sha256'])
        name = 'english_extra' if writers['english_common'].complete else 'english_common'
        writers[name].append(parent, position, text, alg.encode_content(tokenizer, text))
        if writers['english_extra'].complete:
            break
    streams = {name: writer.finish() for name, writer in writers.items()}
    require(not {r['parent_id'] for r in writers['english_common'].index_rows} & {r['parent_id'] for r in writers['english_extra'].index_rows}, 'English halves share parent IDs')
    protein_path = store.raw('data/corpora/protein/train.jsonl.gz')
    entries, parent_file, parent_report = alg.build_protein(list(rows(protein_path)), canonical, forbidden, tokenizer, out, out)
    streams.update(entries)
    write_json(out / 'protein_control_audit.json', parent_report)
    schedules = {}
    (out / 'schedules').mkdir()
    for seed in (0, 1, 2):
        path = out / 'schedules' / f'seed{seed}.npy'
        np.save(path, alg.make_schedule(seed), allow_pickle=False)
        schedules[str(seed)] = {'file': str(path.relative_to(out)), 'sha256': sha(path), 'shape': [2048, 16, 2], 'dtype': 'int64'}
    coverage = {}
    for name in ('protein', 'shuffled'):
        index = list(rows(out / streams[name]['index']))
        coverage[name] = {'used': {r['parent_id'] for r in index}, 'full': {r['parent_id'] for r in index if not r['truncated']},
                          'exposed_residues': sum(r['exposed_characters'] for r in index)}
    coverage_report = {'same_eligible_parent_pool_order': True,
        'used_parent_intersection': len(coverage['protein']['used'] & coverage['shuffled']['used']),
        'complete_parent_intersection': len(coverage['protein']['full'] & coverage['shuffled']['full']),
        'protein_only_used_parents': len(coverage['protein']['used'] - coverage['shuffled']['used']),
        'shuffled_only_used_parents': len(coverage['shuffled']['used'] - coverage['protein']['used']),
        'exposed_residues': {k: v['exposed_residues'] for k, v in coverage.items()}, 'equal_residue_exposure_claimed': False}
    write_json(out / 'parent_prefix_coverage.json', coverage_report)
    return streams, schedules, parent_file, parent_report, coverage_report


def build(protocol_path, protocol_sha256, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Data construction requires CUDA_VISIBLE_DEVICES empty')
    from tokenizers import Tokenizer
    started = time.monotonic(); store = Store(protocol_path, protocol_sha256, output)
    require(str(Path(__file__).resolve()) in store.protocol['code_pins'] and str(Path(alg.__file__).resolve()) in store.protocol['code_pins'], 'Executed builder/algorithm module not pinned')
    design = store.metadata('design'); check_design(design, store.protocol['data'])
    for member in RAW_MEMBERS:
        store.raw(member)
    tokenizer_path = store.raw('tokenizers/mixed_bpe/tokenizer.json')
    tokenizer = Tokenizer.from_file(str(tokenizer_path)); tokenizer.no_padding(); tokenizer.no_truncation()
    require(tokenizer.get_vocab_size() == 32000, 'Wrong tokenizer vocabulary size')
    require([tokenizer.token_to_id(x) for x in ('<|pad_v2|>', '<|eos_v2|>', '<|pair_v2|>', '<|unk_v2|>')] == [0, 1, 2, 3], 'Wrong tokenizer special IDs')
    canonical, by_sequence, clusters = alg.read_canonical_metadata(store.raw('data/sequences/protein_canonical_sequences.tsv.gz'))
    remote_hashes, remote_count = alg.remote_hashes(store.raw('data/sequences/remote_sequences.tsv.gz'))
    canonical_forbidden = {key for key, value in by_sequence.items() if value['split'] != 'train'}
    forbidden = canonical_forbidden | remote_hashes
    split_policy = read_json(store.raw('metadata/protein_split_policy.json'))
    exclusion = read_json(store.raw('metadata/protein_remote_pretraining_exclusion.json'))
    pair_gate = read_json(store.raw('validation/protein_pair_acceptance.json'))
    require(pair_gate['acceptance_pass'] and exclusion['status'] == 'completed', 'Raw source/exclusion gate failed')
    require(len(clusters) == split_policy['cluster_count'], 'Canonical cluster count differs')
    training_records = read_json(store.raw('tokenizers/mixed_bpe/training_records.json'))
    require(all(p in canonical and canonical[p]['split'] == 'train' and canonical[p]['eligible'] for p in training_records['protein']), 'Tokenizer protein training membership differs')
    confirmation = inherit_confirmation(store)
    policy = {'status': 'fixed_before_control_generation', 'created_at_utc': now(), 'data_protocol_sha256': protocol_sha256,
        'selection': store.protocol['data']['source'], 'shuffle_seed_derivation': "int(SHA256('joint-v1:shuffle:0:' + parent_id), 16)",
        'shuffle': 'Python random.Random(full_256_bit_integer).shuffle; one draw per parent; no retries',
        'collision_scope': 'canonical nontraining sequence hashes union every accepted remote sequence hash',
        'duplicate_control_policy': 'Exclude every involved parent from both natural and shuffled eligible pools',
        'unchanged_control_policy': 'Retain unless another fixed exclusion applies',
        'schedule_seed_derivation': "int(SHA256(f'joint-v1:schedule:{seed}:{stream_id}'),16)",
        'schedule_shape': [2048,16,2], 'same_schedule_file_all_conditions': True,
        'english_extra_start': 'Record immediately after final common-English parent; remainder of previous parent discarded',
        'source_test_pair_examples_read': False, 'target_examples_read': False,
        'identity_table_bytes_read_for_exclusion': True, 'model_training_or_inference': False}
    write_json(store.output / 'build_policy.json', policy)
    write_json(store.output / 'confirmation_metadata_inheritance.json', confirmation)
    source, selection, source_overlap, prefix = build_source(store, tokenizer, by_sequence)
    print('New source train/validation constructed and encoded: 8044 / 20276', flush=True)
    inheritance = {'source_train_validation_reencoded': True, 'source_train_validation_overlap': source_overlap,
        'source_test_pair_examples_or_labels_read': False, 'operational_cluster_definition': split_policy['definition'],
        'near_homology_exclusion': exclusion['criterion'], 'near_homology_search_rerun_in_this_stage': False,
        'search_limitations': split_policy['search_limitations'], 'structural_family_overlap': None,
        'structural_family_availability': 'not_provided; unmeasured, not zero',
        'canonical_nontrain_hashes': len(canonical_forbidden), 'remote_domain_rows': remote_count,
        'remote_all_split_hashes': len(remote_hashes), 'combined_exact_exclusion_hashes': len(forbidden),
        'tokenizer_protein_parents_all_train_eligible': True, 'tokenizer_protein_training_records': len(training_records['protein']),
        'identity_tables_physically_read': True, 'heldout_pair_files_opened': False}
    write_json(store.output / 'source_identity_and_exclusions.json', inheritance)
    streams, schedules, parent_file, parent_report, coverage = build_streams(store, tokenizer, canonical, forbidden)
    store.finish_readset()
    outputs = {str(p.relative_to(store.output)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(store.output.rglob('*')) if p.is_file()}
    manifest = {'schema_version': 1, 'status': 'built_pending_independent_audit', 'training_ready': False,
        'training_enabled': False, 'target_scoring_enabled': False,
        'created_at_utc': now(), 'elapsed_seconds': time.monotonic() - started,
        'raw_release_id': store.protocol['release_id'], 'raw_release_root': str(store.root),
        'raw_acceptance_sha256': store.protocol['raw_acceptance']['sha256'],
        'raw_manifest_sha256': store.protocol['raw_manifest']['sha256'],
        'build_protocol_sha256': protocol_sha256, 'data_protocol_sha256': protocol_sha256,
        'inputs': store.inputs, 'outputs': outputs,
        'source': source, 'source_counts': {k: v['count'] for k, v in source.items()},
        'source_selection': selection, 'source_prefix64': prefix, 'smoke_source': prefix,
        'streams': streams, 'schedules': schedules,
        'tokenizer': {'file': str(tokenizer_path), 'sha256': sha(tokenizer_path), 'vocab_size': 32000, 'pad_id': 0, 'eos_id': 1, 'sep_id': 2, 'unk_id': 3},
        'condition_streams': {'EP': ['english_common','protein'], 'ES': ['english_common','shuffled'], 'EE': ['english_common','english_extra']},
        'protein_parent_controls': parent_file, 'parent_control_summary': parent_report,
        'parent_prefix_coverage': coverage, 'source_provenance_and_exclusions': inheritance,
        'confirmation_metadata_inheritance': confirmation,
        'read_boundaries': {'source_test_pair_files_opened': 0, 'qqp_example_or_row_audit_files_opened': 0, 'old_predictions_opened': 0, 'old_prepared_source_files_opened': 0, 'identity_tables_used_for_exclusion': True},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'cuda_used': False}}
    write_json(store.output / 'manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'source_counts': manifest['source_counts'], 'manifest_sha256': sha(store.output / 'manifest.json')}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    build(args.protocol, args.protocol_sha256, args.output)


if __name__ == '__main__':
    main()
