"""Prepare the two fixed held-out roles only after the source-selection barrier."""
from __future__ import annotations

import gzip
import hashlib
import csv
import re
import unicodedata
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from common import atomic, check_source_gate, read, require, sha, M1, NEW_RAW, NEW_PREPARED, COUNTS

ROOT = Path(__file__).resolve().parent
TOKENIZER_SHA = '1bb8092348368961ca211cbf4d066a0e0ebf612b35b401b07ed173476088130b'
EXPECTED_COUNTS = COUNTS
EXPECTED_LABELS = {'source_test': {'0': 10427, '1': 10427}, 'target': {'0': 25227, '1': 14666}}
TARGET_GROUPS = 33128


def records(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def write_records(path, rows):
    require(not path.exists(), 'Refusing to overwrite row metadata')
    with path.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0, filename='') as zipped:
            for row in rows:
                zipped.write((json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n').encode())


def encode_content(tokenizer, text):
    require(isinstance(text, str) and len(text) > 0, 'Empty or invalid endpoint')
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    require(len(ids) > 0, 'Empty tokenization')
    require(all(type(i) is int and 4 <= i < 32000 for i in ids), 'Unknown/special/out-of-range content token')
    require(tokenizer.decode(ids, skip_special_tokens=False) == text, 'Raw text roundtrip failure')
    return ids


def pack(a, b):
    require(bool(a) and bool(b), 'Empty encoded endpoint')
    require(all(type(i) is int and 4 <= i < 32000 for i in a + b), 'Invalid content token')
    ids = np.zeros(512, dtype=np.int64)
    mask = np.zeros(512, dtype=np.int64)
    joined = a[:255] + [2] + b[:255] + [1]
    ids[:len(joined)] = joined
    mask[:len(joined)] = 1
    return ids, mask


def validate_label(value):
    require(type(value) is int and value in (0, 1), 'Label must be unchanged integer 0 or 1')
    return value


def encode_role(role, raw_rows, tokenizer, folder):
    require(role in EXPECTED_COUNTS, 'Unknown held-out role')
    n = len(raw_rows)
    require(n == EXPECTED_COUNTS[role], 'Fixed row count changed')
    ids = np.zeros((n, 512), dtype=np.int64)
    mask = np.zeros((n, 512), dtype=np.int64)
    labels = np.empty(n, dtype=np.int64)
    encoded, row_ids, pair_labels = [], set(), {}
    labels_count, groups, degree, block_degree = Counter(), Counter(), defaultdict(Counter), defaultdict(Counter)
    cache, lengths, truncated, same_label_collisions = {}, [], 0, 0
    unordered_labels, target_digest = {}, hashlib.sha256()
    for i, raw in enumerate(raw_rows):
        label = validate_label(raw['label'])
        fields = ('sentence1', 'sentence2') if role == 'source_test' else ('text_a', 'text_b')
        texts = [raw[field] for field in fields]
        for text in texts:
            if text not in cache:
                cache[text] = encode_content(tokenizer, text)
        a_full, b_full = [cache[text] for text in texts]
        lengths.extend((len(a_full), len(b_full)))
        truncated += sum(len(x) > 255 for x in (a_full, b_full))
        a, b = a_full[:255], b_full[:255]
        rid = raw['row_id'] if role == 'source_test' else raw['id']
        require(isinstance(rid, str) and rid and rid not in row_ids, 'Missing/duplicate row ID')
        row_ids.add(rid)
        if role == 'source_test':
            metadata = raw['metadata']
            require(metadata['split'] == 'test', 'Wrong source split')
            group = metadata['block_id']
            for side, text, content in zip(('a', 'b'), texts, (a, b)):
                require(hashlib.sha256(text.encode()).hexdigest() == metadata['sequence_sha256_' + side], 'Protein sequence identity mismatch')
                degree[(side, tuple(content))][label] += 1
                block_degree[(group, side, tuple(content))][label] += 1
        else:
            canonical_update(target_digest, {'id': rid, 'label': label, 'component_id': raw['component_id'], 'ids_a': a_full, 'ids_b': b_full})
            group = raw['component_id']
            metadata = {key: raw[key] for key in ('upstream_idx', 'endpoint_a_sha256', 'endpoint_b_sha256', 'component_id')}
        require(isinstance(group, str) and bool(group), 'Missing group identity')
        ids[i], mask[i] = pack(a_full, b_full)
        labels[i] = label
        key = tuple(ids[i, :int(mask[i].sum())].tolist())
        if key in pair_labels:
            require(pair_labels[key] == label, 'Conflicting encoded-pair labels')
            same_label_collisions += 1
        pair_labels[key] = label
        unordered = tuple(sorted((tuple(a), tuple(b))))
        require(unordered not in unordered_labels or unordered_labels[unordered] == label, "Conflicting unordered encoded-pair labels")
        unordered_labels[unordered] = label
        labels_count[str(label)] += 1
        groups[group] += 1
        row = {'row_index': i, 'row_id': rid, 'label': label, 'ids_a': a, 'ids_b': b,
               'group_id': group, 'metadata': metadata, 'raw_lengths': [len(t) for t in texts],
               'precap_token_lengths': [len(a_full), len(b_full)]}
        if role == 'target':
            row['component_id'] = group
        encoded.append(row)
    require(truncated == 0, 'Unexpected truncation: retain rows and stop for review')
    if role == 'source_test':
        require(dict(labels_count) == EXPECTED_LABELS[role], 'Source class counts differ')
        require(all(x[0] == x[1] for x in degree.values()), 'Source endpoint role-label imbalance')
        require(all(x[0] == x[1] for x in block_degree.values()), 'Source within-block role-label imbalance')
    else:
        require(dict(labels_count) == EXPECTED_LABELS[role] and len(groups) == TARGET_GROUPS, 'M1 retained population changed')
    npz, rowfile = folder / (role + '.npz'), folder / (role + '.rows.jsonl.gz')
    require(not npz.exists(), 'Refusing to overwrite encoded tensors')
    np.savez_compressed(npz, input_ids=ids, attention_mask=mask, labels=labels)
    write_records(rowfile, encoded)
    return {'npz': str(npz.relative_to(folder.parent)), 'rows': str(rowfile.relative_to(folder.parent)), 'count': n,
            'npz_sha256': sha(npz), 'rows_sha256': sha(rowfile), 'labels': dict(sorted(labels_count.items())),
            'groups': len(groups), 'largest_group_rows': max(groups.values()),
            'shape': [n, 512], 'dtype': 'int64', 'endpoint_content_tokens_min': min(lengths),
            'endpoint_content_tokens_max': max(lengths), 'truncated_endpoints': truncated,
            'unknown_or_special_content_tokens': 0, 'roundtrip_failures': 0,
            'same_label_encoded_pair_collisions': same_label_collisions, 'conflicting_encoded_pair_labels': 0,
            'row_order_and_labels_preserved': True,
            'source_endpoint_role_and_within_block_balance': True if role == 'source_test' else None,
            'previously_observed_evaluation_split': role == 'target',
            'historical_qualified_encoding_content_sha256': target_digest.hexdigest() if role == 'target' else None,
            'qualification_scope': 'new rebuilt operational protein similarity test; same upstream biological corpus' if role == 'source_test'
                                   else 'M1 fixed-rule local QQP confirmation; no global/semantic unseen guarantee'}


def canonical_update(digest, row):
    digest.update((json.dumps(row, sort_keys=True, ensure_ascii=False) + '\n').encode('utf-8'))


def normalized(text):
    if not isinstance(text, str):
        return ''
    return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text).casefold()))


def reconstruct_target(raw, ledger, expected_digest, expected_raw_count=40430):
    """Replay fixed membership, never re-qualify/filter based on current data or scores."""
    require(len(raw) == len(ledger) == expected_raw_count, 'Raw/qualification ledger count differs')
    require(all(set(x) == {'idx', 'question1', 'question2', 'label'} for x in raw), 'QQP raw schema differs')
    require(all(type(x['idx']) is int and x['idx'] >= 0 for x in raw), 'Invalid upstream index')
    require(len({x['idx'] for x in raw}) == len(raw), 'Duplicate raw index')
    raw = sorted(raw, key=lambda x: x['idx'])
    parents = {}
    def find(a):
        parents.setdefault(a, a)
        while a != parents[a]:
            parents[a] = parents[parents[a]]
            a = parents[a]
        return a
    for row in raw:
        validate_label(row['label'])
        a, b = normalized(row['question1']), normalized(row['question2'])
        if a and b:
            x, y = find(a), find(b)
            parents[max(x, y)] = min(x, y)
    output, digest = [], hashlib.sha256()
    for row, fixed in zip(raw, ledger):
        require(fixed['upstream_idx'] == row['idx'] and type(fixed['upstream_idx']) is int, 'Fixed ledger order differs')
        require(type(fixed['retained']) is bool and isinstance(fixed['reasons'], list), 'Fixed membership schema differs')
        a, b = normalized(row['question1']), normalized(row['question2'])
        cid = hashlib.sha256(find(a).encode()).hexdigest() if a and b else None
        require(fixed['component_id'] == cid, 'Fixed component reconstruction differs')
        require(fixed['retained'] == (not fixed['reasons']), 'Fixed retained/reason state inconsistent')
        if fixed['retained']:
            require(bool(a) and bool(b), 'Fixed retained endpoint is invalid')
            item = dict(id=f"qqp_validation:{row['idx']}", upstream_idx=row['idx'], label=row['label'],
                        text_a=row['question1'], text_b=row['question2'],
                        endpoint_a_sha256=hashlib.sha256(a.encode()).hexdigest(),
                        endpoint_b_sha256=hashlib.sha256(b.encode()).hexdigest(), component_id=cid)
            canonical_update(digest, item)
            output.append(item)
    require(digest.hexdigest() == expected_digest, 'Fixed qualified raw-row content hash differs')
    return output


def source_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    result, pairs = [], set()
    metadata_keys = {'block_id', 'split', 'official_split', 'shape_id', 'head_number', 'nearest_distractor_number', 'verb_root'}
    metadata_keys.update(f'{key}_{side}' for side in ('a', 'b') for key in ('sequence_sha256', 'accession', 'sequence_id', 'cluster', 'superfamily', 'family'))
    for row in rows:
        require(row['label'] in ('0', '1'), 'Raw source label differs')
        require(row['split'] == 'test', 'Raw source split differs')
        for side, key in (('a', 'sentence1'), ('b', 'sentence2')):
            require(bool(row[key]) and set(row[key]) <= set('ACDEFGHIKLMNPQRSTVWY'), 'Noncanonical protein endpoint')
            require(hashlib.sha256(row[key].encode()).hexdigest() == row['sequence_sha256_' + side], 'Raw source sequence hash differs')
        pair = tuple(sorted((row['sequence_sha256_a'], row['sequence_sha256_b'])))
        require(pair[0] != pair[1] and pair not in pairs, 'Duplicate/self source pair')
        pairs.add(pair)
        result.append(dict(row_id=row.get('row_id', row['pair_id']), label=int(row['label']), sentence1=row['sentence1'], sentence2=row['sentence2'],
                           metadata={key: row[key] for key in sorted(metadata_keys) if key in row}))
    return result


def check_metadata(specs):
    """Close new data and archived qualification pins without loading old encoded rows."""
    acceptance = read(specs['m1_acceptance']['file'])
    require(acceptance['status'] == 'accepted_for_local_confirmation' and acceptance['confirmation_data_ready'] is True and acceptance['retained_rows'] == 39893, 'Archived qualification not accepted')
    for name in ('m1_manifest', 'm1_references', 'm1_verification', 'm1_source_lock', 'm1_build_report', 'target_raw', 'target_membership'):
        file = Path(specs[name]['file'])
        require(file.is_relative_to(M1) and acceptance['files'][str(file.relative_to(M1))] == specs[name]['sha256'], 'Archived qualification dependency differs')
    historical = read(specs['m1_manifest']['file'])
    for relative in ('data/confirmation.jsonl', 'data/encoded.jsonl'):
        require(historical['outputs'][relative] == acceptance['files'][relative], 'Archived qualified artifact identity differs')
    require(read(specs['m1_verification']['file'])['status'] == 'passed', 'Archived independent qualification failed')
    require(read(specs['m1_build_report']['file'])['qualification_passed'] is True, 'Archived build qualification failed')
    lock = read(specs['m1_source_lock']['file'])
    require(lock['selected_artifact']['expected_sha256'] == specs['target_raw']['sha256'] and lock['dataset']['configuration'] == 'qqp' and lock['dataset']['split'] == 'validation', 'Fixed upstream QQP identity differs')
    source = read(specs['source_prepared_manifest']['file'])
    require(source['source_counts'] == {'train': 8044, 'validation': 20276} and source['tokenizer']['sha256'] == TOKENIZER_SHA, 'New source/tokenizer identity differs')
    for short, key in [('acceptance', 'm1_acceptance'), ('manifest', 'm1_manifest'), ('references', 'm1_references'), ('audit', 'm1_verification')]:
        prior = source['inputs']['confirmation_' + short]
        require(prior['path'] == specs[key]['file'] and prior['sha256'] == specs[key]['sha256'], 'New source archived qualification lineage differs')
    for split in ('train', 'validation'):
        item = source['source'][split]
        require(specs['source_' + split + '_rows']['file'] == str(NEW_PREPARED / item['rows']) and specs['source_' + split + '_rows']['sha256'] == source['outputs'][item['rows']]['sha256'], 'Current source row identities differ')
    inherited = source['confirmation_metadata_inheritance']
    require(inherited['historical_target_has_already_been_scored'] is True and inherited['new_blind_confirmation_claimed'] is False, 'Target exposure disclosure differs')
    raw = read(specs['raw_manifest']['file'])
    files = {x['file']: x for x in raw['files']}
    for name in ('source_test', 'tokenizer', 'source_pair_manifest', 'source_pair_acceptance'):
        spec = specs[name]; relative = str(Path(spec['file']).relative_to(NEW_RAW))
        require(files[relative]['sha256'] == spec['sha256'] and files[relative]['bytes'] == spec['bytes'], 'New raw manifest binding differs')
    require(specs['tokenizer']['sha256'] == TOKENIZER_SHA, 'Shared tokenizer changed')
    require(read(specs['raw_acceptance']['file'])['status'] == 'accepted_new_raw_release_training_disabled', 'Historical raw gate changed')
    pair = read(specs['source_pair_manifest']['file'])
    require(pair['data_files'][Path(specs['source_test']['file']).name] == specs['source_test']['sha256'], 'Protein pair manifest differs')
    require(read(specs['source_pair_acceptance']['file'])['pair_counts']['test'] == {'0': 10427, '1': 10427}, 'New source test counts differ')
    return acceptance


def build(root=ROOT):
    root = Path(root).resolve()
    require(root == ROOT, 'This namespace has fixed absolute provenance')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    protocol = check_source_gate(root)  # First: no held-out hash/read before the source barrier.
    protocol_sha, barrier_sha = sha(root / 'protocol.json'), sha(root / 'source_barrier.json')
    require(not (root / 'prepared').exists(), 'Refusing to overwrite prepared data')
    started = time.monotonic()
    code_pins = {str(root / name): sha(root / name) for name in ('prepare_data.py', 'audit_data.py', 'test_data.py')}
    specs, inputs = protocol['data_inputs'], {}
    for name, item in specs.items():
        path = Path(item['file'])
        require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'Invalid pinned input path: ' + name)
        require(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'Changed pinned input: ' + name)
        inputs[str(path)] = item['sha256']
    acceptance = check_metadata(specs)
    from tokenizers import Tokenizer
    tokenizer_path = Path(specs['tokenizer']['file'])
    tokenizer = Tokenizer.from_file(str(tokenizer_path)); tokenizer.no_padding(); tokenizer.no_truncation()
    require(tokenizer.get_vocab_size() == 32000, 'Tokenizer vocabulary changed')
    require([tokenizer.token_to_id(s) for s in ('<|pad_v2|>', '<|eos_v2|>', '<|pair_v2|>', '<|unk_v2|>')] == [0, 1, 2, 3], 'Special IDs changed')
    from pyarrow.parquet import read_table
    target = reconstruct_target(read_table(specs['target_raw']['file']).to_pylist(), list(records(specs['target_membership']['file'])), acceptance['files']['data/confirmation.jsonl'])
    folder = root / 'prepared'; folder.mkdir()
    roles = {'source_test': encode_role('source_test', source_rows(specs['source_test']['file']), tokenizer, folder),
             'target': encode_role('target', target, tokenizer, folder)}
    require(roles['target']['historical_qualified_encoding_content_sha256'] == acceptance['files']['data/encoded.jsonl'], 'Fresh encoding differs from fixed historical target content identity')
    for role, key in [('source_test', 'source_test'), ('target', 'target_raw')]:
        roles[role].update(raw_input=specs[key]['file'], raw_input_sha256=specs[key]['sha256'])
    roles['target'].update(membership_input=specs['target_membership']['file'], membership_input_sha256=specs['target_membership']['sha256'],
                           reconstructed_confirmation_content_sha256=acceptance['files']['data/confirmation.jsonl'])
    for path, digest in (inputs | code_pins).items():
        require(sha(path) == digest, 'Input changed during preparation: ' + path)
    require(sha(root / 'protocol.json') == protocol_sha and sha(root / 'source_barrier.json') == barrier_sha, 'Gate changed during preparation')
    outputs = {str(p.relative_to(root)): sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    manifest = dict(schema_version=1, status='built_pending_independent_data_audit', created_at_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.monotonic() - started, source_barrier_sha256=barrier_sha, preparation_protocol_sha256=protocol_sha,
                    inputs=inputs, outputs=outputs, roles=roles, preparation_code=code_pins,
                    tokenizer=dict(file=str(tokenizer_path), sha256=TOKENIZER_SHA, vocab_size=32000, pad_id=0, eos_id=1, sep_id=2, unk_id=3),
                    source_prepared_manifest=specs['source_prepared_manifest']['file'], confirmation_acceptance=specs['m1_acceptance']['file'],
                    row_selection_or_filtering_performed=False, fixed_historical_membership_replayed=True,
                    old_heldout_encoded_or_confirmation_rows_read=False, historical_target_already_observed=True,
                    target_calibration_or_training_performed=False, text_redistribution_cleared=False,
                    group_metadata_use='Audit/report only; never included in model tokens')
    atomic(folder / 'manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'roles': {k:v['count'] for k,v in roles.items()}, 'elapsed_seconds': manifest['elapsed_seconds']}), flush=True)
    return manifest


if __name__ == '__main__':
    try:
        build()
    except Exception as error:
        failure = ROOT / 'verification/data_preparation_failure.json'
        if not failure.exists():
            atomic(failure, {'status':'failed', 'error':type(error).__name__ + ': ' + str(error), 'time_utc':datetime.now(timezone.utc).isoformat()})
        raise
