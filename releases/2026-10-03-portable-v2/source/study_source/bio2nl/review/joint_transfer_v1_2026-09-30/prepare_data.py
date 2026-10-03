"""Prepare the two fixed held-out roles only after the source-selection barrier."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from common import atomic, check_source_gate, read, require, sha

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
M2 = ROOT.parent / 'joint_pretraining_v1_2026-09-29'
M1 = ROOT.parent / 'confirmation_data_v1_2026-09-29'
EVAL = ROOT.parent / 'eval_v2/prepared'
V2 = WORKSPACE / 'data_rebuild/2026-09-25-v2'
TOKENIZER_SHA = '1bb8092348368961ca211cbf4d066a0e0ebf612b35b401b07ed173476088130b'
EXPECTED_COUNTS = {'source_test': 20808, 'target': 39893}


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


def check_target_alignment(raw, prior, a, b):
    require(prior['id'] == raw['id'], 'M1 target row order/ID changed')
    require(prior['label'] == raw['label'], 'M1 target label changed')
    require(prior['component_id'] == raw['component_id'], 'M1 target component changed')
    require(prior['ids_a'] == a and prior['ids_b'] == b, 'M1 target tokenization changed')


def validate_label(value):
    require(type(value) is int and value in (0, 1), 'Label must be unchanged integer 0 or 1')
    return value


def encode_role(role, raw_rows, tokenizer, folder, prior_rows=None):
    require(role in EXPECTED_COUNTS, 'Unknown held-out role')
    n = len(raw_rows)
    require(n == EXPECTED_COUNTS[role], 'Fixed row count changed')
    if role == 'target':
        require(prior_rows is not None and len(prior_rows) == n, 'Missing/unequal M1 encoded rows')
    ids = np.zeros((n, 512), dtype=np.int64)
    mask = np.zeros((n, 512), dtype=np.int64)
    labels = np.empty(n, dtype=np.int64)
    encoded, row_ids, pair_labels = [], set(), {}
    labels_count, groups, degree, block_degree = Counter(), Counter(), defaultdict(Counter), defaultdict(Counter)
    cache, lengths, truncated, same_label_collisions = {}, [], 0, 0
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
            check_target_alignment(raw, prior_rows[i], a_full, b_full)
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
        require(all(x[0] == x[1] for x in degree.values()), 'Source endpoint role-label imbalance')
        require(all(x[0] == x[1] for x in block_degree.values()), 'Source within-block role-label imbalance')
    else:
        require(dict(labels_count) == {'0': 25227, '1': 14666} and len(groups) == 33128, 'M1 retained population changed')
    npz, rowfile = folder / (role + '.npz'), folder / (role + '.rows.jsonl.gz')
    require(not npz.exists(), 'Refusing to overwrite encoded tensors')
    np.savez_compressed(npz, input_ids=ids, attention_mask=mask, labels=labels)
    write_records(rowfile, encoded)
    return {'npz': str(npz.relative_to(ROOT)), 'rows': str(rowfile.relative_to(ROOT)), 'count': n,
            'npz_sha256': sha(npz), 'rows_sha256': sha(rowfile), 'labels': dict(sorted(labels_count.items())),
            'groups': len(groups), 'largest_group_rows': max(groups.values()),
            'shape': [n, 512], 'dtype': 'int64', 'endpoint_content_tokens_min': min(lengths),
            'endpoint_content_tokens_max': max(lengths), 'truncated_endpoints': truncated,
            'unknown_or_special_content_tokens': 0, 'roundtrip_failures': 0,
            'same_label_encoded_pair_collisions': same_label_collisions, 'conflicting_encoded_pair_labels': 0,
            'row_order_and_labels_preserved': True,
            'source_endpoint_role_and_within_block_balance': True if role == 'source_test' else None,
            'previously_observed_evaluation_split': role == 'source_test',
            'qualification_scope': 'historical operational protein similarity test' if role == 'source_test'
                                   else 'M1 fixed-rule local QQP confirmation; no global/semantic unseen guarantee'}


def build(root=ROOT):
    root = Path(root).resolve()
    require(root == ROOT, 'This namespace has fixed absolute provenance')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    check_source_gate(root)  # Must precede any held-out hash/read/imported checker.
    protocol_sha = sha(root / 'protocol.json')
    barrier_sha = sha(root / 'source_barrier.json')
    require(not (root / 'prepared').exists(), 'Refusing to overwrite prepared data')
    started = time.monotonic()
    inputs = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        digest = sha(path)
        require(expected is None or digest == expected, 'Upstream hash mismatch: ' + str(path))
        inputs[str(path)] = digest
        return path

    acceptance = read(pin(M1 / 'confirmation_acceptance.json'))
    require(acceptance['status'] == 'accepted_for_local_confirmation', 'M1 not accepted')
    gate_script = pin(M1 / 'check_confirmation.py', acceptance['files']['check_confirmation.py'])
    spec = importlib.util.spec_from_file_location('m3_confirmation_gate', gate_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    qualification = module.check(M1)
    require(qualification['status'] == 'passed', 'M1 hash gate failed')
    old_manifest = read(pin(EVAL / 'manifest.json'))
    old_acceptance = read(pin(EVAL / 'acceptance.json'))
    require(old_acceptance['status'] == 'passed' and old_acceptance['manifest_sha256'] == sha(EVAL / 'manifest.json'), 'Accepted v2 prepared gate mismatch')
    source_spec = old_manifest['tasks']['protein_sequence_similarity']['splits']['test']
    source_path = pin(EVAL / source_spec['file'], source_spec['sha256'])
    target_path = pin(M1 / 'data/confirmation.jsonl', acceptance['files']['data/confirmation.jsonl'])
    target_encoded_path = pin(M1 / 'data/encoded.jsonl', acceptance['files']['data/encoded.jsonl'])
    source_prepared = read(pin(M2 / 'prepared/manifest.json'))
    for dependency in (EVAL / 'manifest.json', M1 / 'confirmation_acceptance.json'):
        require(source_prepared['inputs'][str(dependency)] == inputs[str(dependency)],
                'M2 provenance dependency changed: ' + str(dependency))
    tokenizer_path = pin(V2 / 'tokenizers/mixed_bpe/tokenizer.json', TOKENIZER_SHA)
    require(source_prepared['tokenizer']['sha256'] == TOKENIZER_SHA, 'Source and target tokenizer differ')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    require(tokenizer.get_vocab_size() == 32000, 'Tokenizer vocabulary changed')
    require([tokenizer.token_to_id(s) for s in ('<|pad_v2|>', '<|eos_v2|>', '<|pair_v2|>', '<|unk_v2|>')] == [0, 1, 2, 3], 'Special IDs changed')
    folder = root / 'prepared'
    folder.mkdir()
    roles = {}
    roles['source_test'] = encode_role('source_test', list(records(source_path)), tokenizer, folder)
    roles['target'] = encode_role('target', list(records(target_path)), tokenizer, folder, list(records(target_encoded_path)))
    for role, path in [('source_test', source_path), ('target', target_path)]:
        roles[role].update(raw_input=str(path), raw_input_sha256=inputs[str(path)])
    roles['target'].update(prior_encoded_input=str(target_encoded_path), prior_encoded_sha256=inputs[str(target_encoded_path)])
    for split in ('train', 'validation'):
        entry = source_prepared['source'][split]
        pin(M2 / entry['rows'], source_prepared['outputs'][entry['rows']])
    require(sha(root / 'protocol.json') == protocol_sha and sha(root / 'source_barrier.json') == barrier_sha, 'Gate documents changed during preparation')
    outputs = {str(p.relative_to(root)): sha(p) for p in sorted(folder.iterdir()) if p.is_file()}
    manifest = {'schema_version': 1, 'status': 'built_pending_independent_data_audit',
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic() - started,
                'source_barrier_sha256': barrier_sha, 'preparation_protocol_sha256': protocol_sha,
                'inputs': inputs, 'outputs': outputs, 'roles': roles,
                'tokenizer': {'file': str(tokenizer_path), 'sha256': TOKENIZER_SHA, 'vocab_size': 32000,
                              'pad_id': 0, 'eos_id': 1, 'sep_id': 2, 'unk_id': 3},
                'm1_confirmation_gate': qualification, 'source_prepared_manifest': str(M2 / 'prepared/manifest.json'),
                'source_old_eval_manifest': str(EVAL / 'manifest.json'),
                'confirmation_acceptance': str(M1 / 'confirmation_acceptance.json'),
                'row_selection_or_filtering_performed': False, 'target_calibration_or_training_performed': False,
                'text_redistribution_cleared': False,
                'group_metadata_use': 'Audit/report only; never included in model tokens'}
    atomic(folder / 'manifest.json', manifest)
    print(json.dumps({'status': manifest['status'], 'roles': {k: v['count'] for k, v in roles.items()},
                      'elapsed_seconds': manifest['elapsed_seconds']}, sort_keys=True), flush=True)
    return manifest


if __name__ == '__main__':
    build()
