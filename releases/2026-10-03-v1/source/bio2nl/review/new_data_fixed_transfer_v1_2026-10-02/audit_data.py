"""Independent held-out reconstruction; does not import the data producer."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time
import unicodedata

import numpy as np

from common import atomic, check_source_gate, path_inside, read, require, sha, NEW_PREPARED, COUNTS

ROOT = Path(__file__).resolve().parent
EXPECTED_LABELS = {'source_test': {'0':10427, '1':10427}, 'target': {'0':25227, '1':14666}}
TARGET_GROUPS = 33128


def read_lines(path):
    path = Path(path)
    open_file = gzip.open if path.name.endswith('.gz') else open
    with open_file(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def reconstruct_input(a, b):
    require(len(a) > 0 and len(b) > 0, 'Empty endpoint')
    require(all(type(v) is int and v in range(4, 32000) for v in a + b), 'Invalid endpoint ID')
    expected = np.concatenate((np.asarray(a[:255], dtype=np.int64), [2],
                               np.asarray(b[:255], dtype=np.int64), [1])).astype(np.int64)
    full = np.pad(expected, (0, 512 - len(expected)))
    mask = (np.arange(512) < len(expected)).astype(np.int64)
    return full, mask


def compare_encoded(ids, mask, label, a, b, expected_label):
    expected_ids, expected_mask = reconstruct_input(a, b)
    require(ids.dtype == mask.dtype == np.dtype('int64'), 'Array dtype changed')
    require(np.array_equal(ids, expected_ids), 'Independent packed-input reconstruction failed')
    require(np.array_equal(mask, expected_mask), 'Independent attention mask reconstruction failed')
    require(type(expected_label) is int and expected_label in (0, 1), 'Raw label is not binary integer')
    require(int(label) == expected_label, 'Label changed')


def normalized_hash(text):
    normalized = ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text).casefold()))
    return hashlib.sha256(normalized.encode()).hexdigest()


def identities(rows):
    result = {k: set() for k in ('row_ids', 'groups', 'sequences', 'clusters', 'encoded_inputs')}
    for row in rows:
        result['row_ids'].add(row['row_id'])
        result['groups'].add(row['group_id'] if 'group_id' in row else row['metadata']['block_id'])
        result['encoded_inputs'].add(tuple(row['ids_a'] + [2] + row['ids_b'] + [1]))
        for side in ('a', 'b'):
            result['sequences'].add(row['metadata']['sequence_sha256_' + side])
            result['clusters'].add(row['metadata']['cluster_' + side])
    return result


def verify_role(root, role, entry, tokenizer, raw, expected_encoding_hash=None):
    require(role in ('source_test', 'target'), 'Unknown heldout role')
    stored = read_lines(path_inside(root, entry['rows']))
    n = COUNTS[role]
    require(len(raw) == len(stored) == entry['count'] == n, 'Population count mismatch')
    fields = ('sentence1', 'sentence2') if role == 'source_test' else ('text_a', 'text_b')
    unique_texts = sorted({row[key] for row in raw for key in fields})
    independent = {}
    for start in range(0, len(unique_texts), 256):
        batch = unique_texts[start:start + 256]
        encodings = tokenizer.encode_batch(batch, add_special_tokens=False)
        require(len(encodings) == len(batch), 'Tokenizer batch length changed')
        for text, encoding in zip(batch, encodings):
            require(bool(text) and len(encoding.ids) > 0, 'Empty raw text or encoding')
            require(all(4 <= v < 32000 for v in encoding.ids), 'Unknown/special/range failure')
            require(tokenizer.decode(encoding.ids, skip_special_tokens=False) == text, 'Independent text roundtrip failure')
            independent[text] = encoding.ids
    encoding_digest = hashlib.sha256()
    unordered_labels = {}
    labels_count, groups, lengths, row_ids, pair_labels = Counter(), Counter(), [], set(), {}
    degree, within = defaultdict(Counter), defaultdict(Counter)
    same_label_collisions, truncated = 0, 0
    with np.load(path_inside(root, entry['npz']), allow_pickle=False) as archive:
        # NPZ is lazy: materialize each matrix once before the row-wise audit.
        # Indexing archive['input_ids'] per row would repeatedly decompress N*512.
        require(set(archive.files) == {'input_ids', 'attention_mask', 'labels'}, 'Unexpected NPZ arrays')
        arrays = {name: archive[name] for name in archive.files}
        require(arrays['input_ids'].shape == arrays['attention_mask'].shape == (n, 512), 'Tensor shape changed')
        require(arrays['labels'].shape == (n,), 'Label shape changed')
        require(all(arrays[k].dtype == np.dtype('int64') for k in arrays), 'Tensor dtype changed')
        for i, (original, output) in enumerate(zip(raw, stored)):
            a, b = [independent[original[key]] for key in fields]
            compare_encoded(arrays['input_ids'][i], arrays['attention_mask'][i], arrays['labels'][i], a, b, original['label'])
            rid = original['row_id'] if role == 'source_test' else original['id']
            require(isinstance(rid, str) and bool(rid) and rid not in row_ids, 'Duplicate/missing row identity')
            row_ids.add(rid)
            require(output['row_index'] == i and output['row_id'] == rid and output['label'] == original['label'], 'Stored row identity/order/label mismatch')
            require(output['ids_a'] == a[:255] and output['ids_b'] == b[:255], 'Stored endpoint tokens changed')
            require(output['precap_token_lengths'] == [len(a), len(b)], 'Stored token lengths changed')
            require(output['raw_lengths'] == [len(original[f]) for f in fields], 'Stored raw lengths changed')
            lengths.extend((len(a), len(b)))
            truncated += int(len(a) > 255) + int(len(b) > 255)
            if role == 'source_test':
                metadata = original['metadata']
                require(metadata['split'] == 'test' and output['metadata'] == metadata, 'Source metadata changed')
                group = metadata['block_id']
                for side, field, content in zip(('a', 'b'), fields, (a[:255], b[:255])):
                    require(hashlib.sha256(original[field].encode()).hexdigest() == metadata['sequence_sha256_' + side], 'Source sequence identity changed')
                    degree[(side, tuple(content))][original['label']] += 1
                    within[(group, side, tuple(content))][original['label']] += 1
            else:
                group = original['component_id']
                require(output['component_id'] == group, 'Target component ID changed')
                item = {'id': rid, 'label': original['label'], 'component_id': group, 'ids_a': a, 'ids_b': b}
                encoding_digest.update((json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8'))
                expected_metadata = {key: original[key] for key in ('upstream_idx', 'endpoint_a_sha256', 'endpoint_b_sha256', 'component_id')}
                require(output['metadata'] == expected_metadata, 'Target identity metadata changed')
                for side, field in zip(('a', 'b'), fields):
                    require(normalized_hash(original[field]) == original['endpoint_' + side + '_sha256'], 'M1 normalized endpoint identity changed')
            require(output['group_id'] == group, 'Group ID changed')
            require(not any(key in output for key in ('text_a', 'text_b', 'sentence1', 'sentence2')), 'Raw text unexpectedly copied')
            labels_count[str(original['label'])] += 1
            groups[group] += 1
            joined = tuple(a[:255] + [2] + b[:255] + [1])
            if joined in pair_labels:
                require(pair_labels[joined] == original['label'], 'Conflicting label collision')
                same_label_collisions += 1
            pair_labels[joined] = original['label']
            unordered = frozenset((tuple(a), tuple(b)))
            require(unordered not in unordered_labels or unordered_labels[unordered] == original['label'], 'Conflicting unordered encoded-pair labels')
            unordered_labels[unordered] = original['label']
    expected_labels = EXPECTED_LABELS[role]
    require(dict(labels_count) == expected_labels, 'Fixed labels changed')
    if role == 'target':
        require(len(groups) == TARGET_GROUPS, 'Fixed target groups changed')
        require(encoding_digest.hexdigest() == expected_encoding_hash == entry['historical_qualified_encoding_content_sha256'], 'Independent fixed target encoding content hash differs')
    require(truncated == 0, 'Unexpected heldout truncation')
    if role == 'source_test':
        require(all(c[0] == c[1] for c in degree.values()) and all(c[0] == c[1] for c in within.values()), 'Independent source role-label balance failed')
    findings = {'count': n, 'labels': dict(sorted(labels_count.items())), 'groups': len(groups),
                'largest_group_rows': max(groups.values()), 'endpoint_content_tokens_min': min(lengths),
                'endpoint_content_tokens_max': max(lengths), 'truncated_endpoints': truncated,
                'unknown_or_special_content_tokens': 0, 'roundtrip_failures': 0,
                'same_label_encoded_pair_collisions': same_label_collisions, 'conflicting_encoded_pair_labels': 0}
    for field, value in findings.items():
        require(entry[field] == value, 'Producer statistic differs: ' + role + '/' + field)
    findings.update(independent_reencoding_passed=True, row_order_labels_and_group_ids_exact=True,
                    unique_raw_texts_reencoded=len(independent), source_history='New rebuilt pairs; underlying biological corpus and related prior study were observed.' if role == 'source_test' else 'Exact fixed QQP cohort was already model-scored.')
    return findings, stored


def reconstruct_source(path):
    # Independent TSV parsing and metadata projection, including all accepted identity fields.
    result = []
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as handle:
        for value in csv.DictReader(handle, delimiter='\t'):
            require(value['label'] in ('0', '1') and value['split'] == 'test', 'Invalid raw source split/label')
            metadata = {}
            for key in value:
                if key in ('block_id', 'split', 'official_split', 'shape_id', 'head_number', 'nearest_distractor_number', 'verb_root') or any(key == base + '_' + side for base in ('sequence_sha256', 'accession', 'sequence_id', 'cluster', 'superfamily', 'family') for side in ('a', 'b')):
                    metadata[key] = value[key]
            for column in ('sentence1', 'sentence2'):
                require(bool(value[column]) and all(c in 'ACDEFGHIKLMNPQRSTVWY' for c in value[column]), 'Noncanonical source sequence')
            result.append(dict(row_id=value['row_id'] if 'row_id' in value else value['pair_id'], label=int(value['label']),
                               sentence1=value['sentence1'], sentence2=value['sentence2'], metadata=metadata))
    unique = set()
    for row in result:
        hashes = frozenset((row['metadata']['sequence_sha256_a'], row['metadata']['sequence_sha256_b']))
        require(len(hashes) == 2 and hashes not in unique, 'Source self/duplicate pair')
        unique.add(hashes)
    return result


def reconstruct_target_independent(raw, membership, expected_digest, expected_raw_count=40430):
    require(len(raw) == len(membership) == expected_raw_count, 'Independent raw or ledger population differs')
    require(all(set(row) == {'idx', 'question1', 'question2', 'label'} for row in raw), 'Independent QQP schema differs')
    def normalize(value):
        if not isinstance(value, str):
            return ''
        return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', value).casefold()))
    def digest(value):
        return hashlib.sha256(value.encode('utf-8')).hexdigest()
    # Adjacency traversal, independent of the producer's union-find and root bookkeeping.
    adjacency, by_id = defaultdict(set), {}
    for row in raw:
        require(type(row['idx']) is int and row['idx'] >= 0 and row['idx'] not in by_id, 'Invalid/duplicate raw index')
        require(type(row['label']) is int and row['label'] in (0, 1), 'Invalid raw binary label')
        by_id[row['idx']] = row
        a, b = normalize(row['question1']), normalize(row['question2'])
        if a and b:
            adjacency[a].add(b); adjacency[b].add(a)
    components = {}
    for initial in adjacency:
        if initial in components:
            continue
        seen, stack = {initial}, [initial]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor); stack.append(neighbor)
        identity = digest(min(seen))
        for endpoint in seen:
            components[endpoint] = identity
    output, content = [], hashlib.sha256()
    ordered = sorted(by_id)
    for idx, decision in zip(ordered, membership):
        require(type(decision['upstream_idx']) is int and decision['upstream_idx'] == idx, 'Independent membership order differs')
        require(type(decision['retained']) is bool and isinstance(decision['reasons'], list), 'Independent membership schema differs')
        row = by_id[idx]; a, b = normalize(row['question1']), normalize(row['question2'])
        identity = components[a] if a and b else None
        require(decision['component_id'] == identity, 'Independent component graph differs')
        require(decision['retained'] == (len(decision['reasons']) == 0), 'Independent qualification state inconsistent')
        if decision['retained']:
            require(bool(a) and bool(b), 'Invalid retained text')
            item = {'id': 'qqp_validation:' + str(idx), 'upstream_idx': idx, 'label': row['label'],
                    'text_a': row['question1'], 'text_b': row['question2'],
                    'endpoint_a_sha256': digest(a), 'endpoint_b_sha256': digest(b), 'component_id': identity}
            content.update((json.dumps(item, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8'))
            output.append(item)
    require(content.hexdigest() == expected_digest, 'Independent reconstructed fixed target content hash differs')
    return output


def audit(root=ROOT):
    root = Path(root).resolve()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    protocol = check_source_gate(root)  # No held-out reads or hashes before this barrier.
    output = root / 'verification/data_audit.json'
    require(not output.exists(), 'Refusing to overwrite independent data audit')
    start = time.monotonic()
    manifest_file = root / 'prepared/manifest.json'; manifest = read(manifest_file)
    require(manifest['status'] == 'built_pending_independent_data_audit' and set(manifest['roles']) == {'source_test', 'target'}, 'Producer status/role set differs')
    require(manifest['source_barrier_sha256'] == sha(root / 'source_barrier.json') and manifest['preparation_protocol_sha256'] == sha(root / 'protocol.json'), 'Preparation gate changed')
    specs = protocol['data_inputs']
    expected_inputs = {item['file']: item['sha256'] for item in specs.values()}
    require(manifest['inputs'] == expected_inputs, 'Protocol/manifest input set differs')
    require(set(manifest['preparation_code']) == {str(root / name) for name in ('prepare_data.py', 'audit_data.py', 'test_data.py')}, 'Preparation code closure differs')
    bindings = {}
    for mapping in (manifest['inputs'], manifest['outputs'], manifest['preparation_code']):
        for name, digest in mapping.items():
            path = Path(name) if Path(name).is_absolute() else path_inside(root, name)
            require(path.is_file() and not path.is_symlink() and sha(path) == digest, 'Changed prepared dependency: ' + str(path))
            bindings[str(path)] = digest
    for item in specs.values():
        require(Path(item['file']).stat().st_size == item['bytes'], 'Changed input size')
    token_file = Path(specs['tokenizer']['file'])
    require(manifest['tokenizer']['file'] == str(token_file) and bindings[str(token_file)] == manifest['tokenizer']['sha256'] == '1bb8092348368961ca211cbf4d066a0e0ebf612b35b401b07ed173476088130b', 'Tokenizer identity changed')
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(token_file)); tokenizer.no_padding(); tokenizer.no_truncation()
    require(tokenizer.get_vocab_size() == 32000, 'Tokenizer vocabulary mismatch')
    require([tokenizer.token_to_id(s) for s in ('<|pad_v2|>', '<|eos_v2|>', '<|pair_v2|>', '<|unk_v2|>')] == [0, 1, 2, 3], 'Special tokenizer IDs changed')
    acceptance = read(specs['m1_acceptance']['file'])
    require(manifest['confirmation_acceptance'] == specs['m1_acceptance']['file'] and acceptance['status'] == 'accepted_for_local_confirmation' and acceptance['retained_rows'] == 39893, 'Fixed target acceptance differs')
    require(acceptance['files']['raw/qqp_validation.parquet'] == specs['target_raw']['sha256'] and acceptance['files']['audit/exclusions.jsonl'] == specs['target_membership']['sha256'], 'Fixed raw/qualification pins differ')
    from pyarrow.parquet import read_table
    raw_roles = {'source_test': reconstruct_source(specs['source_test']['file']),
                 'target': reconstruct_target_independent(read_table(specs['target_raw']['file']).to_pylist(), read_lines(specs['target_membership']['file']), acceptance['files']['data/confirmation.jsonl'])}
    results, all_rows = {}, {}
    for role in ('source_test', 'target'):
        entry = manifest['roles'][role]
        raw_key = 'source_test' if role == 'source_test' else 'target_raw'
        require(entry['raw_input'] == specs[raw_key]['file'] and entry['raw_input_sha256'] == specs[raw_key]['sha256'], 'Unbound raw input')
        require(entry['npz'] in manifest['outputs'] and entry['rows'] in manifest['outputs'], 'Unbound prepared role output')
        for kind in ('npz', 'rows'):
            require(entry[kind + '_sha256'] == manifest['outputs'][entry[kind]], 'Role artifact descriptor differs')
        if role == 'target':
            require(entry['membership_input'] == specs['target_membership']['file'] and entry['membership_input_sha256'] == specs['target_membership']['sha256'], 'Unbound target membership')
            require(entry['reconstructed_confirmation_content_sha256'] == acceptance['files']['data/confirmation.jsonl'], 'Producer raw reconstruction identity differs')
        results[role], all_rows[role] = verify_role(root, role, entry, tokenizer, raw_roles[role], acceptance['files']['data/encoded.jsonl'] if role == 'target' else None)
    source_prepared_path = Path(specs['source_prepared_manifest']['file']); source_manifest = read(source_prepared_path)
    require(manifest['source_prepared_manifest'] == str(source_prepared_path) and source_prepared_path == NEW_PREPARED / 'manifest.json', 'Source prepared root differs')
    require(source_manifest['source_counts'] == {'train': 8044, 'validation': 20276}, 'Source populations differ')
    heldout_ids, isolation = identities(all_rows['source_test']), {}
    for split in ('train', 'validation'):
        entry = source_manifest['source'][split]; source_file = NEW_PREPARED / entry['rows']
        require(str(source_file) == specs['source_' + split + '_rows']['file'] and bindings[str(source_file)] == source_manifest['outputs'][entry['rows']]['sha256'], 'Unpinned current source identity rows')
        rows = read_lines(source_file)
        require(len(rows) == source_manifest['source_counts'][split] and all(row['metadata']['split'] == split for row in rows), 'Current source count/split differs')
        current_ids = identities(rows)
        isolation[split + '/source_test'] = {key: len(heldout_ids[key] & current_ids[key]) for key in heldout_ids}
        require(not any(isolation[split + '/source_test'].values()), 'Source heldout/train-validation identity overlap')
    require(manifest['row_selection_or_filtering_performed'] is False and manifest['target_calibration_or_training_performed'] is False and manifest['fixed_historical_membership_replayed'] is True and manifest['old_heldout_encoded_or_confirmation_rows_read'] is False and manifest['historical_target_already_observed'] is True, 'Preparation boundary changed')
    bindings[str(manifest_file)] = sha(manifest_file)
    for relative in ('prepare_data.py', 'audit_data.py', 'common.py', 'protocol.json', 'source_barrier.json'):
        bindings[str(root / relative)] = sha(root / relative)
    for name, digest in bindings.items():
        require(sha(name) == digest, 'Dependency changed during independent audit: ' + name)
    report = dict(schema_version=1, status='passed', created_at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start,
                  prepared_manifest_sha256=bindings[str(manifest_file)], source_barrier_sha256=sha(root/'source_barrier.json'), roles=results,
                  source_cross_split_identity_overlap=isolation, bindings=bindings, verified_unique_files=len(bindings),
                  retained_rows_and_groups_unchanged=True, confirmation_examples_read_after_source_barrier=True,
                  data_acceptance_only_no_model_scoring=True, text_redistribution_cleared=False,
                  target_reconstructed_from_raw_and_fixed_qualification=True, old_heldout_encoded_or_confirmation_rows_read=False,
                  target_confirmation_content_sha256=acceptance['files']['data/confirmation.jsonl'], target_encoding_content_sha256=acceptance['files']['data/encoded.jsonl'],
                  limitations=['New rebuilt protein pairs were not model-scored in this round; underlying proteins and related historical tests were observed.',
                               'The exact fixed QQP cohort was previously model-scored; this is exploratory, not new blind confirmation.',
                               'Inherited lexical qualification does not establish semantic/global unseen status.',
                               'Operational cluster separation is not comprehensive structural-family independence.'])
    atomic(output, report)
    (root/'DATA_REPORT.md').write_text('\n'.join([
        '# Fixed held-out data reconstruction', '',
        'All 27 source classifiers and the selected surface reference were frozen before held-out reconstruction.', '',
        '| Role | Rows | Label 0 / 1 | Groups | Content tokens per endpoint |',
        '| --- | ---: | ---: | ---: | ---: |',
        *[f"| {role} | {v['count']:,} | {v['labels']['0']:,} / {v['labels']['1']:,} | {v['groups']:,} | {v['endpoint_content_tokens_min']}–{v['endpoint_content_tokens_max']} |" for role,v in results.items()], '',
        'Every unique raw endpoint was independently reencoded. All packed inputs, masks, labels, row order and group IDs agree; no truncation, unknown tokens, roundtrip failures or contradictory encoded labels occurred.', '',
        'The new protein test has zero row, block, sequence-hash, operational-cluster and encoded-input overlap with the current source train/validation sets. This does not establish structural-family independence. These rebuilt pairs have not previously been model-scored in this round; their upstream biological corpus and related historical experiments were observed.', '',
        'QQP preserves the exact 39,893-row historical qualified cohort and its original components. It is reconstructed from pinned upstream parquet plus the fixed qualification ledger, with independent graph reconstruction. Canonical raw-row and fresh-tokenization SHA256 values match the historical qualification identities without opening old encoded inputs. No new target filtering or qualification is performed.', '',
        'QQP was already model-scored in earlier experiments. This remains exploratory; inherited lexical qualification is not a semantic or global unseen-data guarantee. Raw text is not copied here; reversible token artifacts remain local and are not cleared for redistribution.', '',
        'This report validates data only and contains no model scores or parameter updates.', '']))
    print(json.dumps({'status':'passed', 'verified_unique_files':len(bindings), 'rows_reconstructed':sum(v['count'] for v in results.values()), 'elapsed_seconds':report['elapsed_seconds']}), flush=True)
    return report


if __name__ == '__main__':
    try:
        audit()
    except Exception as error:
        failure = ROOT / 'verification/data_audit_failure.json'
        if not failure.exists():
            atomic(failure, {'status':'failed', 'error':type(error).__name__ + ': ' + str(error), 'time_utc':datetime.now(timezone.utc).isoformat()})
        raise
