"""Independent source-only input audit; grants bounded technical smoke only.

No builder functions are imported. Source selection, token encoding, stream
membership and schedules are recomputed against pinned new raw inputs. Target
and source-test pair examples are outside this reader's closed input allowlist.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import numpy as np

RELEASE_ID = '2026-10-01-portable-v2'
RAW_MEMBERS = (
    'data/pairs/protein_sequence_similarity_train.tsv.gz',
    'data/pairs/protein_sequence_similarity_validation.tsv.gz',
    'data/corpora/english/train.jsonl.gz', 'data/corpora/protein/train.jsonl.gz',
    'data/sequences/protein_canonical_sequences.tsv.gz', 'data/sequences/remote_sequences.tsv.gz',
    'tokenizers/mixed_bpe/tokenizer.json', 'tokenizers/mixed_bpe/training_records.json',
    'tokenizers/mixed_bpe/metadata.json', 'metadata/corpora_english.json',
    'metadata/corpora_protein.json', 'validation/corpora_english.json',
    'validation/corpora_protein.json', 'metadata/protein_split_policy.json',
    'metadata/protein_remote_pretraining_exclusion.json', 'metadata/protein_pair_construction.json',
    'validation/protein_pair_acceptance.json', 'validation/cross_dataset_homology.json')
META_SUFFIXES = {'confirmation_acceptance': 'confirmation_acceptance.json',
    'confirmation_manifest': 'manifest.json', 'confirmation_references': 'references.json',
    'confirmation_audit': 'verification/independent_verification.json',
    'design': 'configs/joint_pretraining_design.json'}
SOURCE_POLICY = {'target_rows': 8000, 'selection_seed': 20260925, 'expected_train_rows': 8044,
    'expected_train_groups': 63, 'expected_validation_rows': 20276, 'expected_raw_train_rows': 99818}
CONDITIONS = {'EP': ['english_common', 'protein'], 'ES': ['english_common', 'shuffled'],
              'EE': ['english_common', 'english_extra']}
STREAMS = ('english_common', 'english_extra', 'protein', 'shuffled')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def text_sha(value):
    return digest(value.encode('utf-8'))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def identical(a, b, message):
    require(canonical(a) == canonical(b), message)


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, 'Duplicate JSON key: ' + key)
        obj[key] = value
    return obj


def loads(text):
    def bad(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(text, object_pairs_hook=unique_object, parse_constant=bad)


def read_json(path):
    return loads(Path(path).read_text())


def identity(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)),
            'Missing/nonabsolute/symlinked input: ' + str(path))
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'sha256': sha, 'bytes': path.stat().st_size}


def inside(root, name):
    require(isinstance(name, str) and name and '\\' not in name and '\x00' not in name, 'Invalid relative path')
    rel = PurePosixPath(name)
    require(not rel.is_absolute() and '..' not in rel.parts and rel.as_posix() == name, 'Escaping/noncanonical relative path')
    path = Path(root) / name
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlinked path')
    return path


def json_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            require(bool(line.strip()), 'Blank JSONL row')
            yield loads(line)


def tsv_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), 'Invalid TSV header')
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()), 'Malformed TSV row')
            yield row


class ReadSet:
    def __init__(self):
        self.files = {}

    def pin(self, path, expected=None):
        path = Path(path).absolute()
        current = identity(path)
        if expected is not None:
            identical(current, {k: expected[k] for k in ('sha256', 'bytes')}, 'Pinned file differs: ' + str(path))
        if str(path) in self.files:
            identical(current, self.files[str(path)], 'Input changed between reads: ' + str(path))
        self.files[str(path)] = current
        return path

    def finish(self):
        for path, expected in self.files.items():
            identical(identity(path), expected, 'Input changed during audit: ' + path)


def independently_selected_groups(group_counts, target=8000, seed=20260925):
    """Nearest cumulative group prefix, with larger prefix chosen on exact ties.

    Independent formulation: minimize distance over all cumulative prefixes;
    positive group sizes make this equivalent to checking the crossing prefix.
    """
    require(group_counts and all(type(n) is int and n > 0 for n in group_counts.values()), 'Invalid group counts')
    def order(key):
        encoded = json.dumps([seed, 'protein_sequence_similarity', key], ensure_ascii=False, separators=(',', ':')).encode()
        return digest(encoded), key
    groups = sorted(group_counts, key=order)
    totals = [0]
    for key in groups:
        totals.append(totals[-1] + group_counts[key])
    cutoff = min(range(len(totals)), key=lambda i: (abs(totals[i] - target), -i))
    require(cutoff > 0, 'Whole-group selection is empty')
    return set(groups[:cutoff]), {'selected_rows': totals[cutoff], 'selected_groups': cutoff,
                                 'source_groups': len(groups), 'source_rows': totals[-1]}


def project_raw(row):
    allowed = {'block_id', 'split', 'official_split', 'shape_id', 'head_number',
               'nearest_distractor_number', 'verb_root'}
    allowed |= {key + '_' + side for side in ('a', 'b') for key in
                ('sequence_sha256', 'accession', 'sequence_id', 'cluster', 'superfamily', 'family')}
    label = int(row['label'])
    require(label in (0, 1), 'Nonbinary raw label')
    return {'row_id': row.get('row_id', row['pair_id']), 'label': label,
            'sentence1': row['sentence1'], 'sentence2': row['sentence2'],
            'metadata': {key: row[key] for key in sorted(allowed) if key in row}}


def content_ids(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    require(bool(ids) and all(type(i) is int and 4 <= i < 32000 for i in ids), 'Unknown/special/empty content tokens')
    require(tokenizer.decode(ids, skip_special_tokens=False) == text, 'Tokenizer full text roundtrip differs')
    return ids


def canonical_metadata(path):
    parents, sequences, clusters = {}, {}, {}
    for row in tsv_rows(path):
        parent = 'swissprot:' + row['accession'] + ':' + row['sequence_version']
        sha, split, cluster = row['sequence_sha256'], row['split'], row['cluster_id']
        require(parent not in parents and sha not in sequences and split in ('train', 'validation', 'test'), 'Duplicate/bad canonical identity')
        require(cluster not in clusters or clusters[cluster] == split, 'Canonical cluster crosses splits')
        # Full canonical table is physically read; heldout sequence text is not
        # retained or used. Only identity, split and exclusion metadata are used.
        value = {'sequence_sha256': sha, 'split': split, 'cluster_id': cluster,
                 'length': int(row['length']), 'eligible': row['pretrain_eligible'].lower() in ('1', 'true', 'yes')}
        parents[parent] = value
        sequences[sha] = value
        clusters[cluster] = split
    require(bool(parents), 'Empty canonical table')
    return parents, sequences, clusters


def audit_source(prepared, manifest, raw_paths, tokenizer, canonical, policy=SOURCE_POLICY):
    identities, details, prefix = {}, {}, {}
    selection_report = read_json(prepared / 'source_selection.json')
    identical(selection_report['policy'], policy, 'Selection policy differs')
    for split in ('train', 'validation'):
        counts = Counter(row['block_id'] for row in tsv_rows(raw_paths[split]))
        if split == 'train':
            selected, selection = independently_selected_groups(counts, policy['target_rows'], policy['selection_seed'])
            require(selection['source_rows'] == policy['expected_raw_train_rows'] and selection['selected_groups'] == policy['expected_train_groups'], 'New raw train population/selection differs')
        else:
            selected = set(counts)
            selection = {'selected_rows': sum(counts.values()), 'selected_groups': len(counts),
                         'source_groups': len(counts), 'source_rows': sum(counts.values())}
        expected = [project_raw(row) for row in tsv_rows(raw_paths[split]) if row['block_id'] in selected]
        require(len(expected) == policy['expected_' + split + '_rows'], 'New source row count differs')
        actual = list(json_rows(prepared / 'source' / (split + '.jsonl')))
        identical(actual, expected, 'Selected full rows/order differ from independent complete-block selection: ' + split)
        info = manifest['source'][split]
        require(info['npz'] == f'source/{split}.npz' and info['rows'] == f'source/{split}.rows.jsonl.gz', 'Unexpected source tensor path')
        rows = list(json_rows(prepared / info['rows']))
        require(len(rows) == len(expected), 'Encoded metadata row count differs')
        seen = {key: set() for key in ('row_ids', 'groups', 'sequences', 'clusters', 'inputs')}
        raw_pairs, pair_labels = set(), {}
        degrees, blocks = defaultdict(Counter), defaultdict(Counter)
        label_counts, truncation, collisions = Counter(), Counter(), 0
        with np.load(prepared / info['npz'], allow_pickle=False) as arrays:
            require(set(arrays.files) == {'input_ids', 'attention_mask', 'labels'}, 'Unexpected NPZ tensor members')
            ids, masks, labels = arrays['input_ids'], arrays['attention_mask'], arrays['labels']
            require(ids.dtype == masks.dtype == labels.dtype == np.dtype('int64'), 'Source tensor dtype differs')
            require(ids.shape == masks.shape == (len(expected), 512) and labels.shape == (len(expected),), 'Source tensor shapes differ')
            for index, (raw, encoded) in enumerate(zip(expected, rows)):
                meta, label = raw['metadata'], raw['label']
                require(meta['split'] == split and raw['row_id'] not in seen['row_ids'], 'Source split/duplicate ID')
                full = [content_ids(tokenizer, raw[k]) for k in ('sentence1', 'sentence2')]
                a, b = [x[:255] for x in full]
                joined = a + [2] + b + [1]
                require(np.array_equal(ids[index, :len(joined)], joined) and not ids[index, len(joined):].any(), 'Independent input token encoding/padding differs')
                require(np.array_equal(masks[index], np.arange(512) < len(joined)), 'Attention mask differs')
                require(int(labels[index]) == label and ids[index, int(masks[index].sum()) - 1] == 1, 'Label/EOS pooling position differs')
                expected_metadata = {'row_index': index, 'row_id': raw['row_id'], 'label': label,
                    'ids_a': a, 'ids_b': b, 'metadata': meta,
                    'raw_lengths': [len(raw['sentence1']), len(raw['sentence2'])],
                    'precap_token_lengths': [len(x) for x in full]}
                identical(encoded, expected_metadata, 'Encoded row provenance differs')
                hashes = []
                for role, text, tokens, all_tokens in zip(('a', 'b'), (raw['sentence1'], raw['sentence2']), (a, b), full):
                    sha = text_sha(text); hashes.append(sha)
                    require(sha == meta['sequence_sha256_' + role] and set(text) <= set('ACDEFGHIKLMNPQRSTVWY'), 'Protein endpoint identity/alphabet differs')
                    c = canonical[sha]
                    require(c['split'] == split and c['cluster_id'] == meta['cluster_' + role] and c['length'] == len(text), 'Source canonical assignment differs')
                    degrees[(role, tuple(tokens))][label] += 1
                    blocks[(meta['block_id'], role, tuple(tokens))][label] += 1
                    truncation[str(label)] += len(all_tokens) > 255
                    seen['sequences'].add(sha); seen['clusters'].add(c['cluster_id'])
                pair = tuple(sorted(hashes))
                require(pair[0] != pair[1] and pair not in raw_pairs, 'Duplicate or self unordered source pair')
                require(raw['row_id'] == text_sha('|'.join(pair)), 'Pair row ID differs from sequence identities')
                raw_pairs.add(pair)
                key = tuple(joined)
                if key in pair_labels:
                    require(pair_labels[key] == label, 'Conflicting encoded labels')
                    collisions += 1
                pair_labels[key] = label
                seen['row_ids'].add(raw['row_id']); seen['groups'].add(meta['block_id']); seen['inputs'].add(key)
                label_counts[str(label)] += 1
        require(label_counts['0'] == label_counts['1'] and all(c[0] == c[1] for c in degrees.values()) and all(c[0] == c[1] for c in blocks.values()), 'Source label/role/block degree imbalance')
        identical(info['labels'], dict(label_counts), 'Source reported label counts differ')
        identical(info['truncated_endpoints_by_label'], dict(truncation), 'Source truncation report differs')
        require(info['same_label_encoded_pair_collisions'] == collisions and info['conflicting_encoded_pair_labels'] == 0, 'Source collision report differs')
        identical(info['identity_counts'], {k: len(v) for k,v in seen.items()}, 'Source identity counts differ')
        require(info['count'] == len(expected) and info['shape'] == [len(expected), 512] and info['dtype'] == 'int64', 'Source manifest count/schema differs')
        require(info['endpoint_role_label_balance'] is True and info['within_block_role_label_balance'] is True, 'Source balance flags differ')
        ids64 = [r['row_id'] for r in expected[:64]]
        p = {'indices': list(range(64)), 'row_ids': ids64, 'rows': 64,
             'row_ids_sha256': text_sha(json.dumps(ids64, separators=(',', ':'))),
             'purpose': 'bounded technical smoke only; not full-group scientific training subset'}
        identical(manifest['source_prefix64'][split], p, 'Source prefix64 identity differs')
        identical(manifest['smoke_source'][split], p, 'Smoke source prefix differs')
        selection_row = selection_report['roles'][split]
        for field, wanted in [('raw_rows', selection['source_rows']), ('raw_groups', len(counts)), ('selected_rows', len(expected)), ('selected_groups', len(selected))]:
            require(selection_row[field] == wanted, 'Selection summary differs: ' + field)
        require(selection_row['ordered_row_ids_sha256'] == text_sha(json.dumps([r['row_id'] for r in expected], separators=(',', ':'))), 'Selected row identity digest differs')
        identities[split] = seen
        details[split] = {'rows': len(expected), 'labels': dict(label_counts), 'blocks': len(selected),
                          'independently_reencoded_rows': len(expected), 'truncated_endpoints_by_label': dict(truncation),
                          'same_label_encoded_pair_collisions': collisions}
        prefix[split] = p
        print('Audited source role:', split, len(expected), flush=True)
    overlap = {k: len(identities['train'][k] & identities['validation'][k]) for k in identities['train']}
    require(not any(overlap.values()), 'Cross-split row/group/sequence/cluster/input overlap')
    identical(selection_report, manifest['source_selection'], 'Selection file and manifest differ')
    identical(selection_report['cross_split_overlap'], {k:v for k,v in overlap.items() if k != 'inputs'}, 'Selection overlap summary differs')
    return details, overlap, prefix


class StreamAudit:
    """Compare physical uint16 bins/index to independently tokenized raw records."""
    def __init__(self, root, name, descriptor, tokenizer, budget=8388608):
        self.name, self.tokenizer, self.budget = name, tokenizer, budget
        require(descriptor['bin'] == 'streams/' + name + '.bin' and descriptor['index'] == 'streams/' + name + '.index.jsonl', 'Unexpected stream path')
        path = root / descriptor['bin']
        require(path.stat().st_size == 2 * budget, 'Stream byte budget differs')
        self.values = np.memmap(path, mode='r', dtype='<u2')
        self.index = list(json_rows(root / descriptor['index']))
        self.descriptor = descriptor
        self.position, self.offset = 0, 0
        self.seen, self.full, self.characters = set(), set(), 0

    @property
    def complete(self):
        return self.offset == self.budget

    def take(self, parent, source_index, text, ids=None):
        require(not self.complete and self.position < len(self.index) and parent not in self.seen, 'Stream index exhaustion/duplicate parent')
        ids = content_ids(self.tokenizer, text) if ids is None else ids
        used = min(len(ids), self.budget - self.offset - 1)
        tokens = ids[:used] + [1]
        require(np.array_equal(self.values[self.offset:self.offset+len(tokens)], tokens), 'Stream tokens differ from raw parent reencoding: ' + self.name)
        expected = {'parent_id': parent, 'source_record_index': source_index, 'text_sha256': text_sha(text),
            'content_tokens_full': len(ids), 'content_tokens_used': used, 'offset': self.offset,
            'length': len(tokens), 'eos_offset': self.offset+used, 'truncated': used < len(ids),
            'source_characters': len(text), 'exposed_characters': len(self.tokenizer.decode(ids[:used], skip_special_tokens=False))}
        identical(self.index[self.position], expected, 'Stream index record differs: ' + self.name)
        self.offset += len(tokens); self.position += 1; self.seen.add(parent)
        self.characters += expected['exposed_characters']
        if not expected['truncated']:
            self.full.add(parent)

    def finish(self):
        require(self.complete and self.position == len(self.index), 'Stream incomplete/index has extra rows')
        d = self.descriptor
        expected = {'tokens': self.budget, 'blocks': self.budget // 512, 'dtype': '<u2', 'records': len(self.index),
            'content_tokens': self.budget-len(self.index), 'eos_tokens': len(self.index),
            'clipped_records': sum(r['truncated'] for r in self.index), 'exposed_characters': self.characters,
            'ordered_parent_ids_sha256': text_sha(json.dumps([r['parent_id'] for r in self.index], separators=(',', ':'))),
            'last_source_record_index': self.index[-1]['source_record_index']}
        for key, value in expected.items():
            identical(d[key], value, 'Stream manifest summary differs: ' + self.name + ':' + key)
        return expected


def audit_schedules(root, schedules, seeds=(0,1,2), blocks=16384):
    require(set(schedules) == {str(s) for s in seeds}, 'Schedule seed coverage differs')
    reports = {}
    for seed in seeds:
        item = schedules[str(seed)]
        require(item['file'] == f'schedules/seed{seed}.npy', 'Schedule path differs')
        observed = np.load(root / item['file'], allow_pickle=False)
        require(observed.dtype == np.dtype('int64') and observed.shape == (blocks//8, 16, 2), 'Schedule dtype/shape differs')
        for stream in (0,1):
            slot = observed[:,8*stream:8*(stream+1),:]
            require(np.all(slot[:,:,0] == stream), 'Schedule domain slot differs')
            flat = slot[:,:,1].ravel()
            require(np.array_equal(np.sort(flat), np.arange(blocks)), 'Schedule duplicates or omits blocks')
            seed_value = int(hashlib.sha256(f'joint-v1:schedule:{seed}:{stream}'.encode()).hexdigest(),16)
            expected = np.random.default_rng(seed_value).permutation(blocks)
            require(np.array_equal(flat, expected), 'Frozen seeded schedule order differs')
        identical(item['shape'], list(observed.shape), 'Schedule reported shape differs')
        require(item['dtype'] == 'int64', 'Schedule reported dtype differs')
        reports[str(seed)] = {'microbatches': blocks//8, 'stream_blocks_each': blocks, 'complete_permutations': True}
    return reports


def audit_streams(prepared, manifest, english_path, protein_path, canonical, forbidden, tokenizer, budget=8388608):
    require(set(manifest['streams']) == set(STREAMS), 'Four-stream coverage differs')
    readers = {name: StreamAudit(prepared, name, manifest['streams'][name], tokenizer, budget) for name in STREAMS}
    seen_ids, seen_texts = set(), set()
    for position, row in enumerate(json_rows(english_path)):
        parent, text = row['record_id'], row['text']
        sha = text_sha(text)
        require(row['split'] == 'train' and row['text_sha256'] == sha and parent not in seen_ids and sha not in seen_texts, 'Invalid/repeated English input record')
        seen_ids.add(parent); seen_texts.add(sha)
        name = 'english_extra' if readers['english_common'].complete else 'english_common'
        readers[name].take(parent, position, text)
        if readers['english_extra'].complete:
            break
    require(not readers['english_common'].seen & readers['english_extra'].seen, 'EE additional English shares a common-half parent')
    print('Audited both English streams', flush=True)
    originals = list(json_rows(protein_path))
    allowed = {key for key,v in canonical.items() if v['split'] == 'train' and v['eligible']}
    require(len(originals) == len(allowed) and {r['record_id'] for r in originals} == allowed, 'Protein mother pool differs from every train-eligible canonical parent')
    controls, natural_hashes, control_hashes, seeds = [], [], [], []
    for row in originals:
        parent, text = row['record_id'], row['text']
        c = canonical[parent]
        require(row['split'] == 'train' and row['cluster_id'] == c['cluster_id'], 'Protein mother split/cluster differs')
        require(text_sha(text) == row['text_sha256'] == c['sequence_sha256'] and len(text) == c['length'], 'Protein mother identity differs')
        seed = hashlib.sha256(('joint-v1:shuffle:0:' + parent).encode()).hexdigest()
        control = list(text)
        random.Random(int(seed,16)).shuffle(control)
        control = ''.join(control)
        require(Counter(control) == Counter(text), 'Generated control composition differs')
        controls.append(control); natural_hashes.append(text_sha(text)); control_hashes.append(text_sha(control)); seeds.append(seed)
    require(len(set(natural_hashes)) == len(originals), 'Duplicate natural protein text')
    frequencies = Counter(control_hashes)
    ledger_name = manifest['protein_parent_controls']
    require(ledger_name == 'protein_parent_controls.jsonl.gz', 'Unexpected parent ledger path')
    ledger = list(json_rows(prepared / ledger_name))
    require(len(ledger) == len(originals), 'Parent ledger omits/duplicates rows')
    eligible_count, reason_counts, lengths, totals = 0, Counter(), {'natural': [], 'shuffled': []}, {'protein':0,'shuffled':0}
    for position, row in enumerate(originals):
        parent, text, control = row['record_id'], row['text'], controls[position]
        reasons = []
        if natural_hashes[position] in forbidden: reasons.append('natural_heldout_collision')
        if control_hashes[position] in forbidden: reasons.append('shuffled_heldout_collision')
        if frequencies[control_hashes[position]] > 1: reasons.append('duplicate_shuffled_control')
        a, b = content_ids(tokenizer, text), content_ids(tokenizer, control)
        lengths['natural'].append(len(a)); lengths['shuffled'].append(len(b))
        expected = {'parent_id': parent, 'source_record_index': position, 'natural_sha256': natural_hashes[position],
            'shuffled_sha256': control_hashes[position], 'shuffle_seed_hex': seeds[position],
            'source_characters': len(text), 'eligible': not reasons, 'exclusion_reasons': sorted(reasons),
            'unchanged': text == control, 'eligible_pool_index': eligible_count if not reasons else None,
            'natural_content_tokens': len(a), 'shuffled_content_tokens': len(b)}
        identical(ledger[position], expected, 'Protein common-pool/control ledger differs')
        reason_counts.update(reasons)
        if not reasons:
            eligible_count += 1; totals['protein'] += len(a)+1; totals['shuffled'] += len(b)+1
            for name, value, encoded in [('protein',text,a),('shuffled',control,b)]:
                if not readers[name].complete:
                    readers[name].take(parent,position,value,encoded)
        if (position + 1) % 25000 == 0:
            print('Audited protein parent/control records:', position + 1, flush=True)
    natural_set = set(natural_hashes)
    report = {'original_parents': len(originals), 'eligible_common_parents': eligible_count,
        'excluded_parents': len(originals)-eligible_count, 'reason_counts': dict(reason_counts),
        'duplicate_generated_control_groups': sum(n>1 for n in frequencies.values()),
        'unchanged_controls': sum(a['text']==b for a,b in zip(originals,controls)),
        'controls_equal_other_natural_training_parent': sum(x in natural_set and x != natural_hashes[i] for i,x in enumerate(control_hashes)),
        'eligible_available_tokens_including_eos': totals,
        'content_token_length_all_parents': {name:{'min':min(v),'max':max(v),'mean':float(np.mean(v)),'median':float(np.median(v))} for name,v in lengths.items()}}
    identical(report, manifest['parent_control_summary'], 'Parent control summary differs')
    identical(report, read_json(prepared/'protein_control_audit.json'), 'Parent control report file differs')
    result = {name: reader.finish() for name,reader in readers.items()}
    coverage = {'same_eligible_parent_pool_order': True,
        'used_parent_intersection': len(readers['protein'].seen & readers['shuffled'].seen),
        'complete_parent_intersection': len(readers['protein'].full & readers['shuffled'].full),
        'protein_only_used_parents': len(readers['protein'].seen - readers['shuffled'].seen),
        'shuffled_only_used_parents': len(readers['shuffled'].seen - readers['protein'].seen),
        'exposed_residues': {k:readers[k].characters for k in ('protein','shuffled')}, 'equal_residue_exposure_claimed':False}
    identical(coverage,manifest['parent_prefix_coverage'],'Parent prefix coverage differs')
    identical(coverage,read_json(prepared/'parent_prefix_coverage.json'),'Parent prefix report file differs')
    return result, report, coverage


def expected_output_names():
    names = {'source_selection.json','build_policy.json','confirmation_metadata_inheritance.json',
             'source_identity_and_exclusions.json','protein_control_audit.json',
             'parent_prefix_coverage.json','protein_parent_controls.jsonl.gz'}
    names |= {'source/' + split + suffix for split in ('train','validation') for suffix in ('.jsonl','.npz','.rows.jsonl.gz')}
    names |= {'streams/' + stream + suffix for stream in STREAMS for suffix in ('.bin','.index.jsonl')}
    names |= {f'schedules/seed{seed}.npy' for seed in (0,1,2)}
    return names


def validate_bound_inputs(protocol_path, protocol_sha, prepared, watch):
    protocol_path = watch.pin(protocol_path)
    require(watch.files[str(protocol_path)]['sha256'] == protocol_sha, 'Data protocol hash differs')
    protocol = read_json(protocol_path)
    require(protocol['schema_version'] == 1 and protocol['release_id'] == RELEASE_ID, 'Wrong data protocol/release version')
    require(protocol['training_enabled'] is False and protocol['target_scoring_enabled'] is False and protocol['source_test_examples_allowed'] is False, 'Build protocol role scope differs')
    require(protocol['output_root'] == str(prepared), 'Prepared directory differs from frozen protocol')
    identical(protocol['data']['source'], SOURCE_POLICY, 'Source scientific parameters differ')
    require(protocol['data']['token_budget_per_stream'] == 8388608 and protocol['data']['seeds'] == [0,1,2], 'Stream budget/seeds differ')
    raw_root = Path(protocol['raw_release_root'])
    require(raw_root.is_absolute() and raw_root != prepared and not prepared.is_relative_to(raw_root), 'Unsafe prepared/raw relationship')
    for key, rel in [('raw_acceptance','portable_validation/acceptance.json'),('raw_manifest','manifest.json')]:
        spec = protocol[key]
        require(Path(spec['path']) == inside(raw_root,rel), 'Wrong raw gate path')
        watch.pin(spec['path'],spec)
    acceptance, raw_manifest = read_json(protocol['raw_acceptance']['path']), read_json(protocol['raw_manifest']['path'])
    for obj in (acceptance,raw_manifest):
        require(obj['status'] == 'accepted_new_raw_release_training_disabled' and obj['release_id'] == RELEASE_ID, 'Raw acceptance identity differs')
        for key in ('training_enabled','training_gate_pass','target_scoring_enabled','derived_reconstruction_authorized'):
            require(obj[key] is False, 'Raw gate authority changed: ' + key)
    require(raw_manifest['acceptance']['sha256'] == protocol['raw_acceptance']['sha256'], 'Raw manifest does not bind acceptance')
    inventory = {v['file']:v for v in raw_manifest['files']}
    require(len(inventory) == len(raw_manifest['files']) == len(acceptance['files']) == 361, 'Raw closed inventory differs')
    expected = {'raw:'+rel for rel in RAW_MEMBERS} | set(META_SUFFIXES)
    require(set(protocol['inputs']) == expected, 'Source-only input allowlist differs')
    inputs = {}
    for key,spec in protocol['inputs'].items():
        path = Path(spec['path'])
        if key.startswith('raw:'):
            rel = key[4:]
            require(path == inside(raw_root,rel), 'Raw input path differs from versioned release')
            for declared in (inventory[rel],acceptance['files'][rel]):
                identical({k:spec[k] for k in ('sha256','bytes')},{k:declared[k] for k in ('sha256','bytes')},'Raw own-file binding differs')
            require(acceptance['files'][rel]['equivalent_to_declared_reference'] is True,'Raw referent not accepted')
        else:
            require(path.as_posix().endswith('/'+META_SUFFIXES[key]),'Unexpected aggregate metadata input path')
            require(not any(s in path.parts for s in ('raw','data','predictions','checkpoints')),'Example/model input forbidden')
        inputs[key] = watch.pin(path,spec)
    require(str(Path(__file__).resolve()) in protocol['code_pins'], 'Independent audit source is not frozen')
    for path,pin in protocol['code_pins'].items():
        watch.pin(path,pin)
    manifest_path = watch.pin(prepared/'manifest.json')
    manifest = read_json(manifest_path)
    require(manifest['schema_version'] == 1 and manifest['status'] == 'built_pending_independent_audit', 'Unrecognized prepared manifest state')
    require(manifest['training_ready'] is False and manifest['training_enabled'] is False and manifest['target_scoring_enabled'] is False,'Builder cannot grant training authority')
    require(manifest['raw_release_id'] == RELEASE_ID and manifest['raw_release_root'] == str(raw_root),'Prepared raw identity differs')
    for key,pkey in [('raw_acceptance_sha256','raw_acceptance'),('raw_manifest_sha256','raw_manifest')]:
        require(manifest[key] == protocol[pkey]['sha256'],'Prepared raw gate hash differs')
    require(manifest['build_protocol_sha256'] == manifest['data_protocol_sha256'] == protocol_sha,'Prepared data protocol binding differs')
    expected_inputs = dict(protocol['inputs'])
    expected_inputs.update(raw_acceptance=protocol['raw_acceptance'],raw_manifest=protocol['raw_manifest'],
                           data_protocol={'path':str(protocol_path),**watch.files[str(protocol_path)]})
    expected_inputs.update({'code:'+path:{'path':path,**pin} for path,pin in protocol['code_pins'].items()})
    identical(manifest['inputs'],expected_inputs,'Prepared input inventory is not the exact frozen allowlist')
    require(set(manifest['outputs']) == expected_output_names(),'Prepared output coverage differs')
    actual = {p.relative_to(prepared).as_posix() for p in prepared.rglob('*') if p.is_file()}
    require(actual == expected_output_names() | {'manifest.json'},'Unexpected/missing prepared files')
    for name,pin in manifest['outputs'].items():
        watch.pin(inside(prepared,name),pin)
    for split in ('train','validation'):
        item = manifest['source'][split]
        for field,key in [('npz','npz_sha256'),('rows','rows_sha256')]:
            require(item[key] == manifest['outputs'][item[field]]['sha256'],'Source artifact SHA field differs')
        raw_name = f'source/{split}.jsonl'
        require(item['raw_input'] == str(prepared/raw_name) and item['raw_input_sha256'] == manifest['outputs'][raw_name]['sha256'],'Source raw row provenance differs')
    for item in manifest['streams'].values():
        for field,key in [('bin','bin_sha256'),('index','index_sha256')]:
            require(item[key] == manifest['outputs'][item[field]]['sha256'],'Stream descriptor hash differs')
    for item in manifest['schedules'].values():
        require(item['sha256'] == manifest['outputs'][item['file']]['sha256'],'Schedule hash descriptor differs')
    identical(manifest['condition_streams'],CONDITIONS,'Condition/domain stream layout differs')
    identical(manifest['source_counts'],{'train':8044,'validation':20276},'New source role counts differ')
    return protocol, manifest, acceptance, inputs


def audit_confirmation_metadata(prepared, manifest, acceptance, inputs):
    qualified = read_json(inputs['confirmation_acceptance'])
    old_manifest = read_json(inputs['confirmation_manifest'])
    refs = read_json(inputs['confirmation_references'])
    old_audit = read_json(inputs['confirmation_audit'])
    require(qualified['status'] == 'accepted_for_local_confirmation' and qualified['confirmation_data_ready'] is True,'Historical qualification absent')
    require(old_audit['status'] == 'passed' and old_audit['confirmation_data_qualified'] is True,'Historical qualification audit absent')
    for name, rel in [('confirmation_manifest','manifest.json'),('confirmation_references','references.json'),('confirmation_audit','verification/independent_verification.json')]:
        require(identity(inputs[name])['sha256'] == qualified['files'][rel],'Historical aggregate metadata not bound')
    bindings = {}
    for name in ('data/corpora/english/train.jsonl.gz','tokenizers/mixed_bpe/tokenizer.json'):
        prior, current = refs['files'][name], acceptance['files'][name]
        require(current['reference_kind'] == 'accepted_v2' and current['equivalent_to_declared_reference'] is True,'Inherited exposure reference changed')
        require(current['reference_sha256'] == prior['sha256'] == old_manifest['inputs'][prior['path']] == old_audit['file_sha256'][prior['path']],'Historical aggregate exposure identity differs')
        bindings[name] = {'current_sha256':current['sha256'],'historical_reference_sha256':prior['sha256'],
            'raw_gate_comparison_mode':current['comparison_mode'],'same_scientific_content':True}
    expected = {'acceptance_sha256':identity(inputs['confirmation_acceptance'])['sha256'],
        'same_full_english_train_pool_and_tokenizer_scientific_content':True,'reference_bindings':bindings,
        'retained_lexical_hits_zero_inherited_from_M1':True,'target_raw_or_encoded_or_row_ledger_files_opened':False,
        'qualification_scope':qualified['qualification_scope'],'historical_target_has_already_been_scored':True,
        'new_blind_confirmation_claimed':False,'current_target_bytes_verification_deferred_to_separate_scoring_gate':True}
    identical(manifest['confirmation_metadata_inheritance'],expected,'Exposure/target-observation scope differs')
    identical(read_json(prepared/'confirmation_metadata_inheritance.json'),expected,'Exposure inheritance report differs')
    return expected


def audit(protocol_path, protocol_sha, prepared, output):
    prepared, output = Path(prepared).absolute(), Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), 'Fresh audit output required')
    require(not any(p.is_symlink() for p in output.parents),'Symlinked audit destination')
    require(not output.is_relative_to(prepared),'Audit output must be outside immutable prepared files')
    watch = ReadSet()
    protocol, manifest, acceptance, inputs = validate_bound_inputs(protocol_path,protocol_sha,prepared,watch)
    require(not output.is_relative_to(Path(protocol['raw_release_root'])),'Cannot write into raw release')
    from tokenizers import Tokenizer
    tok_path = inputs['raw:tokenizers/mixed_bpe/tokenizer.json']
    identical(manifest['tokenizer'],{'file':str(tok_path),'sha256':identity(tok_path)['sha256'],'vocab_size':32000,'pad_id':0,'eos_id':1,'sep_id':2,'unk_id':3},'Tokenizer descriptor differs')
    tokenizer = Tokenizer.from_file(str(tok_path)); tokenizer.no_padding(); tokenizer.no_truncation()
    require(tokenizer.get_vocab_size() == 32000 and [tokenizer.token_to_id(t) for t in ('<|pad_v2|>','<|eos_v2|>','<|pair_v2|>','<|unk_v2|>')] == [0,1,2,3],'Tokenizer vocabulary/special IDs differ')
    parents, sequences, clusters = canonical_metadata(inputs['raw:data/sequences/protein_canonical_sequences.tsv.gz'])
    remote_rows = list(tsv_rows(inputs['raw:data/sequences/remote_sequences.tsv.gz']))
    remote = {row['sequence_sha256'] for row in remote_rows}
    require(remote and all(len(s)==64 for s in remote),'Empty/malformed remote identity table')
    remote_count = len(remote_rows); del remote_rows
    nontrain = {key for key,v in sequences.items() if v['split'] != 'train'}
    forbidden = nontrain | remote
    train_records = read_json(inputs['raw:tokenizers/mixed_bpe/training_records.json'])
    require(all(p in parents and parents[p]['split']=='train' and parents[p]['eligible'] for p in train_records['protein']),'Tokenizer training parents are not train eligible')
    selection, overlap, prefixes = audit_source(prepared,manifest,
        {s:inputs[f'raw:data/pairs/protein_sequence_similarity_{s}.tsv.gz'] for s in ('train','validation')},tokenizer,sequences)
    confirmation = audit_confirmation_metadata(prepared,manifest,acceptance,inputs)
    streams, controls, coverage = audit_streams(prepared,manifest,inputs['raw:data/corpora/english/train.jsonl.gz'],
        inputs['raw:data/corpora/protein/train.jsonl.gz'],parents,forbidden,tokenizer)
    schedules = audit_schedules(prepared,manifest['schedules'])
    split_policy = read_json(inputs['raw:metadata/protein_split_policy.json'])
    exclusion = read_json(inputs['raw:metadata/protein_remote_pretraining_exclusion.json'])
    require(len(clusters)==split_policy['cluster_count'] and exclusion['status']=='completed','Canonical/exclusion metadata differs')
    expected_inheritance = {'source_train_validation_reencoded':True,'source_train_validation_overlap':overlap,
        'source_test_pair_examples_or_labels_read':False,'operational_cluster_definition':split_policy['definition'],
        'near_homology_exclusion':exclusion['criterion'],'near_homology_search_rerun_in_this_stage':False,
        'search_limitations':split_policy['search_limitations'],'structural_family_overlap':None,
        'structural_family_availability':'not_provided; unmeasured, not zero','canonical_nontrain_hashes':len(nontrain),
        'remote_domain_rows':remote_count,'remote_all_split_hashes':len(remote),'combined_exact_exclusion_hashes':len(forbidden),
        'tokenizer_protein_parents_all_train_eligible':True,'tokenizer_protein_training_records':len(train_records['protein']),
        'identity_tables_physically_read':True,'heldout_pair_files_opened':False}
    identical(manifest['source_provenance_and_exclusions'],expected_inheritance,'Source exclusion scope/report differs')
    identical(read_json(prepared/'source_identity_and_exclusions.json'),expected_inheritance,'Source exclusion file differs')
    identical(manifest['read_boundaries'],{'source_test_pair_files_opened':0,'qqp_example_or_row_audit_files_opened':0,
        'old_predictions_opened':0,'old_prepared_source_files_opened':0,'identity_tables_used_for_exclusion':True},'Declared read boundary differs')
    policy = read_json(prepared/'build_policy.json')
    require(policy['data_protocol_sha256']==protocol_sha and policy['same_schedule_file_all_conditions'] is True and
            policy['source_test_pair_examples_read'] is False and policy['target_examples_read'] is False and
            policy['model_training_or_inference'] is False,'Builder policy scope differs')
    watch.finish()
    require({p.relative_to(prepared).as_posix() for p in prepared.rglob('*') if p.is_file()} == expected_output_names() | {'manifest.json'},'Prepared inventory changed during audit')
    report = {'schema_version':1,'status':'source_only_smoke_inputs_accepted','all_checks_passed':True,
        'data_protocol_sha256':protocol_sha,'prepared_manifest_sha256':watch.files[str(prepared/'manifest.json')]['sha256'],
        'raw_release_id':RELEASE_ID,'raw_release_manifest_sha256':protocol['raw_manifest']['sha256'],
        'raw_release_acceptance_sha256':protocol['raw_acceptance']['sha256'],'prepared_root':str(prepared),
        'bounded_smoke_training_enabled':True,'full_training_enabled':False,'full_budget_training_enabled':False,
        'target_scoring_enabled':False,'source_test_scoring_enabled':False,'new_blind_confirmation_claimed':False,
        'source_counts':{k:v['rows'] for k,v in selection.items()},'source':selection,'source_overlap':overlap,
        'source_prefix64':prefixes,'streams':streams,'schedules':schedules,'protein_control_summary':controls,
        'parent_prefix_coverage':coverage,'confirmation_metadata_inheritance':confirmation,
        'outputs':manifest['outputs'],'verified_read_files':watch.files,'all_read_files_final_rehashed':len(watch.files),
        'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'limitations':['Technical smoke input qualification only; full-budget training and heldout scoring remain disabled.',
                       'Identity tables were read for exclusions; no source-test pair or QQP example/label/prediction files were opened.',
                       'Historical QQP has already been scored; aggregate exposure inheritance does not create a new blind target.',
                       'Independent audit does not rerun near-homology alignment or certify structural-family independence.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--protocol-sha256',required=True)
    parser.add_argument('--prepared',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    result = audit(args.protocol,args.protocol_sha256,args.prepared,args.output)
    print(json.dumps({'status':result['status'],'source_counts':result['source_counts'],'output':args.output}),flush=True)


if __name__ == '__main__':
    main()
