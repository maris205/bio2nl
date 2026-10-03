"""Build the fixed joint-v1 data on CPU; target and source-test examples forbidden."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V2 = ROOT / 'data_rebuild/2026-09-25-v2'
EVAL = ROOT / 'bio2nl/review/eval_v2/prepared'
M1 = ROOT / 'bio2nl/review/confirmation_data_v1_2026-09-29'
DRAFT = ROOT / 'bio2nl/review/next_round_preparation_v1_2026-09-29'
BUDGET = 8_388_608
RELEASE_SHA = '34f3bd78b6a4d923fafe3d76a53d0741886e39bc0af61245ff5bd6b34c511317'
GATE_SHA = '60912c70e78ada4d263712e54a38669f550d5ea246a30c8c9a62d2be7fb98320'
TOKENIZER_SHA = '1bb8092348368961ca211cbf4d066a0e0ebf612b35b401b07ed173476088130b'


def now():
    return datetime.now(timezone.utc).isoformat()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def guard(path):
    path = Path(path).resolve()
    if path.is_relative_to(M1):
        require(path.relative_to(M1).parts[0] not in ('raw', 'data'), 'QQP content forbidden')
        require(path.name not in ('exclusions.jsonl', 'overlap_hits.jsonl'), 'QQP row ledger forbidden')
    require('predictions' not in path.parts and 'checkpoints' not in path.parts, 'Model artifacts forbidden')
    require(not path.name.startswith('protein_sequence_similarity_test.'), 'Source test examples forbidden')
    require(not (path.name.startswith('test.') and 'protein_sequence_similarity' in path.parts), 'Source test examples forbidden')
    return path


def sha(path):
    h = hashlib.sha256()
    with guard(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def read(path):
    return json.loads(guard(path).read_text())


def rows(path):
    path = guard(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def write_json(path, value):
    path = Path(path)
    require(not path.exists(), 'Refusing overwrite: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')


def write_rows(path, records):
    path = Path(path)
    require(not path.exists(), 'Refusing overwrite: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'wt', encoding='utf-8') as f:
        for item in records:
            f.write(json.dumps(item, sort_keys=True, separators=(',', ':'), ensure_ascii=False) + '\n')


class Inputs:
    def __init__(self):
        self.hashes = {}
        self.release_files = {}

    def pin(self, path, expected=None):
        path = guard(path)
        if path.is_relative_to(V2) and path != V2 / 'manifest.json':
            rel = str(path.relative_to(V2))
            require(rel in self.release_files, 'Not an accepted v2 artifact: ' + rel)
            accepted = self.release_files[rel]
            require(expected is None or expected == accepted, 'Conflicting expected hash')
            expected = accepted
        actual = sha(path)
        require(expected is None or expected == actual, 'Input hash changed: ' + str(path))
        self.hashes[str(path)] = actual
        return path

    def js(self, path, expected=None):
        return read(self.pin(path, expected))


def shuffle_record(parent_id, sequence):
    seed_hex = text_sha('joint-v1:shuffle:0:' + parent_id)
    chars = list(sequence)
    random.Random(int(seed_hex, 16)).shuffle(chars)
    return ''.join(chars), seed_hex


def encode_content(tokenizer, text):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    ids = encoded.ids
    require(ids and all(4 <= i < 32000 for i in ids), 'Empty, unknown or special content token')
    require(tokenizer.decode(ids, skip_special_tokens=False) == text, 'Full input roundtrip failed')
    return ids


def make_schedule(seed, blocks=16384, half=8):
    require(blocks % half == 0, 'Incomplete microbatch')
    schedule = np.empty((blocks // half, half * 2, 2), dtype=np.int64)
    for stream_id in (0, 1):
        derived = int(text_sha(f'joint-v1:schedule:{seed}:{stream_id}'), 16)
        values = np.random.default_rng(derived).permutation(blocks).reshape(-1, half)
        schedule[:, stream_id * half:(stream_id + 1) * half, 0] = stream_id
        schedule[:, stream_id * half:(stream_id + 1) * half, 1] = values
    return schedule


class StreamWriter:
    def __init__(self, folder, name, tokenizer, budget=BUDGET):
        self.name, self.tokenizer, self.budget = name, tokenizer, budget
        self.binary, self.index = folder / (name + '.bin'), folder / (name + '.index.jsonl')
        require(not self.binary.exists() and not self.index.exists(), 'Existing stream artifacts')
        folder.mkdir(parents=True, exist_ok=True)
        self.handle = self.binary.open('wb')
        self.index_rows, self.offset = [], 0

    @property
    def complete(self):
        return self.offset == self.budget

    def append(self, parent, position, text, ids):
        require(not self.complete, 'Stream already complete')
        used = min(len(ids), self.budget - self.offset - 1)
        selected = ids[:used] + [1]
        np.asarray(selected, dtype='<u2').tofile(self.handle)
        self.index_rows.append({'parent_id': parent, 'source_record_index': position,
            'text_sha256': text_sha(text), 'content_tokens_full': len(ids), 'content_tokens_used': used,
            'offset': self.offset, 'length': len(selected), 'eos_offset': self.offset + used,
            'truncated': used < len(ids), 'source_characters': len(text),
            'exposed_characters': len(self.tokenizer.decode(ids[:used], skip_special_tokens=False))})
        self.offset += len(selected)

    def finish(self):
        self.handle.close()
        require(self.complete, 'Insufficient distinct eligible tokens: ' + self.name)
        write_rows(self.index, self.index_rows)
        require(len({r['parent_id'] for r in self.index_rows}) == len(self.index_rows), 'Repeated stream parent')
        return {'bin': str(self.binary.relative_to(HERE)), 'index': str(self.index.relative_to(HERE)),
            'bin_sha256': sha(self.binary), 'index_sha256': sha(self.index), 'tokens': self.offset,
            'blocks': self.offset // 512, 'dtype': '<u2', 'records': len(self.index_rows),
            'content_tokens': self.offset - len(self.index_rows), 'eos_tokens': len(self.index_rows),
            'clipped_records': sum(r['truncated'] for r in self.index_rows),
            'exposed_characters': sum(r['exposed_characters'] for r in self.index_rows),
            'ordered_parent_ids_sha256': text_sha(json.dumps([r['parent_id'] for r in self.index_rows], separators=(',', ':'))),
            'last_source_record_index': self.index_rows[-1]['source_record_index']}


def read_canonical_metadata(path):
    parents, sequences, clusters = {}, {}, {}
    with gzip.open(path, 'rt') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            # Deliberately do not access the sequence column or any pair labels.
            parent = f"swissprot:{row['accession']}:{row['sequence_version']}"
            split, cluster = row['split'], row['cluster_id']
            require(parent not in parents, 'Duplicate canonical parent')
            require(split in ('train', 'validation', 'test'), 'Unexpected canonical split')
            require(cluster not in clusters or clusters[cluster] == split, 'Canonical cluster crosses splits')
            clusters[cluster] = split
            value = {'sequence_sha256': row['sequence_sha256'], 'split': split, 'cluster_id': cluster,
                'length': int(row['length']), 'eligible': row['pretrain_eligible'].lower() in ('1', 'true', 'yes')}
            previous = sequences.get(value['sequence_sha256'])
            require(previous is None or previous['split'] == split, 'Canonical sequence crosses splits')
            parents[parent] = value
            sequences[value['sequence_sha256']] = value
    return parents, sequences, clusters


def remote_hashes(path):
    found = set()
    count = 0
    with gzip.open(path, 'rt') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            found.add(row['sequence_sha256'])
            count += 1
    require(count > 0 and all(len(x) == 64 for x in found), 'Invalid remote identities')
    return found, count


def make_source(split, raw_path, tokenizer, canonical_sequences, folder):
    require(split in ('train', 'validation'), 'Source-test encoding forbidden')
    raw_rows = list(rows(raw_path))
    n = len(raw_rows)
    ids_array = np.zeros((n, 512), dtype=np.int64)
    masks = np.zeros((n, 512), dtype=np.int64)
    labels = np.empty(n, dtype=np.int64)
    degree, block_degree = defaultdict(Counter), defaultdict(Counter)
    output, identity, pair_labels = [], {k: set() for k in ('row_ids', 'groups', 'sequences', 'clusters', 'inputs')}, {}
    truncated, collisions = Counter(), 0
    for index, raw in enumerate(raw_rows):
        metadata, label = raw['metadata'], raw['label']
        require(type(label) is int and label in (0, 1), 'Invalid source label')
        require(raw['row_id'] not in identity['row_ids'], 'Duplicate source row')
        require(metadata['split'] == split, 'Wrong source split')
        original = [encode_content(tokenizer, raw[field]) for field in ('sentence1', 'sentence2')]
        a, b = [x[:255] for x in original]
        combined = a + [2] + b + [1]
        require(len(combined) <= 512, 'Source input overflow')
        key = tuple(combined)
        if key in pair_labels:
            require(pair_labels[key] == label, 'Conflicting encoded pair labels')
            collisions += 1
        pair_labels[key] = label
        for side, text, content, full in zip(('a', 'b'), (raw['sentence1'], raw['sentence2']), (a, b), original):
            digest = text_sha(text)
            require(digest == metadata['sequence_sha256_' + side], 'Source sequence hash mismatch')
            c = canonical_sequences[digest]
            require(c['split'] == split and c['cluster_id'] == metadata['cluster_' + side], 'Source canonical identity differs')
            degree[(side, tuple(content))][label] += 1
            block_degree[(metadata['block_id'], side, tuple(content))][label] += 1
            truncated[str(label)] += len(full) > 255
            identity['sequences'].add(digest)
            identity['clusters'].add(c['cluster_id'])
        identity['row_ids'].add(raw['row_id'])
        identity['groups'].add(metadata['block_id'])
        identity['inputs'].add(key)
        ids_array[index, :len(combined)] = combined
        masks[index, :len(combined)] = 1
        labels[index] = label
        output.append({'row_index': index, 'row_id': raw['row_id'], 'label': label, 'ids_a': a, 'ids_b': b,
            'metadata': metadata, 'raw_lengths': [len(raw['sentence1']), len(raw['sentence2'])],
            'precap_token_lengths': [len(x) for x in original]})
    require(n and all(v[0] == v[1] for v in degree.values()), 'Endpoint role-label balance failed')
    require(all(v[0] == v[1] for v in block_degree.values()), 'Within-block role-label balance failed')
    folder.mkdir(parents=True, exist_ok=True)
    npz, rowfile = folder / (split + '.npz'), folder / (split + '.rows.jsonl.gz')
    require(not npz.exists(), 'Existing source tensors')
    np.savez_compressed(npz, input_ids=ids_array, attention_mask=masks, labels=labels)
    write_rows(rowfile, output)
    entry = {'npz': str(npz.relative_to(HERE)), 'rows': str(rowfile.relative_to(HERE)), 'count': n,
        'npz_sha256': sha(npz), 'rows_sha256': sha(rowfile), 'raw_input': str(raw_path), 'raw_input_sha256': sha(raw_path),
        'labels': dict(Counter(map(str, labels.tolist()))), 'shape': [n, 512], 'dtype': 'int64',
        'endpoint_role_label_balance': True, 'within_block_role_label_balance': True,
        'truncated_endpoints_by_label': dict(truncated), 'same_label_encoded_pair_collisions': collisions,
        'conflicting_encoded_pair_labels': 0, 'identity_counts': {k: len(v) for k, v in identity.items()}}
    return entry, identity


def build_protein(records, canonical, forbidden_hashes, tokenizer, folder):
    allowed = {key for key, value in canonical.items() if value['split'] == 'train' and value['eligible']}
    require(len(records) == len(allowed) and {r['record_id'] for r in records} == allowed, 'Natural raw parent pool differs')
    natural_hashes, controls, hashes, seeds = [], [], [], []
    for row in records:
        parent, text = row['record_id'], row['text']
        c = canonical[parent]
        require(row['split'] == 'train' and row['cluster_id'] == c['cluster_id'], 'Natural split/cluster mismatch')
        digest = text_sha(text)
        require(digest == c['sequence_sha256'] == row['text_sha256'] and len(text) == c['length'], 'Natural canonical sequence mismatch')
        control, seed = shuffle_record(parent, text)
        require(Counter(control) == Counter(text), 'Shuffle composition differs')
        natural_hashes.append(digest); controls.append(control); hashes.append(text_sha(control)); seeds.append(seed)
    require(len(set(natural_hashes)) == len(records), 'Duplicate natural raw text')
    frequencies = Counter(hashes)
    writers = {name: StreamWriter(folder / 'streams', name, tokenizer) for name in ('protein', 'shuffled')}
    parent_rows, excluded, reason_counts, order = [], set(), Counter(), 0
    length_stats = {'natural': [], 'shuffled': []}
    for index, row in enumerate(records):
        reasons = []
        if natural_hashes[index] in forbidden_hashes:
            reasons.append('natural_heldout_collision')
        if hashes[index] in forbidden_hashes:
            reasons.append('shuffled_heldout_collision')
        if frequencies[hashes[index]] > 1:
            reasons.append('duplicate_shuffled_control')
        eligible = not reasons
        natural_tokens = encode_content(tokenizer, row['text'])
        shuffled_tokens = encode_content(tokenizer, controls[index])
        length_stats['natural'].append(len(natural_tokens))
        length_stats['shuffled'].append(len(shuffled_tokens))
        parent_rows.append({'parent_id': row['record_id'], 'source_record_index': index,
            'natural_sha256': natural_hashes[index], 'shuffled_sha256': hashes[index],
            'shuffle_seed_hex': seeds[index], 'source_characters': len(row['text']),
            'eligible': eligible, 'exclusion_reasons': sorted(reasons), 'unchanged': controls[index] == row['text'],
            'eligible_pool_index': order if eligible else None,
            'natural_content_tokens': len(natural_tokens), 'shuffled_content_tokens': len(shuffled_tokens)})
        reason_counts.update(reasons)
        if eligible:
            order += 1
            for name, text, tokens in [('protein', row['text'], natural_tokens), ('shuffled', controls[index], shuffled_tokens)]:
                if not writers[name].complete:
                    writers[name].append(row['record_id'], index, text, tokens)
        else:
            excluded.add(row['record_id'])
        if (index + 1) % 25000 == 0:
            print('Protein parent/control processed:', index + 1, flush=True)
    entries = {name: writer.finish() for name, writer in writers.items()}
    parent_file = folder / 'protein_parent_controls.jsonl.gz'
    write_rows(parent_file, parent_rows)
    natural_set = set(natural_hashes)
    report = {'original_parents': len(records), 'eligible_common_parents': order, 'excluded_parents': len(excluded),
        'reason_counts': dict(reason_counts), 'duplicate_generated_control_groups': sum(n > 1 for n in frequencies.values()),
        'unchanged_controls': sum(r['unchanged'] for r in parent_rows),
        'controls_equal_other_natural_training_parent': sum(x in natural_set and x != natural_hashes[i] for i, x in enumerate(hashes)),
        'eligible_available_tokens_including_eos': {name: sum(r[key] + 1 for r in parent_rows if r['eligible']) for name, key in [('protein','natural_content_tokens'),('shuffled','shuffled_content_tokens')]},
        'content_token_length_all_parents': {name: {'min': min(values), 'max': max(values), 'mean': float(np.mean(values)), 'median': float(np.median(values))} for name, values in length_stats.items()}}
    return entries, str(parent_file.relative_to(HERE)), report


def main():
    from tokenizers import Tokenizer
    started = time.monotonic()
    folder = HERE / 'prepared'
    require(not folder.exists() or not any(folder.iterdir()), 'Existing prepared attempt; preserve before a new build')
    folder.mkdir(exist_ok=True)
    inputs = Inputs()
    release = inputs.js(V2 / 'manifest.json', RELEASE_SHA)
    inputs.release_files = {row['file']: row['sha256'] for row in release['files']}
    gate = inputs.js(V2 / 'validation/integrated_release_acceptance.json', GATE_SHA)
    require(release['training_gate_pass'] and gate['training_gate_pass'], 'Accepted v2 gate failed')
    design = inputs.js(DRAFT / 'configs/joint_pretraining_design.json')
    inputs.pin(DRAFT / 'JOINT_PRETRAINING_PROTOCOL.md')
    require(design['pretraining']['seeds'] == [0, 1, 2] and design['pretraining']['input_tokens_per_run'] == 2 * BUDGET, 'Fixed budget changed')
    inputs.pin(HERE / 'prepare_data.py')
    inputs.pin(HERE / 'test_prepare_data.py')
    policy = {'status': 'fixed_before_control_generation', 'fixed_at_utc': now(), 'python_version': platform.python_version(),
        'shuffle_seed_derivation': "int(SHA256('joint-v1:shuffle:0:' + parent_id), 16)",
        'shuffle': 'Python random.Random(full_256_bit_integer).shuffle(list(sequence)); one draw per parent; no retries',
        'raw_duplicate_ids_or_text': 'integrity_failure',
        'natural_heldout': 'natural_heldout_collision', 'shuffled_heldout': 'shuffled_heldout_collision',
        'duplicate_shuffled': 'duplicate_shuffled_control',
        'collision_scope': 'canonical nontraining sequence hashes union every accepted remote domain sequence hash',
        'duplicate_control_policy': 'Exclude every involved parent from both natural and shuffled eligible pools',
        'unchanged_control_policy': 'Report and retain unless another fixed exclusion applies',
        'control_equal_other_natural_train_policy': 'Report only',
        'source_test_examples_read': False, 'qqp_examples_read': False,
        'index_exposed_characters': 'Decoded used token prefix character count; English Unicode clipping may include replacement and is not exact original character coverage',
        'schedule_seed_derivation': "int(SHA256(f'joint-v1:schedule:{seed}:{stream_id}'),16)",
        'schedule_rng': 'numpy.random.default_rng independently per stream',
        'schedule_shape': [2048,16,2], 'same_schedule_file_all_conditions': True,
        'english_extra_start': 'Original row immediately after the final common-English parent; discard any remaining tokens of that parent'}
    write_json(folder / 'build_policy.json', policy)
    print('Fixed policy written; verifying source metadata', flush=True)

    english_path = inputs.pin(V2 / 'data/corpora/english/train.jsonl.gz')
    protein_path = inputs.pin(V2 / 'data/corpora/protein/train.jsonl.gz')
    tokenizer_path = inputs.pin(V2 / 'tokenizers/mixed_bpe/tokenizer.json', TOKENIZER_SHA)
    tokenizer = Tokenizer.from_file(str(tokenizer_path)); tokenizer.no_padding(); tokenizer.no_truncation()
    require(tokenizer.get_vocab_size() == 32000, 'Tokenizer vocabulary mismatch')
    require([tokenizer.token_to_id(x) for x in ['<|pad_v2|>','<|eos_v2|>','<|pair_v2|>','<|unk_v2|>']] == [0,1,2,3], 'Tokenizer special IDs differ')
    tokenizer_records = inputs.js(V2 / 'tokenizers/mixed_bpe/training_records.json')
    tokenizer_meta = inputs.js(V2 / 'tokenizers/mixed_bpe/metadata.json')
    for name in ('protein','english'):
        inputs.pin(V2 / f'metadata/corpora_{name}.json')
        inputs.pin(V2 / f'validation/corpora_{name}.json')
    split_policy = inputs.js(V2 / 'metadata/protein_split_policy.json')
    remote_exclusion = inputs.js(V2 / 'metadata/protein_remote_pretraining_exclusion.json')
    pair_gate = inputs.js(V2 / 'validation/protein_pair_acceptance.json')
    inputs.pin(V2 / 'metadata/protein_pair_construction.json')
    inputs.pin(V2 / 'validation/cross_dataset_homology.json')
    require(pair_gate['acceptance_pass'] and remote_exclusion['status'] == 'completed', 'Source/exclusion metadata failed')
    canonical_path = inputs.pin(V2 / 'data/sequences/protein_canonical_sequences.tsv.gz')
    remote_path = inputs.pin(V2 / 'data/sequences/remote_sequences.tsv.gz')
    canonical, by_sequence, clusters = read_canonical_metadata(canonical_path)
    require(len(clusters) == split_policy['cluster_count'], 'Canonical cluster count mismatch')
    canonical_forbidden = {key for key, val in by_sequence.items() if val['split'] != 'train'}
    remote_forbidden, remote_count = remote_hashes(remote_path)
    forbidden = canonical_forbidden | remote_forbidden
    require(set(tokenizer_records['protein']) <= set(canonical), 'Tokenizer protein parent absent')
    require(all(canonical[p]['split'] == 'train' and canonical[p]['eligible'] for p in tokenizer_records['protein']), 'Tokenizer protein heldout exposure')

    acceptance = inputs.js(M1 / 'confirmation_acceptance.json')
    require(acceptance['confirmation_data_ready'] and acceptance['status'] == 'accepted_for_local_confirmation', 'Target metadata acceptance failed')
    m1manifest = inputs.js(M1 / 'manifest.json', acceptance['files']['manifest.json'])
    references = inputs.js(M1 / 'references.json', acceptance['files']['references.json'])
    m1audit = inputs.js(M1 / 'verification/independent_verification.json', acceptance['files']['verification/independent_verification.json'])
    require(m1audit['status'] == 'passed' and m1audit['confirmation_data_qualified'], 'Independent target metadata audit failed')
    for rel, path in [('data/corpora/english/train.jsonl.gz',english_path),('tokenizers/mixed_bpe/tokenizer.json',tokenizer_path)]:
        entry = references['files'][rel]
        require(Path(entry['path']) == path and entry['sha256'] == inputs.hashes[str(path)] == m1manifest['inputs'][str(path)] == m1audit['file_sha256'][str(path)], 'Target metadata reference mismatch')
    confirmation = {'acceptance_sha256': inputs.hashes[str(M1 / 'confirmation_acceptance.json')],
        'same_full_english_train_pool_and_tokenizer': True, 'retained_lexical_hits_zero_inherited_from_M1': True,
        'target_raw_or_encoded_or_row_ledger_files_opened': False, 'qualification_scope': acceptance['qualification_scope'],
        'current_target_bytes_verification_deferred_to_scoring_gate': True}

    parent_manifest = inputs.js(EVAL / 'manifest.json')
    parent_gate = inputs.js(EVAL / 'acceptance.json')
    require(parent_gate['status'] == 'passed' and parent_gate['manifest_sha256'] == inputs.hashes[str(EVAL / 'manifest.json')], 'Parent source gate failed')
    require(parent_manifest['release_manifest_sha256'] == RELEASE_SHA and parent_manifest['release_gate_sha256'] == GATE_SHA, 'Parent source release mismatch')
    task = parent_manifest['tasks']['protein_sequence_similarity']
    for scope in ('source_isolation','prepared_isolation'):
        require(all(all(v == 0 for v in item['cross_split_overlap'].values()) for item in task[scope].values()), 'Inherited source split isolation failed')
    source, source_ids = {}, {}
    for split, count in [('train',8002),('validation',20338)]:
        entry = task['splits'][split]
        raw_path = inputs.pin(EVAL / entry['file'], entry['sha256'])
        source[split], source_ids[split] = make_source(split, raw_path, tokenizer, by_sequence, folder / 'source')
        require(source[split]['count'] == entry['rows'] == count, 'Source candidate membership count changed')
    source_overlap = {key: len(source_ids['train'][key] & source_ids['validation'][key]) for key in source_ids['train']}
    require(not any(source_overlap.values()), 'Source train/validation overlap')
    inheritance = {'source_train_validation_reencoded': True, 'source_train_validation_overlap': source_overlap,
        'source_test_rows': task['splits']['test']['rows'], 'source_test_identity_sha256': task['splits']['test']['sha256'],
        'source_test_pair_examples_or_labels_read': False, 'inherited_prepared_isolation': task['prepared_isolation'],
        'operational_cluster_definition': split_policy['definition'], 'source_positive_definition': split_policy['positive_definition'],
        'near_homology_exclusion': remote_exclusion['criterion'], 'near_homology_search_rerun': False,
        'search_limitations': split_policy['search_limitations'],
        'structural_family_overlap': None, 'structural_family_availability': 'not_provided; unmeasured, not zero',
        'canonical_nontrain_hashes': len(canonical_forbidden), 'remote_domain_rows': remote_count,
        'remote_all_split_hashes': len(remote_forbidden), 'combined_exact_exclusion_hashes': len(forbidden),
        'tokenizer_protein_parents_all_train_eligible': True, 'tokenizer_protein_training_records': len(tokenizer_records['protein']),
        'tokenizer_training_byte_budget': tokenizer_meta['training_utf8_bytes']}
    write_json(folder / 'source_identity_and_exclusions.json', inheritance)
    write_json(folder / 'confirmation_metadata_inheritance.json', confirmation)
    print('Source train/validation encoded; no source-test or target examples opened', flush=True)

    english_writers = {name: StreamWriter(folder / 'streams', name, tokenizer) for name in ('english_common','english_extra')}
    seen_ids, seen_text = set(), set()
    for position, row in enumerate(rows(english_path)):
        parent, text = row['record_id'], row['text']
        require(row['split'] == 'train' and parent not in seen_ids and row['text_sha256'] not in seen_text, 'English source identity repeats')
        require(text_sha(text) == row['text_sha256'], 'English source text hash mismatch')
        seen_ids.add(parent); seen_text.add(row['text_sha256'])
        name = 'english_extra' if english_writers['english_common'].complete else 'english_common'
        english_writers[name].append(parent, position, text, encode_content(tokenizer, text))
        if english_writers['english_extra'].complete:
            break
    streams = {name: writer.finish() for name, writer in english_writers.items()}
    require(not {r['parent_id'] for r in english_writers['english_common'].index_rows} & {r['parent_id'] for r in english_writers['english_extra'].index_rows}, 'English halves share record IDs')
    print('English streams built', flush=True)
    protein_entries, parent_table, parent_report = build_protein(list(rows(protein_path)), canonical, forbidden, tokenizer, folder)
    streams.update(protein_entries)
    write_json(folder / 'protein_control_audit.json', parent_report)
    schedules = {}
    for seed in (0,1,2):
        path = folder / 'schedules' / f'seed{seed}.npy'; path.parent.mkdir(exist_ok=True)
        np.save(path, make_schedule(seed), allow_pickle=False)
        schedules[str(seed)] = {'file': str(path.relative_to(HERE)), 'shape': [2048,16,2], 'dtype':'int64', 'sha256':sha(path)}
    coverage = {}
    for name in ('protein','shuffled'):
        idx = list(rows(HERE / streams[name]['index']))
        coverage[name] = {'used_parent_ids': {r['parent_id'] for r in idx}, 'full_parent_ids': {r['parent_id'] for r in idx if not r['truncated']}, 'exposed_residues': sum(r['exposed_characters'] for r in idx)}
    coverage_report = {'same_eligible_parent_pool_order': True, 'used_parent_intersection':len(coverage['protein']['used_parent_ids'] & coverage['shuffled']['used_parent_ids']),
        'complete_parent_intersection':len(coverage['protein']['full_parent_ids'] & coverage['shuffled']['full_parent_ids']),
        'protein_only_used_parents':len(coverage['protein']['used_parent_ids'] - coverage['shuffled']['used_parent_ids']),
        'shuffled_only_used_parents':len(coverage['shuffled']['used_parent_ids'] - coverage['protein']['used_parent_ids']),
        'exposed_residues':{k:v['exposed_residues'] for k,v in coverage.items()}, 'equal_residue_exposure_claimed':False}
    write_json(folder / 'parent_prefix_coverage.json', coverage_report)
    write_json(folder / 'acceptance_draft.json', {'status':'built_pending_independent_audit','training_ready':False,
        'builder_checks_passed':True,'source_test_or_target_examples_opened':False,'model_training_or_inference':False})
    outputs = {str(p.relative_to(HERE)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    for path, expected in inputs.hashes.items():
        require(sha(path) == expected, 'Input changed during build')
    manifest = {'schema_version':1,'status':'built_pending_independent_audit','training_ready':False,'created_at':now(),
        'elapsed_seconds':time.monotonic()-started,'inputs':inputs.hashes,'outputs':outputs,
        'corpora':{'english':str(english_path),'protein':str(protein_path)},'canonical_metadata_file':str(canonical_path),
        'remote_metadata_file':str(remote_path),'tokenizer':{'file':str(tokenizer_path),'sha256':TOKENIZER_SHA,'vocab_size':32000,'pad_id':0,'eos_id':1,'sep_id':2,'unk_id':3},
        'protein_parent_controls':parent_table,'streams':streams,'schedules':schedules,'source':source,
        'condition_streams':{'EP':['english_common','protein'],'ES':['english_common','shuffled'],'EE':['english_common','english_extra']},
        'build_policy':'prepared/build_policy.json','source_provenance_and_exclusions':inheritance,
        'confirmation_metadata_inheritance':confirmation,'parent_control_summary':parent_report,'parent_prefix_coverage':coverage_report,
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'cuda_used':False},
        'read_boundaries':{'source_test_pair_files_opened':0,'qqp_example_or_row_audit_files_opened':0,'old_predictions_opened':0},
        'protocol_dependency':'Frozen prior design plus build policy only; future execution protocol may bind this manifest without a circular dependency'}
    write_json(folder / 'manifest.json', manifest)
    text = ['# Joint-v1 data build', '', '**Built; independent data audit is still required. No model training was started.**','',
        f"Four streams contain {BUDGET:,} tokens / 16,384 blocks each. Source train/validation: {source['train']['count']:,}/{source['validation']['count']:,}. Three schedules have shape (2048,16,2).",'',
        'Policy was saved before generated controls. Duplicate natural IDs/text fail. Natural or generated exact held-out/remote collisions and every parent involved in duplicate generated controls are excluded from both protein conditions. Unchanged controls are retained unless another rule applies. No shuffle retries.','',
        f"Original protein parents: {parent_report['original_parents']:,}; common eligible: {parent_report['eligible_common_parents']:,}; excluded: {parent_report['excluded_parents']:,}; unchanged controls: {parent_report['unchanged_controls']:,}.",'',
        'The additional English half starts at the next raw record after the common half; the common last record remainder is discarded. Every record ends in EOS, including clipped records. English exposed-character counts are decoded token-prefix lengths and may include Unicode replacement; protein residue exposure is reported separately.','',
        'Train/validation raw source rows were reencoded and their roles, groups and operational clusters checked. Source-test identity/isolation and the old MMseqs2 thresholds are inherited from pinned accepted metadata; source-test pair examples/labels were not opened. Canonical and remote tables supplied identity columns only for exact generated-control screening. Structural-family overlap is unmeasured, not zero.','',
        'QQP qualification is inherited through accepted M1 metadata that binds the same complete English pool and tokenizer. No QQP raw/text/encoded/row-ledger file was opened. This does not claim semantic or global unseen status.','',
        'All inputs, bins, record tables, schedules and source tensors are hashed in `prepared/manifest.json`. The future execution protocol is intentionally not an input dependency. See `prepared/build_policy.json`, `source_identity_and_exclusions.json`, `protein_control_audit.json`, and `parent_prefix_coverage.json` for counts and limitations.','']
    write_report = HERE / 'DATA_BUILD_REPORT.md'
    require(not write_report.exists(), 'Existing report')
    write_report.write_text('\n'.join(text))
    print(json.dumps({'status':manifest['status'],'elapsed_seconds':manifest['elapsed_seconds'],'manifest_sha256':sha(folder/'manifest.json'),'source_rows':{k:v['count'] for k,v in source.items()}}), flush=True)


if __name__ == '__main__':
    main()
