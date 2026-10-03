"""Unchanged source-only pure algorithms copied from pinned portable derived code.
No historical execution module is imported. See algorithm_sources.json.
"""
from collections import Counter, defaultdict
import csv, gzip, hashlib, json, math, random
from pathlib import Path
import numpy as np
try:
    from .common import require, sha, text_sha, rows, write_rows
except ImportError:
    from common import require, sha, text_sha, rows, write_rows
BUDGET = 8_388_608
TASKS = {'protein_sequence_similarity': {'kind': 'protein', 'group': 'block_id', 'inputs': ['sentence1','sentence2']}}

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
    def __init__(self, base, folder, name, tokenizer, budget=BUDGET):
        self.base = Path(base)
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
        return {'bin': str(self.binary.relative_to(self.base)), 'index': str(self.index.relative_to(self.base)),
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

def make_source(split, raw_path, tokenizer, canonical_sequences, folder, base):
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
    entry = {'npz': str(npz.relative_to(base)), 'rows': str(rowfile.relative_to(base)), 'count': n,
        'npz_sha256': sha(npz), 'rows_sha256': sha(rowfile), 'raw_input': str(raw_path), 'raw_input_sha256': sha(raw_path),
        'labels': dict(Counter(map(str, labels.tolist()))), 'shape': [n, 512], 'dtype': 'int64',
        'endpoint_role_label_balance': True, 'within_block_role_label_balance': True,
        'truncated_endpoints_by_label': dict(truncated), 'same_label_encoded_pair_collisions': collisions,
        'conflicting_encoded_pair_labels': 0, 'identity_counts': {k: len(v) for k, v in identity.items()}}
    return entry, identity

def build_protein(records, canonical, forbidden_hashes, tokenizer, folder, base):
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
    writers = {name: StreamWriter(base, folder / 'streams', name, tokenizer) for name in ('protein', 'shuffled')}
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
    return entries, str(parent_file.relative_to(base)), report

def stable_hash(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()

def select_groups(rows: list[dict], group_key: str, target: int, seed: int, task: str) -> tuple[list[dict], dict]:
    """Uniform deterministic group order, nearest prefix to requested row budget.

    Unequal groups are never split or greedily skipped by size. Thus selected row
    counts may exceed the budget slightly, and we do not claim row-uniform sampling.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    ordered = sorted(groups, key=lambda key: (stable_hash(seed, task, key), key))
    if len(rows) <= target:
        selected = set(ordered)
    else:
        count = 0
        cutoff = 0
        for index, key in enumerate(ordered):
            after = count + len(groups[key])
            if after >= target:
                cutoff = index + int(after - target <= target - count)
                break
            count = after
        selected = set(ordered[:cutoff])
    result = [row for row in rows if row[group_key] in selected]
    if not result:
        raise ValueError("Whole-group budget selects no rows; increase target")
    return result, {"method": "seeded_sha256_group_order_nearest_prefix", "group_key": group_key, "source_groups": len(groups), "selected_groups": len(selected), "target_rows": target, "selected_rows": len(result), "whole_groups": True}

def distribution(values: list[int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {"min": ordered[0], "median": ordered[(len(ordered) - 1) // 2], "p95": ordered[math.ceil(.95 * len(ordered)) - 1], "max": ordered[-1], "mean": sum(ordered) / len(ordered)}

def audit_split(rows: list[dict], task: str, split: str) -> dict:
    spec = TASKS[task]
    kind = spec["kind"]
    if not rows or len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError(f"Empty split or duplicate row IDs: {task}/{split}")
    if any(row["split"] != split for row in rows):
        raise ValueError(f"Changed split assignment: {task}/{split}")
    labels = Counter(row["label"] for row in rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row[spec["group"]]].append(row)
    info = {"rows": len(rows), "labels": dict(sorted(labels.items())), "groups": len(groups)}
    if kind != "nlp":
        if labels[0] != labels[1]:
            raise ValueError(f"Unequal classes: {task}/{split}")
        for group in groups.values():
            count = Counter(row["label"] for row in group)
            if count[0] != count[1]:
                raise ValueError(f"Unbalanced group: {task}/{split}")
            if kind in ("dyck", "longrange") and len(group) != 2:
                raise ValueError(f"Incomplete synthetic contrast: {task}/{split}")
        info["balanced_classes_and_complete_groups"] = True
    if kind in ("protein", "remote"):
        degrees = defaultdict(Counter)
        pair_keys = set()
        for row in rows:
            hashes = [row[f"sequence_sha256_{role}"] for role in ("a", "b")]
            if hashes[0] == hashes[1]:
                raise ValueError(f"Self pair: {task}/{split}")
            pair_key = tuple(sorted(hashes))
            if pair_key in pair_keys:
                raise ValueError(f"Duplicate unordered pair: {task}/{split}")
            pair_keys.add(pair_key)
            for role, seq_hash in zip(("a", "b"), hashes):
                text = row["sentence1" if role == "a" else "sentence2"]
                if hashlib.sha256(text.encode()).hexdigest() != seq_hash:
                    raise ValueError(f"Sequence hash mismatch: {task}/{split}")
                if not set(text) <= set("ACDEFGHIKLMNPQRSTVWY"):
                    raise ValueError(f"Invalid protein symbols: {task}/{split}")
                degrees[(role, seq_hash)][row["label"]] += 1
        if any(count[0] != count[1] for count in degrees.values()):
            raise ValueError(f"Unequal per-role endpoint class degree: {task}/{split}")
        info["endpoint_role_degree_balance"] = True
        info["endpoint_roles_checked"] = len(degrees)
    inputs = [[row.get("sentence", row.get("sentence1"))] if len(spec["inputs"]) == 1 else [row["sentence1"], row["sentence2"]] for row in rows]
    if any(not isinstance(text, str) or not text for item in inputs for text in item):
        raise ValueError(f"Missing model input: {task}/{split}")
    info["input_characters"] = {"combined": distribution([sum(map(len, item)) for item in inputs]), "side1": distribution([len(item[0]) for item in inputs])}
    if len(spec["inputs"]) == 2:
        info["input_characters"]["side2"] = distribution([len(item[1]) for item in inputs])
    info["pure_byte_input_plus_two_special_tokens_exceed_512"] = sum(sum(len(s.encode()) for s in item) + 2 > 512 for item in inputs)
    if kind == "longrange":
        cells = Counter((r["head_number"], r["nearest_distractor_number"]) for r in rows)
        info["design_cells"] = {"/".join(k): v for k, v in sorted(cells.items())}
        info["nearest_noun_rule_accuracy"] = sum(r["head_number"] == r["nearest_distractor_number"] for r in rows) / len(rows)
        if len(set(cells.values())) != 1:
            raise ValueError(f"Unbalanced longrange design cells: {task}/{split}")
    return info

def prepared_row(row: dict, task: str) -> dict:
    spec = TASKS[task]
    result = {"row_id": row["row_id"], "label": row["label"]}
    for key in spec["inputs"]:
        result[key] = row.get(key, row.get("sentence1")) if key == "sentence" else row[key]
    keys = {spec["group"], "split", "official_split", "shape_id", "head_number", "nearest_distractor_number", "verb_root"}
    if spec["kind"] in ("protein", "remote"):
        keys.update(f"{key}_{role}" for role in ("a", "b") for key in ("sequence_sha256", "accession", "sequence_id", "cluster", "superfamily", "family"))
    result["metadata"] = {key: row[key] for key in sorted(keys) if key in row}
    return result
