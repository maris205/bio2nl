"""CPU-only, predeclared QQP qualification. No model or prediction imports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def text_hash(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def norm(s):
    if not isinstance(s, str):
        return ''
    return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', s).casefold()))


def patterns(s):
    words = s.split()
    if len(words) < 8:
        return {s} if s else set()
    return {' '.join(words[i:i + 8]) for i in range(len(words) - 7)}


def scan_document(automaton, have_patterns, text):
    if not have_patterns:
        return set()
    return {p for _, p in automaton.iter(' ' + norm(text) + ' ')}


class Graph:
    def __init__(self):
        self.parent = {}

    def root(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def validate_references(refs):
    assert refs['status'] == 'passed'
    release = refs['release_manifest']
    assert sha(release['path']) == release['sha256']
    for item in refs['files'].values():
        p = Path(item['path'])
        if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise ValueError(f'Changed accepted reference: {p}')


def iter_reference_docs(refs):
    import pyarrow.parquet as pq
    for entry in refs['old_nlp_files']:
        path = refs['files'][entry['file']]['path']
        domain = f"old_nlp:{entry['task']}:{entry['stage']}:{entry['split']}"
        if entry['format'] == 'parquet':
            rows = pq.read_table(path, columns=entry['text_fields']).to_pylist()
        else:
            rows = (json.loads(line) for line in Path(path).read_text().splitlines())
        count = 0
        for i, row in enumerate(rows):
            count += 1
            for field in entry['text_fields']:
                s = row[field]
                if s is None:
                    continue
                if not isinstance(s, str):
                    raise TypeError(f'Nontext historical reference {path}:{i}:{field}')
                yield domain, f"{entry['file']}:{i}:{field}", s
        assert count == entry['rows'], (path, count, entry['rows'])
    for entry in refs['english_corpora']:
        count = 0
        with gzip.open(refs['files'][entry['file']]['path'], 'rt') as f:
            for line in f:
                row = json.loads(line)
                count += 1
                yield f"english:{entry['split']}", row[entry['id_field']], row[entry['text_field']]
        assert count == entry['rows']
    prefix = refs['tokenizer_prefix']
    used, nbytes = [], 0
    budget = prefix['utf8_bytes']
    expected_ids = json.loads(Path(refs['files'][prefix['record_ids_file']]['path']).read_text())[prefix['record_ids_key']]
    with gzip.open(refs['files'][prefix['source_corpus_file']]['path'], 'rt') as f:
        for line in f:
            if nbytes >= budget:
                break
            row = json.loads(line)
            text = row['text'].encode()[:budget - nbytes].decode('utf-8', errors='ignore')
            if text:
                used.append(row['record_id'])
                nbytes += len(text.encode())
                yield 'tokenizer:english_prefix', row['record_id'], text
    assert used == expected_ids and nbytes == budget and len(used) == prefix['records']


def freeze(root=ROOT):
    if (root / 'FREEZE.json').exists() or (root / 'raw/qqp_validation.parquet').exists():
        raise FileExistsError('Refuse to replace freeze or previously acquired candidate')
    files = ['DATA_PROTOCOL.md', 'sources/source_lock.json', 'references.json',
             'pin_references.py', 'build_confirmation.py', 'verify_confirmation.py', 'test_confirmation.py', 'test_builder.py']
    lock = json.loads((root / 'sources/source_lock.json').read_text())
    for item in lock['metadata_evidence'].values():
        rel = 'sources/' + item['file']
        assert sha(root / rel) == item['sha256']
        files.append(rel)
    refs = json.loads((root / 'references.json').read_text())
    validate_references(refs)
    env = {p: importlib.metadata.version(p) for p in ['pyahocorasick', 'tokenizers', 'pyarrow', 'requests', 'pytest']}
    env.update(python=sys.version, executable=sys.executable, unicode_version=unicodedata.unidata_version,
               cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write_json(root / 'environment.json', env)
    files.append('environment.json')
    write_json(root / 'FREEZE.json', dict(created_at_utc=now(), stage='pre_candidate_access',
               candidate_downloaded=False, model_inference_performed=False, acquisition_enabled_for_pinned_validation_only=True,
               files={p: sha(root / p) for p in files}, expected_raw_sha256=lock['selected_artifact']['expected_sha256']))
    print(json.dumps({'stage': 'frozen', 'freeze_sha256': sha(root / 'FREEZE.json')}), flush=True)


def check_freeze(root):
    frozen = json.loads((root / 'FREEZE.json').read_text())
    for name, digest in frozen['files'].items():
        assert sha(root / name) == digest, f'Changed preaccess file {name}'
    return frozen


def acquire(root, lock):
    import requests
    info = lock['selected_artifact']
    destination = root / 'raw/qqp_validation.parquet'
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise FileExistsError('Candidate already downloaded; preserve failed run, do not silently resume')
    start = now()
    partial = destination.with_suffix('.parquet.partial')
    with requests.get(info['download_url'], stream=True, timeout=(30, 90)) as response:
        response.raise_for_status()
        with partial.open('xb') as out:
            for chunk in response.iter_content(1 << 20):
                out.write(chunk)
    assert partial.stat().st_size == info['expected_bytes']
    assert sha(partial) == info['expected_sha256']
    partial.replace(destination)
    write_json(root / 'audit/download.json', dict(started_at_utc=start, completed_at_utc=now(),
        url=info['download_url'], sha256=sha(destination), bytes=destination.stat().st_size,
        freeze_sha256=sha(root / 'FREEZE.json')))
    return destination


def run(root=ROOT):
    import ahocorasick
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES='' for data-only work")
    begin = time.monotonic()
    frozen = check_freeze(root)
    refs = json.loads((root / 'references.json').read_text())
    lock = json.loads((root / 'sources/source_lock.json').read_text())
    validate_references(refs)
    write_json(root / 'execution_status.json', dict(status='running', stage='acquisition', started_at_utc=now()))
    raw = acquire(root, lock)
    write_json(root / 'audit/access_log.json', dict(first_candidate_read_at_utc=now(),
        access_purpose='predeclared_data_integrity_and_qualification', no_model_inference=True,
        training_or_selection_access=False, sample_text_rendered_to_console=False))
    rows = pq.read_table(raw).to_pylist()
    assert len(rows) == lock['selected_artifact']['expected_rows']
    assert all(set(r) == {'idx', 'question1', 'question2', 'label'} for r in rows)
    assert all(type(r['idx']) is int and type(r['label']) is int and r['label'] in (0, 1) for r in rows)
    assert len({r['idx'] for r in rows}) == len(rows)
    rows.sort(key=lambda r: r['idx'])
    graph = Graph()
    valid, invalid = {}, set()
    for row in rows:
        a, b = norm(row['question1']), norm(row['question2'])
        if not a or not b:
            invalid.add(row['idx'])
            continue
        valid[row['idx']] = (a, b)
        graph.join(a, b)
    components = {n: text_hash(graph.root(n)) for n in graph.parent}
    component_reasons = defaultdict(set)
    pair_labels, first_pair_label, duplicates = defaultdict(set), {}, set()
    for row in rows:
        if row['idx'] not in valid:
            continue
        a, b = valid[row['idx']]
        pair = tuple(sorted((a, b)))
        pair_labels[pair].add(row['label'])
        key = (pair, row['label'])
        if key in first_pair_label:
            duplicates.add(row['idx'])
        else:
            first_pair_label[key] = row['idx']
        if a == b and row['label'] == 0:
            component_reasons[components[a]].add('negative_self_pair')
    for (a, b), labels in pair_labels.items():
        if len(labels) > 1:
            component_reasons[components[a]].add('conflicting_pair_labels')

    # All raw valid endpoints participate, including rows subsequently excluded.
    lookup = defaultdict(set)
    for endpoint in sorted(components):
        for p in patterns(endpoint):
            lookup[p].add(endpoint)
    automaton = ahocorasick.Automaton()
    for p in sorted(lookup):
        automaton.add_word(' ' + p + ' ', p)
    automaton.make_automaton()
    hits = {}
    reference_counts = Counter()
    reference_chars = Counter()
    scan_start = time.monotonic()
    for domain, ref_id, text in iter_reference_docs(refs):
        reference_counts[domain] += 1
        reference_chars[domain] += len(text)
        found = scan_document(automaton, bool(lookup), text)
        for p in sorted(found):
            for endpoint in sorted(lookup[p]):
                key = (domain, endpoint)
                if key not in hits:
                    hits[key] = dict(endpoint_sha256=text_hash(endpoint), domain=domain,
                        reference_id=ref_id, pattern_sha256=text_hash(p), pattern_words=len(p.split()))
                    component_reasons[components[endpoint]].add('reference_overlap')
        if sum(reference_counts.values()) % 25000 == 0:
            print(json.dumps({'stage': 'reference_scan', 'documents': sum(reference_counts.values()),
                'endpoint_domain_hits': len(hits), 'elapsed_seconds': round(time.monotonic() - scan_start, 1)}), flush=True)
    write_rows(root / 'audit/overlap_hits.jsonl', [hits[k] for k in sorted(hits)])

    tokenizer = Tokenizer.from_file(refs['files']['tokenizers/mixed_bpe/tokenizer.json']['path'])
    assert tokenizer.get_vocab_size() == 32000
    special = {'<|pad_v2|>': 0, '<|eos_v2|>': 1, '<|pair_v2|>': 2, '<|unk_v2|>': 3}
    assert all(tokenizer.token_to_id(k) == v for k, v in special.items())
    texts = sorted({r[k] for r in rows if r['idx'] in valid for k in ('question1', 'question2')})
    encoded, bad_text = {}, {}
    for start in range(0, len(texts), 1024):
        batch = texts[start:start + 1024]
        for text, enc in zip(batch, tokenizer.encode_batch(batch, add_special_tokens=False)):
            ids = enc.ids
            encoded[text] = ids
            reasons = []
            if not 1 <= len(ids) <= 255:
                reasons.append('content_length')
            if any(i < 4 or i >= 32000 for i in ids):
                reasons.append('special_or_unknown')
            if tokenizer.decode(ids, skip_special_tokens=False) != text:
                reasons.append('roundtrip')
            if reasons:
                bad_text[text] = reasons
                component_reasons[components[norm(text)]].add('encoding_failure')

    kept, encoded_rows, ledger = [], [], []
    encoding_labels = defaultdict(set)
    sequential = Counter()
    order = ['invalid_text', 'conflicting_pair_labels', 'negative_self_pair', 'reference_overlap', 'encoding_failure', 'duplicate_same_label']
    for row in rows:
        idx = row['idx']
        if idx in invalid:
            cid = None
            reasons = {'invalid_text'}
        else:
            a, b = valid[idx]
            cid = components[a]
            reasons = set(component_reasons[cid])
            if idx in duplicates:
                reasons.add('duplicate_same_label')
        ledger.append(dict(upstream_idx=idx, component_id=cid, retained=not reasons, reasons=sorted(reasons)))
        if reasons:
            sequential[next(r for r in order if r in reasons)] += 1
            continue
        a, b = valid[idx]
        ids_a, ids_b = encoded[row['question1']], encoded[row['question2']]
        encoding_labels[tuple(sorted((tuple(ids_a), tuple(ids_b))))].add(row['label'])
        rid = f'qqp_validation:{idx}'
        kept.append(dict(id=rid, upstream_idx=idx, label=row['label'], text_a=row['question1'],
                         text_b=row['question2'], endpoint_a_sha256=text_hash(a), endpoint_b_sha256=text_hash(b), component_id=cid))
        encoded_rows.append(dict(id=rid, label=row['label'], component_id=cid, ids_a=ids_a, ids_b=ids_b))
    assert all(len(x) == 1 for x in encoding_labels.values()), 'Retained conflicting encoding collision'
    assert len(kept) + sum(sequential.values()) == len(rows)
    write_rows(root / 'data/confirmation.jsonl', kept)
    write_rows(root / 'data/encoded.jsonl', encoded_rows)
    write_rows(root / 'audit/exclusions.jsonl', ledger)
    labels = Counter(r['label'] for r in kept)
    sizes = Counter(r['component_id'] for r in kept)
    qualification = len(kept) >= 2000 and labels[0] >= 200 and labels[1] >= 200 and len(sizes) >= 100
    all_lengths = [len(ids) for ids in encoded.values()]
    final_lengths = [len(r[k]) for r in encoded_rows for k in ('ids_a', 'ids_b')]
    report = dict(schema_version=1, status='built_pending_independent_verification',
        completed_at_utc=now(), raw_rows=len(rows), raw_labels=dict(sorted(Counter(r['label'] for r in rows).items())),
        valid_raw_rows=len(valid), unique_raw_normalized_endpoints=len(components), raw_components=len(set(components.values())),
        raw_unordered_pairs=len(pair_labels), raw_conflicting_pairs=sum(len(v) > 1 for v in pair_labels.values()),
        raw_same_label_duplicate_rows=len(duplicates), raw_negative_self_pairs=sum(valid.get(r['idx'], (None, 1))[0] == valid.get(r['idx'], (None, 1))[1] and r['label'] == 0 for r in rows),
        retained_rows=len(kept), retained_labels={str(c): labels[c] for c in (0, 1)}, retained_components=len(sizes),
        retained_unique_normalized_endpoints=len({r[k] for r in kept for k in ('endpoint_a_sha256','endpoint_b_sha256')}),
        retained_largest_component_rows=max(sizes.values(), default=0),
        retained_component_size_histogram=dict(sorted(Counter(sizes.values()).items())),
        retained_component_rows_squared_sum=sum(v*v for v in sizes.values()),
        excluded_rows=len(rows)-len(kept), sequential_exclusions={reason: sequential[reason] for reason in order},
        overlapping_row_exclusion_reasons=dict(sorted(Counter(r for entry in ledger for r in entry['reasons']).items())),
        components_by_exclusion_reason=dict(sorted(Counter(reason for reasons in component_reasons.values() for reason in reasons).items())),
        reference_documents=dict(sorted(reference_counts.items())), reference_characters=dict(sorted(reference_chars.items())),
        overlap_endpoint_domain_hits=dict(sorted(Counter(v['domain'] for v in hits.values()).items())),
        overlap_unique_endpoints=len({ep for domain,ep in hits}), overlap_hit_records=len(hits),
        encoded_original_unique_endpoints=len(texts), encoding_failure_endpoints=len(bad_text),
        encoding_failure_reasons=dict(sorted(Counter(r for reasons in bad_text.values() for r in reasons).items())),
        raw_endpoint_content_token_min=min(all_lengths,default=0), raw_endpoint_content_token_max=max(all_lengths,default=0),
        retained_endpoint_content_token_min=min(final_lengths,default=0), retained_endpoint_content_token_max=max(final_lengths,default=0),
        retained_conflicting_encoding_pairs=0, qualification_passed=qualification,
        acceptance_thresholds={'rows':2000,'each_class':200,'components':100},
        model_inference_performed=False, target_training_or_model_selection=False, training_ready=False,
        data_text_redistribution='not_cleared', formal_wall_seconds=time.monotonic()-begin,
        exposure_claim='Zero retained lexical-rule hits in pinned references; not semantic or global unseen certification')
    write_json(root / 'results/build_report.json', report)
    inputs = {k:v for k,v in frozen['files'].items()}
    inputs.update({item['path']:item['sha256'] for item in refs['files'].values()})
    inputs[refs['release_manifest']['path']] = refs['release_manifest']['sha256']
    artifacts = ['raw/qqp_validation.parquet','audit/download.json','audit/access_log.json','audit/overlap_hits.jsonl',
                 'audit/exclusions.jsonl','data/confirmation.jsonl','data/encoded.jsonl','results/build_report.json']
    write_json(root / 'manifest.json', dict(schema_version=1, inputs=inputs,
        outputs={name:sha(root/name) for name in artifacts}, freeze_sha256=sha(root/'FREEZE.json'),
        status='built_pending_independent_verification', redistribution_not_cleared=['raw/','data/'],
        training_authorized_by_this_manifest=False))
    write_json(root / 'execution_status.json', dict(status='built_pending_independent_verification',
        completed_at_utc=now(), qualification_passed=qualification, training_started=False))
    print(json.dumps({k:report[k] for k in ['raw_rows','retained_rows','retained_labels','retained_components','qualification_passed','formal_wall_seconds']}), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--freeze', action='store_true')
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    assert args.freeze != args.run, 'Choose exactly one action'
    if args.freeze:
        freeze()
    else:
        try:
            run()
        except Exception as error:
            write_json(ROOT / 'execution_status.json', dict(status='failed', failed_at_utc=now(),
                error_type=type(error).__name__, error=str(error), training_started=False))
            raise


if __name__ == '__main__':
    main()
