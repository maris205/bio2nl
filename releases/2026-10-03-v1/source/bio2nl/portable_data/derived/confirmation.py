"""Offline replay of the fixed historical M1 qualification rules."""
from collections import Counter, defaultdict
import time
import json
import ahocorasick
import pyarrow.parquet as pq
from tokenizers import Tokenizer
from .common import write_json, write_rows, now
from .algorithms import norm, patterns, scan_document, Graph, text_hash

def construct(root, raw, lock, refs, iter_reference_docs):
    begin = time.monotonic()
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
    return report
