#!/usr/bin/env python3
"""Independent audit of two deterministic pair replays; never a training gate."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'

SPLITS = ('train', 'validation', 'test')
PAIR_FILES = tuple(f'data/pairs/protein_sequence_similarity_{s}.tsv.gz' for s in ('all', *SPLITS))
DEGREES = 'validation/protein_endpoint_degrees.tsv'
BASELINES = 'validation/protein_pair_baselines.csv'
CONSTRUCTION = 'metadata/protein_pair_construction.json'
ACCEPTANCE = 'validation/protein_pair_acceptance.json'
LEGACY = 'metadata/original_builder_manifest.json'
OUTPUTS = (*PAIR_FILES, DEGREES, BASELINES, CONSTRUCTION, ACCEPTANCE, LEGACY)
CANONICAL = 'data/sequences/protein_canonical_sequences.tsv.gz'
CANDIDATES = 'work/protein/positive_candidates.jsonl'
MANIFEST = 'replay_manifest.json'
HEADERS = ('pair_id split block_id label accession_a accession_b cluster_a cluster_b '
           'sequence_sha256_a sequence_sha256_b sentence1 sentence2 length_a length_b '
           'mmseqs_identity mmseqs_query_coverage mmseqs_target_coverage mmseqs_evalue mmseqs_alignment_length '
           'global_identity global_alignment_length global_score local_identity local_query_coverage '
           'local_target_coverage label_basis').split()
POS_BASIS = 'MMseqs2 >=40% <=90% identity, >=80% both coverage, E<=1e-3'
NEG_BASIS = 'different operational 30%-identity component; global identity <=25%; no >=30% local identity with >=80% coverage on both ends'
HISTORICAL_NOTE = 'Original builder hardcoded historical label only; not the current replay identity or accepted release.'
AA = set('ACDEFGHIKLMNPQRSTVWY')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def describe(path):
    path = Path(path)
    return {'sha256': digest(path), 'bytes': path.stat().st_size}


def read_json(path):
    return json.loads(Path(path).read_text())


def binding(path, expected, label):
    actual = describe(path)
    require(expected.get('sha256') == actual['sha256'], f'{label}: SHA mismatch')
    require(expected.get('bytes', actual['bytes']) == actual['bytes'], f'{label}: byte count mismatch')
    return actual


def data_bytes(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rb') as f:
        return f.read()


def table(path, delimiter='\t'):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fields = reader.fieldnames
        rows = list(reader)
    require(all(None not in r and all(v is not None for v in r.values()) for r in rows), f'Malformed rows: {path}')
    return fields, rows


def finite(value, name, low=None, high=None):
    x = float(value)
    require(math.isfinite(x), f'{name}: nonfinite')
    require(low is None or x >= low, f'{name}: below range')
    require(high is None or x <= high, f'{name}: above range')
    return x


def validate_manifest(root):
    m = read_json(root / MANIFEST)
    require(m.get('schema_version') == 1 and m.get('status') == 'completed_intermediate_input_replay', 'Incomplete replay manifest')
    require(m.get('scope') == 'protein_pair_replay_from_pinned_intermediate_inputs', 'Wrong replay scope')
    require(isinstance(m.get('release_id'), str) and m['release_id'] and m['release_id'] != '2026-09-25-v2', 'Replay impersonates historical release')
    for k in ('training_enabled', 'data_release_gate_passed', 'complete_raw_reconstruction'):
        require(m.get(k) is False, f'Forbidden gate flag: {k}')
    require(m.get('seed') == 20260925, 'Unexpected replay seed')
    require(set(m['outputs']) == set(OUTPUTS), 'Replay output coverage mismatch')
    bound = {rel: binding(root / rel, m['outputs'][rel], rel) for rel in OUTPUTS}
    acceptance = read_json(root / ACCEPTANCE)
    require(acceptance.get('acceptance_pass') is True and acceptance.get('failures') == [], 'Source acceptance rejected')
    ma = m['acceptance']
    require(ma['file'] == ACCEPTANCE and ma.get('acceptance_pass') is True, 'Manifest acceptance identity')
    binding(root / ACCEPTANCE, ma, 'Manifest acceptance')
    historical = m['legacy_builder_manifest']
    require(historical['file'] == LEGACY and historical['embedded_release_id'] == '2026-09-25-v2', 'Historical manifest attribution')
    require(historical.get('interpretation') == HISTORICAL_NOTE, 'Unqualified historical release label')
    binding(root / LEGACY, historical, 'Historical builder manifest')
    old = read_json(root / LEGACY)
    require(old['release_id'] == '2026-09-25-v2' and old['acceptance'] == acceptance, 'Builder acceptance identity')
    require(not (root / 'metadata/protein_pair_manifest.json').exists(), 'Ambiguous old-named release manifest')
    require(set(old['data_files']) == {Path(p).name for p in PAIR_FILES}, 'Builder manifest pair coverage')
    for rel in PAIR_FILES:
        require(old['data_files'][Path(rel).name] == bound[rel]['sha256'], 'Builder current gzip binding mismatch')
        decoded = data_bytes(root / rel)
        rec = m['outputs'][rel]
        require(rec.get('decoded_sha256') == hashlib.sha256(decoded).hexdigest(), 'Decoded pair SHA mismatch')
        require(rec.get('records') == decoded.count(b'\n') - 1, 'Decoded pair row count mismatch')
    can = m['canonicalization']
    require(can.get('key') == ['a', 'b'] and can.get('duplicate_policy') == 'reject' and can.get('score_values_changed') is False, 'Canonicalization policy mismatch')
    require(can['sorted_candidate_file'] == CANDIDATES, 'Wrong candidate file')
    require(digest(root / CANDIDATES) == can['sorted_candidate_sha256'], 'Canonical candidates SHA mismatch')
    require(can['local_table']['file'] == CANONICAL, 'Wrong canonical table')
    binding(root / CANONICAL, can['local_table'], 'Local canonical table')
    return m


def check_row_semantics(root, rows):
    """Recompute sequence identities, structural invariants and alignment metrics."""
    from Bio.Align import PairwiseAligner, substitution_matrices
    aligners = {}
    for mode in ('global', 'local'):
        a = PairwiseAligner(mode=mode)
        a.substitution_matrix = substitution_matrices.load('BLOSUM62')
        a.open_gap_score = -11
        a.extend_gap_score = -1
        aligners[mode] = a
    wanted = {r[k] for r in rows for k in ('accession_a', 'accession_b')}
    canonical = {}
    canonical_count = 0
    with gzip.open(root / CANONICAL, 'rt', newline='') as f:
        for r in csv.DictReader(f, delimiter='\t'):
            canonical_count += 1
            if r['accession'] in wanted:
                require(r['accession'] not in canonical, 'Duplicate requested canonical accession')
                canonical[r['accession']] = r
    require(set(canonical) == wanted, 'Missing pair endpoints in canonical input')
    pair_ids = set()
    degrees = Counter()
    split_sequences = defaultdict(set)
    split_clusters = defaultdict(set)
    role_by_sequence = defaultdict(set)
    block_rows = defaultdict(list)
    per_a = defaultdict(dict)
    counts = {s: Counter() for s in SPLITS}
    selected_candidates = {}
    for n, r in enumerate(rows, 1):
        split = r['split']
        require(split in SPLITS and r['label'] in ('0', '1'), f'Bad split/label at {n}')
        label = int(r['label'])
        counts[split][r['label']] += 1
        require(r['block_id'].startswith(split + '_cycle_'), 'Block split label mismatch')
        block_rows[(split, r['block_id'])].append(r)
        for role, sentence in [('a', 'sentence1'), ('b', 'sentence2')]:
            seq = r[sentence]
            require(40 <= len(seq) <= 250 and not set(seq) - AA, 'Noncanonical sequence or out-of-protocol length')
            sh = hashlib.sha256(seq.encode()).hexdigest()
            require(r[f'sequence_sha256_{role}'] == sh and int(r[f'length_{role}']) == len(seq), 'Sequence SHA/length mismatch')
            c = canonical[r[f'accession_{role}']]
            require(c['sequence'] == seq and c['sequence_sha256'] == sh and int(c['length']) == len(seq), 'Canonical sequence disagreement')
            require(c['split'] == split and c['cluster_id'] == r[f'cluster_{role}'], 'Canonical split/cluster disagreement')
            degrees[(split, sh, role, label)] += 1
            split_sequences[sh].add(split)
            split_clusters[c['cluster_id']].add(split)
            role_by_sequence[sh].add(role)
        sa, sb = r['sequence_sha256_a'], r['sequence_sha256_b']
        require(sa != sb, 'Self pair')
        pid = hashlib.sha256('|'.join(sorted((sa, sb))).encode()).hexdigest()
        require(r['pair_id'] == pid and pid not in pair_ids, 'Pair ID mismatch or duplicate unordered pair')
        pair_ids.add(pid)
        require(label not in per_a[(split, sa)], 'A endpoint repeated within a label')
        per_a[(split, sa)][label] = (r['length_b'], r['block_id'])
        for mode, engine in aligners.items():
            alignment = engine.align(r['sentence1'], r['sentence2'])[0]
            # Independent API: counts()/length, rather than the builder's coordinate-walk identity implementation.
            identity = alignment.counts().identities / alignment.length
            require(finite(r[f'{mode}_identity'], f'{mode} identity', 0, 1) == identity, f'{mode} alignment identity differs')
            if mode == 'global':
                require(int(r['global_alignment_length']) == alignment.length, 'Global alignment length differs')
                require(finite(r['global_score'], 'global score') == float(alignment.score), 'Global score differs')
            else:
                q = int(alignment.coordinates[0, -1] - alignment.coordinates[0, 0]) / len(r['sentence1'])
                t = int(alignment.coordinates[1, -1] - alignment.coordinates[1, 0]) / len(r['sentence2'])
                require(finite(r['local_query_coverage'], 'local qcov', 0, 1) == q and finite(r['local_target_coverage'], 'local tcov', 0, 1) == t, 'Local coverage differs')
        if label:
            require(r['cluster_a'] == r['cluster_b'] and r['label_basis'] == POS_BASIS, 'Positive cluster/basis mismatch')
            identity = finite(r['mmseqs_identity'], 'positive identity', .4, .9)
            qcov = finite(r['mmseqs_query_coverage'], 'positive qcov', .8, 1)
            tcov = finite(r['mmseqs_target_coverage'], 'positive tcov', .8, 1)
            evalue = finite(r['mmseqs_evalue'], 'positive evalue', 0, .001)
            alnlen = int(r['mmseqs_alignment_length'])
            require(alnlen > 0, 'Positive alignment length invalid')
            a, b = r['accession_a'], r['accession_b']
            if a > b:
                a, b, qcov, tcov = b, a, tcov, qcov
            require((a, b) not in selected_candidates, 'Duplicate selected positive candidate')
            selected_candidates[(a, b)] = (identity, qcov, tcov, evalue, alnlen, split, r['cluster_a'])
        else:
            require(r['cluster_a'] != r['cluster_b'] and r['label_basis'] == NEG_BASIS, 'Negative cluster/basis mismatch')
            require(all(r[k] == '' for k in HEADERS if k.startswith('mmseqs_')), 'Negative has positive-search fields')
            require(float(r['global_identity']) <= .25, 'Negative global threshold violation')
            require(not (float(r['local_identity']) >= .3 and min(float(r['local_query_coverage']), float(r['local_target_coverage'])) >= .8), 'Negative local threshold violation')
    require(all(len(x) == 1 for x in split_sequences.values()), 'Sequence cross-split leakage')
    require(all(len(x) == 1 for x in split_clusters.values()), 'Cluster cross-split leakage')
    require(all(len(x) == 1 for x in role_by_sequence.values()), 'Endpoint role changed')
    for s in SPLITS:
        require(counts[s]['0'] == counts[s]['1'] > 0, f'Empty/imbalanced split {s}')
    require(all(set(v) == {0, 1} and v[0] == v[1] for v in per_a.values()), 'Per-A B length or block mismatch')
    expected_degrees = []
    for s, seq, role in sorted({k[:3] for k in degrees}):
        require(degrees[(s, seq, role, 0)] == degrees[(s, seq, role, 1)] == 1, 'Endpoint role degree violation')
        expected_degrees.append({'split': s, 'sequence_sha256': seq, 'role': role, 'positive_degree': '1', 'negative_degree': '1'})
    for (split, block), members in block_rows.items():
        pos = [r for r in members if r['label'] == '1']
        neg = [r for r in members if r['label'] == '0']
        require(len(pos) == len(neg) >= 2, 'Invalid balanced rewiring block')
        owners = {r['sequence_sha256_b']: r['sequence_sha256_a'] for r in pos}
        require({r['sequence_sha256_a'] for r in pos} == {r['sequence_sha256_a'] for r in neg}, 'Block A membership differs')
        require(set(owners) == {r['sequence_sha256_b'] for r in neg}, 'Block B membership differs')
        links = {r['sequence_sha256_a']: owners[r['sequence_sha256_b']] for r in neg}
        start = next(iter(links))
        visited = set()
        cur = start
        while cur not in visited:
            visited.add(cur)
            cur = links[cur]
        require(cur == start and len(visited) == len(links), 'Block is not one permutation cycle')
    fields, degree_rows = table(root / DEGREES)
    require(fields == ['split', 'sequence_sha256', 'role', 'positive_degree', 'negative_degree'] and degree_rows == expected_degrees, 'Degree file does not match recomputation')
    previous = None
    candidate_count = 0
    found = set()
    with (root / CANDIDATES).open() as f:
        for line in f:
            p = json.loads(line)
            candidate_count += 1
            key = (p['a'], p['b'])
            require(key[0] < key[1] and (previous is None or previous < key), 'Candidate order/uniqueness violation')
            previous = key
            if key in selected_candidates:
                value = (p['identity'], p['query_coverage'], p['target_coverage'], p['evalue'], p['alignment_length'], p['split'], p['cluster_id'])
                require(value == selected_candidates[key], 'Positive evidence differs from pinned candidate')
                found.add(key)
    require(found == set(selected_candidates), 'Selected positive candidate absent from input')
    summary = {'pair_counts': {s: dict(counts[s]) for s in SPLITS}, 'unique_unordered_pairs': len(pair_ids), 'unique_sequences': len(split_sequences), 'endpoint_rows': len(expected_degrees), 'A_endpoints_checked_for_length_match': len(per_a), 'block_count': len(block_rows), 'canonical_records': canonical_count, 'candidate_records': candidate_count, 'alignment_rows_recomputed': len(rows)}
    acceptance = read_json(root / ACCEPTANCE)
    expected = {k: summary[k] for k in ('pair_counts', 'unique_unordered_pairs', 'unique_sequences', 'endpoint_rows', 'A_endpoints_checked_for_length_match')}
    expected.update({'acceptance_pass': True, 'failures': [], 'sequence_cross_split_overlap': 0, 'endpoint_role_degree_balance': True, 'length_difference_matched_per_A': True, 'A_endpoints_with_length_mismatch': 0, 'gap_or_noncanonical_input_count': 0})
    require(all(acceptance.get(k) == v for k, v in expected.items()), 'Acceptance counters/booleans disagree with independent audit')
    construction = read_json(root / CONSTRUCTION)
    require(construction['seed'] == 20260925 and construction['rows'] == len(rows) and construction['split_label_counts'] == summary['pair_counts'] and construction['block_count'] == len(block_rows), 'Construction scientific counts disagree')
    finite(construction['runtime_seconds'], 'runtime_seconds', 0)
    require(all(isinstance(v, int) and v >= 0 for v in construction['filter_counts'].values()), 'Invalid filter counts')
    fields, baselines = table(root / BASELINES, ',')
    require(fields == ['method', 'split', 'n', 'accuracy', 'auroc', 'mcc'], 'Baseline schema mismatch')
    methods = ['negative_length_difference', 'negative_composition_l1', '3mer_jaccard', 'global_identity', 'length_logistic', 'composition_logistic', 'length_composition_kmer_logistic']
    require([(r['method'], r['split']) for r in baselines] == [(m, s) for m in methods for s in SPLITS], 'Baseline cells/order mismatch')
    for r in baselines:
        require(int(r['n']) == sum(counts[r['split']].values()), 'Baseline count mismatch')
        finite(r['auroc'], 'baseline AUROC', 0, 1)
        for k, low in [('accuracy', 0), ('mcc', -1)]:
            if r[k] != '':
                finite(r[k], k, low, 1)
        require((r['accuracy'] == r['mcc'] == '') == (r['method'] in methods[:4]), 'Baseline metric presence mismatch')
    return summary


def audit_core(root_a, root_b):
    """Check full outputs and their input semantics without granting formal status."""
    roots = [Path(root_a).resolve(), Path(root_b).resolve()]
    require(roots[0] != roots[1], 'Distinct replay roots required')
    manifests = [validate_manifest(r) for r in roots]
    require(manifests[0]['release_id'] == manifests[1]['release_id'], 'Replay release IDs differ')
    comparisons = {}
    for rel in (*PAIR_FILES, DEGREES, BASELINES):
        a, b = (data_bytes(r / rel) for r in roots)
        require(a == b, f'Ordered decoded content differs: {rel}')
        comparisons[rel] = {'comparison': 'decoded_bytes_exact' if rel.endswith('.gz') else 'bytes_exact', 'equivalent': True, 'decoded_sha256': hashlib.sha256(a).hexdigest()}
    for rel in (CONSTRUCTION, ACCEPTANCE):
        a, b = (read_json(r / rel) for r in roots)
        if rel == CONSTRUCTION:
            for v in (a, b):
                finite(v.pop('runtime_seconds'), 'runtime_seconds', 0)
        require(a == b, f'Scientific JSON differs: {rel}')
        comparisons[rel] = {'comparison': 'json_exact_except_runtime_seconds' if rel == CONSTRUCTION else 'json_exact', 'equivalent': True}
    require(data_bytes(roots[0] / CANONICAL) == data_bytes(roots[1] / CANONICAL), 'Canonical inputs differ')
    require(digest(roots[0] / CANDIDATES) == digest(roots[1] / CANDIDATES), 'Canonical candidate outputs differ')
    headers, rows = table(roots[0] / PAIR_FILES[0])
    require(headers == HEADERS and rows, 'Pair schema/empty table')
    for root in roots:
        for s in SPLITS:
            h, split_rows = table(root / f'data/pairs/protein_sequence_similarity_{s}.tsv.gz')
            require(h == HEADERS and split_rows == [r for r in rows if r['split'] == s], 'Split file is not exact ordered all-file projection')
    scientific = check_row_semantics(roots[0], rows)
    for m in manifests:
        require(m['canonicalization']['records'] == scientific['candidate_records'], 'Candidate manifest count mismatch')
        require(m['canonicalization']['local_table']['records'] == scientific['canonical_records'], 'Canonical manifest count mismatch')
        require(m['inputs']['positive_candidates']['records'] == scientific['candidate_records'] and m['inputs']['canonical_sequences']['records'] == scientific['canonical_records'], 'Source input manifest counts mismatch')
    return {'status': 'pair_replay_scientific_checks_passed_without_execution_binding', 'training_enabled': False, 'data_release_gate_passed': False, 'complete_raw_reconstruction': False, 'target_scoring_enabled': False, 'comparisons': comparisons, 'scientific_audit': scientific, 'scope': 'deterministic_pair_replay_only', 'independence': 'Output alignments, IDs, source endpoint identities, candidate evidence, cycles, degrees and splits independently recomputed; baseline CSV compared fully, logistic probes not refitted.'}


def verify_bound(protocol_path, protocol_sha256, queue_path):
    protocol_path, queue_path = Path(protocol_path).resolve(), Path(queue_path).resolve()
    require(digest(protocol_path) == protocol_sha256, 'Protocol SHA mismatch')
    protocol, queue = read_json(protocol_path), read_json(queue_path)
    require(protocol['schema_version'] == 1 and set(protocol['runs']) == {'order_a', 'order_b'}, 'Protocol run schema mismatch')
    require(protocol['seed'] == 20260925 and protocol['training_enabled'] is False and protocol['complete_raw_reconstruction'] is False, 'Protocol scientific scope mismatch')
    require(protocol['scope'] == 'protein_pair_replay_from_pinned_intermediate_inputs', 'Protocol scope mismatch')
    require(protocol['runs']['order_a']['inputs']['positive_candidates']['sha256'] != protocol['runs']['order_b']['inputs']['positive_candidates']['sha256'], 'Candidate input orders must differ')
    require(protocol['runs']['order_a']['inputs']['canonical_sequences']['sha256'] == protocol['runs']['order_b']['inputs']['canonical_sequences']['sha256'], 'Canonical source must be held fixed')
    require(queue['status'] == 'completed' and queue['protocol_sha256'] == protocol_sha256 and set(queue['runs']) == set(protocol['runs']), 'Replay queue incomplete/unbound')
    watch = {str(protocol_path): describe(protocol_path), str(queue_path): describe(queue_path), str(Path(__file__).resolve()): describe(Path(__file__))}
    require(str(Path(__file__).resolve()) in protocol['execution_code'], 'Independent auditor not protocol-pinned')
    for filename, expected in protocol['execution_code'].items():
        watch[filename] = binding(filename, expected, 'Execution code')
    roots = {}
    for name, spec in protocol['runs'].items():
        root = Path(spec['root']).resolve()
        roots[name] = root
        observed = queue['runs'][name]
        require(observed['status'] == 'completed' and observed['exit_code'] == 0, 'Replay stage failed/incomplete')
        for key in ('root', 'code', 'inputs'):
            require(observed[key] == spec[key], f'Queue protocol disagreement: {key}')
        m = read_json(root / MANIFEST)
        require(m['release_id'] == protocol['release_id'] and m['protocol_run_id'] == name, 'Run release/protocol identity')
        require(Path(m['protocol']['path']).resolve() == protocol_path and m['protocol']['sha256'] == protocol_sha256, 'Adapter protocol binding')
        require(m['worker_runtime']['file'] == 'worker_runtime.json', 'Worker runtime path mismatch')
        runtime_path = root / 'worker_runtime.json'
        watch[str(runtime_path)] = binding(runtime_path, m['worker_runtime'], 'Actual worker runtime')
        runtime = read_json(runtime_path)
        require(runtime['python_hash_seed'] == str(spec['python_hash_seed']), 'Actual worker hash seed mismatch')
        require(runtime['hash_probe_text'] == 'deterministic_pair_worker_probe_v1', 'Wrong worker hash probe')
        probe_env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
        probe_env['PYTHONHASHSEED'] = str(spec['python_hash_seed'])
        probe = json.loads(subprocess.check_output([sys.executable, '-P', '-s', '-B', '-c', "import json,sys; print(json.dumps([hash('deterministic_pair_worker_probe_v1'), sys.flags.hash_randomization]))"], env=probe_env, text=True))
        require([runtime['hash_probe'], runtime['hash_randomization']] == probe, 'Worker effective hash-seed probe mismatch')
        require(all(m['environment'].get(k) == v for k, v in runtime.items()), 'Worker environment attribution mismatch')
        require(set(m['implementation']) == {'adapter', 'builder', 'builder_dependency'}, 'Implementation coverage mismatch')
        implementations = {str(Path(v['path']).resolve()): v for v in m['implementation'].values()}
        require(set(implementations) == set(spec['code']), 'Code binding coverage mismatch')
        for filename, expected in spec['code'].items():
            require(implementations[filename]['sha256'] == expected['sha256'], 'Adapter code SHA differs from protocol')
            watch[filename] = binding(filename, expected, 'Stage code')
        require(read_json(root / LEGACY)['builder_sha256'] == m['implementation']['builder']['sha256'], 'Legacy builder identity mismatch')
        require(set(spec['inputs']) == {'positive_candidates', 'canonical_sequences'} and set(m['inputs']) == set(spec['inputs']), 'Input binding coverage mismatch')
        for key, expected in spec['inputs'].items():
            actual = m['inputs'][key]
            require(str(Path(actual['path']).resolve()) == expected['file'], 'Source input path mismatch')
            require(actual['sha256'] == expected['sha256'] and actual['bytes'] == expected['bytes'], 'Adapter source binding differs from protocol')
            watch[expected['file']] = binding(expected['file'], expected, 'Stage input')
        require(digest(root / CANONICAL) == spec['inputs']['canonical_sequences']['sha256'], 'Local canonical table differs from pinned source')
        required = set(OUTPUTS) | {MANIFEST}
        require(required <= set(observed['outputs']), 'Queue output binding coverage incomplete')
        for rel, expected in observed['outputs'].items():
            p = (root / rel).resolve()
            require(p.is_relative_to(root) and not Path(rel).is_absolute(), 'Output escapes run root')
            watch[str(p)] = binding(p, expected, 'Queue output')
        for rel in (CANONICAL, CANDIDATES):
            watch[str(root / rel)] = describe(root / rel)
    result = audit_core(roots['order_a'], roots['order_b'])
    require(result['scientific_audit']['candidate_records'] == protocol['candidate_records_per_run'], 'Protocol candidate count mismatch')
    for filename, before in watch.items():
        require(describe(filename) == before, f'Input/code/output changed during audit: {filename}')
    result.update({'status': 'pair_replay_determinism_passed', 'completed_utc': datetime.now(timezone.utc).isoformat(), 'protocol': {'file': str(protocol_path), 'sha256': protocol_sha256}, 'queue_status': {'file': str(queue_path), **describe(queue_path)}, 'verified_files': watch})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--protocol-sha256', required=True)
    p.add_argument('--queue-status', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists(), 'Refusing to overwrite existing audit')
    try:
        result = verify_bound(a.protocol, a.protocol_sha256, a.queue_status)
        code = 0
    except Exception as e:
        result = {'status': 'pair_replay_determinism_rejected', 'training_enabled': False, 'data_release_gate_passed': False, 'complete_raw_reconstruction': False, 'target_scoring_enabled': False, 'error': f'{type(e).__name__}: {e}'}
        code = 1
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(a.output)}), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
