"""Actual frozen-builder fixtures plus fail-closed independent audit regressions."""
import csv
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load('independent_pair_replay_auditor', HERE / 'verify_pair_replays.py')
helper = load('original_adapter_test_helpers', WORKSPACE / 'biopaws/portable_build/deterministic_v1/tests/test_replay_pairs.py')


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def rewrite_table(path, rows, fields=None):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), delimiter='\t')
    writer.writeheader()
    writer.writerows(rows)
    path.write_bytes(gzip.compress(stream.getvalue().encode(), mtime=42))


def refresh(root):
    """Rebind tampered fixtures so scientific checks, not stale hashes, reject."""
    m = audit.read_json(root / audit.MANIFEST)
    old = audit.read_json(root / audit.LEGACY)
    old['data_files'] = {Path(p).name: audit.digest(root / p) for p in audit.PAIR_FILES}
    old['acceptance'] = audit.read_json(root / audit.ACCEPTANCE)
    write_json(root / audit.LEGACY, old)
    m['outputs'] = {p: helper.adapter.output_identity(root, p) for p in audit.OUTPUTS}
    m['legacy_builder_manifest']['sha256'] = audit.digest(root / audit.LEGACY)
    m['acceptance']['sha256'] = audit.digest(root / audit.ACCEPTANCE)
    write_json(root / audit.MANIFEST, m)


@pytest.fixture(scope='module')
def actual_replays(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('actual_pair_replay_integration')
    table, source, candidates = helper.inputs.__wrapped__(tmp)
    fields, rows = audit.table(table)
    for row in rows:
        row['length'] = str(len(row['sequence']))
    rewrite_table(table, rows, fields + ['length'])
    reverse = tmp / 'reverse.jsonl'
    helper.write_candidates(reverse, list(reversed(candidates)))
    roots = [tmp / 'order_a', tmp / 'order_b']
    protocol_path = helper.protocol_for(tmp, table, [source, reverse], roots)
    protocol = audit.read_json(protocol_path)
    protocol['runs'] = dict(zip(('order_a', 'order_b'), protocol['runs'].values()))
    protocol.update(scope='protein_pair_replay_from_pinned_intermediate_inputs', training_enabled=False,
                    complete_raw_reconstruction=False, candidate_records_per_run=len(candidates),
                    execution_code={str(HERE / 'verify_pair_replays.py'): audit.describe(HERE / 'verify_pair_replays.py')})
    write_json(protocol_path, protocol)
    for seed, src, root in zip(('0', '12345'), (source, reverse), roots):
        result = subprocess.run(helper.command(table, src, root, protocol_path),
                                env=dict(os.environ, PYTHONHASHSEED=seed, CUDA_VISIBLE_DEVICES=''),
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    queue = {'status': 'completed', 'protocol_sha256': audit.digest(protocol_path), 'runs': {}}
    for name, root in zip(('order_a', 'order_b'), roots):
        queue['runs'][name] = dict(protocol['runs'][name], status='completed', exit_code=0,
                                  outputs={str(p.relative_to(root)): audit.describe(p) for p in root.rglob('*') if p.is_file()})
    queue_path = tmp / 'pair_queue_status.json'
    write_json(queue_path, queue)
    return roots, protocol_path, queue_path


@pytest.fixture
def copies(tmp_path, actual_replays):
    roots, _, _ = actual_replays
    out = [tmp_path / 'a', tmp_path / 'b']
    for src, dst in zip(roots, out):
        shutil.copytree(src, dst)
    return out


def test_real_adapter_bound_integration(actual_replays):
    roots, protocol, queue = actual_replays
    report = audit.verify_bound(protocol, audit.digest(protocol), queue)
    assert report['status'] == 'pair_replay_determinism_passed'
    assert report['scientific_audit']['candidate_records'] == 27
    assert report['scientific_audit']['alignment_rows_recomputed'] == 18
    assert not report['training_enabled'] and not report['complete_raw_reconstruction']


def test_gzip_container_time_and_only_runtime_may_differ(copies):
    for rel in audit.PAIR_FILES:
        path = copies[1] / rel
        path.write_bytes(gzip.compress(gzip.decompress(path.read_bytes()), mtime=999999))
    c = audit.read_json(copies[1] / audit.CONSTRUCTION)
    c['runtime_seconds'] += 99
    write_json(copies[1] / audit.CONSTRUCTION, c)
    refresh(copies[1])
    assert audit.audit_core(*copies)['scientific_audit']['unique_unordered_pairs'] == 18


@pytest.mark.parametrize('field,value', [('pair_id', '0' * 64), ('sequence_sha256_a', '0' * 64),
                                         ('global_identity', '.0'), ('mmseqs_evalue', '1.0')])
def test_both_outputs_same_bad_row_still_rejected(copies, field, value):
    for root in copies:
        _, rows = audit.table(root / audit.PAIR_FILES[0])
        target = next(r for r in rows if r['label'] == '1')
        target[field] = value
        rewrite_table(root / audit.PAIR_FILES[0], rows)
        for s in audit.SPLITS:
            rewrite_table(root / f'data/pairs/protein_sequence_similarity_{s}.tsv.gz', [r for r in rows if r['split'] == s])
        refresh(root)
    with pytest.raises(ValueError):
        audit.audit_core(*copies)


def test_both_bad_degree_files_rejected(copies):
    for root in copies:
        path = root / audit.DEGREES
        path.write_text(path.read_text().replace('\t1\t1\n', '\t2\t2\n', 1))
        refresh(root)
    with pytest.raises(ValueError, match='Degree file'):
        audit.audit_core(*copies)


def test_both_reordered_split_projection_rejected(copies):
    for root in copies:
        p = root / 'data/pairs/protein_sequence_similarity_train.tsv.gz'
        _, rows = audit.table(p)
        rewrite_table(p, list(reversed(rows)))
        refresh(root)
    with pytest.raises(ValueError, match='projection'):
        audit.audit_core(*copies)


def test_both_false_acceptance_counter_rejected(copies):
    for root in copies:
        x = audit.read_json(root / audit.ACCEPTANCE)
        x['unique_sequences'] += 1
        write_json(root / audit.ACCEPTANCE, x)
        refresh(root)
    with pytest.raises(ValueError, match='Acceptance counters'):
        audit.audit_core(*copies)


def test_construction_scientific_difference_rejected(copies):
    x = audit.read_json(copies[1] / audit.CONSTRUCTION)
    x['filter_counts']['invented'] = 1
    write_json(copies[1] / audit.CONSTRUCTION, x)
    refresh(copies[1])
    with pytest.raises(ValueError, match='Scientific JSON'):
        audit.audit_core(*copies)


def test_current_gzip_manifest_sha_must_match(copies):
    p = copies[1] / audit.MANIFEST
    m = audit.read_json(p)
    m['outputs'][audit.PAIR_FILES[0]]['sha256'] = '0' * 64
    write_json(p, m)
    with pytest.raises(ValueError, match='SHA mismatch'):
        audit.audit_core(*copies)


def test_legacy_label_cannot_impersonate_current(copies):
    p = copies[1] / audit.MANIFEST
    m = audit.read_json(p)
    m['release_id'] = '2026-09-25-v2'
    write_json(p, m)
    with pytest.raises(ValueError, match='historical release'):
        audit.audit_core(*copies)


def test_queue_missing_binding_fails(actual_replays, tmp_path):
    _, protocol, queue = actual_replays
    q = audit.read_json(queue)
    del q['runs']['order_a']['outputs'][audit.PAIR_FILES[0]]
    path = tmp_path / 'queue.json'
    write_json(path, q)
    with pytest.raises(ValueError, match='coverage incomplete'):
        audit.verify_bound(protocol, audit.digest(protocol), path)
