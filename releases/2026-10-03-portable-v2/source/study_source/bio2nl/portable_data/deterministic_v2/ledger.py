"""Reconstruct the exact new conflict-exclusion ledger from pinned references.

Caller authenticates physical input files. This module only reads files/objects;
it never edits source records, generates task examples or grants a release gate.
"""
import hashlib
import json
import os
from pathlib import Path

TASKS = ('pawsx_en', 'cola', 'rte')
FIELDS = ('task', 'row_id', 'official_split', 'reason', 'input_group_id')
CONFLICT = 'contradictory_labels_for_same_semantic_input'
REASONS = {CONFLICT, 'input_group_reserved_by_higher_priority_evaluation_split', 'duplicate_semantic_input'}
SPLITS = {'train', 'validation', 'test'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def unique_keys(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, 'Duplicate JSON key')
        obj[key] = value
    return obj


def read_path(value):
    path = Path(os.path.abspath(value))
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Missing/symlinked ledger input')
    return path.read_bytes()


def rows(value):
    if isinstance(value, (str, Path)):
        lines = read_path(value).decode('utf-8').splitlines()
        require(all(line.strip() for line in lines), 'Blank ledger row')
        value = [json.loads(line, object_pairs_hook=unique_keys) for line in lines]
    require(isinstance(value, list), 'Ledger must be an ordered row list')
    result, seen = [], set()
    for row in value:
        require(isinstance(row, dict) and set(row) == set(FIELDS), 'Unexpected ledger fields')
        require(all(isinstance(row[k], str) and row[k] for k in FIELDS), 'Invalid ledger field value')
        require(row['task'] in TASKS and row['official_split'] in SPLITS and row['reason'] in REASONS, 'Invalid ledger task/split/reason')
        require(row['row_id'] not in seen, 'Duplicate ledger row ID')
        seen.add(row['row_id'])
        result.append({k: row[k] for k in FIELDS})
    return result


def expected_ledger(reference_ledger, reference_conflicts):
    """Return full row dicts in the fixed new order, preserving within-group rows."""
    source = rows(reference_ledger)
    if isinstance(reference_conflicts, (str, Path)):
        reference_conflicts = json.loads(read_path(reference_conflicts), object_pairs_hook=unique_keys)
    require(isinstance(reference_conflicts, dict) and set(reference_conflicts) == set(TASKS), 'Conflict task coverage differs')
    byid = {r['row_id']: r for r in source}
    expected, seen_conflicts = [], set()
    # Original producer writes each task's conflict segment, then its remaining
    # exclusions. Reference conflict-group order itself was hash-seed dependent.
    previous_task = -1
    nonconflict_seen = set()
    for row in source:
        pos = TASKS.index(row['task'])
        require(pos >= previous_task, 'Reference ledger task segments are interleaved')
        previous_task = pos
        if row['reason'] != CONFLICT:
            nonconflict_seen.add(row['task'])
        else:
            require(row['task'] not in nonconflict_seen, 'Reference conflict row appears after other exclusions')
    for task in TASKS:
        groups = reference_conflicts[task]
        require(isinstance(groups, list), 'Conflict groups must be a list')
        semantic_ids = set()
        for group in groups:
            require(isinstance(group, dict) and set(group) == {'semantic_input_id', 'rows'}, 'Unexpected conflict group fields')
            key = group['semantic_input_id']
            require(isinstance(key, str) and key and key not in semantic_ids, 'Duplicate/invalid conflict semantic ID')
            semantic_ids.add(key)
        for group in sorted(groups, key=lambda g: g['semantic_input_id']):
            require(isinstance(group['rows'], list) and len(group['rows']) >= 2, 'Incomplete conflict group')
            labels = set()
            for item in group['rows']:
                require(isinstance(item, dict) and set(item) == {'row_id', 'official_split', 'label'}, 'Unexpected conflict member fields')
                require(type(item['label']) is int and item['label'] in (0, 1), 'Invalid conflicting label')
                labels.add(item['label'])
                key = item['row_id']
                require(isinstance(key, str) and key in byid, 'Conflict row is missing from the reference ledger')
                require(key not in seen_conflicts, 'Duplicate conflict row ID')
                row = byid[key]
                require(row['task'] == task and row['reason'] == CONFLICT and row['official_split'] == item['official_split'], 'Conflict member and ledger disagree')
                seen_conflicts.add(key)
                expected.append(dict(row))
            require(labels == {0, 1}, 'Conflict group has no contradictory labels')
        expected.extend(dict(r) for r in source if r['task'] == task and r['reason'] != CONFLICT)
    require(seen_conflicts == {r['row_id'] for r in source if r['reason'] == CONFLICT}, 'Conflict metadata omits ledger rows')
    require(len(expected) == len(source) and {r['row_id'] for r in expected} == set(byid), 'Ledger reconstruction changed membership')
    return expected


def serialized(value):
    return ''.join(json.dumps({k: row[k] for k in FIELDS}, ensure_ascii=False, separators=(',', ':')) + '\n' for row in value).encode('utf-8')


def compare_ledger(reference_ledger, reference_conflicts, current_ledger):
    expected = expected_ledger(reference_ledger, reference_conflicts)
    current = rows(current_ledger)
    require(current == expected, 'Current ledger differs from the exact deterministic row order/content')
    data = serialized(expected)
    if isinstance(current_ledger, (str, Path)):
        require(read_path(current_ledger) == data, 'Current ledger serialization differs from the fixed writer')
    return {'comparison': 'exact_deterministic_conflict_ledger', 'equivalent': True,
            'rows': len(expected), 'conflict_rows': sum(r['reason'] == CONFLICT for r in expected),
            'expected_sha256': hashlib.sha256(data).hexdigest(), 'expected_bytes': len(data),
            'order': 'pawsx_en,cola,rte; sorted semantic conflict IDs, original group-member order, original nonconflict order',
            'membership_changed': False}
