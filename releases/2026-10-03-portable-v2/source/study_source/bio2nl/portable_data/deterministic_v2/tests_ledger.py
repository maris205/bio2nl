import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ledger_test_module', HERE / 'ledger.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    rows = [{'task': 'pawsx_en', 'row_id': name, 'official_split': 'train',
             'reason': m.CONFLICT, 'input_group_id': name[:1]} for name in ('z1', 'z2', 'a1', 'a2')]
    rows.append({'task': 'pawsx_en', 'row_id': 'duplicate', 'official_split': 'train',
                 'reason': 'duplicate_semantic_input', 'input_group_id': 'other'})
    groups = {task: [] for task in m.TASKS}
    groups['pawsx_en'] = [{'semantic_input_id': key, 'rows': [
        {'row_id': key + '1', 'official_split': 'train', 'label': 1},
        {'row_id': key + '2', 'official_split': 'train', 'label': 0}]} for key in ('z', 'a')]
    return rows, groups


def test_only_conflict_group_order_changes_and_inputs_unchanged(tmp_path):
    rows, groups = fixture(); before = copy.deepcopy((rows, groups))
    expected = m.expected_ledger(rows, groups)
    assert [r['row_id'] for r in expected] == ['a1', 'a2', 'z1', 'z2', 'duplicate']
    assert (rows, groups) == before
    path = tmp_path / 'ledger.jsonl'; path.write_bytes(m.serialized(expected))
    assert m.compare_ledger(rows, groups, path)['rows'] == 5
    assert m.expected_ledger(rows, {**groups, 'pawsx_en': list(reversed(groups['pawsx_en']))}) == expected


@pytest.mark.parametrize('case', ['duplicate_ledger', 'missing_ledger', 'missing_group', 'duplicate_group',
                                'duplicate_member', 'bad_label', 'wrong_split', 'extra_field', 'current_order', 'current_value'])
def test_inconsistent_inputs_and_changed_current_fail(case):
    rows, groups = fixture(); current = m.expected_ledger(rows, groups)
    if case == 'duplicate_ledger':rows.append(dict(rows[0]))
    if case == 'missing_ledger':rows.pop(0)
    if case == 'missing_group':groups['pawsx_en'].pop(0)
    if case == 'duplicate_group':groups['pawsx_en'].append(copy.deepcopy(groups['pawsx_en'][0]))
    if case == 'duplicate_member':groups['pawsx_en'][0]['rows'][1] = dict(groups['pawsx_en'][0]['rows'][0])
    if case == 'bad_label':groups['pawsx_en'][0]['rows'][1]['label'] = 1
    if case == 'wrong_split':groups['pawsx_en'][0]['rows'][0]['official_split'] = 'test'
    if case == 'extra_field':rows[0]['ignored'] = 'not allowed'
    if case == 'current_order':current.reverse()
    if case == 'current_value':current[0]['input_group_id'] = 'changed'
    with pytest.raises(ValueError):m.compare_ledger(rows, groups, current)


def test_current_serialization_and_duplicate_json_key_fail(tmp_path):
    rows, groups = fixture(); current = m.expected_ledger(rows, groups)
    path = tmp_path / 'current.jsonl'
    path.write_text(''.join(json.dumps(r) + '\n' for r in current))
    with pytest.raises(ValueError, match='serialization'):m.compare_ledger(rows, groups, path)
    path.write_text(m.serialized(current).decode().replace('"task":"pawsx_en"', '"task":"pawsx_en","task":"pawsx_en"', 1))
    with pytest.raises(ValueError, match='Duplicate JSON'):m.compare_ledger(rows, groups, path)
