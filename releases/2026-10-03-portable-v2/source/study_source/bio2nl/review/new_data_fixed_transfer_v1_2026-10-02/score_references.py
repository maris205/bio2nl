"""Apply one source-selected surface head and two hard constant references."""
from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
ROLES = ('source_test', 'target')
CE_REASON = 'deterministic_wrong_predictions_have_infinite_CE'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # The pinned feature module has dataclasses with postponed annotations.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def apply_surface(matrix, state, feature_columns):
    require(state['feature_columns'] == list(feature_columns), 'Surface feature columns changed')
    require(state['source_train_only'] is True and state['dtype'] == 'float64', 'Surface state must be source-only float64')
    mean, scale, weight = (np.asarray(state[key], dtype=np.float64) for key in ('mean', 'scale', 'coefficients'))
    require(mean.shape == scale.shape == weight.shape == (9,), 'Surface scaler/head shape differs')
    require(matrix.dtype == np.float64 and matrix.ndim == 2 and matrix.shape[1] == 9 and np.isfinite(matrix).all(), 'Invalid surface matrix')
    require(np.isfinite(mean).all() and np.isfinite(scale).all() and np.isfinite(weight).all() and np.all(scale > 0), 'Invalid source scaler/head')
    require(math.isfinite(state['bias']), 'Nonfinite source bias')
    margin = ((matrix - mean) / scale) @ weight + state['bias']
    logp = np.column_stack((-np.logaddexp(0., margin), -np.logaddexp(0., -margin)))
    require(np.isfinite(logp).all(), 'Nonfinite fixed surface score')
    return logp


def constant_predictions(labels, label):
    labels = np.asarray(labels)
    require(type(label) is int and label in (0, 1), 'Unknown constant class')
    require(labels.dtype == np.int64 and labels.ndim == 1 and set(labels.tolist()) == {0, 1}, 'Constant CE policy expects both binary labels')
    decisions = np.full(len(labels), label, dtype=np.int64)
    scores = np.zeros(len(labels), dtype=np.float64)
    metrics = {'rows': len(labels), 'accuracy': float(accuracy_score(labels, decisions)),
               'balanced_accuracy': float(balanced_accuracy_score(labels, decisions)),
               'mcc': float(matthews_corrcoef(labels, decisions)), 'auroc': float(roc_auc_score(labels, scores)),
               'cross_entropy': None, 'predicted_positive_fraction': float(decisions.mean()),
               'confusion_matrix': confusion_matrix(labels, decisions, labels=[0, 1]).tolist()}
    return metrics, decisions, scores


def save_constant(path, labels, predictions, scores, rows):
    path = Path(path)
    require(not path.exists(), 'Refuse to overwrite constant predictions')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.npz.partial')
    with temp.open('xb') as stream:
        np.savez_compressed(stream, labels=labels, predictions=predictions, scores=scores,
                            row_ids=np.asarray([r['row_id'] for r in rows], dtype=str),
                            group_ids=np.asarray([r['group_id'] for r in rows], dtype=str))
    temp.replace(path)


def run(root=ROOT):
    root = Path(root).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Reference scoring must be CPU-only')
    common = import_file('_new_data_references_common', root / 'common.py')
    protocol = common.check_gate(root, mode='full')
    input_hashes = common.gate_file_hashes(root, mode='full')
    directory = root / 'results/references'
    require(not directory.exists(), 'Preserve existing reference outputs')
    started = time.monotonic()
    refs = {key: protocol[key] for key in ('source_selection', 'feature_loader', 'surface_checkpoint')}
    for name, ref in refs.items():
        require(Path(ref['file']).is_absolute() and common.sha(ref['file']) == ref['sha256'], 'Changed upstream reference: ' + name)
    require(Path(refs['source_selection']['file']) == common.M2_RESULTS / 'source_selection.json' and
            Path(refs['feature_loader']['file']) == common.M2 / 'code_snapshot/surface_features.py',
            'Reference inputs must come from this new-data source stage')
    for ref in refs.values():
        require(input_hashes.setdefault(str(ref['file']), ref['sha256']) == ref['sha256'], 'Conflicting reference input')
    source = common.read(refs['source_selection']['file'])
    require(source['status'] == 'frozen' and source['conditions_filtered_by_score'] is False, 'Source selection not frozen')
    m2_root = common.M2_RESULTS
    source_surface_ref = source['surface_selection']
    source_surface_path = common.path_inside(m2_root, source_surface_ref['file'])
    require(common.sha(source_surface_path) == source_surface_ref['sha256'], 'Source surface selection changed')
    input_hashes[str(source_surface_path)] = source_surface_ref['sha256']
    source_surface = common.read(source_surface_path)
    expected_checkpoint = common.path_inside(m2_root, source_surface['checkpoint'])
    require(Path(refs['surface_checkpoint']['file']) == expected_checkpoint and refs['surface_checkpoint']['sha256'] == source_surface['checkpoint_sha256'],
            'Surface scorer must use the unique source-selected head')
    state = common.read(expected_checkpoint)
    require(state['protocol_sha256'] == source['protocol_sha256'] and state['prepared_manifest_sha256'] == source['prepared_manifest_sha256'],
            'Surface source provenance differs')
    features = import_file('_new_data_pinned_surface_features', refs['feature_loader']['file'])
    manifest = common.read(root / 'prepared/manifest.json')
    protocol_sha, manifest_sha = common.sha(root / 'protocol.json'), common.sha(root / 'prepared/manifest.json')
    records = {}
    for name in ('surface', 'constant0', 'constant1'):
        record = {'status': 'completed', 'kind': 'surface' if name == 'surface' else 'constant',
                  'source_selection_sha256': refs['source_selection']['sha256'], 'protocol_sha256': protocol_sha,
                  'prepared_manifest_sha256': manifest_sha, 'roles': {}, 'labels_used_for_training_or_selection': False}
        if name == 'surface':
            record['checkpoint'] = refs['surface_checkpoint']
            record['feature_loader'] = refs['feature_loader']
        else:
            record['constant_label'] = int(name[-1])
        records[name] = record
    for role in ROLES:
        arrays, rows = common.load_role(root, role, manifest=manifest)
        matrix = np.asarray([features.vector_features(r['ids_a'], r['ids_b']) for r in rows], dtype=np.float64)
        logp = apply_surface(matrix, state, features.FEATURE_COLUMNS)
        path = directory / 'surface' / (role + '.predictions.npz')
        common.save_predictions(path, arrays['labels'], logp, [r['row_id'] for r in rows], [r['group_id'] for r in rows])
        records['surface']['roles'][role] = {'metrics': common.metrics(arrays['labels'], logp),
                                           'predictions': {'file': str(path.relative_to(root)), 'sha256': common.sha(path)},
                                           'cross_entropy_status': 'finite'}
        for label in (0, 1):
            metrics, decisions, scores = constant_predictions(arrays['labels'], label)
            path = directory / ('constant' + str(label)) / (role + '.predictions.npz')
            save_constant(path, arrays['labels'], decisions, scores, rows)
            records['constant' + str(label)]['roles'][role] = {'metrics': metrics,
                'predictions': {'file': str(path.relative_to(root)), 'sha256': common.sha(path)},
                'cross_entropy_status': 'infinite', 'cross_entropy_reason': CE_REASON}
        print('Fixed references scored ' + role + ': ' + str(len(rows)) + ' rows', flush=True)
    require(common.sha(expected_checkpoint) == refs['surface_checkpoint']['sha256'], 'Surface source checkpoint changed during scoring')
    for name, record in records.items():
        common.check_hashes(root, input_hashes)
        record['file_sha256'] = input_hashes
        record['all_input_hashes_unchanged'] = True
        record['elapsed_seconds'] = time.monotonic() - started
        common.atomic(directory / name / 'metrics.json', record)
    return records


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        run()
