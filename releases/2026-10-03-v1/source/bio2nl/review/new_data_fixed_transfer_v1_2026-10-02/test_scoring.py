"""Synthetic-only scorer identity/checkpoint/metric tests; no held-out data."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).absolute().parent
FROZEN_MODEL = HERE.parent / 'new_data_full_training_v1_2026-10-02/code_snapshot/model.py'
FROZEN_FEATURES = HERE.parent / 'new_data_full_training_v1_2026-10-02/code_snapshot/surface_features.py'


def module(name, path=None):
    path = path or HERE / (name + '.py')
    spec = importlib.util.spec_from_file_location('_synthetic_fixed_transfer_' + name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


score = module('score_models')
references = module('score_references')
torch.set_num_threads(1)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def path_inside(root, name):
    if Path(name).is_absolute():
        raise ValueError('relative path required')
    path = (Path(root) / name).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('outside root')
    return path


def check_hashes(root, pins):
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('hash changed')


def full_selection():
    return {'status': 'frozen', 'conditions_filtered_by_score': False,
            'selected': [{'job': {'condition': c, 'pt_seed': p, 'ft_seed': f, 'smoke': False}}
                         for c in ('EP', 'ES', 'EE') for p in range(3) for f in range(3)]}


class TinyFixture:
    def __init__(self, root):
        self.common = SimpleNamespace(M2=root, M2_RESULTS=root / 'results', sha=sha, read=read,
                                      path_inside=path_inside, check_hashes=check_hashes)
        self.model_file = root / 'code_snapshot/model.py'
        self.model_file.parent.mkdir(parents=True)
        self.model_file.write_bytes(FROZEN_MODEL.read_bytes())
        self.loader = module('tiny_model', self.model_file)
        design = {'pretraining': {'architecture': {'model_type': 'gpt2', 'n_layer': 1, 'n_head': 2,
                    'n_embd': 16, 'n_inner': 32, 'attn_pdrop': 0., 'embd_pdrop': 0., 'resid_pdrop': 0.},
                    'context_length': 16}, 'tokenizer': {'vocab_size': 32, 'pad_token_id': 0, 'eos_token_id': 1}}
        self.source_protocol_file = root / 'protocol.json'
        write(self.source_protocol_file, {'design': design})
        self.original = self.loader.create_classifier(self.loader.create_pretraining_model(design, 0), 0).eval()
        state = self.loader.snapshot_state(self.original)
        self.job = {'condition': 'EP', 'pt_seed': 0, 'ft_seed': 0, 'smoke': False}
        self.payload = {'schema_version': 1, 'kind': 'source_classifier', 'config': self.original.backbone.config.to_dict(),
                        'state_dict': state, 'state_sha256': self.loader.state_digest(state),
                        'metadata': {'job': self.job, 'epoch': 2, 'protocol_sha256': sha(self.source_protocol_file),
                                     'prepared_manifest_sha256': 'b' * 64}}
        self.directory = root / 'results/full/source/EP/pt0/ft0'
        self.directory.mkdir(parents=True)
        self.cp_file = self.directory / 'best.pt'
        self.metadata_file = self.directory / 'metadata.json'
        self.selection_file = root / 'results/source_selection.json'
        self.selection = full_selection()
        self.selection.update(protocol_sha256=sha(self.source_protocol_file), prepared_manifest_sha256='b' * 64)
        self.protocol = {'source_selection': {'file': str(self.selection_file)},
                         'source_protocol': {'file': str(self.source_protocol_file)},
                         'model_loader': {'file': str(self.model_file), 'sha256': sha(self.model_file)}}
        self.repin()

    def repin(self):
        torch.save(self.payload, self.cp_file)
        cp = {'file': 'best.pt', 'kind': 'source_classifier', 'sha256': sha(self.cp_file),
              'bytes': self.cp_file.stat().st_size, 'state_sha256': self.payload['state_sha256']}
        metadata = {'status': 'completed', 'kind': 'source_sft', 'run_kind': 'full', 'job': self.job,
                    'best_epoch': 2, 'protocol_sha256': self.selection['protocol_sha256'],
                    'prepared_manifest_sha256': self.selection['prepared_manifest_sha256'], 'checkpoint': cp}
        write(self.metadata_file, metadata)
        chosen = {'job': self.job, 'best_epoch': 2,
                  'checkpoint': {**cp, 'file': str(self.cp_file.relative_to(self.common.M2_RESULTS))},
                  'source_result': {'file': str(self.metadata_file.relative_to(self.common.M2_RESULTS)),
                                    'sha256': sha(self.metadata_file), 'bytes': self.metadata_file.stat().st_size}}
        self.selection['selected'][0] = chosen
        write(self.selection_file, self.selection)
        self.protocol['source_selection']['sha256'] = sha(self.selection_file)
        self.protocol['source_protocol']['sha256'] = sha(self.source_protocol_file)

    def load(self):
        return score.load_selected_model(self.common, self.protocol, 'EP', 0, 0)


def tiny_arrays():
    return {'input_ids': np.array([[4, 2, 5, 1, 0], [6, 7, 2, 8, 1], [9, 2, 10, 1, 0]], dtype=np.int64),
            'attention_mask': np.array([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0]], dtype=np.int64),
            'labels': np.array([0, 1, 0], dtype=np.int64)}


def test_complete_new_classifier_safe_reload_preserves_exact_predictions(tmp_path):
    fixture = TinyFixture(tmp_path)
    model, loader, chosen, path, pins = fixture.load()
    expected = fixture.loader.predict_source(fixture.original, tiny_arrays(), device='cpu')
    actual = loader.predict_source(model, tiny_arrays(), device='cpu')
    np.testing.assert_array_equal(actual['log_probabilities'], expected['log_probabilities'])
    np.testing.assert_array_equal(actual['predictions'], expected['predictions'])
    assert loader.state_digest(model.state_dict()) == chosen['checkpoint']['state_sha256']
    assert all(p.dtype == torch.float32 and not p.requires_grad for p in model.parameters())
    assert path == tmp_path / 'results/full/source/EP/pt0/ft0/best.pt'
    assert str(fixture.metadata_file) in pins
    assert len(pins) == 5


def test_selected_checkpoint_hash_tamper_fails_before_deserialization(tmp_path):
    fixture = TinyFixture(tmp_path)
    with fixture.cp_file.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Selected classifier'):
        fixture.load()


def test_selected_epoch_is_checked_inside_complete_checkpoint(tmp_path):
    fixture = TinyFixture(tmp_path)
    fixture.payload['metadata']['epoch'] = 3
    fixture.repin()
    with pytest.raises(ValueError, match='epoch differs'):
        fixture.load()


def test_selected_prepared_lineage_checked_inside_checkpoint(tmp_path):
    fixture = TinyFixture(tmp_path)
    fixture.payload['metadata']['prepared_manifest_sha256'] = 'c' * 64
    fixture.repin()
    with pytest.raises(ValueError, match='manifest differs'):
        fixture.load()


def test_actual_checkpoint_architecture_bound_to_source_protocol(tmp_path):
    fixture = TinyFixture(tmp_path)
    source = read(fixture.source_protocol_file)
    source['design']['pretraining']['architecture']['n_embd'] = 32
    write(fixture.source_protocol_file, source)
    fixture.selection['protocol_sha256'] = sha(fixture.source_protocol_file)
    fixture.payload['metadata']['protocol_sha256'] = sha(fixture.source_protocol_file)
    fixture.repin()
    with pytest.raises(ValueError, match='config differs'):
        fixture.load()


def test_no_historical_loader_can_replace_new_source_loader(tmp_path):
    fixture = TinyFixture(tmp_path)
    fixture.protocol['model_loader']['file'] = str(tmp_path / 'old_loader.py')
    with pytest.raises(ValueError, match='Wrong current source reference'):
        fixture.load()


def test_all_conditions_required_without_source_score_filter():
    selection = full_selection()
    assert score.exact_selected(selection, 'ES', 2, 1)['job']['ft_seed'] == 1
    for mutation in ('missing', 'duplicate', 'filtered', 'smoke'):
        value = copy.deepcopy(selection)
        if mutation == 'missing': value['selected'].pop()
        elif mutation == 'duplicate': value['selected'][-1] = value['selected'][0]
        elif mutation == 'filtered': value['conditions_filtered_by_score'] = True
        else: value['selected'][0]['job']['smoke'] = True
        with pytest.raises(ValueError):
            score.exact_selected(value, 'EP', 0, 0)


def test_smoke_is_exact_first64_in_both_roles():
    rows = [{'row_id': str(i)} for i in range(100)]
    arrays = {'input_ids': np.arange(100 * 8).reshape(100, 8), 'attention_mask': np.ones((100, 8), dtype=np.int64),
              'labels': np.arange(100, dtype=np.int64) % 2}
    selected, identities = score.subset(arrays, rows, True)
    assert identities == rows[:64]
    for key in arrays: np.testing.assert_array_equal(selected[key], arrays[key][:64])
    assert score.subset(arrays, rows, False)[1] == rows


def test_constant_references_have_fixed_decisions_and_no_artificial_finite_ce(tmp_path):
    labels = np.array([0, 0, 0, 1], dtype=np.int64)
    rows = [{'row_id': str(i), 'group_id': 'g'} for i in range(4)]
    for label in (0, 1):
        metrics, decisions, scores = references.constant_predictions(labels, label)
        assert metrics['accuracy'] == (.75 if label == 0 else .25)
        assert metrics['balanced_accuracy'] == metrics['auroc'] == .5
        assert metrics['mcc'] == 0. and metrics['cross_entropy'] is None
        assert np.all(decisions == label) and np.all(scores == 0.)
        path = tmp_path / f'constant{label}.npz'
        references.save_constant(path, labels, decisions, scores, rows)
        with np.load(path) as loaded:
            np.testing.assert_array_equal(loaded['predictions'], decisions)
            assert 'log_probabilities' not in loaded
            assert loaded['row_ids'].tolist() == ['0', '1', '2', '3']
        with pytest.raises(ValueError, match='overwrite'):
            references.save_constant(path, labels, decisions, scores, rows)


def test_surface_source_scaler_and_head_unchanged():
    columns = [str(i) for i in range(9)]
    matrix = np.arange(27, dtype=np.float64).reshape(3, 9)
    state = {'feature_columns': columns, 'source_train_only': True, 'dtype': 'float64',
             'mean': list(range(9)), 'scale': [3.] * 9, 'coefficients': [.2] * 9, 'bias': -.1}
    before = copy.deepcopy(state)
    actual = references.apply_surface(matrix, state, columns)
    margin = ((matrix - np.arange(9)) / 3) @ np.full(9, .2) - .1
    expected = np.column_stack((-np.logaddexp(0., margin), -np.logaddexp(0., -margin)))
    np.testing.assert_array_equal(actual, expected)
    assert state == before
    with pytest.raises(ValueError, match='columns'):
        references.apply_surface(matrix, state, list(reversed(columns)))


def test_surface_fixed_direction_and_argmax_zero_tie():
    columns = list('abcdefghi')
    state = {'feature_columns': columns, 'source_train_only': True, 'dtype': 'float64',
             'mean': [0.] * 9, 'scale': [1.] * 9, 'coefficients': [1.] + [0.] * 8, 'bias': 0.}
    x = np.zeros((3, 9)); x[:, 0] = [-2, 0, 2]
    logp = references.apply_surface(x, state, columns)
    np.testing.assert_array_equal(logp.argmax(1), [0, 0, 1])
    np.testing.assert_allclose(logp[:, 1] - logp[:, 0], [-2, 0, 2], rtol=0, atol=1e-15)
    state['scale'][0] = 0
    with pytest.raises(ValueError, match='scaler/head'):
        references.apply_surface(x, state, columns)


def test_pinned_new_feature_module_imports_dataclass_and_preserves_symmetry():
    features = references.import_file('_synthetic_current_features', FROZEN_FEATURES)
    for a, b in (([4, 4, 5], [4, 5, 5]), ([7], [8]), ([7], [7, 8])):
        value = features.vector_features(a, b)
        assert value.shape == (9,) and value.dtype == np.float64
        np.testing.assert_array_equal(value, features.vector_features(b, a))
    np.testing.assert_array_equal(features.vector_features([7], [8])[6:], [1., 1., 1.])
