"""Independent audit fault tests and complete miniature 9+27 result integration."""
import copy
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from bio2nl.new_data_full_training_v1 import audit_runtime as replay
from bio2nl.new_data_full_training_v1 import audit_source as audit
from bio2nl.new_data_full_training_v1 import model
from bio2nl.new_data_full_training_v1 import runtime
from bio2nl.new_data_full_training_v1.runtime_data import canonical_hash, file_sha256


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return path


def pin(path):
    return {'sha256': file_sha256(path), 'bytes': path.stat().st_size}


def design():
    return {'pretraining': {'context_length': 512, 'architecture': {'model_type': 'gpt2', 'n_layer': 1,
            'n_head': 2, 'n_embd': 8, 'n_inner': 16, 'activation_function': 'gelu_new',
            'resid_pdrop': .1, 'embd_pdrop': .1, 'attn_pdrop': .1, 'layer_norm_epsilon': 1e-5,
            'initializer_range': .02, 'tie_word_embeddings': True}},
            'tokenizer': {'vocab_size': 17, 'pad_token_id': 0, 'eos_token_id': 1}}


class MiniData:
    def __init__(self, root):
        self.output_root = root / 'results'
        self.root = root / 'prepared'
        self.protocol_path = put(root / 'protocol.json', {'test': 'miniature-only'})
        self.protocol_sha256 = file_sha256(self.protocol_path)
        self.manifest_sha256 = 'b' * 64
        self.design = design()
        self.mode = 'full'
        self.full_training_accepted = True
        self.source_counts = {'train': 4, 'validation': 4}
        self.manifest = {'raw_manifest_sha256': 'a' * 64, 'raw_release_id': 'new-fixture',
                         'source': {}, 'streams': {}, 'schedules': {}, 'outputs': {}}
        self.accessed = {str(self.protocol_path): pin(self.protocol_path)}
        self.base = copy.deepcopy(self.accessed)
        self.execution_protocol = {'output_root': str(self.output_root)}
        self.rows = {r: [{'row_id': f'{r}{i}'} for i in range(4)] for r in ('train', 'validation')}
        for role in self.rows:
            self.manifest['source'][role] = {}
            for kind in ('npz', 'rows'):
                rel = f'source/{role}.{kind}'
                self.manifest['source'][role][kind] = rel
                self.add(rel)
        for name in ('english_common', 'english_extra', 'protein', 'shuffled'):
            rel = f'streams/{name}.bin'
            self.manifest['streams'][name] = {'bin': rel}
            self.add(rel)
        for seed in (0, 1, 2):
            rel = f'schedules/seed{seed}.npy'
            self.manifest['schedules'][str(seed)] = {'file': rel}
            self.add(rel)

    def add(self, rel):
        p = put(self.root / rel, {'fixture': rel})
        self.manifest['outputs'][rel] = pin(p)

    def output_path(self, rel):
        p = self.root / rel
        if pin(p) != self.manifest['outputs'][rel]:
            raise ValueError('fixture input tampered')
        self.accessed[str(p)] = pin(p)
        return p

    def verify_current_inputs(self):
        for name, expected in self.accessed.items():
            if pin(Path(name)) != expected:
                raise ValueError('fixture input changed')
        return dict(self.accessed)

    def load_source(self, role, smoke=False):
        for key in ('npz', 'rows'):
            self.output_path(self.manifest['source'][role][key])
        return {'labels': np.asarray([0, 1, 0, 1], dtype=np.int64)}, self.rows[role]

    def load_pretrain(self, condition, seed):
        for name in audit.CONDITION_STREAMS[condition]:
            self.output_path(self.manifest['streams'][name]['bin'])
        self.output_path(self.manifest['schedules'][str(seed)]['file'])
        schedule = np.zeros((1, 16, 2), dtype=np.int64)
        schedule[:, 8:, 0] = 1
        return {'streams': (np.full((1, 512), 4), np.full((1, 512), 5)), 'schedule': schedule}


def base_record(data, job, kind, inputs):
    return {'status': 'completed', 'kind': kind, 'job': job, 'run_kind': 'full',
            'protocol_sha256': data.protocol_sha256, 'prepared_manifest_sha256': data.manifest_sha256,
            'raw_manifest_sha256': data.manifest['raw_manifest_sha256'], 'raw_release_id': data.manifest['raw_release_id'],
            'source_only': True, 'target_examples_read': 0, 'source_test_examples_read': 0, 'old_weights_loaded': False,
            'final_metrics_are_smoke_only': False, 'tf32_enabled': False, 'cpu_threads': 4, 'inputs': inputs,
            'portable_training_module_sha256': file_sha256(Path(runtime.__file__))}


def cp_metadata(data, job):
    return {'job': job, 'protocol_sha256': data.protocol_sha256, 'prepared_manifest_sha256': data.manifest_sha256}


def fresh_history():
    return [{'update': i + 1, 'learning_rate_used': 3e-4 * (i / 21 if i < 21 else .5 * (1 + math.cos(math.pi * (i - 21) / 1003))),
             'mean_microbatch_ce': 1., 'unscaled_gradient_norm': 1., 'loss_scale_before': 1024., 'loss_scale_after': 1024.}
            for i in range(1024)]


def build_fixture(root):
    data = MiniData(root)
    parents = {}
    for seed in audit.SEEDS:
        for condition in audit.CONDITIONS:
            folder = data.output_root / 'full/pretrain' / condition / f'pt{seed}'
            folder.mkdir(parents=True)
            job = {'condition': condition, 'pt_seed': seed, 'smoke': False}
            inp = dict(data.base)
            for rel in [data.manifest['streams'][n]['bin'] for n in audit.CONDITION_STREAMS[condition]] + [f'schedules/seed{seed}.npy']:
                inp[str(data.root / rel)] = data.manifest['outputs'][rel]
            if condition != 'EP':
                prior = parents['EP', seed][0] / 'metadata.json'
                inp[str(prior)] = pin(prior)
            meta = base_record(data, job, 'pretraining', inp)
            lm = model.create_pretraining_model(data.design, seed)
            initial = model.state_digest(lm.state_dict())
            with torch.no_grad():
                lm.transformer.wte.weight.add_((1 + audit.CONDITIONS.index(condition)) * .001)
            cp = runtime.save_checkpoint(lm, folder / 'model.pt', 'causal_lm', cp_metadata(data, job))
            meta.update(checkpoint=cp, initial_state_sha256=initial, final_state_sha256=cp['state_sha256'],
                        updates=1024, input_tokens=16777216, causal_loss_positions=16744448, scheduler_total_steps=1024,
                        skipped_optimizer_updates=0, precision='float32_parameters_float16_autocast',
                        fresh_random_initialization=True, loss_scaler=runtime.SCALER, online_training_ce=1.,
                        history=fresh_history(), loss_history=fresh_history())
            if condition != 'EP':
                meta['paired_EP_initialization'] = {'file': str(prior), 'sha256': file_sha256(prior)}
            ids = np.stack([np.full(512, 4), np.full(512, 5)]).astype(np.int64)
            with (folder / 'probe.npz').open('wb') as f:
                np.savez_compressed(f, input_ids=ids, token_negative_log_likelihood=np.ones((2, 511), dtype=np.float64))
            meta['probe'] = {'file': 'probe.npz', 'sha256': file_sha256(folder / 'probe.npz'), 'mean_nll': 1.}
            put(folder / 'metadata.json', meta)
            parents[condition, seed] = folder, meta, model.snapshot_state(lm)
            del lm
    for seed in audit.SEEDS:
        for condition in audit.CONDITIONS:
            parent_dir, parent, parent_state = parents[condition, seed]
            for ft in audit.SEEDS:
                folder = data.output_root / 'full/source' / condition / f'pt{seed}' / f'ft{ft}'
                folder.mkdir(parents=True)
                job = {'condition': condition, 'pt_seed': seed, 'ft_seed': ft, 'smoke': False}
                inp = dict(data.base)
                for role in data.rows:
                    for key in ('npz', 'rows'):
                        rel = data.manifest['source'][role][key]
                        inp[str(data.root / rel)] = data.manifest['outputs'][rel]
                for name in ('model.pt', 'metadata.json'):
                    inp[str(parent_dir / name)] = pin(parent_dir / name)
                meta = base_record(data, job, 'source_sft', inp)
                lm = model.create_pretraining_model(data.design, seed)
                lm.load_state_dict(parent_state)
                classifier = model.create_classifier(lm, ft)
                initial = model.snapshot_state(classifier)
                meta.update(initial_state_sha256=model.state_digest(initial),
                            initial_head_sha256=model.state_digest({'score.weight': initial['score.weight']}),
                            initial_backbone_sha256=model.state_digest({k: v for k, v in initial.items() if k.startswith('backbone.')}),
                            pretraining_checkpoint={**parent['checkpoint'], 'file': str(parent_dir / 'model.pt')},
                            new_parent_metadata={'file': str(parent_dir / 'metadata.json'), 'sha256': file_sha256(parent_dir / 'metadata.json')},
                            source_counts=data.source_counts, source_row_ids_sha256={r: canonical_hash([v['row_id'] for v in rows]) for r, rows in data.rows.items()},
                            new_parent_backbone_exactly_inherited=True, fresh_head_initialization=True, only_source_validation_selected_checkpoint=True,
                            scientific_model_selection_performed=True, technical_smoke_single_epoch=False, backbone_and_head_updated=True,
                            precision='float32_parameters_bfloat16_autocast', updates=5, scheduler_total_steps=5, sample_presentations=20,
                            update_history=[{'update': i + 1, 'learning_rate_used': 2e-5 * (1 - i / 5), 'online_dropout_training_ce': 1., 'unscaled_gradient_norm': 1.} for i in range(5)])
                history = []
                for epoch in range(1, 6):
                    with torch.no_grad():
                        classifier.backbone.wte.weight.add_(.001)
                        classifier.score.weight.add_(.001)
                    state = model.snapshot_state(classifier)
                    item = {'epoch': epoch, 'updates': epoch, 'sample_presentations': 4 * epoch, 'epoch_presentations': 4,
                            'selection_metric': 'source_validation_cross_entropy', 'online_dropout_training_ce': 1., 'last_unscaled_gradient_norm': 1.,
                            'state_sha256': model.state_digest(state), 'head_sha256': model.state_digest({'score.weight': state['score.weight']}),
                            'backbone_sha256': model.state_digest({k: v for k, v in state.items() if k.startswith('backbone.')} )}
                    for role, rows in data.rows.items():
                        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
                        # Best CE ties at epochs 2/3, making earlier-epoch selection testable.
                        quality = .8 if epoch in (2, 3) else .65
                        probs = np.where(labels[:, None] == np.arange(2), quality, 1 - quality)
                        logp = np.log(probs).astype(np.float64)
                        ids = np.asarray([r['row_id'] for r in rows])
                        pred = {'labels': labels, 'predictions': logp.argmax(1), 'log_probabilities': logp, 'row_ids': ids}
                        rel = f'epochs/epoch{epoch:02d}/{role}.predictions.npz'
                        path = folder / rel
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with path.open('wb') as f:
                            np.savez_compressed(f, **pred)
                        item[role] = {'metrics': audit.metric_values(labels, logp),
                                      'predictions': {'file': rel, 'sha256': file_sha256(path), 'row_ids_sha256': canonical_hash(ids.tolist())}}
                    if epoch == 2:
                        meta['checkpoint'] = runtime.save_checkpoint(classifier, folder / 'best.pt', 'source_classifier', {**cp_metadata(data, job), 'epoch': epoch})
                    put(folder / f'epochs/epoch{epoch:02d}/metrics.json', item)
                    history.append(item)
                best = history[1]
                meta.update(history=history, best_epoch=2, best_validation_cross_entropy=best['validation']['metrics']['cross_entropy'],
                            scoring={r: best[r]['predictions'] for r in data.rows}, last_scoring={r: history[-1][r]['predictions'] for r in data.rows},
                            final_state_sha256=history[-1]['state_sha256'], final_epoch_checkpoint_retained=False)
                put(folder / 'metadata.json', meta)
                del lm, classifier
    candidates = []
    files = {}
    for i in range(5):
        path = put(data.output_root / f'surface/lambda{i}.json', {'fixture': i})
        files[str(path)] = file_sha256(path)
        candidates.append({'candidate_id': f'lambda{i}', 'penalty': [0., .001, .01, .1, 1.][i],
                           'checkpoint': f'surface/lambda{i}.json', 'checkpoint_sha256': file_sha256(path),
                           'metrics': {'train': {'cross_entropy': .5}, 'validation': {'cross_entropy': .5 + i * .01}}})
    chosen = candidates[0]
    surface = {'status': 'selected_pending_independent_audit', 'selected_candidate': chosen['candidate_id'],
               'selection_metric': 'source_validation_cross_entropy', 'candidates': candidates,
               'checkpoint': chosen['checkpoint'], 'checkpoint_sha256': chosen['checkpoint_sha256'],
               'protocol_sha256': data.protocol_sha256, 'prepared_manifest_sha256': data.manifest_sha256,
               'source_counts': data.source_counts, 'source_only': True, 'target_examples_read': False}
    surface_path = put(data.output_root / 'surface/selection.json', surface)
    checked = {**surface, 'status': 'passed', 'candidates': 5, 'prediction_artifacts': 10, 'all_read_files_final_rehashed': True,
               'selection_sha256': file_sha256(surface_path), 'selection_file': str(surface_path), 'files': files}
    put(root / 'verification/surface_audit.json', checked)
    data.accessed = dict(data.base)
    return data


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_metrics_fixed_direction_and_ties(self):
        labels = np.asarray([0, 1, 0, 1])
        logp = np.log(np.full((4, 2), .5))
        result = audit.metric_values(labels, logp)
        self.assertEqual(result['confusion_matrix'], [[2, 0], [2, 0]])
        self.assertEqual(result['auroc'], .5)
        self.assertEqual(result['mcc'], 0.)

    def test_metrics_does_not_flip_low_auc(self):
        result = audit.metric_values(np.asarray([0, 1]), np.log([[.1, .9], [.9, .1]]))
        self.assertEqual(result['auroc'], 0.)

    def test_bad_probability_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unnormalized'):
            audit.metric_values(np.asarray([0, 1]), np.zeros((2, 2)))

    def test_bad_label_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid labels'):
            audit.metric_values(np.asarray([0, 2]), np.log(np.full((2, 2), .5)))

    def test_metric_tamper_rejected(self):
        a = audit.metric_values(np.asarray([0, 1]), np.log(np.full((2, 2), .5)))
        b = {**a, 'accuracy': .7}
        with self.assertRaisesRegex(ValueError, 'Metric differs'):
            audit.compare_metrics(a, b)

    def test_select_tie_earlier(self):
        h = [{'epoch': e, 'validation': {'metrics': {'cross_entropy': v}}} for e, v in ((1, .7), (2, .5), (3, .5))]
        self.assertEqual(audit.select_epoch(h)['epoch'], 2)

    def test_select_missing_epoch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Epoch roster'):
            audit.select_epoch([{'epoch': 2, 'validation': {'metrics': {'cross_entropy': .5}}}])

    def test_fixed_reload_threshold_rejects_previous_failure(self):
        with self.assertRaisesRegex(ValueError, 'Fixed replay tolerance'):
            replay.difference(np.asarray([1.049041748046875e-5]), np.asarray([0.]))

    def test_fixed_reload_threshold_accepts_boundary(self):
        self.assertEqual(replay.difference(np.asarray([1e-5]), np.asarray([0.])), 1e-5)

    def test_invalid_reload_comparison_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Invalid numerical'):
            replay.difference(np.asarray([float('nan')]), np.asarray([0.]))

    def test_prediction_tamper_rejected(self):
        values = {'labels': np.asarray([0, 1]), 'predictions': np.asarray([0, 1]),
                  'log_probabilities': np.log(np.full((2, 2), .5)), 'row_ids': np.asarray(['a', 'b'])}
        with self.assertRaisesRegex(ValueError, 'Fixed argmax'):
            replay.validate_predictions(values, values['labels'], values['row_ids'])

    def test_ledger_detects_post_read_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = put(Path(tmp) / 'a.json', {'a': 1})
            ledger = replay.Artifacts()
            ledger.read(path)
            put(path, {'a': 2})
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                ledger.final_check()

    def test_ledger_refuses_traversal(self):
        with self.assertRaisesRegex(ValueError, 'relative'):
            replay.Artifacts().member(Path('/tmp'), {'file': '../a', 'sha256': 'a' * 64})

    def test_nonfinite_tensor_rejected(self):
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            audit.state_digest({'v': torch.tensor([float('nan')])})

    def test_nonfp32_tensor_rejected(self):
        with self.assertRaisesRegex(ValueError, 'non-FP32'):
            audit.state_digest({'v': torch.tensor([1.], dtype=torch.float64)})

    def test_tensor_digest_independent_equivalence(self):
        state = {'b': torch.ones(2, 3), 'a': torch.arange(3, dtype=torch.float32)}
        self.assertEqual(audit.state_digest(state), model.state_digest(state))

    def test_pretraining_budget_rejected(self):
        record = {'updates': 1023}
        with self.assertRaisesRegex(ValueError, 'fixed budget'):
            audit.pretraining_trace(record)

    def test_pretraining_skip_rejected(self):
        record = {'updates': 1024, 'input_tokens': 16777216, 'causal_loss_positions': 16744448,
                  'scheduler_total_steps': 1024, 'skipped_optimizer_updates': 1}
        with self.assertRaisesRegex(ValueError, 'fixed budget'):
            audit.pretraining_trace(record)

    def test_smoke_entry_cli_paths_and_mode(self):
        data = SimpleNamespace(output_root=Path('/fixture/results'))
        with patch.object(replay, 'ReleaseData', return_value=data) as reader, patch.object(replay, 'audit_runtime') as check:
            replay.main(['--protocol', '/fixture/protocol.json', '--protocol-sha256', 'a' * 64])
        reader.assert_called_once_with(Path('/fixture/protocol.json'), 'a' * 64, mode='smoke')
        check.assert_called_once_with(data, Path('/fixture/results/smoke'), Path('/fixture/results/smoke/audit_runtime'))

    def test_smoke_correct_output_path_reaches_checkpoint_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'results'
            data = SimpleNamespace(execution_protocol={'output_root': str(root)}, protocol_sha256='a' * 64,
                                   manifest_sha256='b' * 64, design=design())
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(torch.cuda, 'is_bf16_supported', return_value=True):
                with self.assertRaises(FileNotFoundError):
                    replay.audit_runtime(data, root / 'smoke', root / 'smoke/audit_runtime')
            report = json.loads((root / 'smoke/audit_runtime/metadata.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['scope'], 'new_full_training_smoke_reload')
            self.assertNotIn('original_queue_remains_failed', report)

    def test_smoke_wrong_output_path_rejected_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'results'
            data = SimpleNamespace(execution_protocol={'output_root': str(root)})
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), patch.object(torch.cuda, 'is_available', return_value=True), \
                    patch.object(torch.cuda, 'is_bf16_supported', return_value=True):
                with self.assertRaisesRegex(ValueError, 'dedicated audit directory'):
                    replay.audit_runtime(data, root / 'smoke', Path(tmp) / 'wrong')
            self.assertFalse((Path(tmp) / 'wrong').exists())

    def test_source_audit_rejects_gpu_environment(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}):
            with self.assertRaisesRegex(ValueError, 'CPU-only'):
                audit.audit_source(None)

    def test_source_audit_rejects_missing_full_gate(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            with self.assertRaisesRegex(ValueError, 'acceptance'):
                audit.audit_source(SimpleNamespace(mode='smoke'))

    def test_complete_miniature_report_and_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = build_fixture(Path(tmp))
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), patch.object(audit, 'SOURCE_COUNTS', data.source_counts):
                result = audit.audit_source(data)
            self.assertEqual(result['complete_checkpoints_audited'], 36)
            self.assertEqual(result['prediction_artifacts_audited'], 270)
            self.assertEqual(result['prediction_rows_audited'], 1080)
            self.assertEqual(result['maximum_metric_difference'], 0.)
            selection = json.loads((data.output_root / 'source_selection.json').read_text())
            self.assertEqual(len(selection['selected']), 27)
            self.assertEqual({s['best_epoch'] for s in selection['selected']}, {2})
            self.assertFalse(selection['target_scoring_enabled'])
            digest = selection.pop('selection_sha256')
            self.assertEqual(digest, canonical_hash(selection))
            summary = json.loads((data.output_root / 'report/summary.json').read_text())
            self.assertEqual(len(summary['nested_metrics']), 72)
            self.assertEqual(len(summary['paired_contrasts']), 72)
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), patch.object(audit, 'SOURCE_COUNTS', data.source_counts):
                with self.assertRaisesRegex(ValueError, 'Preserve'):
                    audit.audit_source(data)

    def test_checkpoint_wrong_epoch_and_missing_head_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = MiniData(Path(tmp))
            folder = Path(tmp) / 'one'
            folder.mkdir()
            job = {'condition': 'EP', 'pt_seed': 0, 'ft_seed': 0, 'smoke': False}
            classifier = model.create_classifier(model.create_pretraining_model(data.design, 0), 0)
            checkpoint = runtime.save_checkpoint(classifier, folder / 'best.pt', 'source_classifier', {**cp_metadata(data, job), 'epoch': 2})
            meta = {'checkpoint': checkpoint, 'job': job}
            config = model.make_config(data.design).to_dict()
            expected = {k: config[k] for k in replay.CONFIG_KEYS}
            with self.assertRaisesRegex(ValueError, 'epoch differs'):
                audit.checkpoint_identity(data, replay.Artifacts(), folder, meta, 'source_classifier', expected, 3)
            payload = torch.load(folder / 'best.pt', map_location='cpu', weights_only=True)
            del payload['state_dict']['score.weight']
            payload['state_sha256'] = model.state_digest(payload['state_dict'])
            torch.save(payload, folder / 'best.pt')
            meta['checkpoint'].update(**pin(folder / 'best.pt'), state_sha256=payload['state_sha256'])
            with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                audit.checkpoint_identity(data, replay.Artifacts(), folder, meta, 'source_classifier', expected, 2)

    def test_surface_failure_prevents_selection_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = MiniData(Path(tmp))
            data.output_root.mkdir()
            # The source reporting function fails before any report/selection output.
            with self.assertRaisesRegex(ValueError, 'Incomplete source matrix'):
                audit.write_results(data, replay.Artifacts(), {}, [], [], [], {}, {}, 0., 269, 100)
            self.assertFalse((data.output_root / 'source_selection.json').exists())
            self.assertFalse((data.output_root / 'report').exists())

    def test_bad_selected_epoch_prevents_selection_barrier(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = build_fixture(Path(tmp))
            path = data.output_root / 'full/source/EP/pt0/ft0/metadata.json'
            value = json.loads(path.read_text())
            value['best_epoch'] = 3  # Same CE but later epoch must be rejected.
            put(path, value)
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), patch.object(audit, 'SOURCE_COUNTS', data.source_counts):
                with self.assertRaisesRegex(ValueError, 'Best epoch differs'):
                    audit.audit_source(data)
            self.assertFalse((data.output_root / 'source_selection.json').exists())
            self.assertFalse((data.output_root / 'report').exists())

    def test_nested_seed_aggregation_does_not_pool_nine_iid(self):
        rows = []
        for condition in audit.CONDITIONS:
            for stage in ('best', 'final'):
                for role in ('train', 'validation'):
                    for p in audit.SEEDS:
                        for f in audit.SEEDS:
                            rows.append({'condition': condition, 'stage': stage, 'role': role, 'pt_seed': p, 'ft_seed': f,
                                         **{m: p + f * .1 for m in audit.METRICS}})
        first = audit.nested_summary(rows)[0]
        np.testing.assert_allclose(first['pretraining_seed_values'], [.1, 1.1, 2.1])
        self.assertAlmostEqual(first['pretraining_sample_sd'], 1.)
        self.assertAlmostEqual(first['within_pretraining_sft_sample_sd'][0], .1)
        contrasts = audit.paired_contrasts(rows)
        self.assertTrue(all(x['mean'] == 0 for x in contrasts))

    def test_nested_missing_seed_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            audit.nested_summary([])


if __name__ == '__main__':
    unittest.main()
