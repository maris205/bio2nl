"""Independent CPU source-result audit and immutable all-condition selection."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

try:
    from .runtime_data import ReleaseData, CONDITION_STREAMS, canonical_hash, file_sha256, relative_path, require
    from .audit_runtime import Artifacts, CONFIG_KEYS, load_new, validate_predictions, save_json
    from . import model as models
except ImportError:
    from runtime_data import ReleaseData, CONDITION_STREAMS, canonical_hash, file_sha256, relative_path, require
    from audit_runtime import Artifacts, CONFIG_KEYS, load_new, validate_predictions, save_json
    import model as models

CONDITIONS = ("EP", "ES", "EE")
SEEDS = (0, 1, 2)
SOURCE_COUNTS = {"train": 8044, "validation": 20276}

METRICS = ('accuracy', 'balanced_accuracy', 'auroc', 'cross_entropy', 'mcc', 'predicted_positive_fraction')


def metric_values(labels, logp):
    labels = np.asarray(labels)
    logp = np.asarray(logp, dtype=np.float64)
    require(labels.ndim == 1 and len(labels) > 0 and np.isin(labels, [0, 1]).all(), 'Invalid labels')
    require(logp.shape == (len(labels), 2) and np.isfinite(logp).all(), 'Invalid log probabilities')
    require(np.max(np.abs(np.logaddexp(logp[:, 0], logp[:, 1]))) <= 2e-6, 'Unnormalized log probabilities')
    pred = logp.argmax(1)
    tn = int(((labels == 0) & (pred == 0)).sum()); fp = int(((labels == 0) & (pred == 1)).sum())
    fn = int(((labels == 1) & (pred == 0)).sum()); tp = int(((labels == 1) & (pred == 1)).sum())
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    recalls = [v / n for v, n in ((tn, tn + fp), (tp, tp + fn)) if n]
    return {'rows': len(labels), 'accuracy': (tp + tn) / len(labels), 'balanced_accuracy': float(np.mean(recalls)),
            'mcc': (tp * tn - fp * fn) / denominator if denominator else 0.0,
            'auroc': float(roc_auc_score(labels, logp[:, 1] - logp[:, 0])) if len(set(labels)) == 2 else None,
            'cross_entropy': float(-logp[np.arange(len(labels)), labels.astype(int)].mean()),
            'predicted_positive_fraction': float(pred.mean()), 'confusion_matrix': [[tn, fp], [fn, tp]]}


def compare_metrics(actual, saved, atol=1e-7):
    require(set(actual) == set(saved), 'Metric field set differs')
    largest = 0.0
    for key, value in actual.items():
        if key in ('rows', 'confusion_matrix') or value is None:
            require(saved[key] == value, 'Metric differs: ' + key)
        else:
            difference = abs(value - saved[key]); largest = max(largest, difference)
            require(math.isfinite(saved[key]) and difference <= atol, 'Metric differs: ' + key)
    return largest


def select_epoch(history):
    require(history and [row['epoch'] for row in history] == list(range(1, len(history) + 1)), 'Epoch roster differs')
    return min(history, key=lambda row: (row['validation']['metrics']['cross_entropy'], row['epoch']))


def nested_summary(records):
    """Average SFT seeds first; sample SD across three independently pretrained seeds."""
    result = []
    for condition in CONDITIONS:
        for stage in ('best', 'final'):
            for role in ('train', 'validation'):
                for metric in METRICS:
                    means, within = [], []
                    for pt_seed in SEEDS:
                        group = sorted((r for r in records if r['condition'] == condition and r['pt_seed'] == pt_seed
                                        and r['stage'] == stage and r['role'] == role), key=lambda r: r['ft_seed'])
                        require([r['ft_seed'] for r in group] == list(SEEDS), 'Incomplete nested seed group')
                        values = [r[metric] for r in group]
                        means.append(float(np.mean(values))); within.append(float(np.std(values, ddof=1)))
                    result.append({'condition': condition, 'stage': stage, 'role': role, 'metric': metric,
                                   'pretraining_seed_values': means, 'within_pretraining_sft_sample_sd': within,
                                   'mean': float(np.mean(means)), 'pretraining_sample_sd': float(np.std(means, ddof=1))})
    return result


def paired_contrasts(records):
    out = []
    keyed = {(r['condition'], r['pt_seed'], r['ft_seed'], r['stage'], r['role']): r for r in records}
    for a, b in (('EP', 'ES'), ('EP', 'EE'), ('ES', 'EE')):
        for stage in ('best', 'final'):
            for role in ('train', 'validation'):
                for metric in METRICS:
                    effects = [float(np.mean([keyed[a, p, f, stage, role][metric] - keyed[b, p, f, stage, role][metric]
                                              for f in SEEDS])) for p in SEEDS]
                    out.append({'contrast': a + '-' + b, 'stage': stage, 'role': role, 'metric': metric,
                                'pretraining_seed_effects': effects, 'mean': float(np.mean(effects)),
                                'pretraining_sample_sd': float(np.std(effects, ddof=1))})
    return out


def write_csv(path, rows):
    require(rows, 'Empty report table')
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)


def state_digest(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        require(value.dtype == torch.float32 and torch.isfinite(value).all().item(), 'Checkpoint nonfinite/non-FP32 tensor: ' + name)
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        data = value.numpy().tobytes()
        digest.update(len(header).to_bytes(8, 'big')); digest.update(header)
        digest.update(len(data).to_bytes(8, 'big')); digest.update(data)
    return digest.hexdigest()



def descriptor(path, root=None):
    path = Path(path)
    return {'file': str(path.relative_to(root)) if root else str(path),
            'sha256': file_sha256(path), 'bytes': path.stat().st_size}


def read_record(data, artifacts, directory, job, kind, expected_inputs):
    path = artifacts.read(directory / 'metadata.json')
    value = json.loads(path.read_text())
    require(value['status'] == 'completed' and value['kind'] == kind and value['job'] == job and
            value['run_kind'] == 'full', 'Run incomplete or job identity differs')
    require(value['protocol_sha256'] == data.protocol_sha256 and
            value['prepared_manifest_sha256'] == data.manifest_sha256 and
            value['raw_manifest_sha256'] == data.manifest['raw_manifest_sha256'] and
            value['raw_release_id'] == data.manifest['raw_release_id'], 'Run provenance differs')
    require(value['source_only'] is True and value['target_examples_read'] == 0 and
            value['source_test_examples_read'] == 0 and value['old_weights_loaded'] is False,
            'Non-source or historical model access recorded')
    require(value['final_metrics_are_smoke_only'] is False and value['tf32_enabled'] is False and
            value['cpu_threads'] == 4, 'Formal numerical mode differs')
    require(value['inputs'] == expected_inputs, 'Run read set or input identity differs')
    code = Path(__file__).absolute().parent / 'runtime.py'
    require(value['portable_training_module_sha256'] == file_sha256(code), 'Training module identity differs')
    return value, descriptor(path, data.output_root)


def checkpoint_identity(data, artifacts, directory, metadata, kind, config, epoch=None):
    model, payload = load_new(data, artifacts, directory, metadata, kind, config, epoch)
    actual = state_digest(payload['state_dict'])
    require(actual == payload['state_sha256'] == metadata['checkpoint']['state_sha256'],
            'Independently framed checkpoint tensor digest differs')
    identity = {'state_sha256': actual}
    if kind == 'source_classifier':
        require(model.score.weight.shape == (2, config['n_embd']) and model.score.bias is None,
                'Complete bias-free source head differs')
        identity['head_sha256'] = state_digest({'score.weight': payload['state_dict']['score.weight']})
        identity['backbone_sha256'] = state_digest({k: v for k, v in payload['state_dict'].items()
                                                   if k.startswith('backbone.')})
    else:
        identity['backbone_sha256'] = state_digest({'backbone.' + k[len('transformer.'):]: v
                                                  for k, v in payload['state_dict'].items()
                                                  if k.startswith('transformer.')})
    del model, payload
    gc.collect()
    return identity


def expected_input_members(data, base, relatives):
    output = dict(base)
    for relative in relatives:
        path = data.output_path(relative)
        output[str(path)] = data.manifest['outputs'][relative]
    return output


def pretraining_trace(record):
    require(record['updates'] == 1024 and record['input_tokens'] == 16777216 and
            record['causal_loss_positions'] == 16744448 and record['scheduler_total_steps'] == 1024 and
            record['skipped_optimizer_updates'] == 0, 'Pretraining fixed budget differs')
    require(record['precision'] == 'float32_parameters_float16_autocast' and
            record['fresh_random_initialization'] is True, 'Pretraining precision/initialization differs')
    require(record['loss_scaler'] == {'init_scale': 1024.0, 'growth_factor': 2.0,
            'backoff_factor': 0.5, 'growth_interval': 2000}, 'Pretraining scaler differs')
    history = record['history']
    require(history == record['loss_history'] and [h['update'] for h in history] == list(range(1, 1025)),
            'Pretraining update history differs')
    for row in history:
        step = row['update'] - 1
        multiplier = step / 21 if step < 21 else .5 * (1 + math.cos(math.pi * (step - 21) / 1003))
        require(abs(row['learning_rate_used'] - 3e-4 * multiplier) <= 1e-14, 'Pretraining LR differs')
        require(all(math.isfinite(row[k]) for k in ('mean_microbatch_ce', 'unscaled_gradient_norm',
                'loss_scale_before', 'loss_scale_after')), 'Nonfinite optimizer trace')
        require(row['loss_scale_after'] >= row['loss_scale_before'] > 0 and row['unscaled_gradient_norm'] >= 0,
                'Skipped/nonfinite pretraining update')
    require(abs(record['online_training_ce'] - np.mean([h['mean_microbatch_ce'] for h in history])) <= 1e-10,
            'Pretraining online CE differs')


def source_trace(record, train_rows):
    updates_per_epoch = math.ceil(train_rows / 32)
    total = updates_per_epoch * 5
    require(record['updates'] == record['scheduler_total_steps'] == total and
            record['sample_presentations'] == train_rows * 5, 'Source full budget differs')
    require(record['precision'] == 'float32_parameters_bfloat16_autocast' and
            record['only_source_validation_selected_checkpoint'] is True and
            record['scientific_model_selection_performed'] is True and
            record['technical_smoke_single_epoch'] is False and record['backbone_and_head_updated'] is True,
            'Source precision or source-only selection differs')
    history = record['update_history']
    require([h['update'] for h in history] == list(range(1, total + 1)), 'SFT update roster differs')
    for row in history:
        expected_lr = 2e-5 * (1 - (row['update'] - 1) / total)
        require(abs(row['learning_rate_used'] - expected_lr) <= 1e-14, 'SFT linear LR differs')
        require(math.isfinite(row['online_dropout_training_ce']) and
                math.isfinite(row['unscaled_gradient_norm']) and row['unscaled_gradient_norm'] >= 0,
                'Nonfinite SFT optimizer trace')
    return updates_per_epoch


def surface_reference(data, artifacts):
    root = data.output_root
    selection_path = artifacts.read(root / 'surface/selection.json')
    audit_path = artifacts.read(data.protocol_path.parent / 'verification/surface_audit.json')
    selected, checked = json.loads(selection_path.read_text()), json.loads(audit_path.read_text())
    require(checked['status'] == 'passed' and checked['candidates'] == 5 and checked['prediction_artifacts'] == 10 and
            checked['all_read_files_final_rehashed'] is True, 'Surface independent audit incomplete')
    for value in (selected, checked):
        require(value['protocol_sha256'] == data.protocol_sha256 and
                value['prepared_manifest_sha256'] == data.manifest_sha256 and
                value['source_counts'] == data.source_counts and value['source_only'] is True and
                value['target_examples_read'] is False, 'Surface provenance/boundary differs')
    require(selected['status'] == 'selected_pending_independent_audit' and
            selected['selection_metric'] == 'source_validation_cross_entropy' and len(selected['candidates']) == 5,
            'Surface candidate inventory/selection differs')
    best = min(selected['candidates'], key=lambda row: (row['metrics']['validation']['cross_entropy'], row['candidate_id']))
    require(selected['selected_candidate'] == checked['selected_candidate'] == best['candidate_id'],
            'Surface selection differs')
    require(checked['selection_sha256'] == file_sha256(selection_path), 'Surface selection audit hash differs')
    require(selected['checkpoint'] == checked['checkpoint'] == best['checkpoint'] and
            selected['checkpoint_sha256'] == checked['checkpoint_sha256'] == best['checkpoint_sha256'],
            'Surface selected checkpoint binding differs')
    require(Path(checked['selection_file']).absolute() == selection_path, 'Surface audited selection path differs')
    for filename, digest in checked['files'].items():
        artifacts.read(filename, digest)
    artifacts.member(root, {'file': relative_path(best['checkpoint']), 'sha256': best['checkpoint_sha256']})
    return {'lambda': best['penalty'], 'selected_candidate': best['candidate_id'], 'metrics': best['metrics'],
            'checkpoint': best['checkpoint'], 'checkpoint_sha256': best['checkpoint_sha256'], 'seed_sd': None}, \
        descriptor(selection_path, root), descriptor(audit_path)


def make_report(summary):
    index = {(r['condition'], r['stage'], r['role'], r['metric']): r for r in summary['nested_metrics']}
    lines = ['# 新数据全量训练：源阶段审计', '',
             '完成 9 个从头预训练模型和 27 个源任务分类器；本阶段只使用源训练/验证数据。', '',
             '每个预训练 seed 内先平均 3 个微调 seed，再对 3 个预训练 seed 报告均值 ± 样本标准差。', '',
             '| 条件 | 选中模型训练准确率 | 验证准确率 | 验证 AUC | 验证 CE |',
             '|---|---:|---:|---:|---:|']
    for condition in CONDITIONS:
        cells = []
        for role, metric in [('train', 'accuracy'), ('validation', 'accuracy'), ('validation', 'auroc'), ('validation', 'cross_entropy')]:
            row = index[condition, 'best', role, metric]
            factor = 100 if metric == 'accuracy' else 1
            cells.append(f"{row['mean'] * factor:.4f} ± {row['pretraining_sample_sd'] * factor:.4f}" + ('%' if factor == 100 else ''))
        lines.append('| ' + condition + ' | ' + ' | '.join(cells) + ' |')
    surface = summary['surface_reference']
    lines.extend(['', f"确定性表面特征参考：源验证 CE 选择 λ={surface['lambda']}；不报告 seed SD。", '',
                  '保留所有条件与种子，包括弱源域表现或恒定预测。验证 CE 在 5 个 epoch 中选模，平手选更早 epoch；验证结果含选模偏差。', '',
                  '完整逐 epoch、选中与最终 epoch、嵌套种子和配对差值分别见 epoch_metrics.csv、per_job_metrics.csv、nested_metrics.csv、paired_contrasts.csv。最终 epoch 的指标保留；只保存源验证选中的完整分类器权重。', '',
                  f"独立 CPU 审计重算 {summary['prediction_artifacts']} 份预测、{summary['prediction_rows']:,} 行指标，并检查 36 个完整 checkpoint 的架构、有限值和来源；没有重复全量模型推理。新代码的独立 GPU smoke 重载提供 4 个模型的小规模前向复现。", '',
                  '这是已观察过历史结果后的探索性重训。本报告没有源测试或英文目标评分，没有英文迁移结论、置信区间或显著性检验。预训练在线训练损失仅用于优化诊断。'])
    return '\n'.join(lines) + '\n'


def write_results(data, artifacts, pretrains, source_selected, epochs, selected_rows, init_by_seed,
                  head_initials, maximum_difference, prediction_count, prediction_rows):
    require(prediction_count == 270 and len(source_selected) == 27 and len(pretrains) == 9, 'Incomplete source matrix')
    root, report_dir = data.output_root, data.output_root / 'report'
    selection_path = root / 'source_selection.json'
    require(not report_dir.exists() and not selection_path.exists(), 'Preserve existing source report/selection')
    surface, surface_selection, surface_audit = surface_reference(data, artifacts)
    nested, contrasts = nested_summary(selected_rows), paired_contrasts(selected_rows)
    summary = {'status': 'completed', 'source_only': True, 'pretraining_runs': 9, 'source_fits': 27,
               'prediction_artifacts': prediction_count, 'prediction_rows': prediction_rows, 'complete_checkpoints': 36,
               'protocol_sha256': data.protocol_sha256, 'prepared_manifest_sha256': data.manifest_sha256,
               'source_counts': data.source_counts, 'surface_reference': surface,
               'nested_metrics': nested, 'paired_contrasts': contrasts,
               'target_scoring_enabled': False, 'source_test_scoring_enabled': False,
               'heldout_or_target_predictions': 0, 'all_conditions_retained': True,
               'uncertainty': 'descriptive_sample_SD_over_3_pretraining_seeds_after_3_SFT_seed_mean; no_CI_or_pvalues'}
    ledger = artifacts.final_check()
    input_ledger = data.verify_current_inputs()
    report_dir.mkdir(parents=True, exist_ok=False)
    write_csv(report_dir / 'epoch_metrics.csv', epochs)
    write_csv(report_dir / 'per_job_metrics.csv', selected_rows)
    write_csv(report_dir / 'nested_metrics.csv', nested)
    write_csv(report_dir / 'paired_contrasts.csv', contrasts)
    save_json(report_dir / 'summary.json', summary)
    with (report_dir / 'REPORT.md').open('x') as stream:
        stream.write(make_report(summary))
    report_hashes = {str(p.relative_to(root)): descriptor(p) for p in sorted(report_dir.iterdir()) if p.is_file()}
    audit_value = {'status': 'passed', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                   'protocol_sha256': data.protocol_sha256,
                   'prepared_manifest_sha256': data.manifest_sha256, 'pretraining_runs': 9, 'source_fits': 27,
                   'prediction_artifacts_audited': 270, 'prediction_rows_audited': prediction_rows,
                   'complete_checkpoints_audited': 36, 'maximum_metric_difference': maximum_difference,
                   'paired_initial_state_sha256': init_by_seed, 'paired_initial_head_sha256': head_initials,
                   'artifact_inputs': ledger, 'data_and_code_inputs': input_ledger, 'outputs': report_hashes,
                   'all_read_files_final_rehashed': True, 'source_only': True, 'new_model_inference': False,
                   'target_or_source_test_examples_read': 0, 'target_scoring_enabled': False,
                   'source_test_scoring_enabled': False}
    # Recheck after report computation and before creating the selection barrier.
    artifacts.final_check()
    data.verify_current_inputs()
    audit_path = report_dir / 'audit.json'
    save_json(audit_path, audit_value)
    selection = {'status': 'frozen', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                 'protocol_sha256': data.protocol_sha256,
                 'prepared_manifest_sha256': data.manifest_sha256, 'selected': source_selected,
                 'conditions_filtered_by_score': False, 'source_audit': descriptor(audit_path, root),
                 'surface_selection': surface_selection, 'surface_audit': surface_audit,
                 'target_scoring_started': False, 'target_scoring_enabled': False,
                 'source_test_scoring_enabled': False}
    selection['selection_sha256'] = canonical_hash(selection)
    save_json(selection_path, selection)
    return audit_value


def audit_source(data):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Final source audit must be CPU-only')
    require(data.mode == 'full' and data.full_training_accepted, 'Full training acceptance required')
    require(data.source_counts == SOURCE_COUNTS, 'New-data source counts differ')
    root = data.output_root
    require(not (root / 'report').exists() and not (root / 'source_selection.json').exists(), 'Preserve old report/selection')
    torch.set_num_threads(4)
    artifacts = Artifacts()
    base_inputs = dict(data.accessed)
    config_all = models.make_config(data.design).to_dict()
    config = {key: config_all[key] for key in CONFIG_KEYS}
    labels, row_ids, row_hashes = {}, {}, {}
    for role in ('train', 'validation'):
        arrays, rows = data.load_source(role, smoke=False)
        labels[role] = arrays['labels'].copy()
        row_ids[role] = np.asarray([r['row_id'] for r in rows], dtype=str)
        row_hashes[role] = canonical_hash(row_ids[role].tolist())
        del arrays, rows
    source_members = [data.manifest['source'][role][key] for role in ('train', 'validation') for key in ('npz', 'rows')]
    source_inputs = expected_input_members(data, base_inputs, source_members)
    initial_references = {}
    head_references = {}
    for seed in SEEDS:
        fresh = models.create_pretraining_model(data.design, seed)
        initial_references[str(seed)] = state_digest(fresh.state_dict())
        del fresh
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            head = torch.nn.Linear(config['n_embd'], 2, bias=False, dtype=torch.float32)
            torch.nn.init.normal_(head.weight, std=.02)
            head_references[str(seed)] = state_digest({'score.weight': head.weight})
        del head
        gc.collect()
    require(len(set(initial_references.values())) == len(set(head_references.values())) == 3,
            'Independent fresh seeds have identical initialization states')
    pretrains, init_by_seed, source_selected, epochs, selected_rows = {}, {}, [], [], []
    head_initials = {str(seed): set() for seed in SEEDS}
    biggest_difference = 0.0
    prediction_count = prediction_rows = 0
    for pt_seed in SEEDS:
        for condition in CONDITIONS:
            directory = root / 'full/pretrain' / condition / f'pt{pt_seed}'
            job = {'condition': condition, 'pt_seed': pt_seed, 'smoke': False}
            members = [data.manifest['streams'][name]['bin'] for name in CONDITION_STREAMS[condition]]
            members.append(data.manifest['schedules'][str(pt_seed)]['file'])
            expected = expected_input_members(data, base_inputs, members)
            if condition != 'EP':
                paired = pretrains['EP', pt_seed]['ref']
                expected[str(root / paired['file'])] = {k: paired[k] for k in ('sha256', 'bytes')}
            record, ref = read_record(data, artifacts, directory, job, 'pretraining', expected)
            pretraining_trace(record)
            require(record['initial_state_sha256'] == initial_references[str(pt_seed)], 'Fresh seed initialization differs')
            if condition != 'EP':
                parent = pretrains['EP', pt_seed]
                require(record['paired_EP_initialization'] == {'file': str(root / parent['ref']['file']),
                        'sha256': parent['ref']['sha256']}, 'Paired EP initialization provenance differs')
            identity = checkpoint_identity(data, artifacts, directory, record, 'causal_lm', config)
            require(identity['state_sha256'] == record['final_state_sha256'] != record['initial_state_sha256'],
                    'Pretraining final state unchanged/different')
            # Probe identity is audited; final CPU reporting does not execute LM inference.
            loaded = data.load_pretrain(condition, pt_seed)
            expected_ids = np.stack([loaded['streams'][int(s)][int(i)] for s, i in loaded['schedule'][0][[0, 8]]]).astype(np.int64)
            require(record['probe']['file'] == 'probe.npz', 'Pretraining probe path differs')
            probe = artifacts.npz(directory, record['probe'])
            require(set(probe) == {'input_ids', 'token_negative_log_likelihood'} and
                    np.array_equal(probe['input_ids'], expected_ids) and probe['input_ids'].dtype == np.int64 and
                    probe['token_negative_log_likelihood'].dtype == np.float64 and
                    probe['token_negative_log_likelihood'].shape == (2, 511) and
                    np.isfinite(probe['token_negative_log_likelihood']).all(), 'Final pretraining probe identity differs')
            require(abs(float(probe['token_negative_log_likelihood'].mean()) - record['probe']['mean_nll']) <= 1e-12,
                    'Final pretraining probe mean differs')
            del loaded, probe
            pretrains[condition, pt_seed] = {'record': record, 'ref': ref, 'identity': identity}
            init_by_seed[str(pt_seed)] = record['initial_state_sha256']
            print(f'Audited complete pretraining checkpoint {condition} pt{pt_seed}', flush=True)
    for pt_seed in SEEDS:
        for condition in CONDITIONS:
            parent = pretrains[condition, pt_seed]
            parent_directory = root / 'full/pretrain' / condition / f'pt{pt_seed}'
            parent_cp = {**parent['record']['checkpoint'], 'file': str(parent_directory / 'model.pt')}
            for ft_seed in SEEDS:
                directory = root / 'full/source' / condition / f'pt{pt_seed}' / f'ft{ft_seed}'
                job = {'condition': condition, 'pt_seed': pt_seed, 'ft_seed': ft_seed, 'smoke': False}
                expected_source_inputs = {**source_inputs, str(parent_directory / 'model.pt'): {k: parent_cp[k] for k in ('sha256', 'bytes')},
                                         str(root / parent['ref']['file']): {k: parent['ref'][k] for k in ('sha256', 'bytes')}}
                record, source_ref = read_record(data, artifacts, directory, job, 'source_sft', expected_source_inputs)
                updates_per_epoch = source_trace(record, data.source_counts['train'])
                require(record['source_counts'] == data.source_counts and record['source_row_ids_sha256'] == row_hashes,
                        'Full-source row membership/order differs')
                require(record['pretraining_checkpoint'] == parent_cp and
                        record['new_parent_metadata'] == {'file': str(root / parent['ref']['file']), 'sha256': parent['ref']['sha256']},
                        'SFT new pretraining parent identity differs')
                require(record['initial_backbone_sha256'] == parent['identity']['backbone_sha256'] and
                        record['new_parent_backbone_exactly_inherited'] is True and record['fresh_head_initialization'] is True and
                        record['initial_head_sha256'] == head_references[str(ft_seed)], 'Source initialization differs')
                head_initials[str(ft_seed)].add(record['initial_head_sha256'])
                history = record['history']
                require(len(history) == 5 and [v['epoch'] for v in history] == [1, 2, 3, 4, 5], 'Source epoch roster differs')
                for item in history:
                    epoch = item['epoch']
                    require(item['updates'] == updates_per_epoch * epoch and
                            item['sample_presentations'] == data.source_counts['train'] * epoch and
                            item['epoch_presentations'] == data.source_counts['train'] and
                            item['selection_metric'] == 'source_validation_cross_entropy', 'Source epoch budget/selection differs')
                    require(math.isfinite(item['online_dropout_training_ce']) and
                            math.isfinite(item['last_unscaled_gradient_norm']) and item['last_unscaled_gradient_norm'] >= 0,
                            'Nonfinite epoch optimizer summary')
                    metric_path = artifacts.read(directory / 'epochs' / f'epoch{epoch:02d}' / 'metrics.json')
                    require(json.loads(metric_path.read_text()) == item, 'Epoch JSON differs from run history')
                    for role in ('train', 'validation'):
                        reference = item[role]['predictions']
                        require(reference['file'] == f'epochs/epoch{epoch:02d}/{role}.predictions.npz' and
                                reference['row_ids_sha256'] == row_hashes[role], 'Prediction path or row digest differs')
                        values = artifacts.npz(directory, reference)
                        validate_predictions(values, labels[role], row_ids[role])
                        metrics = metric_values(values['labels'], values['log_probabilities'])
                        largest = compare_metrics(metrics, item[role]['metrics'])
                        biggest_difference = max(biggest_difference, largest)
                        prediction_count += 1
                        prediction_rows += len(labels[role])
                        epochs.append({'condition': condition, 'pt_seed': pt_seed, 'ft_seed': ft_seed,
                                       'epoch': epoch, 'role': role, 'online_dropout_training_ce': item['online_dropout_training_ce'], **metrics})
                best = select_epoch(history)
                require(record['best_epoch'] == best['epoch'] and
                        record['best_validation_cross_entropy'] == best['validation']['metrics']['cross_entropy'], 'Best epoch differs')
                require(record['scoring'] == {r: best[r]['predictions'] for r in ('train', 'validation')} and
                        record['last_scoring'] == {r: history[-1][r]['predictions'] for r in ('train', 'validation')},
                        'Selected/final prediction binding differs')
                identity = checkpoint_identity(data, artifacts, directory, record, 'source_classifier', config, best['epoch'])
                require(identity == {k: best[k] for k in ('state_sha256', 'head_sha256', 'backbone_sha256')},
                        'Source-selected complete state differs from scored epoch')
                require(record['final_state_sha256'] == history[-1]['state_sha256'] and
                        record['final_epoch_checkpoint_retained'] == (record['best_epoch'] == 5), 'Final state retention disclosure differs')
                require(history[-1]['head_sha256'] != record['initial_head_sha256'] and
                        history[-1]['backbone_sha256'] != record['initial_backbone_sha256'], 'Source backbone/head failed to update')
                for stage, item in (('best', best), ('final', history[-1])):
                    for role in ('train', 'validation'):
                        selected_rows.append({'condition': condition, 'pt_seed': pt_seed, 'ft_seed': ft_seed,
                                              'stage': stage, 'epoch': item['epoch'], 'role': role, **item[role]['metrics']})
                checkpoint = {**record['checkpoint'], 'file': str((directory / record['checkpoint']['file']).relative_to(root))}
                source_selected.append({'job': job, 'checkpoint': checkpoint, 'best_epoch': best['epoch'],
                                        'source_result': source_ref, 'best_validation': best['validation']['metrics']})
                print(f'Audited source classifier {condition} pt{pt_seed} ft{ft_seed}', flush=True)
    require(all(len(values) == 1 for values in head_initials.values()), 'Paired source head initializations differ')
    heads = {seed: next(iter(values)) for seed, values in head_initials.items()}
    require(heads == head_references and init_by_seed == initial_references, 'Independently regenerated initialization differs')
    result = write_results(data, artifacts, pretrains, source_selected, epochs, selected_rows, init_by_seed,
                           heads, biggest_difference, prediction_count, prediction_rows)
    print('Source audit passed: 9 pretraining / 27 selected classifiers / 270 predictions; no held-out scoring', flush=True)
    return result


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args(argv)
    data = ReleaseData(args.protocol, args.protocol_sha256, mode='full')
    audit_source(data)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
