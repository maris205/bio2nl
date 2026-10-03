"""Independent metric, identity and surface-forward audit of fixed new-data transfer outputs.

Neural forward equivalence is checked by the separately frozen GPU smoke audit.
This CPU reporter recomputes all 60 full prediction artifacts without refitting.
"""
from __future__ import annotations

from collections import Counter
import csv
import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
CONDITIONS = ('EP', 'ES', 'EE')
ROLES = ('source_test', 'target')
METRICS = ('accuracy', 'balanced_accuracy', 'auroc', 'cross_entropy', 'mcc', 'predicted_positive_fraction')
CE_REASON = 'deterministic_wrong_predictions_have_infinite_CE'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def independent_metrics(labels, decisions, scores, logp=None):
    """Fixed-label binary metrics; tied ranks, zero MCC denominator => 0."""
    y = np.asarray(labels)
    p = np.asarray(decisions)
    s = np.asarray(scores)
    require(y.dtype == np.int64 and p.dtype == np.int64, 'Labels/decisions must be int64')
    require(y.ndim == p.ndim == s.ndim == 1 and y.shape == p.shape == s.shape and len(y) > 0, 'Metric vector shape differs')
    require(np.isin(y, [0, 1]).all() and np.isin(p, [0, 1]).all(), 'Metric labels are not binary')
    require(np.isfinite(s).all() and len(np.unique(y)) == 2, 'AUC requires finite scores and both labels')
    tn = int(np.sum((y == 0) & (p == 0)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    tp = int(np.sum((y == 1) & (p == 1)))
    negatives, positives = tn + fp, tp + fn
    ranks = rankdata(s, method='average')
    auc = (ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    ce = None
    if logp is not None:
        require(logp.dtype == np.float64 and logp.shape == (len(y), 2) and np.isfinite(logp).all(), 'Log-probability shape/dtype/values differ')
        require(np.max(np.abs(np.logaddexp(logp[:, 0], logp[:, 1]))) <= 2e-6, 'Scores are not normalized log probabilities')
        require(np.array_equal(p, np.argmax(logp, axis=1)), 'Argmax decisions differ; ties must select label 0')
        ce = float(-np.mean(logp[np.arange(len(y)), y]))
    return {'rows': len(y), 'accuracy': (tn + tp) / len(y),
            'balanced_accuracy': .5 * (tn / negatives + tp / positives),
            'auroc': float(auc), 'cross_entropy': ce,
            'mcc': (tp * tn - fp * fn) / denominator if denominator else 0.,
            'predicted_positive_fraction': (tp + fp) / len(y),
            'confusion_matrix': [[tn, fp], [fn, tp]]}


def independent_surface_vector(a, b):
    require(len(a) > 0 and len(b) > 0 and all(type(x) is int and 4 <= x < 32000 for x in list(a) + list(b)),
            'Empty/invalid surface endpoint content IDs')
    shorter, longer = sorted((len(a), len(b)))
    values = [math.log1p(shorter), math.log1p(longer), shorter / longer]
    for width in (1, 2):
        left = Counter(tuple(a[i:i + width]) for i in range(len(a) - width + 1))
        right = Counter(tuple(b[i:i + width]) for i in range(len(b) - width + 1))
        if not left and not right:
            values.extend((1., 1., 1.))
        elif not left or not right:
            values.extend((0., 0., 0.))
        else:
            overlap = left.keys() & right.keys()
            values.extend((len(overlap) / len(left.keys() | right.keys()),
                           2 * sum(min(left[token], right[token]) for token in overlap) / (sum(left.values()) + sum(right.values())),
                           sum(left[token] * right[token] for token in overlap) /
                           math.sqrt(sum(v * v for v in left.values()) * sum(v * v for v in right.values()))))
    return values


def surface_logp(rows, state):
    weight, center, scale = (np.asarray(state[key], dtype=np.float64) for key in ('coefficients', 'mean', 'scale'))
    require(weight.shape == center.shape == scale.shape == (9,), 'Surface feature dimension differs')
    require(np.isfinite(weight).all() and np.isfinite(center).all() and np.isfinite(scale).all() and np.all(scale > 0), 'Invalid surface scaler/head')
    require(state['source_train_only'] is True and state['dtype'] == 'float64', 'Surface provenance differs')
    bias = state['bias']
    require(math.isfinite(bias), 'Nonfinite surface bias')
    matrix = np.asarray([independent_surface_vector(r['ids_a'], r['ids_b']) for r in rows], dtype=np.float64)
    margin = ((matrix - center) / scale) @ weight + bias
    return np.stack((-np.logaddexp(0., margin), -np.logaddexp(0., -margin)), axis=1)


def compare_metrics(recorded, actual, tolerance=1e-12):
    require(set(recorded) == set(actual), 'Metric keys differ')
    biggest = 0.
    for key, expected in actual.items():
        value = recorded[key]
        if expected is None:
            require(value is None, 'Deterministic constant CE must be null')
        elif key in ('rows', 'confusion_matrix'):
            require(value == expected, 'Exact metric differs: ' + key)
        else:
            require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite recorded metric: ' + key)
            difference = abs(value - expected)
            require(difference <= tolerance, 'Metric differs: ' + key)
            biggest = max(biggest, difference)
    return biggest


def audit_prediction(path, labels, rows, recorded_metrics, constant_label=None, replay_logp=None):
    with np.load(path, allow_pickle=False) as stored:
        expected_keys = {'labels', 'predictions', 'row_ids', 'group_ids', 'scores' if constant_label is not None else 'log_probabilities'}
        require(set(stored.files) == expected_keys, 'Prediction columns differ')
        values = {key: stored[key] for key in stored.files}
    require(values['row_ids'].dtype.kind == values['group_ids'].dtype.kind == 'U', 'Row/group identity must be unicode')
    require(np.array_equal(values['labels'], labels), 'Prediction label/order mismatch')
    require(values['row_ids'].tolist() == [r['row_id'] for r in rows], 'Prediction row identity/order mismatch')
    require(values['group_ids'].tolist() == [r['group_id'] for r in rows], 'Prediction component identity/order mismatch')
    require(len(set(values['row_ids'].tolist())) == len(labels), 'Duplicated prediction row identities')
    replay_difference = 0.
    if constant_label is not None:
        require(constant_label in (0, 1), 'Unknown constant baseline')
        scores = values['scores']
        require(scores.dtype == np.float64 and np.all(scores == 0), 'Constant scores must be explicit tied zero scores')
        require(np.all(values['predictions'] == constant_label), 'Constant decisions changed')
        actual = independent_metrics(values['labels'], values['predictions'], scores)
    else:
        logp = values['log_probabilities']
        actual = independent_metrics(values['labels'], values['predictions'], logp[:, 1] - logp[:, 0], logp)
        if replay_logp is not None:
            require(logp.shape == replay_logp.shape, 'Surface replay shape differs')
            replay_difference = float(np.max(np.abs(logp - replay_logp)))
            require(replay_difference <= 1e-10, 'Independent surface score replay differs')
            require(np.array_equal(values['predictions'], replay_logp.argmax(1)), 'Independent surface decisions differ')
    metric_difference = compare_metrics(recorded_metrics, actual)
    return actual, metric_difference, replay_difference


def validate_grid(cells):
    expected = {(c, p, f, r) for c in CONDITIONS for p in range(3) for f in range(3) for r in ROLES}
    actual = [(x['condition'], x['pt_seed'], x['ft_seed'], x['role']) for x in cells]
    require(len(actual) == 54 and len(set(actual)) == 54 and set(actual) == expected, 'Incomplete/duplicated neural result grid')


def nested_summary(cells):
    """Average FT seeds within PT seed; SD over independent PT-seed means."""
    validate_grid(cells)
    lookup = {(x['condition'], x['pt_seed'], x['ft_seed'], x['role']): x['metrics'] for x in cells}
    summary = []
    for condition in CONDITIONS:
        for role in ROLES:
            for metric in METRICS:
                seed_values = [[lookup[condition, p, f, role][metric] for f in range(3)] for p in range(3)]
                means = [float(np.mean(values)) for values in seed_values]
                summary.append({'condition': condition, 'role': role, 'metric': metric,
                                'pt0_mean': means[0], 'pt1_mean': means[1], 'pt2_mean': means[2],
                                'mean': float(np.mean(means)), 'pretraining_sample_sd': float(np.std(means, ddof=1)),
                                'pt0_ft_sample_sd': float(np.std(seed_values[0], ddof=1)),
                                'pt1_ft_sample_sd': float(np.std(seed_values[1], ddof=1)),
                                'pt2_ft_sample_sd': float(np.std(seed_values[2], ddof=1)),
                                'pretraining_seeds': 3, 'fine_tuning_seeds_per_pretraining_seed': 3})
    return summary


def paired_contrasts(cells):
    validate_grid(cells)
    lookup = {(x['condition'], x['pt_seed'], x['ft_seed'], x['role']): x['metrics'] for x in cells}
    results = []
    for left, right in (('EP', 'ES'), ('EP', 'EE'), ('ES', 'EE')):
        for role in ROLES:
            for metric in METRICS:
                means = [float(np.mean([lookup[left, p, f, role][metric] - lookup[right, p, f, role][metric]
                                        for f in range(3)])) for p in range(3)]
                result = {'contrast': left + '-' + right, 'role': role, 'metric': metric,
                          'pt0_difference': means[0], 'pt1_difference': means[1], 'pt2_difference': means[2],
                          'mean_difference': float(np.mean(means)), 'pretraining_sample_sd': float(np.std(means, ddof=1)),
                          'comparison_role': 'primary' if (left, right) == ('EP', 'ES') else 'secondary',
                          'all_three_pt_differences_positive': all(value > 0 for value in means)}
                results.append(result)
    return results


def write_csv(path, rows):
    require(bool(rows) and not path.exists(), 'Empty or existing CSV report')
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_text(summary):
    lookup = {(x['condition'], x['role'], x['metric']): x for x in summary['nested_metrics']}
    lines = ['# 新数据：固定源分类器迁移评分', '',
             '全部 27 个源验证选中分类器及固定参考已完成评分；没有英语标签训练、目标调参、极性翻转或按结果筛选模型。', '',
             'EP 为英语＋天然蛋白，ES 为同一英语＋打乱蛋白，EE 为双倍英语。先平均每个预训练 seed 的 3 个 SFT seed，再报告 3 个预训练 seed 均值 ± 样本 SD。', '',
             '| 条件 | 蛋白源测试准确率 | 蛋白源测试 AUC | QQP 准确率 | QQP 平衡准确率 | QQP AUC | QQP CE |',
             '|---|---:|---:|---:|---:|---:|---:|']
    fields = [('source_test', 'accuracy'), ('source_test', 'auroc'), ('target', 'accuracy'),
              ('target', 'balanced_accuracy'), ('target', 'auroc'), ('target', 'cross_entropy')]
    for condition in CONDITIONS:
        formatted = []
        for role, metric in fields:
            item = lookup[condition, role, metric]
            factor = 100 if metric in ('accuracy', 'balanced_accuracy') else 1
            formatted.append(f"{item['mean'] * factor:.4f} ± {item['pretraining_sample_sd'] * factor:.4f}" + ('%' if factor == 100 else ''))
        lines.append('| ' + condition + ' | ' + ' | '.join(formatted) + ' |')
    lines.extend(['', '| 固定参考 | 角色 | 准确率 | 平衡准确率 | AUC | CE |', '|---|---|---:|---:|---:|---:|'])
    for item in summary['reference_cells']:
        metric = item['metrics']
        ce = f"{metric['cross_entropy']:.4f}" if metric['cross_entropy'] is not None else '∞（JSON 为 null，原因已记录）'
        lines.append(f"| {item['reference']} | {item['role']} | {metric['accuracy'] * 100:.4f}% | {metric['balanced_accuracy'] * 100:.4f}% | {metric['auroc']:.4f} | {ce} |")
    primary = summary['primary_target_auc_contrast']
    lines.extend(['',
        f"预先固定的主要对比 EP−ES：QQP AUC 差值 {primary['mean_difference']:.6f} ± {primary['pretraining_sample_sd']:.6f}，",
        f"三个预训练 seed 内先平均 SFT seed 的差值分别为 {primary['pt0_difference']:.6f}、{primary['pt1_difference']:.6f}、{primary['pt2_difference']:.6f}。",
        f"三个方向均为正且均值为正的描述性标志：{summary['primary_direction_consistency_flag']}。该标志不代表显著性。", '',
        '本轮只提供描述性统计，不报告 p 值或置信区间；SFT seed 不作为额外独立预训练重复。QQP 连通组件身份随每行预测保留，未按独立行抽样计算区间。', '',
        '源测试分数是判断蛋白源任务能力的必要背景。若源域能力仍弱，即使目标 AUC 出现小幅差异，也不能据此认定学到了可迁移的蛋白关系语义。EP−ES 同时包含 BPE token 覆盖与残基顺序差异；EE 的英语暴露量加倍，不能视为单因素机制识别。', '',
        'QQP 在历史实验中已经被评分；这是源训练数据和模型重建后的探索性固定迁移评估，不是新的盲确认。排除与暴露限制依数据协议披露，字符串排除不等于全球或语义未见。保留所有模型与参考，无目标阈值校准或标签方向优化。', '',
        '独立 CPU 审计重算 60 份预测的逐行身份、组件、固定方向指标，并从原始 token ID 独立重放 surface 特征、源 scaler 与 head。神经预测的全量审计不重复模型前向；独立 GPU smoke 提供固定 checkpoint 前向重放。完整数据见 per_job_metrics.csv、nested_metrics.csv 与 paired_contrasts.csv。'])
    return '\n'.join(lines) + '\n'


def validate_input_closure(record, gate_inputs):
    require(record['all_input_hashes_unchanged'] is True, 'Scoring did not close input hashes')
    values = record['file_sha256']
    require(isinstance(values, dict) and bool(values), 'Scoring input ledger absent')
    require(all(values.get(path) == digest for path, digest in gate_inputs.items()),
            'Scoring input ledger omits or changes a frozen gate binding')


def audit(root=ROOT):
    root = Path(root).resolve()
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Independent full reporting is CPU-only')
    common = import_file('_m3_report_common', root / 'common.py')
    protocol = common.check_gate(root, mode='full')
    report_dir = root / 'results/report'
    require(not report_dir.exists(), 'Preserve the existing complete report')
    protocol_sha = common.sha(root / 'protocol.json')
    manifest_path = root / 'prepared/manifest.json'
    manifest_sha = common.sha(manifest_path)
    manifest = common.read(manifest_path)
    source_ref = protocol['source_selection']
    source_path = Path(source_ref['file'])
    require(common.sha(source_path) == source_ref['sha256'], 'Source selection changed')
    selection = common.read(source_path)
    require(selection['status'] == 'frozen' and selection['conditions_filtered_by_score'] is False, 'Source selection status differs')
    expected_jobs = {(c, p, f) for c in CONDITIONS for p in range(3) for f in range(3)}
    chosen = {(x['job']['condition'], x['job']['pt_seed'], x['job']['ft_seed']): x for x in selection['selected']}
    require(len(selection['selected']) == len(chosen) == 27 and set(chosen) == expected_jobs, 'Frozen selected grid differs')
    require(all(item['job']['smoke'] is False for item in chosen.values()), 'Smoke classifiers cannot enter the full matrix')
    m2_root = source_path.parent
    ledger = dict(common.gate_file_hashes(root, mode='full'))
    gate_inputs = dict(ledger)
    ledger.update({str(source_path): source_ref['sha256'], str(root / 'protocol.json'): protocol_sha, str(manifest_path): manifest_sha})
    role_arrays, role_rows = {}, {}
    for role in ROLES:
        role_arrays[role], role_rows[role] = common.load_role(root, role, manifest=manifest)
        require(len(role_rows[role]) == common.COUNTS[role], 'Fixed cohort count differs')
    cells, references = [], []
    maximum_metric_difference = maximum_surface_difference = 0.
    prediction_count = prediction_rows = 0

    def bound(path, digest):
        path = Path(path).absolute()
        if str(path) not in ledger:
            require(common.sha(path) == digest, 'Artifact hash changed: ' + str(path))
        require(ledger.setdefault(str(path), digest) == digest, 'Conflicting artifact identity')
        return path

    def check_record(record, path):
        require(record['status'] == 'completed', 'Scoring run is incomplete')
        require(record['source_selection_sha256'] == source_ref['sha256'], 'Scoring selection differs')
        require(record['protocol_sha256'] == protocol_sha and record['prepared_manifest_sha256'] == manifest_sha,
                'Scoring protocol/data differs')
        require(set(record['roles']) == set(ROLES), 'Scoring roles differ')
        require(record['labels_used_for_training_or_selection'] is False, 'Scoring consumed labels for fitting or selection')
        validate_input_closure(record, gate_inputs)
        for name, digest in record['file_sha256'].items():
            bound(Path(name), digest)
        bound(path, common.sha(path))

    for condition in CONDITIONS:
        for pt_seed in range(3):
            for ft_seed in range(3):
                directory = root / 'results/neural' / f'{condition}__pt{pt_seed}__ft{ft_seed}'
                path = directory / 'metrics.json'
                record = common.read(path)
                check_record(record, path)
                selected = chosen[condition, pt_seed, ft_seed]
                require(record['job'] == {'condition': condition, 'pt_seed': pt_seed, 'ft_seed': ft_seed, 'smoke': False}, 'Scoring job identity differs')
                checkpoint = selected['checkpoint']
                checkpoint_path = common.path_inside(m2_root, checkpoint['file'])
                require(record['checkpoint'] == {'file': str(checkpoint_path), 'sha256': checkpoint['sha256'],
                                                 'state_sha256': checkpoint['state_sha256']}, 'Scoring complete checkpoint differs')
                bound(checkpoint_path, checkpoint['sha256'])
                require(record['state_before_sha256'] == record['state_after_sha256'] == checkpoint['state_sha256'], 'Scoring changed the full source head/backbone')
                require(record['source_best_epoch'] == selected['best_epoch'], 'Scoring checkpoint epoch differs')
                require(record['model_eval'] is True and record['weights_frozen'] is True and record['labels_used_for_training_or_selection'] is False,
                        'Scoring must remain frozen/eval-only')
                require(record['batch_size'] == 32 and record['parameter_dtype'] == 'float32' and record['autocast_dtype'] == 'bfloat16'
                        and record['tf32_enabled'] is False, 'Numerical regime differs')
                for role in ROLES:
                    ref = record['roles'][role]['predictions']
                    prediction_path = common.path_inside(root, ref['file'])
                    require(prediction_path.parent == directory, 'Prediction belongs to a different job')
                    bound(prediction_path, ref['sha256'])
                    actual, difference, _ = audit_prediction(prediction_path, role_arrays[role]['labels'], role_rows[role], record['roles'][role]['metrics'])
                    cells.append({'condition': condition, 'pt_seed': pt_seed, 'ft_seed': ft_seed, 'role': role, 'metrics': actual})
                    maximum_metric_difference = max(maximum_metric_difference, difference)
                    prediction_count += 1
                    prediction_rows += actual['rows']

    surface_selection_ref = selection['surface_selection']
    surface_selection_path = bound(common.path_inside(m2_root, surface_selection_ref['file']), surface_selection_ref['sha256'])
    surface_selection = common.read(surface_selection_path)
    surface_checkpoint_path = bound(common.path_inside(m2_root, surface_selection['checkpoint']), surface_selection['checkpoint_sha256'])
    surface_state = common.read(surface_checkpoint_path)
    require(surface_state['protocol_sha256'] == selection['protocol_sha256'] and surface_state['prepared_manifest_sha256'] == selection['prepared_manifest_sha256'],
            'Surface source protocol/data differs')
    for name in ('surface', 'constant0', 'constant1'):
        directory = root / 'results/references' / name
        path = directory / 'metrics.json'
        record = common.read(path)
        check_record(record, path)
        constant_label = None if name == 'surface' else int(name[-1])
        require(record['kind'] == ('surface' if constant_label is None else 'constant'), 'Reference kind differs')
        if constant_label is None:
            require(record['checkpoint'] == {'file': str(surface_checkpoint_path), 'sha256': surface_selection['checkpoint_sha256']}, 'Surface checkpoint differs')
        else:
            require(record['constant_label'] == constant_label, 'Reference fixed label differs')
        for role in ROLES:
            part = record['roles'][role]
            require(part['cross_entropy_status'] == ('finite' if constant_label is None else 'infinite'), 'Reference CE status differs')
            if constant_label is not None:
                require(part['cross_entropy_reason'] == CE_REASON, 'Constant CE reason missing/different')
            ref = part['predictions']
            prediction_path = common.path_inside(root, ref['file'])
            require(prediction_path.parent == directory, 'Reference predictions belong to another baseline')
            bound(prediction_path, ref['sha256'])
            replay = surface_logp(role_rows[role], surface_state) if constant_label is None else None
            actual, difference, replay_difference = audit_prediction(prediction_path, role_arrays[role]['labels'], role_rows[role],
                                                                    part['metrics'], constant_label=constant_label, replay_logp=replay)
            references.append({'reference': name, 'role': role, 'metrics': actual,
                               'cross_entropy_status': part['cross_entropy_status'],
                               'cross_entropy_reason': CE_REASON if constant_label is not None else None})
            maximum_metric_difference = max(maximum_metric_difference, difference)
            maximum_surface_difference = max(maximum_surface_difference, replay_difference)
            prediction_count += 1
            prediction_rows += actual['rows']
    require(prediction_count == 60 and prediction_rows == 30 * sum(common.COUNTS.values()), 'Incomplete full prediction matrix')
    nested = nested_summary(cells)
    contrasts = paired_contrasts(cells)
    primary = next(x for x in contrasts if x['contrast'] == 'EP-ES' and x['role'] == 'target' and x['metric'] == 'auroc')
    summary = {'status': 'completed', 'source_selection_sha256': source_ref['sha256'], 'protocol_sha256': protocol_sha,
               'prepared_manifest_sha256': manifest_sha, 'neural_cells': cells, 'reference_cells': references,
               'nested_metrics': nested, 'paired_contrasts': contrasts, 'primary_target_auc_contrast': primary,
               'primary_direction_consistency_flag': bool(primary['all_three_pt_differences_positive'] and primary['mean_difference'] > 0),
               'descriptive_only': True, 'confidence_intervals': False, 'hypothesis_tests': False,
               'conditions_filtered_by_score': False, 'target_calibration': False, 'target_training': False,
               'groups_preserved': True, 'role_counts': dict(common.COUNTS), 'exploratory_after_prior_target_use': True,
               'new_blind_confirmation_claimed': False, 'pretraining_seeds': 3, 'fine_tuning_seeds_per_pretraining_seed': 3,
               'neural_forward_repeated_in_full_cpu_audit': False,
               'qualification': 'Source competence and surface controls constrain any protein semantic-transfer interpretation.'}
    common.check_hashes(root, ledger)
    report_dir.mkdir(parents=True, exist_ok=False)
    common.atomic(report_dir / 'summary.json', summary)
    flat_cells = [{key: value for key, value in cell.items() if key != 'metrics'} | cell['metrics'] for cell in cells]
    write_csv(report_dir / 'per_job_metrics.csv', flat_cells)
    write_csv(report_dir / 'nested_metrics.csv', nested)
    write_csv(report_dir / 'paired_contrasts.csv', contrasts)
    write_csv(report_dir / 'reference_metrics.csv', [{key: value for key, value in cell.items() if key != 'metrics'} | cell['metrics'] for cell in references])
    (report_dir / 'REPORT.md').write_text(report_text(summary))
    report_hashes = {str(path.relative_to(root)): common.sha(path) for path in report_dir.iterdir() if path.is_file()}
    audit_record = {'status': 'passed', 'completed_at_utc': common.now(), 'source_selection_sha256': source_ref['sha256'],
                    'protocol_sha256': protocol_sha, 'prepared_manifest_sha256': manifest_sha,
                    'neural_classifiers': 27, 'neural_cells': 54, 'reference_cells': 6,
                    'prediction_artifacts_audited': prediction_count, 'prediction_rows_audited': prediction_rows,
                    'complete_checkpoint_hashes_verified': 27, 'surface_replay_cells': 2,
                    'maximum_metric_difference': maximum_metric_difference,
                    'maximum_surface_log_probability_difference': maximum_surface_difference,
                    'independent_metric_implementation': 'rank_sum_auc_and_confusion_counts',
                    'independent_surface_implementation': 'counter_based_ngrams',
                    'groups_and_row_order_exact': True, 'new_neural_inference': False,
                    'file_sha256': ledger, 'report_sha256': report_hashes, 'all_input_hashes_unchanged': True,
                    'role_counts': dict(common.COUNTS), 'new_training_performed': False,
                    'exploratory_after_prior_target_use': True, 'new_blind_confirmation_claimed': False}
    common.check_hashes(root, ledger)
    common.atomic(report_dir / 'audit.json', audit_record)
    print('New-data fixed-transfer independent audit passed: 27 classifiers / 60 predictions', flush=True)
    return audit_record


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        audit()
