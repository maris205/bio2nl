"""Independent complete-classifier reload and pooling replay for fixed transfer.

Only the prespecified EP/pt0/ft0 source-selected classifier is replayed on each
role's first64 rows. No training, selection, calibration or direction change.
"""
from __future__ import annotations

import gc
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
ATOL = 1e-5
CONFIG_KEYS = ('model_type', 'vocab_size', 'n_positions', 'n_ctx', 'n_embd', 'n_layer', 'n_head', 'n_inner',
               'activation_function', 'resid_pdrop', 'embd_pdrop', 'attn_pdrop', 'layer_norm_epsilon',
               'initializer_range', 'tie_word_embeddings', 'bos_token_id', 'eos_token_id', 'pad_token_id')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def exact_selected(selection):
    entries = selection['selected']
    expected = {(c, p, f) for c in ('EP', 'ES', 'EE') for p in range(3) for f in range(3)}
    identities = [(e['job']['condition'], e['job']['pt_seed'], e['job']['ft_seed']) for e in entries]
    require(selection['status'] == 'frozen' and selection['conditions_filtered_by_score'] is False and
            len(entries) == len(set(identities)) == 27 and set(identities) == expected and
            all(e['job']['smoke'] is False for e in entries), 'Source classifier grid is incomplete or filtered')
    return entries[identities.index(('EP', 0, 0))]


def verify_replay(logp, saved, labels, rows):
    expected_keys = {'labels', 'predictions', 'log_probabilities', 'row_ids', 'group_ids'}
    require(set(saved) == expected_keys, 'Smoke prediction columns differ')
    n = len(rows)
    require(n == 64 and labels.dtype == np.int64 and labels.shape == (n,), 'Smoke must use first64 rows')
    require(saved['labels'].dtype == saved['predictions'].dtype == np.int64 and
            saved['log_probabilities'].dtype == np.float64 and
            saved['row_ids'].dtype.kind == saved['group_ids'].dtype.kind == 'U', 'Smoke prediction dtypes differ')
    require(saved['labels'].shape == saved['predictions'].shape == saved['row_ids'].shape == saved['group_ids'].shape == (n,),
            'Smoke prediction vector shape differs')
    require(np.array_equal(saved['labels'], labels) and
            saved['row_ids'].tolist() == [r['row_id'] for r in rows] and
            saved['group_ids'].tolist() == [r['group_id'] for r in rows], 'Smoke labels/row/component order differ')
    original = saved['log_probabilities']
    require(logp.dtype == np.float64 and logp.shape == original.shape == (n, 2) and
            np.isfinite(logp).all() and np.isfinite(original).all(), 'Nonfinite or invalid smoke scores')
    for values in (logp, original):
        require(np.max(np.abs(np.logaddexp(values[:, 0], values[:, 1]))) <= 2e-6, 'Unnormalized smoke scores')
    require(np.array_equal(original.argmax(1), saved['predictions']), 'Original predictions violate fixed argmax')
    maximum = float(np.max(np.abs(logp - original)))
    require(maximum <= ATOL, 'Independent scoring replay exceeded fixed1e-5 tolerance')
    require(np.array_equal(logp.argmax(1), saved['predictions']), 'Independent scoring decisions differ')
    return maximum


@torch.inference_mode()
def independent_logp(model, arrays, device):
    model.eval()
    parts = []
    require(len(arrays['labels']) == 64, 'Independent forward is restricted to64 smoke rows')
    for first in range(0, 64, 32):
        mask_np = arrays['attention_mask'][first:first + 32]
        width = min(mask_np.shape[1], (int(mask_np.sum(1).max()) + 7) // 8 * 8)
        mask = torch.tensor(mask_np[:, :width], dtype=torch.long, device=device)
        ids = torch.tensor(arrays['input_ids'][first:first + 32, :width], dtype=torch.long, device=device)
        require(torch.all(mask.sum(1) > 0).item() and torch.all(mask[:, 1:] <= mask[:, :-1]).item(),
                'Independent pooling requires nonempty right-padded inputs')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            hidden = model.backbone(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
            # Separate from TransferClassifier's masked-position maximum implementation.
            last = mask.sum(1) - 1
            pooled = hidden[torch.arange(len(ids), device=device), last]
            logits = F.linear(pooled, model.score.weight)
        parts.append(F.log_softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64))
    return np.concatenate(parts)


def validate_input_closure(record, gate_inputs):
    require(record['all_input_hashes_unchanged'] is True, 'Scoring did not close input hashes')
    values = record['file_sha256']
    require(isinstance(values, dict) and bool(values), 'Scoring input ledger absent')
    require(all(values.get(path) == digest for path, digest in gate_inputs.items()),
            'Scoring input ledger omits or changes a frozen gate binding')


def audit(root=ROOT):
    root = Path(root).resolve()
    common = import_file('_new_transfer_smoke_common', root / 'common.py')
    protocol = common.check_gate(root, mode='smoke')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and torch.cuda.is_available() and
            torch.cuda.is_bf16_supported(), 'Explicit GPU0 BF16 runtime required')
    output = root / 'verification/smoke_audit.json'
    require(not output.exists(), 'Preserve existing independent smoke audit')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    files = dict(common.gate_file_hashes(root, mode='smoke'))
    gate_inputs = dict(files)

    def bound(path, digest=None):
        path = Path(path).absolute()
        actual = common.sha(path)
        require(digest is None or actual == digest, 'Smoke artifact hash differs: ' + str(path))
        require(files.setdefault(str(path), actual) == actual, 'Conflicting smoke artifact identity')
        return path

    recordpath = root / 'results/smoke/EP__pt0__ft0/metrics.json'
    record = common.read(bound(recordpath))
    require(record['status'] == 'completed' and record['job'] == dict(condition='EP', pt_seed=0, ft_seed=0, smoke=True),
            'Smoke scoring job differs')
    require(record['protocol_sha256'] == common.sha(root / 'protocol.json') and
            record['prepared_manifest_sha256'] == common.sha(root / 'prepared/manifest.json') and
            record['source_selection_sha256'] == protocol['source_selection']['sha256'], 'Smoke scoring provenance differs')
    require(record['model_eval'] is record['weights_frozen'] is True and
            record['labels_used_for_training_or_selection'] is False and record['tf32_enabled'] is False and
            record['batch_size'] == 32 and record['parameter_dtype'] == 'float32' and
            record['autocast_dtype'] == 'bfloat16', 'Smoke numerical/parameter regime differs')
    validate_input_closure(record, gate_inputs)
    for name, digest in record['file_sha256'].items():
        bound(name, digest)
    selection_path = bound(protocol['source_selection']['file'], protocol['source_selection']['sha256'])
    selection = common.read(selection_path)
    selected = exact_selected(selection)
    cp = bound(common.path_inside(selection_path.parent, selected['checkpoint']['file']), selected['checkpoint']['sha256'])
    require(cp.stat().st_size == selected['checkpoint']['bytes'], 'Selected checkpoint byte count differs')
    require(record['checkpoint'] == {'file': str(cp), 'sha256': selected['checkpoint']['sha256'],
            'state_sha256': selected['checkpoint']['state_sha256']} and record['source_best_epoch'] == selected['best_epoch'],
            'Smoke used another complete classifier/epoch')
    loader_path = bound(protocol['model_loader']['file'], protocol['model_loader']['sha256'])
    source_protocol = common.read(bound(protocol['source_protocol']['file'], protocol['source_protocol']['sha256']))
    require(protocol['source_protocol']['sha256'] == selection['protocol_sha256'], 'Source configuration protocol differs')
    loader = import_file('_new_transfer_replay_model', loader_path)
    expected_all = loader.make_config(source_protocol['design']).to_dict()
    config = {k: expected_all[k] for k in CONFIG_KEYS}
    model, payload = loader.load_checkpoint(cp, expected_sha256=selected['checkpoint']['sha256'],
        expected_kind='source_classifier', expected_job=selected['job'],
        expected_state_sha256=selected['checkpoint']['state_sha256'], expected_protocol_sha256=selection['protocol_sha256'],
        expected_manifest_sha256=selection['prepared_manifest_sha256'], expected_epoch=selected['best_epoch'], expected_config=config)
    require(model.score.bias is None and tuple(model.score.weight.shape) == (2, config['n_embd']), 'Complete classifier head differs')
    before = loader.state_digest(model.state_dict())
    require(before == payload['state_sha256'] == selected['checkpoint']['state_sha256'] ==
            record['state_before_sha256'] == record['state_after_sha256'], 'Complete source state differs')
    del payload
    model.requires_grad_(False).to('cuda:0').eval()
    require(all(p.dtype == torch.float32 and not p.requires_grad for p in model.parameters()), 'FP32 frozen weights required')
    roles = {}
    require(set(record['roles']) == {'source_test', 'target'}, 'Smoke scored roles differ')
    for role in ('source_test', 'target'):
        full_arrays, rows = common.load_role(root, role)
        arrays = {k: v[:64] for k, v in full_arrays.items()}
        rows = rows[:64]
        logp = independent_logp(model, arrays, torch.device('cuda:0'))
        reference = record['roles'][role]['predictions']
        path = common.path_inside(root, reference['file'])
        require(path == recordpath.parent / f'{role}.predictions.npz', 'Smoke prediction path differs')
        bound(path, reference['sha256'])
        with np.load(path, allow_pickle=False) as got:
            saved = {k: got[k] for k in got.files}
        maximum = verify_replay(logp, saved, arrays['labels'], rows)
        computed = common.metrics(arrays['labels'], logp)
        recorded = record['roles'][role]['metrics']
        require(set(computed) == set(recorded), 'Smoke metric columns differ')
        for key, value in computed.items():
            original = recorded[key]
            require(original is None if value is None else np.allclose(value, original, atol=ATOL, rtol=0),
                    'Smoke replay metric differs: ' + key)
        roles[role] = {'rows': 64, 'maximum_log_probability_difference': maximum, 'all_decisions_equal': True,
                       'row_ids_sha256': common.canonical([r['row_id'] for r in rows]),
                       'group_ids_sha256': common.canonical([r['group_id'] for r in rows])}
        del arrays, full_arrays, saved
    after = loader.state_digest(model.state_dict())
    require(before == after and common.sha(cp) == selected['checkpoint']['sha256'], 'Replay changed source weights')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    common.check_hashes(root, files)
    report = {'status': 'passed', 'completed_at_utc': common.now(), 'protocol_sha256': common.sha(root / 'protocol.json'),
              'execution_freeze_sha256': common.sha(root / 'execution_freeze.json'),
              'prepared_manifest_sha256': common.sha(root / 'prepared/manifest.json'),
              'source_selection_sha256': protocol['source_selection']['sha256'], 'roles': roles, 'files': files,
              'checkpoint': record['checkpoint'], 'state_before_sha256': before, 'state_after_sha256': after,
              'complete_head_retained': True, 'training_or_selection_performed': False,
              'optimizer_updates_performed': 0, 'fixed_absolute_tolerance': ATOL,
              'all_input_hashes_unchanged': True, 'target_calibration': False, 'polarity_flipping': False}
    common.atomic(output, report)
    print('Independent new-data fixed-transfer smoke passed', roles, flush=True)
    return report


if __name__ == '__main__':
    audit()
