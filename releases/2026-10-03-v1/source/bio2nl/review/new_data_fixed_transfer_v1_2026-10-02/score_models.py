"""Score one new-data source-selected complete classifier with fixed weights."""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
CONDITIONS = ('EP', 'ES', 'EE')
ROLES = ('source_test', 'target')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exact_selected(selection, condition, pt_seed, ft_seed):
    """Reject missing, duplicated, extra, smoke or score-filtered selections."""
    require(selection['status'] == 'frozen', 'Source selection is not frozen')
    require(selection['conditions_filtered_by_score'] is False, 'Source arms were filtered')
    expected = {(c, p, f) for c in CONDITIONS for p in range(3) for f in range(3)}
    entries = selection['selected']
    identities = [(e['job']['condition'], e['job']['pt_seed'], e['job']['ft_seed']) for e in entries]
    require(len(entries) == 27 and len(set(identities)) == 27 and set(identities) == expected,
            'Source selection does not contain the complete 27-classifier grid')
    require(all(e['job']['smoke'] is False for e in entries), 'Smoke weights cannot be held-out classifiers')
    key = (condition, pt_seed, ft_seed)
    require(key in expected, 'Unknown scoring job')
    return entries[identities.index(key)]


CONFIG_KEYS = ("model_type", "vocab_size", "n_positions", "n_ctx", "n_embd", "n_layer", "n_head", "n_inner",
               "activation_function", "resid_pdrop", "embd_pdrop", "attn_pdrop", "layer_norm_epsilon",
               "initializer_range", "tie_word_embeddings", "bos_token_id", "eos_token_id", "pad_token_id")


def load_selected_model(common, protocol, condition, pt_seed, ft_seed):
    """CPU reload only, with trusted source selection, full provenance and config."""
    expected_paths = {"source_selection": common.M2_RESULTS / "source_selection.json",
                      "source_protocol": common.M2 / "protocol.json",
                      "model_loader": common.M2 / "code_snapshot/model.py"}
    pins = {}
    for key, expected in expected_paths.items():
        descriptor = protocol[key]
        path = Path(descriptor["file"])
        require(path.is_absolute() and path == expected, "Wrong current source reference: " + key)
        require(common.sha(path) == descriptor["sha256"], "Pinned source reference changed: " + key)
        pins[str(path)] = descriptor["sha256"]
    selection = common.read(expected_paths["source_selection"])
    chosen = exact_selected(selection, condition, pt_seed, ft_seed)
    source_protocol = common.read(expected_paths["source_protocol"])
    require(selection["protocol_sha256"] == protocol["source_protocol"]["sha256"], "Source numerical protocol differs")
    checkpoint_path = common.path_inside(common.M2_RESULTS, chosen["checkpoint"]["file"])
    result_path = common.path_inside(common.M2_RESULTS, chosen["source_result"]["file"])
    expected_job_path = Path("full/source") / condition / f"pt{pt_seed}" / f"ft{ft_seed}"
    require(chosen["checkpoint"]["file"] == str(expected_job_path / "best.pt") and
            chosen["source_result"]["file"] == str(expected_job_path / "metadata.json"), "Selected source job path differs")
    for path, descriptor in ((checkpoint_path, chosen["checkpoint"]), (result_path, chosen["source_result"])):
        require(common.sha(path) == descriptor["sha256"] and path.stat().st_size == descriptor["bytes"],
                "Selected classifier or source metadata changed")
        pins[str(path)] = descriptor["sha256"]
    source_result = common.read(result_path)
    require(source_result["status"] == "completed" and source_result["kind"] == "source_sft" and
            source_result["run_kind"] == "full" and source_result["job"] == chosen["job"] and
            source_result["best_epoch"] == chosen["best_epoch"] and
            source_result["protocol_sha256"] == selection["protocol_sha256"] and
            source_result["prepared_manifest_sha256"] == selection["prepared_manifest_sha256"],
            "Selected source result provenance differs")
    require(source_result["checkpoint"] == {**chosen["checkpoint"], "file": "best.pt"},
            "Source selected complete checkpoint descriptor differs")
    loader = import_file("_new_data_fixed_transfer_loader", expected_paths["model_loader"])
    config = loader.make_config(source_protocol["design"]).to_dict()
    model, payload = loader.load_checkpoint(checkpoint_path,
        expected_sha256=chosen["checkpoint"]["sha256"], expected_kind="source_classifier",
        expected_job=chosen["job"], expected_state_sha256=chosen["checkpoint"]["state_sha256"],
        expected_protocol_sha256=selection["protocol_sha256"],
        expected_manifest_sha256=selection["prepared_manifest_sha256"], expected_epoch=chosen["best_epoch"],
        expected_config={key: config[key] for key in CONFIG_KEYS})
    require(isinstance(model, loader.TransferClassifier) and model.score.bias is None and
            tuple(model.score.weight.shape) == (2, config["n_embd"]), "Complete source head differs")
    require(payload["state_sha256"] == loader.state_digest(model.state_dict()) == chosen["checkpoint"]["state_sha256"],
            "Complete source classifier tensor state differs")
    require(all(p.dtype == torch.float32 and not p.requires_grad for p in model.parameters()),
            "Reloaded source weights must be frozen FP32")
    del payload
    common.check_hashes(common.M2, pins)
    return model, loader, chosen, checkpoint_path, pins


def subset(arrays, rows, smoke):
    limit = 64 if smoke else len(rows)
    require(len(rows) >= limit and len(arrays['labels']) == len(rows), 'Invalid scoring row count')
    return {name: value[:limit] for name, value in arrays.items()}, rows[:limit]


def run(condition, pt_seed, ft_seed, smoke=False, root=ROOT):
    root = Path(root).resolve()
    common = import_file('_new_data_scoring_common', root / 'common.py')
    mode = 'smoke' if smoke else 'full'
    protocol = common.check_gate(root, mode=mode)
    input_hashes = common.gate_file_hashes(root, mode=mode)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'Scoring requires the local GPU 0')
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'Required BF16 CUDA regime unavailable')
    require(not smoke or (condition, pt_seed, ft_seed) == ('EP', 0, 0), 'Only EP pt0 ft0 scoring smoke is frozen')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    model, loader, chosen, checkpoint_path, source_pins = load_selected_model(common, protocol, condition, pt_seed, ft_seed)
    for path, digest in source_pins.items():
        require(input_hashes.setdefault(path, digest) == digest, "Conflicting scoring input identity")
    source_ref = protocol["source_selection"]
    before = loader.state_digest(model.state_dict())
    model.requires_grad_(False)
    model.to('cuda:0').eval()
    require(all(p.dtype == torch.float32 and not p.requires_grad for p in model.parameters()), 'FP32 frozen weights required')
    manifest_path = root / 'prepared/manifest.json'
    manifest = common.read(manifest_path)
    job = {'condition': condition, 'pt_seed': pt_seed, 'ft_seed': ft_seed, 'smoke': bool(smoke)}
    directory = root / 'results' / ('smoke' if smoke else 'neural') / f'{condition}__pt{pt_seed}__ft{ft_seed}'
    directory.mkdir(parents=True, exist_ok=False)
    roles = {}
    for role in ROLES:
        arrays, rows = common.load_role(root, role, manifest=manifest)
        arrays, rows = subset(arrays, rows, smoke)
        predicted = loader.predict_source(model, arrays, batch_size=32, device='cuda:0', precision='bfloat16')
        labels, logp = predicted['labels'], predicted['log_probabilities']
        require(np.array_equal(labels, arrays['labels']), 'Scoring reordered labels')
        require(logp.dtype == np.float64 and np.isfinite(logp).all(), 'Invalid classifier log probabilities')
        path = directory / (role + '.predictions.npz')
        common.save_predictions(path, labels, logp, [r['row_id'] for r in rows], [r['group_id'] for r in rows])
        roles[role] = {'metrics': common.metrics(labels, logp),
                       'predictions': {'file': str(path.relative_to(root)), 'sha256': common.sha(path)}}
        print(f"{condition} pt{pt_seed} ft{ft_seed}: {role} scored {len(rows)} rows", flush=True)
    after = loader.state_digest(model.state_dict())
    require(before == after and common.sha(checkpoint_path) == chosen['checkpoint']['sha256'], 'Scoring changed selected weights')
    common.check_hashes(root, input_hashes)
    result = {'status': 'completed', 'job': job, 'checkpoint': {'file': str(checkpoint_path),
              'sha256': chosen['checkpoint']['sha256'], 'state_sha256': before},
              'source_selection_sha256': source_ref['sha256'], 'protocol_sha256': common.sha(root / 'protocol.json'),
              'prepared_manifest_sha256': common.sha(manifest_path), 'roles': roles,
              'state_before_sha256': before, 'state_after_sha256': after,
              'source_best_epoch': chosen['best_epoch'], 'model_eval': True, 'weights_frozen': True,
              'batch_size': 32, 'parameter_dtype': 'float32', 'autocast_dtype': 'bfloat16', 'tf32_enabled': False,
              'labels_used_for_training_or_selection': False, 'file_sha256': input_hashes,
              'all_input_hashes_unchanged': True, 'elapsed_seconds': time.monotonic() - started}
    common.atomic(directory / 'metrics.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--pt-seed', type=int, choices=range(3), required=True)
    parser.add_argument('--ft-seed', type=int, choices=range(3), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.condition, args.pt_seed, args.ft_seed, args.smoke)
