"""Fresh source-only full training on audited rebuilt inputs.

The immutable protocol permits bounded smoke. Full 9-pretraining/27-SFT fits
additionally require the independently accepted source-training gate. Training
never reads old weights, source-test pairs or target examples.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .runtime_data import CONDITION_STREAMS, ReleaseData, file_sha256
    from . import model as models
except ImportError:
    from runtime_data import CONDITION_STREAMS, ReleaseData, file_sha256
    import model as models

SCALER = dict(init_scale=1024.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".partial")
    with pending.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    pending.replace(path)


def seed_all(seed):
    require(type(seed) is int and seed >= 0, "Invalid RNG seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_job(condition, pt_seed, ft_seed=None, run_kind="smoke"):
    require(run_kind in ("smoke", "full"), "Unknown run kind")
    require(condition in CONDITION_STREAMS and type(pt_seed) is int and pt_seed in (0, 1, 2), "Unknown pretraining job")
    require(ft_seed is None or (type(ft_seed) is int and ft_seed in (0, 1, 2)), "Unknown fine-tuning seed")
    smoke = run_kind == "smoke"
    require(not smoke or (pt_seed == 0 and (ft_seed is None or condition == "EP" and ft_seed == 0)),
            "Only EP/ES/EE pt0 pretraining and EP pt0 ft0 SFT smoke are allowed")
    job = {"condition": condition, "pt_seed": pt_seed, "smoke": smoke}
    if ft_seed is not None:
        job["ft_seed"] = ft_seed
    return job


def output_directory(data, condition, pt_seed, ft_seed=None, run_kind="smoke"):
    resolve_job(condition, pt_seed, ft_seed, run_kind)
    branch = "pretrain" if ft_seed is None else "source"
    path = Path(data.execution_protocol["output_root"]).absolute() / run_kind / branch / condition / f"pt{pt_seed}"
    return path if ft_seed is None else path / f"ft{ft_seed}"


def sft_budget(training_rows, run_kind):
    require(type(training_rows) is int and training_rows > 0, "Invalid training count")
    require(run_kind in ("smoke", "full"), "Unknown run kind")
    epochs = 1 if run_kind == "smoke" else 5
    require(training_rows == (64 if run_kind == "smoke" else 8044), "Frozen source training count differs")
    batches = math.ceil(training_rows / 32)
    return {"epochs": epochs, "steps_per_epoch": batches, "updates": epochs * batches,
            "sample_presentations": epochs * training_rows}


def validate_numerical_design(design, source_counts):
    pt, config = design["pretraining"], design["source_sft"]
    architecture = {"model_type": "gpt2", "n_layer": 12, "n_head": 12, "n_embd": 768, "n_inner": 3072,
                    "activation_function": "gelu_new", "attn_pdrop": .1, "embd_pdrop": .1,
                    "resid_pdrop": .1, "layer_norm_epsilon": 1e-5, "initializer_range": .02,
                    "tie_word_embeddings": True}
    require(pt["architecture"] == architecture, "Frozen architecture differs")
    require(pt["grad_scaler"] == SCALER and
            (pt["context_length"], pt["batch_size"], pt["gradient_accumulation"], pt["updates_per_run"]) ==
            (512, 16, 2, 1024), "Pretraining numerical budget differs")
    opt = pt["optimizer"]
    require((opt["learning_rate"], opt["betas"], opt["eps"], opt["weight_decay"], opt["max_grad_norm"],
             opt["warmup_steps"]) == (3e-4, [.9, .999], 1e-8, .01, 1., 21), "Pretraining optimizer differs")
    require((config["epochs"], config["batch_size"], config["learning_rate"], config["betas"],
             config["eps"], config["weight_decay"], config["max_grad_norm"], config["head_initializer_std"],
             config["dropout"]) == (5, 32, 2e-5, [.9, .999], 1e-8, .01, 1., .02, .1),
            "SFT numerical regime differs")
    require(source_counts == {"train": 8044, "validation": 20276}, "New source membership counts differ")
    budget = sft_budget(source_counts["train"], "full")
    require((config["steps_per_epoch"], config["updates_per_fit"], config["presentations_per_fit"]) ==
            (budget["steps_per_epoch"], budget["updates"], budget["sample_presentations"]), "Resolved SFT budget differs")


def optimizer_groups(model, mode, weight_decay):
    require(mode in ("pretrain", "sft"), "Unknown optimizer grouping")
    excluded = set()
    if mode == "pretrain":
        for module_name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                excluded.update((module_name + "." if module_name else "") + name
                                for name, _ in module.named_parameters(recurse=False))
    names, params = {"decay": [], "no_decay": []}, {"decay": [], "no_decay": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        no_decay = ("bias" in name or name in excluded) if mode == "pretrain" else ("bias" in name or parameter.ndim < 2)
        group = "no_decay" if no_decay else "decay"
        names[group].append(name)
        params[group].append(parameter)
    require(sum(map(len, params.values())) == sum(p.requires_grad for p in model.parameters()), "Optimizer omitted parameters")
    return [{"params": params["decay"], "weight_decay": weight_decay},
            {"params": params["no_decay"], "weight_decay": 0.0}], names


def cosine_multiplier(step, total_steps=1024, warmup_steps=21):
    require(total_steps > 0 and 0 <= warmup_steps <= total_steps and step >= 0, "Invalid cosine schedule")
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def linear_multiplier(step, total_steps):
    require(total_steps > 0 and step >= 0, "Invalid linear schedule")
    return max(0.0, 1.0 - step / total_steps)


def gather_blocks(stream0, stream1, schedule_batch):
    require(schedule_batch.dtype == np.int64 and schedule_batch.ndim == 2 and schedule_batch.shape[1] == 2,
            "Invalid scheduled batch descriptor")
    require(stream0.ndim == stream1.ndim == 2 and stream0.shape[1] == stream1.shape[1], "Stream widths differ")
    require(np.isin(schedule_batch[:, 0], [0, 1]).all(), "Unknown stream selector")
    result = np.empty((len(schedule_batch), stream0.shape[1]), dtype=np.int64)
    for stream_id, stream in enumerate((stream0, stream1)):
        positions = np.flatnonzero(schedule_batch[:, 0] == stream_id)
        indices = schedule_batch[positions, 1]
        require(((indices >= 0) & (indices < len(stream))).all(), "Scheduled block out of bounds")
        result[positions] = stream[indices]
    return torch.from_numpy(result)


def source_batch(arrays, indices, device):
    device = torch.device(device)
    index = torch.as_tensor(indices, device=device, dtype=torch.long)
    ids, mask, labels = (arrays[key].index_select(0, index) for key in ("input_ids", "attention_mask", "labels"))
    width = min(ids.shape[1], ((int(mask.sum(1).max().item()) + 7) // 8) * 8)
    return ids[:, :width], mask[:, :width], labels


def checked_gradient_norm(model, max_norm):
    value = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, error_if_nonfinite=True)
    require(torch.isfinite(value).item(), "Nonfinite gradient norm")
    return float(value.item())


def scaled_optimizer_step(model, optimizer, scaler, max_norm):
    scaler.unscale_(optimizer)
    norm = checked_gradient_norm(model, max_norm)
    before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    after = float(scaler.get_scale())
    require(after >= before, "FP16 overflow/skipped optimizer update: stop queue")
    return norm, before, after


def require_checkpoint_space(directory, state):
    # Sum all tensors, including tied aliases, to conservatively cover serialization.
    estimated_bytes = sum(t.numel() * t.element_size() for t in state.values()) + (16 << 20)
    free_bytes = shutil.disk_usage(directory).free
    required_bytes = (8 << 30) + estimated_bytes
    require(free_bytes >= required_bytes, "Checkpoint storage would violate 8 GiB free reserve")
    return {"free_bytes_before_write": free_bytes, "estimated_checkpoint_bytes": estimated_bytes,
            "required_free_bytes": required_bytes}


def save_checkpoint(model, path, kind, metadata, *, replace=False):
    path = Path(path)
    require(replace or not path.exists(), "Refuse to replace existing checkpoint")
    state = models.snapshot_state(model)
    storage = require_checkpoint_space(path.parent, state)
    digest = models.state_digest(state)
    config = model.config if kind == "causal_lm" else model.backbone.config
    payload = {"schema_version": 1, "kind": kind, "config": config.to_dict(), "state_dict": state,
               "state_sha256": digest, "metadata": metadata}
    pending = path.with_name(path.name + ".partial")
    with pending.open("xb") as handle:
        torch.save(payload, handle)
    pending.replace(path)
    return {"file": path.name, "sha256": file_sha256(path), "state_sha256": digest,
            "bytes": path.stat().st_size, "kind": kind, "storage_check": storage}


def metrics(labels, logp):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
    labels, logp = np.asarray(labels), np.asarray(logp)
    require(labels.ndim == 1 and logp.shape == (len(labels), 2) and np.isfinite(logp).all(), "Invalid predictions")
    require(np.isin(labels, [0, 1]).all() and np.allclose(np.exp(logp).sum(1), 1, rtol=0, atol=2e-6), "Invalid prediction probabilities")
    pred = logp.argmax(1)
    return {"rows": len(labels), "accuracy": float((labels == pred).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "mcc": float(matthews_corrcoef(labels, pred)),
            "auroc": float(roc_auc_score(labels, logp[:, 1] - logp[:, 0])) if len(set(labels)) == 2 else None,
            "cross_entropy": float(-logp[np.arange(len(labels)), labels].mean()),
            "predicted_positive_fraction": float(pred.mean()),
            "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1]).tolist()}


def new_output_directory(data, output):
    directory = Path(output).absolute()
    for part in [*reversed(directory.parents), directory]:
        require(not part.is_symlink(), "Symlinked output paths are forbidden")
    directory = directory.resolve()
    require(not directory.is_relative_to(data.root) and not directory.is_relative_to(data.raw_root),
            "Outputs cannot be placed inside immutable data")
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def create_trainable_classifier(lm, ft_seed):
    model = models.create_classifier(lm, ft_seed).requires_grad_(True)
    require(all(p.requires_grad for p in model.parameters()), "SFT backbone and head must both be trainable")
    return model


def _entry(data, output, job, run_kind, kind):
    context = data.source_context()
    require(data.mode == run_kind and (run_kind != "full" or data.full_training_accepted),
            "Full training requires its separate acceptance and matching runtime access mode")
    validate_numerical_design(context["protocol"], data.source_counts)
    require(Path(output).absolute() == output_directory(data, job["condition"], job["pt_seed"], job.get("ft_seed"), run_kind),
            "Output directory differs from frozen job identity")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "Local training requires CUDA_VISIBLE_DEVICES=0")
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), "Explicit CUDA FP16/BF16 runtime required")
    directory = new_output_directory(data, output)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    meta = {"schema_version": 1, "status": "running", "kind": kind, "job": job, "run_kind": run_kind,
            "started_at": now(), "protocol_sha256": context["protocol_sha256"],
            "prepared_manifest_sha256": context["prepared_manifest_sha256"], "raw_release_id": data.manifest["raw_release_id"],
            "raw_manifest_sha256": data.manifest["raw_manifest_sha256"],
            "command": list(sys.argv), "source_only": True, "target_examples_read": 0, "source_test_examples_read": 0,
            "tf32_enabled": False, "cpu_threads": 4, "full_training_reproduction_demonstrated": False,
            "final_metrics_are_smoke_only": run_kind == "smoke", "portable_training_module_sha256": file_sha256(Path(__file__)),
            "full_budget_training_accepted": bool(data.full_training_accepted),
            "training_acceptance_file": data.execution_protocol["training_acceptance_file"] if run_kind == "full" else None}
    atomic_json(directory / "metadata.json", meta)
    return context["protocol"], directory, meta


def _finish(data, directory, meta, start):
    meta.update(status="completed", completed_at=now(), elapsed_seconds=time.monotonic() - start,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                inputs=data.verify_current_inputs())
    atomic_json(directory / "metadata.json", meta)
    print(json.dumps({"status": meta["status"], "job": meta["job"], "updates": meta["updates"],
                      "elapsed_seconds": meta["elapsed_seconds"]}), flush=True)
    return meta


def _failure(directory, meta, exc):
    meta.update(status="failed", failed_at=now(), error=repr(exc))
    atomic_json(directory / "metadata.json", meta)


def pretrain(data, condition, pt_seed, output, run_kind="smoke"):
    job = resolve_job(condition, pt_seed, run_kind=run_kind)
    protocol, directory, meta = _entry(data, output, job, run_kind, "pretraining")
    start = time.monotonic()
    try:
        loaded = data.load_pretrain(condition, pt_seed)
        pt, opt = protocol["pretraining"], protocol["pretraining"]["optimizer"]
        require(pt["grad_scaler"] == SCALER and (pt["context_length"], pt["batch_size"], pt["gradient_accumulation"], pt["updates_per_run"]) == (512, 16, 2, 1024), "Pretraining regime differs")
        seed_all(pt_seed)
        model = models.create_pretraining_model(protocol, pt_seed)
        meta["initial_state_sha256"] = models.state_digest(model.state_dict())
        meta["fresh_random_initialization"] = True
        if condition != "EP":
            previous = output_directory(data, "EP", pt_seed, run_kind=run_kind) / "metadata.json"
            prior_digest = file_sha256(previous)
            data._pin(previous, {"sha256": prior_digest, "bytes": previous.stat().st_size})
            prior = json.loads(previous.read_text())
            require(prior["status"] == "completed" and prior["job"] == {"condition": "EP", "pt_seed": pt_seed, "smoke": run_kind == "smoke"}
                    and prior["protocol_sha256"] == meta["protocol_sha256"]
                    and prior["prepared_manifest_sha256"] == meta["prepared_manifest_sha256"]
                    and prior["initial_state_sha256"] == meta["initial_state_sha256"], "Paired fresh initialization differs")
            require(file_sha256(previous) == prior_digest, "Paired initialization metadata changed")
            meta["paired_EP_initialization"] = {"file": str(previous), "sha256": prior_digest}
        meta["old_weights_loaded"] = False
        meta["parameter_count"] = sum(p.numel() for p in model.parameters())
        model.to("cuda:0")
        groups, names = optimizer_groups(model, "pretrain", opt["weight_decay"])
        optimizer = torch.optim.AdamW(groups, lr=opt["learning_rate"], betas=tuple(opt["betas"]), eps=opt["eps"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: cosine_multiplier(step, pt["updates_per_run"], opt["warmup_steps"]))
        scaler = torch.cuda.amp.GradScaler(**SCALER)
        steps = 2 if run_kind == "smoke" else 1024
        history = []
        for step in range(steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            used_lr = float(optimizer.param_groups[0]["lr"])
            mean_ce = 0.0
            for accumulation in range(2):
                ids = gather_blocks(*loaded["streams"], loaded["schedule"][2 * step + accumulation]).to("cuda:0")
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(input_ids=ids, labels=ids, use_cache=False, return_dict=True).loss
                require(torch.isfinite(loss).item(), "Nonfinite pretraining loss")
                mean_ce += float(loss.detach().item()) / 2
                scaler.scale(loss / 2).backward()
            norm, before, after = scaled_optimizer_step(model, optimizer, scaler, opt["max_grad_norm"])
            scheduler.step()
            history.append({"update": step + 1, "learning_rate_used": used_lr, "mean_microbatch_ce": mean_ce,
                            "unscaled_gradient_norm": norm, "loss_scale_before": before, "loss_scale_after": after})
            if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
                meta.update(updates=step + 1, history=history, elapsed_seconds=time.monotonic() - start)
                atomic_json(directory / "metadata.json", meta)
                print(json.dumps({"status": "running", "job": job, "updates": step + 1,
                                  "total_updates": steps, "last_online_training_ce": mean_ce,
                                  "elapsed_seconds": meta["elapsed_seconds"]}), flush=True)
        checkpoint = save_checkpoint(model, directory / "model.pt", "causal_lm",
                                     {k: meta[k] for k in ("job", "protocol_sha256", "prepared_manifest_sha256")})
        require(checkpoint["state_sha256"] != meta["initial_state_sha256"], "Pretraining did not change parameters")
        meta.update(checkpoint=checkpoint, final_state_sha256=checkpoint["state_sha256"], updates=steps,
                    input_tokens=steps * 32 * 512, causal_loss_positions=steps * 32 * 511,
                    history=history, loss_history=history, online_training_ce=float(np.mean([h["mean_microbatch_ce"] for h in history])),
                    skipped_optimizer_updates=0, loss_scaler=SCALER, scheduler_total_steps=1024,
                    optimizer_parameter_groups=names, precision="float32_parameters_float16_autocast")
        ids = gather_blocks(*loaded["streams"], loaded["schedule"][0][[0, 8]]).to("cuda:0")
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
        nll = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), ids[:, 1:], reduction="none")
        require(torch.isfinite(nll).all().item(), "Nonfinite pretraining probe")
        path = directory / "probe.npz"
        with path.open("xb") as handle:
            np.savez_compressed(handle, input_ids=ids.cpu().numpy(), token_negative_log_likelihood=nll.cpu().numpy().astype(np.float64))
        meta["probe"] = {"file": "probe.npz", "sha256": file_sha256(path), "mean_nll": float(nll.double().mean().item()),
                         "rows": 2, "prediction_positions": 1022, "autocast_dtype": "float16", "logit_loss_dtype": "float32",
                         "loss_expression": "cross_entropy(logits[:, :-1].float().transpose(1, 2), ids[:, 1:], reduction=none)"}
        del model, optimizer, scheduler, scaler, groups
        gc.collect()
        torch.cuda.empty_cache()
        return _finish(data, directory, meta, start)
    except Exception as exc:
        _failure(directory, meta, exc)
        raise


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--job", choices=("pretrain", "sft"), required=True)
    parser.add_argument("--condition", choices=tuple(CONDITION_STREAMS), required=True)
    parser.add_argument("--pt-seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--ft-seed", type=int, choices=(0, 1, 2))
    parser.add_argument("--run-kind", choices=("smoke", "full"), required=True)
    args = parser.parse_args(argv)
    require((args.job == "sft") == (args.ft_seed is not None), "SFT requires ft-seed; pretraining forbids it")
    resolve_job(args.condition, args.pt_seed, args.ft_seed, args.run_kind)
    data = ReleaseData(args.protocol, args.protocol_sha256, mode=args.run_kind)
    output = output_directory(data, args.condition, args.pt_seed, args.ft_seed, args.run_kind)
    if args.job == "pretrain":
        result = pretrain(data, args.condition, args.pt_seed, output, args.run_kind)
    else:
        parent = output_directory(data, args.condition, args.pt_seed, run_kind=args.run_kind)
        result = sft(data, args.condition, args.pt_seed, args.ft_seed, output, args.run_kind, parent_run=parent)
    print(json.dumps({"job": args.job, "status": result["status"], "protocol_sha256": args.protocol_sha256}, sort_keys=True), flush=True)
    return 0


def _score_and_save(model, arrays, rows, path, directory):
    predicted = models.predict_source(model, arrays, batch_size=32, device="cuda:0", precision="bfloat16")
    row_ids = [row["row_id"] for row in rows]
    require(len(row_ids) == len(predicted["labels"]), "Prediction row count differs")
    with path.open("xb") as handle:
        np.savez_compressed(handle, **predicted, row_ids=np.asarray(row_ids, dtype=str))
    return {"metrics": metrics(predicted["labels"], predicted["log_probabilities"]),
            "predictions": {"file": str(path.relative_to(directory)), "sha256": file_sha256(path), "row_ids_sha256": canonical_hash(row_ids)}}


def _parent_checkpoint(data, parent_run, job, protocol, meta):
    require(parent_run is not None, "SFT requires an explicit new pretraining parent_run")
    parent_dir = Path(parent_run).absolute()
    for part in [*reversed(parent_dir.parents), parent_dir]:
        require(not part.is_symlink(), "Symlinked parent paths are forbidden")
    parent_dir = parent_dir.resolve(strict=True)
    require(parent_dir == output_directory(data, job["condition"], job["pt_seed"], run_kind=meta["run_kind"]),
            "SFT parent must be this protocol's exact new condition/seed output")
    metadata_path = parent_dir / "metadata.json"
    digest = file_sha256(metadata_path)
    data._pin(metadata_path, {"sha256": digest, "bytes": metadata_path.stat().st_size})
    with metadata_path.open(encoding="utf-8") as handle:
        parent = json.load(handle)
    expected_job = {k: v for k, v in job.items() if k != "ft_seed"}
    require(parent["status"] == "completed" and parent["kind"] == "pretraining" and parent["job"] == expected_job,
            "Wrong or incomplete new pretraining parent")
    require(parent["run_kind"] == meta["run_kind"] and parent["raw_manifest_sha256"] == data.manifest["raw_manifest_sha256"],
            "New parent run/data differs")
    require(parent["protocol_sha256"] == meta["protocol_sha256"] and parent["prepared_manifest_sha256"] == meta["prepared_manifest_sha256"], "Parent provenance differs")
    checkpoint = parent["checkpoint"]
    require(checkpoint["file"] == "model.pt" and checkpoint["kind"] == "causal_lm", "Parent checkpoint path/kind differs")
    path = parent_dir / "model.pt"
    require(file_sha256(path) == checkpoint["sha256"], "Parent checkpoint bytes differ")
    data._pin(path, {"sha256": checkpoint["sha256"], "bytes": checkpoint["bytes"]})
    config = models.make_config(protocol).to_dict()
    # Optional serialization fields can vary; bind every scientific architecture field.
    config_keys = ("vocab_size", "n_positions", "n_ctx", "n_embd", "n_layer", "n_head", "n_inner", "activation_function",
                   "attn_pdrop", "embd_pdrop", "resid_pdrop", "layer_norm_epsilon", "initializer_range", "tie_word_embeddings",
                   "bos_token_id", "eos_token_id", "pad_token_id", "model_type")
    lm, payload = models.load_checkpoint(path, expected_sha256=checkpoint["sha256"], expected_kind="causal_lm",
        expected_job=expected_job, expected_state_sha256=checkpoint["state_sha256"], expected_protocol_sha256=meta["protocol_sha256"],
        expected_manifest_sha256=meta["prepared_manifest_sha256"], expected_config={k: config[k] for k in config_keys})
    meta["pretraining_checkpoint"] = {**checkpoint, "file": str(path)}
    meta["new_parent_metadata"] = {"file": str(metadata_path), "sha256": digest}
    require(file_sha256(metadata_path) == digest, "Parent metadata changed while loading")
    return lm, payload


def sft(data, condition, pt_seed, ft_seed, output, run_kind="smoke", parent_run=None):
    job = resolve_job(condition, pt_seed, ft_seed, run_kind)
    require(parent_run is not None, "SFT requires an explicit new pretraining parent_run")
    protocol, directory, meta = _entry(data, output, job, run_kind, "source_sft")
    start = time.monotonic()
    try:
        config = protocol["source_sft"]
        require((config["epochs"], config["batch_size"], config["learning_rate"]) == (5, 32, 2e-5), "SFT budget differs")
        arrays, rows = {}, {}
        for split in ("train", "validation"):
            arrays[split], rows[split] = data.load_source(split, smoke=run_kind == "smoke")
        lm, payload = _parent_checkpoint(data, parent_run, job, protocol, meta)
        parent_backbone = models.state_digest({"backbone." + k: v for k, v in lm.transformer.state_dict().items()})
        model = create_trainable_classifier(lm, ft_seed)
        del lm, payload
        initial = models.snapshot_state(model)
        meta["initial_state_sha256"] = models.state_digest(initial)
        meta["initial_head_sha256"] = models.state_digest({"score.weight": initial["score.weight"]})
        meta["initial_backbone_sha256"] = models.state_digest({k: v for k, v in initial.items() if k.startswith("backbone.")})
        require(meta["initial_backbone_sha256"] == parent_backbone, "SFT changed the new parent backbone before training")
        meta["fresh_head_initialization"] = True
        meta["old_weights_loaded"] = False
        meta["new_parent_backbone_exactly_inherited"] = True
        del initial
        seed_all(ft_seed)
        model.to("cuda:0")
        gpu_arrays = {role: {k: torch.tensor(v, device="cuda:0") for k, v in values.items()} for role, values in arrays.items()}
        groups, names = optimizer_groups(model, "sft", config["weight_decay"])
        optimizer = torch.optim.AdamW(groups, lr=config["learning_rate"], betas=tuple(config["betas"]), eps=config["eps"], foreach=True)
        budget = sft_budget(len(rows["train"]), run_kind)
        total_steps = budget["updates"]
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: linear_multiplier(step, total_steps))
        generator = torch.Generator(device="cpu").manual_seed(ft_seed)
        meta.update(optimizer_parameter_groups=names, precision="float32_parameters_bfloat16_autocast",
                    source_counts={k: len(v) for k, v in rows.items()}, scheduler_total_steps=total_steps,
                    planned_budget=budget,
                    source_row_ids_sha256={k: canonical_hash([r["row_id"] for r in v]) for k, v in rows.items()})
        history, update_history = [], []
        step = seen_total = 0
        best_ce = math.inf
        for epoch in range(1, budget["epochs"] + 1):
            order = torch.randperm(len(rows["train"]), generator=generator).numpy()
            online_sum, seen = 0.0, 0
            for begin in range(0, len(order), 32):
                indices = order[begin:begin + 32]
                ids, mask, labels = source_batch(gpu_arrays["train"], indices, "cuda:0")
                model.train()
                optimizer.zero_grad(set_to_none=True)
                used_lr = float(optimizer.param_groups[0]["lr"])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(ids, mask)
                loss = F.cross_entropy(logits.float(), labels)
                require(torch.isfinite(loss).item(), "Nonfinite SFT loss")
                loss.backward()
                norm = checked_gradient_norm(model, config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                step += 1
                update_history.append({"update": step, "learning_rate_used": used_lr, "unscaled_gradient_norm": norm,
                                       "online_dropout_training_ce": float(loss.detach().item()),
                                       "epoch": epoch, "batch_examples": len(indices)})
                online_sum += float(loss.detach().item()) * len(indices)
                seen += len(indices)
                seen_total += len(indices)
                if step == 1 or step % 50 == 0 or step == total_steps:
                    meta.update(updates=step, sample_presentations=seen_total, update_history=update_history,
                                elapsed_seconds=time.monotonic() - start)
                    atomic_json(directory / "metadata.json", meta)
                    print(json.dumps({"status": "running", "job": job, "epoch": epoch, "updates": step,
                                      "total_updates": total_steps, "sample_presentations": seen_total,
                                      "elapsed_seconds": meta["elapsed_seconds"]}), flush=True)
            require(seen == len(rows["train"]), "SFT epoch omitted training examples")
            epoch_dir = directory / "epochs" / f"epoch{epoch:02d}"
            epoch_dir.mkdir(parents=True, exist_ok=False)
            scored = {split: _score_and_save(model, arrays[split], rows[split], epoch_dir / f"{split}.predictions.npz", directory)
                      for split in ("train", "validation")}
            state = models.snapshot_state(model)
            state_digest = models.state_digest(state)
            item = {"epoch": epoch, "updates": step, "sample_presentations": seen_total, "epoch_presentations": seen,
                    "online_dropout_training_ce": online_sum / seen, **scored, "state_sha256": state_digest,
                    "head_sha256": models.state_digest({"score.weight": state["score.weight"]}),
                    "backbone_sha256": models.state_digest({k: v for k, v in state.items() if k.startswith("backbone.")}),
                    "last_unscaled_gradient_norm": norm,
                    "selection_metric": "single_fixed_smoke_checkpoint_no_epoch_search" if run_kind == "smoke" else
                                        "source_validation_cross_entropy"}
            del state
            history.append(item)
            ce = scored["validation"]["metrics"]["cross_entropy"]
            if ce < best_ce:
                best_ce = ce
                cp_metadata = {k: meta[k] for k in ("job", "protocol_sha256", "prepared_manifest_sha256")}
                checkpoint = save_checkpoint(model, directory / "best.pt", "source_classifier", {**cp_metadata, "epoch": epoch}, replace=True)
                require(checkpoint["state_sha256"] == state_digest, "Saved classifier differs from scoring state")
                meta.update(best_epoch=epoch, checkpoint=checkpoint,
                            scoring={role: scored[role]["predictions"] for role in ("train", "validation")})
            atomic_json(epoch_dir / "metrics.json", item)
            meta.update(history=history, update_history=update_history, updates=step, sample_presentations=seen_total,
                        best_validation_cross_entropy=best_ce, last_scoring={role: scored[role]["predictions"] for role in ("train", "validation")})
            atomic_json(directory / "metadata.json", meta)
        require(step == total_steps and seen_total == budget["sample_presentations"], "SFT update/presentation budget differs")
        require(history[-1]["head_sha256"] != meta["initial_head_sha256"] and history[-1]["backbone_sha256"] != meta["initial_backbone_sha256"],
                "SFT head/backbone did not both update")
        meta.update(final_state_sha256=history[-1]["state_sha256"], only_source_validation_selected_checkpoint=run_kind == "full",
                    technical_smoke_single_epoch=run_kind == "smoke", scientific_model_selection_performed=run_kind == "full",
                    final_epoch_checkpoint_retained=meta["best_epoch"] == history[-1]["epoch"], backbone_and_head_updated=True)
        del model, optimizer, scheduler, groups, arrays, gpu_arrays
        gc.collect()
        torch.cuda.empty_cache()
        return _finish(data, directory, meta, start)
    except Exception as exc:
        _failure(directory, meta, exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
