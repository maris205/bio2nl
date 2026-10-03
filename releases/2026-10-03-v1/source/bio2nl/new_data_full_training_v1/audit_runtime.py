"""Independent model reload with the original FP32 channel-CE loss layout.

The model forward, input reconstruction, complete-state and provenance checks
remain independent. The LM loss kernel/layout matches the training contract
after a separately pinned diagnosis isolated last-dimension reduction rounding.
The first failed queue, predictions and 1e-5 threshold remain unchanged.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

try:
    from .runtime_data import ReleaseData, canonical_hash, file_sha256, regular_file, relative_path, require
    from . import model as models
except ImportError:
    from runtime_data import ReleaseData, canonical_hash, file_sha256, regular_file, relative_path, require
    import model as models

ATOL = 1e-5
CONFIG_KEYS = ("model_type", "vocab_size", "n_positions", "n_ctx", "n_embd", "n_layer", "n_head", "n_inner",
               "activation_function", "resid_pdrop", "embd_pdrop", "attn_pdrop", "layer_norm_epsilon",
               "initializer_range", "tie_word_embeddings", "bos_token_id", "eos_token_id", "pad_token_id")


def now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    pending = path.with_name(path.name + ".partial")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    pending.replace(path)


def difference(actual, expected):
    require(actual.shape == expected.shape and actual.size and np.isfinite(actual).all() and np.isfinite(expected).all(),
            "Invalid numerical replay comparison")
    maximum = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    require(maximum <= ATOL, f"Fixed replay tolerance exceeded: {maximum} > {ATOL}")
    return maximum


def validate_predictions(values, labels, row_ids):
    require(set(values) == {"labels", "predictions", "log_probabilities", "row_ids"}, "Prediction fields differ")
    n = len(labels)
    require(values["labels"].dtype == values["predictions"].dtype == np.int64 and
            values["log_probabilities"].dtype == np.float64 and values["row_ids"].dtype.kind == "U", "Prediction dtypes differ")
    require(values["labels"].shape == values["predictions"].shape == values["row_ids"].shape == (n,) and
            values["log_probabilities"].shape == (n, 2), "Prediction shapes differ")
    require(np.array_equal(values["labels"], labels) and np.array_equal(values["row_ids"], row_ids), "Prediction row/label identity differs")
    logp = values["log_probabilities"]
    require(np.isfinite(logp).all() and np.allclose(np.exp(logp).sum(1), 1, rtol=0, atol=2e-6), "Nonfinite or unnormalized probabilities")
    require(np.array_equal(values["predictions"], logp.argmax(1)), "Fixed argmax prediction differs")


@torch.inference_mode()
def independent_lm_nll(model, ids, device):
    model.eval()
    tokens = torch.tensor(ids, dtype=torch.long, device=device)
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(input_ids=tokens, use_cache=False, return_dict=True).logits
    values = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), tokens[:, 1:], reduction="none")
    return values.cpu().numpy().astype(np.float64)


@torch.inference_mode()
def independent_classifier_logp(model, arrays, device):
    model.eval()
    parts = []
    for start in range(0, len(arrays["labels"]), 32):
        masks = arrays["attention_mask"][start:start + 32]
        width = min(512, ((int(masks.sum(1).max()) + 7) // 8) * 8)
        ids = torch.tensor(arrays["input_ids"][start:start + 32, :width], device=device)
        mask = torch.tensor(masks[:, :width], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = model.backbone(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
            indices = mask.sum(1) - 1  # Independent right-padding implementation.
            selected = hidden[torch.arange(len(ids), device=device), indices]
            logits = F.linear(selected, model.score.weight)
        parts.append(F.log_softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64))
    return np.concatenate(parts)


def recompute_metrics(labels, logp):
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, matthews_corrcoef, roc_auc_score
    decisions = logp.argmax(1)
    return {"rows": len(labels), "accuracy": float(np.mean(labels == decisions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, decisions)),
            "mcc": float(matthews_corrcoef(labels, decisions)),
            "auroc": float(roc_auc_score(labels, logp[:, 1] - logp[:, 0])) if len(set(labels)) == 2 else None,
            "cross_entropy": float(-logp[np.arange(len(labels)), labels].mean()),
            "predicted_positive_fraction": float(np.mean(decisions == 1)),
            "confusion_matrix": confusion_matrix(labels, decisions, labels=[0, 1]).tolist()}


def compare_metrics(actual, expected):
    require(set(actual) == set(expected), "Metric inventory differs")
    for name, value in actual.items():
        if isinstance(value, float):
            require(np.isfinite(expected[name]) and abs(value - expected[name]) <= 1e-12, "Saved metric differs: " + name)
        else:
            require(value == expected[name], "Saved metric differs: " + name)


class Artifacts:
    def __init__(self):
        self.pins = {}

    def read(self, path, expected_sha=None):
        path = regular_file(path)
        digest = file_sha256(path)
        require(expected_sha is None or digest == expected_sha, "Artifact SHA-256 differs")
        pin = {"sha256": digest, "bytes": path.stat().st_size}
        require(self.pins.setdefault(str(path), pin) == pin, "Artifact changed during audit")
        return path

    def member(self, root, descriptor):
        path = Path(root) / relative_path(descriptor["file"])
        return self.read(path, descriptor["sha256"])

    def npz(self, root, descriptor):
        with np.load(self.member(root, descriptor), allow_pickle=False) as loaded:
            return {name: loaded[name] for name in loaded.files}

    def final_check(self):
        for path, pin in tuple(self.pins.items()):
            self.read(path, pin["sha256"])
        return dict(self.pins)


def validate_job_metadata(data, metadata, job, kind):
    require(metadata["status"] == "completed" and metadata["job"] == job and metadata["kind"] == kind and
            metadata["run_kind"] == "smoke" and metadata["updates"] == 2, "Wrong or incomplete smoke job")
    require(metadata["protocol_sha256"] == data.protocol_sha256 and
            metadata["prepared_manifest_sha256"] == data.manifest_sha256 and
            metadata["raw_manifest_sha256"] == data.manifest["raw_manifest_sha256"], "Smoke provenance differs")
    require(metadata["source_only"] is True and metadata["target_examples_read"] == 0 and
            metadata["source_test_examples_read"] == 0 and metadata["old_weights_loaded"] is False,
            "Smoke input boundary differs")


def load_new(data, artifacts, directory, meta, kind, config, epoch=None):
    cp = meta["checkpoint"]
    require(cp["kind"] == kind and cp["file"] == ("model.pt" if kind == "causal_lm" else "best.pt"), "Checkpoint descriptor differs")
    path = artifacts.member(directory, cp)
    require(path.stat().st_size == cp["bytes"], "Checkpoint bytes differ")
    return models.load_checkpoint(path, expected_sha256=cp["sha256"], expected_kind=kind, expected_job=meta["job"],
                                  expected_state_sha256=cp["state_sha256"], expected_protocol_sha256=data.protocol_sha256,
                                  expected_manifest_sha256=data.manifest_sha256, expected_epoch=epoch, expected_config=config)


def audit_runtime(data, run_root, output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0" and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            "Explicit local GPU0 FP16/BF16 audit required")
    run_root, output = Path(run_root).absolute(), Path(output).absolute()
    require(run_root == Path(data.execution_protocol["output_root"]).absolute() / "smoke" and
            output == run_root / "audit_runtime",
            "Audit must read this smoke and write its dedicated audit directory")
    for path in [*reversed(output.parents), output]:
        require(not path.is_symlink(), "Symlinked audit output forbidden")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    start = time.monotonic()
    report = {"schema_version": 1, "status": "running", "started_at_utc": now(),
              "scope": "new_full_training_smoke_reload", "protocol_sha256": data.protocol_sha256,
              "loss_kernel_independent": False,
              "model_reload_and_forward_independent": True, "loss_expression": "FP32_channel_cross_entropy_original_layout",
              "training_enabled": False, "full_budget_training_enabled": False,
              "prepared_manifest_sha256": data.manifest_sha256, "fixed_absolute_tolerance": ATOL,
              "optimizer_updates_performed": 0, "old_weights_loaded": False, "target_examples_read": 0,
              "source_test_examples_read": 0, "full_training_enabled": False, "target_scoring_enabled": False}
    save_json(output / "metadata.json", report)
    artifacts, initial_hashes, pretraining = Artifacts(), [], []
    ep_backbone = None
    try:
        config_all = models.make_config(data.design).to_dict()
        config = {key: config_all[key] for key in CONFIG_KEYS}
        for condition in ("EP", "ES", "EE"):
            folder = run_root / "pretrain" / condition / "pt0"
            meta_path = artifacts.read(folder / "metadata.json")
            meta = json.loads(meta_path.read_text())
            job = {"condition": condition, "pt_seed": 0, "smoke": True}
            validate_job_metadata(data, meta, job, "pretraining")
            require(meta["fresh_random_initialization"] is True and meta["input_tokens"] == 32768 and
                    meta["causal_loss_positions"] == 32704 and meta["skipped_optimizer_updates"] == 0 and
                    meta["scheduler_total_steps"] == 1024, "PT smoke budget differs")
            require(len(meta["history"]) == 2 and [v["update"] for v in meta["history"]] == [1, 2], "PT update ledger differs")
            require([v["learning_rate_used"] for v in meta["history"]] == [0.0, 3e-4 / 21], "PT frozen LR schedule differs")
            require(all(np.isfinite(v[key]) for v in meta["history"] for key in
                        ("mean_microbatch_ce", "unscaled_gradient_norm", "loss_scale_before", "loss_scale_after")), "Nonfinite PT update ledger")
            require(meta["checkpoint"]["state_sha256"] == meta["final_state_sha256"] != meta["initial_state_sha256"], "PT state failed to update")
            initial_hashes.append(meta["initial_state_sha256"])
            loaded = data.load_pretrain(condition, 0)
            expected_ids = np.stack([loaded["streams"][int(s)][int(i)] for s, i in loaded["schedule"][0][[0, 8]]]).astype(np.int64)
            saved = artifacts.npz(folder, meta["probe"])
            require(set(saved) == {"input_ids", "token_negative_log_likelihood"} and
                    saved["input_ids"].dtype == np.int64 and np.array_equal(saved["input_ids"], expected_ids) and
                    saved["token_negative_log_likelihood"].dtype == np.float64 and
                    saved["token_negative_log_likelihood"].shape == (2, 511), "PT scheduled probe differs")
            model, payload = load_new(data, artifacts, folder, meta, "causal_lm", config)
            before = models.state_digest(model.state_dict())
            if condition == "EP":
                ep_backbone = models.state_digest({"backbone." + k: v for k, v in model.transformer.state_dict().items()})
            model.to(device)
            replay = independent_lm_nll(model, expected_ids, device)
            maximum = difference(replay, saved["token_negative_log_likelihood"])
            after = models.state_digest(model.state_dict())
            require(before == after == meta["checkpoint"]["state_sha256"], "LM state changed during independent inference")
            with (output / f"{condition}.probe.npz").open("xb") as stream:
                np.savez_compressed(stream, input_ids=expected_ids, token_negative_log_likelihood=replay)
            pretraining.append({"condition": condition, "checkpoint": meta["checkpoint"], "probe_rows": 2,
                                "max_absolute_nll_difference": maximum, "state_before_sha256": before, "state_after_sha256": after})
            del model, payload, loaded
            gc.collect()
            torch.cuda.empty_cache()
        require(len(set(initial_hashes)) == 1, "Three paired random initializations differ")
        folder = run_root / "source/EP/pt0/ft0"
        metadata = json.loads(artifacts.read(folder / "metadata.json").read_text())
        validate_job_metadata(data, metadata, {"condition": "EP", "pt_seed": 0, "ft_seed": 0, "smoke": True}, "source_sft")
        require(metadata["sample_presentations"] == 64 and metadata["source_counts"] == {"train": 64, "validation": 64} and
                metadata["best_epoch"] == 1 and metadata["scheduler_total_steps"] == 2 and
                len(metadata["history"]) == 1 and len(metadata["update_history"]) == 2 and
                [v["learning_rate_used"] for v in metadata["update_history"]] == [2e-5, 1e-5], "SFT smoke budget/schedule differs")
        require(metadata["initial_backbone_sha256"] == ep_backbone and metadata["new_parent_backbone_exactly_inherited"] is True and
                metadata["fresh_head_initialization"] is True, "SFT did not inherit exactly the new EP backbone")
        require(metadata["scientific_model_selection_performed"] is False and metadata["technical_smoke_single_epoch"] is True,
                "Smoke must not claim full source model selection")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            head = torch.nn.Linear(config["n_embd"], 2, bias=False, dtype=torch.float32)
            torch.nn.init.normal_(head.weight, std=0.02)
            require(models.state_digest({"score.weight": head.weight}) == metadata["initial_head_sha256"], "Fresh SFT head seed differs")
        del head
        model, payload = load_new(data, artifacts, folder, metadata, "source_classifier", config, epoch=1)
        before = models.state_digest(model.state_dict())
        head_hash = models.state_digest({"score.weight": model.score.weight})
        backbone_hash = models.state_digest({k: v for k, v in model.state_dict().items() if k.startswith("backbone.")})
        require(head_hash != metadata["initial_head_sha256"] and backbone_hash != metadata["initial_backbone_sha256"] and
                metadata["history"][0]["head_sha256"] == head_hash and
                metadata["history"][0]["backbone_sha256"] == backbone_hash, "SFT complete head/backbone update differs")
        model.to(device)
        cells = {}
        for split in ("train", "validation"):
            arrays, rows = data.load_source(split, smoke=True)
            row_ids = np.asarray([r["row_id"] for r in rows], dtype=str)
            require(metadata["source_row_ids_sha256"][split] == canonical_hash(row_ids.tolist()), "SFT row selection differs")
            saved = artifacts.npz(folder, metadata["scoring"][split])
            validate_predictions(saved, arrays["labels"], row_ids)
            compare_metrics(recompute_metrics(saved["labels"], saved["log_probabilities"]), metadata["history"][0][split]["metrics"])
            logp = independent_classifier_logp(model, arrays, device)
            replay = {"labels": arrays["labels"], "row_ids": row_ids, "log_probabilities": logp, "predictions": logp.argmax(1)}
            validate_predictions(replay, arrays["labels"], row_ids)
            maximum = difference(logp, saved["log_probabilities"])
            require(np.array_equal(replay["predictions"], saved["predictions"]), "Independent SFT decisions differ")
            with (output / f"sft.{split}.predictions.npz").open("xb") as stream:
                np.savez_compressed(stream, **replay)
            cells[split] = {"rows": 64, "row_ids_sha256": canonical_hash(row_ids.tolist()),
                            "max_absolute_log_probability_difference": maximum, "decision_differences": 0}
        after = models.state_digest(model.state_dict())
        require(before == after == metadata["checkpoint"]["state_sha256"], "Classifier state changed during independent inference")
        del model, payload
        gc.collect()
        torch.cuda.empty_cache()
        report.update(status="completed", all_new_checkpoint_replays_passed=True, checkpoints_replayed=4,
                      lm_loss_contract="FP32_cross_entropy_channel_layout", fixed_threshold_unchanged=True,
                      paired_initial_state_sha256=initial_hashes[0], paired_initializations_equal=True,
                      pretraining=pretraining, source_sft={"checkpoint": metadata["checkpoint"], "roles": cells,
                      "state_before_sha256": before, "state_after_sha256": after, "complete_head_retained": True},
                      artifact_inputs=artifacts.final_check(), data_and_code_inputs=data.verify_current_inputs(),
                      pretraining_probe_rows=6, source_prediction_rows=128)
        report["outputs"] = {p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
                             for p in sorted(output.glob("*.npz"))}
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report.update(completed_at_utc=now(), wall_seconds=time.monotonic() - start)
        save_json(output / "metadata.json", report)
    return report


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args(argv)
    data = ReleaseData(args.protocol, args.protocol_sha256, mode="smoke")
    root = data.output_root / "smoke"
    audit_runtime(data, root, root / "audit_runtime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
