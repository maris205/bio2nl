"""Offline CPU inference with independently pinned model catalog and artifacts.

No downloads, training, benchmark scoring or workspace-dependent imports occur.
The catalog digest must be obtained from a trusted release, not from a changed
local catalog. Checkpoints are the original complete weights-only PyTorch files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import numpy as np
import torch
from tokenizers import Tokenizer

import portable_core as core


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_path(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative and "\0" not in relative,
            "Invalid artifact path")
    item = PurePosixPath(relative)
    require(not item.is_absolute() and str(item) == relative and not any(x in ("", ".", "..") for x in item.parts),
            "Artifact path must be canonical and relative")
    root = Path(root).resolve()
    target = (root / relative).resolve()
    require(target.is_relative_to(root), "Artifact escapes artifact root")
    return target


def verify_file(path, descriptor):
    require(Path(path).is_file() and Path(path).stat().st_size == descriptor["bytes"], "Artifact size differs")
    require(file_sha(path) == descriptor["sha256"], "Artifact SHA-256 differs")


def read_catalog(path, expected_sha256):
    require(isinstance(expected_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha256),
            "Trusted catalog SHA-256 is required")
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Catalog SHA-256 differs")
    catalog = json.loads(raw)
    require(catalog.get("schema_version") == 1 and catalog.get("release_id") == "bio2nl-2026-10-02-models-v1",
            "Unknown model catalog")
    entries = catalog["models"]
    expected = {f"{c}-pt{p}" for c in ("EP", "ES", "EE") for p in range(3)}
    expected |= {f"{c}-pt{p}-ft{f}" for c in ("EP", "ES", "EE") for p in range(3) for f in range(3)}
    require(len(entries) == 36 and len({x["id"] for x in entries}) == 36 and {x["id"] for x in entries} == expected,
            "Catalog model grid differs")
    for entry in entries:
        require(entry["config_sha256"] == canonical_sha(catalog["config"]), "Config identity differs")
        require(entry["tokenizer_sha256"] == catalog["tokenizer"]["sha256"], "Tokenizer identity differs")
        job = entry["job"]
        classifier = entry["kind"] == "source_classifier"
        require(entry["kind"] in ("causal_lm", "source_classifier") and job.get("smoke") is False,
                "Unexpected checkpoint kind/job")
        suffix = f"-ft{job['ft_seed']}" if classifier else ""
        require(entry["id"] == f"{job['condition']}-pt{job['pt_seed']}" + suffix, "Model/job identity differs")
        if classifier:
            parents = [p for p in entries if p["id"] == entry["pretraining_parent"]["id"]]
            require(len(parents) == 1, "Missing classifier parent")
            parent = parents[0]
            require(parent["kind"] == "causal_lm" and parent["job"]["condition"] == job["condition"] and
                    parent["job"]["pt_seed"] == job["pt_seed"] and
                    entry["pretraining_parent"]["sha256"] == parent["checkpoint"]["sha256"] and
                    entry["pretraining_parent"]["state_sha256"] == parent["state_sha256"],
                    "Classifier pretraining-parent provenance differs")
            require(type(entry["selected_epoch"]) is int and 1 <= entry["selected_epoch"] <= 5,
                    "Invalid source-selected epoch")
    return catalog


def select(catalog, model_id):
    rows = [entry for entry in catalog["models"] if entry["id"] == model_id]
    require(len(rows) == 1, "Unknown model identifier")
    return rows[0]


def load_tokenizer(catalog, artifact_root):
    descriptor = catalog["tokenizer"]
    path = safe_path(artifact_root, descriptor["path"])
    verify_file(path, descriptor)
    tokenizer = Tokenizer.from_file(str(path))
    require(tokenizer.get_vocab_size() == 32000 and
            [tokenizer.token_to_id(s) for s in descriptor["special_tokens"]] == [0, 1, 2, 3],
            "Tokenizer vocabulary or special-token identities differ")
    require(tokenizer.padding is None and tokenizer.truncation is None, "Unexpected tokenizer padding/truncation")
    return tokenizer


def state_parts(state, kind):
    prefix = "transformer." if kind == "causal_lm" else "backbone."
    backbone = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    head_name = "lm_head.weight" if kind == "causal_lm" else "score.weight"
    require(backbone and head_name in state, "Incomplete backbone/head state")
    return core.state_digest(backbone), core.state_digest({head_name: state[head_name]})


def config_for_core(public_config):
    """Restore integer-keyed config maps without mutating the public catalog."""
    config = json.loads(json.dumps(public_config))
    for field in ("id2label", "pruned_heads"):
        config[field] = {int(key): value for key, value in config[field].items()}
    return config


def original_config_sha_after_load(config):
    """Transformers 4.45.2 from_dict adds only this runtime key in-place."""
    original = dict(config)
    require(original.pop("attn_implementation", None) is None, "Unexpected injected attention implementation")
    return canonical_sha(original)


def load_model(catalog, model_id, artifact_root, checkpoint_override=None):
    entry = select(catalog, model_id)
    path = Path(checkpoint_override) if checkpoint_override is not None else safe_path(artifact_root, entry["checkpoint"]["path"])
    require(path.stat().st_size == entry["checkpoint"]["bytes"], "Checkpoint size differs")
    model, payload = core.load_checkpoint(path,
        expected_sha256=entry["checkpoint"]["sha256"], expected_kind=entry["kind"], expected_job=entry["job"],
        expected_state_sha256=entry["state_sha256"], expected_protocol_sha256=catalog["training_protocol_sha256"],
        expected_manifest_sha256=catalog["prepared_manifest_sha256"], expected_epoch=entry.get("selected_epoch"),
        expected_config=config_for_core(catalog["config"]))
    require(original_config_sha_after_load(payload["config"]) == entry["config_sha256"], "Full checkpoint config differs")
    backbone_sha, head_sha = state_parts(payload["state_dict"], entry["kind"])
    require(backbone_sha == entry["backbone_state_sha256"] and head_sha == entry["head_state_sha256"],
            "Backbone/head identity differs")
    require(len(payload["state_dict"]) == entry["state_tensor_count"], "Tensor inventory differs")
    if entry["kind"] == "source_classifier":
        require(model.score.bias is None and tuple(model.score.weight.shape) == (2, 768), "Classifier head shape differs")
    del payload
    return model


def encode_pairs(tokenizer, pairs):
    require(isinstance(pairs, list) and pairs, "Supply at least one sentence pair")
    ids = np.zeros((len(pairs), 512), dtype=np.int64)
    masks = np.zeros_like(ids)
    details = []
    for index, pair in enumerate(pairs):
        require(isinstance(pair, dict) and set(pair) == {"sentence1", "sentence2"},
                "Each pair requires only sentence1 and sentence2; labels are not accepted")
        original = []
        for field in ("sentence1", "sentence2"):
            text = pair[field]
            require(isinstance(text, str) and bool(text), "Empty/nontext endpoint")
            content = tokenizer.encode(text, add_special_tokens=False).ids
            require(content and all(4 <= token < 32000 for token in content), "Unknown/special content token")
            require(tokenizer.decode(content, skip_special_tokens=False) == text, "Text roundtrip failed")
            original.append(content)
        joined = original[0][:255] + [2] + original[1][:255] + [1]
        ids[index, :len(joined)] = joined
        masks[index, :len(joined)] = 1
        details.append({"original_content_tokens": [len(x) for x in original],
                        "truncated_endpoints": [len(x) > 255 for x in original], "encoded_length": len(joined)})
    return {"input_ids": ids, "attention_mask": masks, "labels": np.zeros(len(pairs), dtype=np.int64)}, details


def predict_pairs(model, tokenizer, pairs, batch_size=1):
    arrays, details = encode_pairs(tokenizer, pairs)
    predicted = core.predict_source(model, arrays, batch_size=batch_size, device="cpu", precision="float32")
    values = []
    for logp, pred, detail in zip(predicted["log_probabilities"], predicted["predictions"], details):
        values.append({**detail, "log_probabilities": logp.tolist(), "probability_label_1": float(np.exp(logp[1])),
                       "logit_margin_label_1_minus_0": float(logp[1] - logp[0]), "predicted_label": int(pred)})
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify", "predict"))
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--catalog-sha256", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True, help="For example EP-pt0-ft0 or EP-pt0")
    parser.add_argument("--checkpoint", type=Path, help="Explicit relocated checkpoint; all hashes still apply")
    parser.add_argument("--pairs", type=Path, help="JSON array with sentence1/sentence2, without labels")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    require(args.cpu_threads > 0, "Invalid CPU thread count")
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision("highest")
    catalog = read_catalog(args.catalog, args.catalog_sha256)
    tokenizer = load_tokenizer(catalog, args.artifact_root)
    model = load_model(catalog, args.model_id, args.artifact_root, args.checkpoint)
    result = {"model_id": args.model_id, "checkpoint_sha256": select(catalog, args.model_id)["checkpoint"]["sha256"],
              "catalog_sha256": args.catalog_sha256, "device": "cpu", "precision": "float32", "verified": True}
    if args.command == "predict":
        require(select(catalog, args.model_id)["kind"] == "source_classifier", "Pair inference requires a complete classifier")
        require(args.pairs is not None, "--pairs is required for prediction")
        result.update(fixed_direction="label_1", decision_rule="argmax_ties_label_0",
                      predictions=predict_pairs(model, tokenizer, json.loads(args.pairs.read_text()), args.batch_size))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
