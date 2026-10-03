"""Project original verified artifacts into a public path-independent catalog.

Maintainer tool only: no network access and no weight copying. The upload map
uses workspace-relative source paths and public repository-relative destinations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import torch

from inference import canonical_sha, file_sha, state_parts, require
import portable_core as core


TRAINING = Path("bio2nl/review/new_data_full_training_v1_2026-10-02")
ASSETS = Path("release_staging/revision_2026-10-03_v1/source/metadata/models_assets.json")
TOKENIZER = Path("data_rebuild/2026-10-01-portable-v2/tokenizers/mixed_bpe/tokenizer.json")


def write(path, value):
    require(not path.exists(), "Refuse to overwrite generated artifact")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run(workspace, output):
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    output.mkdir(exist_ok=True, parents=True)
    require(not (output / "model_catalog.json").exists(), "Catalog already exists")
    assets = json.loads((workspace / ASSETS).read_text())
    selection_path = workspace / TRAINING / "results/source_selection.json"
    selection = json.loads(selection_path.read_text())
    require(file_sha(selection_path) == assets["source_selection_sha256"], "Source selection digest differs")
    require(selection["status"] == "frozen" and selection["conditions_filtered_by_score"] is False,
            "Source selection was not frozen")
    protocol_path = workspace / TRAINING / "protocol.json"
    require(file_sha(protocol_path) == selection["protocol_sha256"], "Training protocol digest differs")
    protocol = json.loads(protocol_path.read_text())
    config = core.make_config(protocol["design"]).to_dict()
    torch.set_num_threads(4)
    selected = {(x["job"]["condition"], x["job"]["pt_seed"], x["job"]["ft_seed"]): x for x in selection["selected"]}
    require(len(selected) == 27 and len(assets["files"]) == 36, "Incomplete model grid")
    tokenizer_file = workspace / TOKENIZER
    tokenizer_sha = file_sha(tokenizer_file)
    require(tokenizer_sha == protocol["design"]["tokenizer"]["sha256"], "Shared tokenizer differs")
    tokenizer_json = json.loads(tokenizer_file.read_text())
    tokens = [x["content"] for x in sorted(tokenizer_json["added_tokens"], key=lambda x: x["id"])]
    require(len(tokens) == 4, "Unexpected tokenizer special tokens")
    catalog = {"schema_version": 1, "release_id": "bio2nl-2026-10-02-models-v1", "model_repository": "dnagpt/bio2nl-models",
        "weight_license_status": "unassigned; no license grant is inferred from code or training-data licenses",
        "training_protocol_sha256": selection["protocol_sha256"], "prepared_manifest_sha256": selection["prepared_manifest_sha256"],
        "source_selection_sha256": file_sha(selection_path), "source_assets_manifest_sha256": file_sha(workspace / ASSETS),
        "frozen_model_implementation_sha256": file_sha(workspace / TRAINING / "code_snapshot/model.py"),
        "config": config, "config_canonical_sha256": canonical_sha(config),
        "conditions": {"EP": "English + natural protein", "ES": "English + residue-shuffled protein", "EE": "English + additional English"},
        "tokenizer": {"path": "tokenizer.json", "sha256": tokenizer_sha, "bytes": tokenizer_file.stat().st_size,
                      "vocab_size": 32000, "special_tokens": tokens,
                      "training_recipe": "Shared byte-level BPE trained on 16 MiB English and 16 MiB protein training text; historical v2 tokenizer reused"},
        "inference": {"endpoint_cap": 255, "context_length": 512, "format": "a[:255] + [SEP=2] + b[:255] + [EOS=1]",
                      "padding": "right, PAD=0", "pooling": "last attention-mask-valid token (EOS)",
                      "score_direction": "label 1", "decision": "argmax; ties label 0", "calibration": "none",
                      "reference_scoring_precision": "FP32 weights with CUDA BF16 autocast",
                      "portable_example_precision": "CPU FP32; exact CUDA BF16 predictions are not promised"},
        "models": []}
    mapping = []
    audit = []
    by_id = {}
    for asset in assets["files"]:
        path = workspace / asset["path"]
        require(path.stat().st_size == asset["bytes"] and file_sha(path) == asset["sha256"], "Original weight digest differs")
        classifier = "ft_seed" in asset
        condition, pt = asset["condition"], asset["pt_seed"]
        model_id = f"{condition}-pt{pt}" + (f"-ft{asset['ft_seed']}" if classifier else "")
        remote = f"checkpoints/source/{condition}/pt{pt}/ft{asset['ft_seed']}/best.pt" if classifier else f"checkpoints/pretrain/{condition}/pt{pt}/model.pt"
        meta_path = path.parent / "metadata.json"
        meta = json.loads(meta_path.read_text())
        require(meta["status"] == "completed" and meta["run_kind"] == "full", "Incomplete original run")
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        kind = "source_classifier" if classifier else "causal_lm"
        job = {"condition": condition, "pt_seed": pt, "smoke": False}
        if classifier:
            job["ft_seed"] = asset["ft_seed"]
        require(payload["schema_version"] == 1 and payload["kind"] == kind and payload["metadata"]["job"] == job,
                "Original checkpoint identity differs")
        require(payload["metadata"]["protocol_sha256"] == selection["protocol_sha256"] and
                payload["metadata"]["prepared_manifest_sha256"] == selection["prepared_manifest_sha256"], "Original provenance differs")
        require(canonical_sha(payload["config"]) == canonical_sha(config), "Original complete config differs")
        state = payload["state_dict"]
        require(all(x.dtype == torch.float32 and torch.isfinite(x).all() for x in state.values()), "Invalid weight tensor")
        digest = core.state_digest(state)
        require(digest == payload["state_sha256"] == meta["checkpoint"]["state_sha256"], "Original state digest differs")
        backbone, head = state_parts(state, kind)
        entry = {"id": model_id, "kind": kind, "job": job,
            "checkpoint": {"path": remote, "sha256": asset["sha256"], "bytes": asset["bytes"]},
            "state_sha256": digest, "state_tensor_count": len(state), "backbone_state_sha256": backbone,
            "head_state_sha256": head, "config_sha256": canonical_sha(config), "tokenizer_sha256": tokenizer_sha,
            "training_result_sha256": file_sha(meta_path),
            "head_description": "two-logit bias-free linear score.weight (2 x 768)" if classifier else "tied 32000 x 768 language-model output embedding"}
        if classifier:
            chosen = selected[(condition, pt, asset["ft_seed"])]
            require(chosen["checkpoint"]["sha256"] == asset["sha256"] and chosen["checkpoint"]["state_sha256"] == digest and
                    chosen["source_result"]["sha256"] == file_sha(meta_path), "Frozen selection differs")
            require(payload["metadata"]["epoch"] == chosen["best_epoch"] == asset["best_epoch"] == meta["best_epoch"], "Selected epoch differs")
            parent = by_id[f"{condition}-pt{pt}"]
            require(meta["pretraining_checkpoint"]["sha256"] == parent["checkpoint"]["sha256"] and
                    meta["pretraining_checkpoint"]["state_sha256"] == parent["state_sha256"] and
                    meta["new_parent_backbone_exactly_inherited"] is True, "Parent link differs")
            require(tuple(state["score.weight"].shape) == (2, 768), "Original classifier head differs")
            entry.update(selected_epoch=chosen["best_epoch"], selection="minimum source validation CE; ties earlier epoch",
                pretraining_parent={"id": parent["id"], "sha256": parent["checkpoint"]["sha256"], "state_sha256": parent["state_sha256"]})
        else:
            require(torch.equal(state["lm_head.weight"], state["transformer.wte.weight"]), "LM tied aliases differ")
            entry.update(pretraining_steps=meta["updates"], input_tokens=meta["input_tokens"])
        catalog["models"].append(entry)
        by_id[model_id] = entry
        mapping.append({"source_relative": asset["path"], "path_in_repo": remote, "sha256": asset["sha256"], "bytes": asset["bytes"]})
        audit.append({"id": model_id, "file_verified": True, "state_verified": True, "config_verified": True,
                      "finite_fp32": True, "source_selection_verified": classifier, "tensor_count": len(state)})
        del payload, state
        print("verified", model_id, flush=True)
    shutil.copyfile(tokenizer_file, output / "tokenizer.json")
    write(output / "config.json", config)
    write(output / "model_catalog.json", catalog)
    write(output / "upload_mapping.json", {"schema_version": 1, "model_repository": "dnagpt/bio2nl-models", "repo_type": "model",
        "scope": "Existing original weights; no second weight copy; package support files added separately", "weights": mapping})
    write(output / "catalog_build_audit.json", {"status": "passed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": audit, "weight_files": len(mapping), "weight_bytes": sum(x["bytes"] for x in mapping),
        "catalog_sha256": file_sha(output / "model_catalog.json"), "weight_files_copied": 0,
        "trained_or_scored_benchmarks": False, "gpu_used": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.workspace, args.output)
