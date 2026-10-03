"""Maintainer CPU replay: three examples, no fitting or benchmark evaluation."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import gc
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

import inference as api


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(workspace, package, output):
    workspace, package, output = Path(workspace).resolve(), Path(package).resolve(), Path(output).resolve()
    api.require(not output.exists(), "Refusing to replace CPU replay evidence")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    catalog_path = package / "model_catalog.json"
    catalog_sha = api.file_sha(catalog_path)
    catalog = api.read_catalog(catalog_path, catalog_sha)
    tokenizer = api.load_tokenizer(catalog, package)
    train_root = workspace / "bio2nl/review/new_data_full_training_v1_2026-10-02"
    reference_path = train_root / "code_snapshot/model.py"
    api.require(api.file_sha(reference_path) == catalog["frozen_model_implementation_sha256"], "Reference code changed")
    reference = import_file("original_frozen_model", reference_path)
    source_path = workspace / "bio2nl/review/new_data_training_smoke_v1_2026-10-01/prepared/source/train.jsonl"
    with source_path.open() as stream:
        row = json.loads(next(stream))
    pairs = [{key: row[key] for key in ("sentence1", "sentence2")},
             {"sentence1": "A small bird sits beside the window.", "sentence2": "A bird is sitting next to a window."},
             {"sentence1": "The protein sequence is an experimental input.", "sentence2": "The train leaves at noon."}]
    arrays, details = api.encode_pairs(tokenizer, pairs)
    encoding_path = workspace / "bio2nl/review/new_data_fixed_transfer_v1_2026-10-02/prepare_data.py"
    module = ast.parse(encoding_path.read_text())
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in {"encode_content", "pack"}]
    api.require(len(functions) == 2, "Reference encoding functions missing")
    namespace = {"np": np, "require": api.require}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(encoding_path), "exec"), namespace)
    for index, pair in enumerate(pairs):
        original_ids, original_mask = namespace["pack"](*[namespace["encode_content"](tokenizer, pair[k]) for k in ("sentence1", "sentence2")])
        api.require(np.array_equal(original_ids, arrays["input_ids"][index]) and
                    np.array_equal(original_mask, arrays["attention_mask"][index]), "Reference encoding differs")
    array_path = source_path.with_suffix(".npz")
    with np.load(array_path, allow_pickle=False) as values:
        api.require(np.array_equal(values["input_ids"][0], arrays["input_ids"][0]) and
                    np.array_equal(values["attention_mask"][0], arrays["attention_mask"][0]), "Original training encoding differs")
    selection = json.loads((train_root / "results/source_selection.json").read_text())
    chosen = next(x for x in selection["selected"] if x["job"] == {"condition": "EP", "pt_seed": 0, "ft_seed": 0, "smoke": False})
    classifier_path = train_root / "results" / chosen["checkpoint"]["file"]
    portable = api.load_model(catalog, "EP-pt0-ft0", package, classifier_path)
    original, payload = reference.load_checkpoint(classifier_path,
        expected_sha256=chosen["checkpoint"]["sha256"], expected_kind="source_classifier", expected_job=chosen["job"],
        expected_state_sha256=chosen["checkpoint"]["state_sha256"], expected_protocol_sha256=selection["protocol_sha256"],
        expected_manifest_sha256=selection["prepared_manifest_sha256"], expected_epoch=chosen["best_epoch"],
        expected_config=api.config_for_core(catalog["config"]))
    del payload
    original_output = reference.predict_source(original, arrays, batch_size=3)
    portable_output = api.core.predict_source(portable, arrays, batch_size=3)
    classifier_diff = float(np.max(np.abs(original_output["log_probabilities"] - portable_output["log_probabilities"])))
    api.require(classifier_diff == 0 and np.array_equal(original_output["predictions"], portable_output["predictions"]),
                "Original/portable classifier replay differs")
    single_output = api.core.predict_source(portable, arrays, batch_size=1)
    padding_diff = float(np.max(np.abs(single_output["log_probabilities"] - portable_output["log_probabilities"])))
    api.require(padding_diff <= 1e-5, "CPU batch/padding replay exceeds fixed tolerance")
    del original, portable
    gc.collect()
    lm_path = train_root / "results/full/pretrain/EP/pt0/model.pt"
    meta = json.loads(lm_path.with_name("metadata.json").read_text())
    portable_lm = api.load_model(catalog, "EP-pt0", package, lm_path)
    original_lm, payload = reference.load_checkpoint(lm_path,
        expected_sha256=meta["checkpoint"]["sha256"], expected_kind="causal_lm", expected_job=meta["job"],
        expected_state_sha256=meta["checkpoint"]["state_sha256"], expected_protocol_sha256=meta["protocol_sha256"],
        expected_manifest_sha256=meta["prepared_manifest_sha256"],
        expected_config=api.config_for_core(catalog["config"]))
    del payload
    with torch.inference_mode():
        ids = torch.as_tensor(arrays["input_ids"][:, :32])
        mask = torch.as_tensor(arrays["attention_mask"][:, :32])
        a = original_lm(input_ids=ids, attention_mask=mask, use_cache=False).logits
        b = portable_lm(input_ids=ids, attention_mask=mask, use_cache=False).logits
        lm_diff = float((a - b).abs().max())
    api.require(lm_diff == 0, "Original/portable pretraining replay differs")
    evidence = {"status": "passed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_sha256": catalog_sha, "cpu_threads": 4, "device": "cpu", "precision": "float32",
        "examples": {"source_training_protein_pairs": 1, "source_training_row_id": row["row_id"], "handwritten_english_pairs": 2},
        "reference_model_code_sha256": api.file_sha(reference_path), "portable_core_sha256": api.file_sha(package / "portable_core.py"),
        "reference_encoding_code_sha256": api.file_sha(encoding_path), "all_encodings_exact": True,
        "original_training_first_row_input_exact": True, "classifier_id": "EP-pt0-ft0", "pretraining_id": "EP-pt0",
        "classifier_max_abs_log_probability_difference": classifier_diff, "classifier_predictions_exact": True,
        "batch_size_1_vs_3_max_abs_log_probability_difference": padding_diff, "fixed_batch_padding_tolerance": 1e-5,
        "pretraining_max_abs_logit_difference": lm_diff,
        "gpu_used": False, "optimizer_updates": 0, "benchmark_scores_computed": False,
        "limitation": "CPU FP32 replay validates portable loading/encoding. It does not promise bitwise equality with prior CUDA BF16 benchmark predictions.",
        "code_sha256": {p.name: api.file_sha(p) for p in (package / "inference.py", package / "portable_core.py", package / "validate_cpu_replay.py")}}
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.workspace, args.package, args.output)
