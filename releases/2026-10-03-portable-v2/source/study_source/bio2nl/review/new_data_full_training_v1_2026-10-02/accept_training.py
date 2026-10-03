"""Close source-only evidence and storage checks before full new-data training.

CPU-only. This producer hashes existing smoke weights but never deserializes or
runs a model. Existing raw, derived and failed historical gates are unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil

from runtime_data import ReleaseData, file_sha256, regular_file, relative_path, require

GIB = 2**30
PREPARATION_MAPS = ("verified_files", "files", "codepins", "code_pins", "checked_files", "reviewed_files", "code_files")
STORAGE_POLICY = {"minimum_projected_peak_free_gib": 8, "checkpoint_size_multiplier": 1.05,
                  "metadata_and_logs_allowance_gib": 2, "prediction_header_bytes_per_file": 4096,
                  "maximum_atomic_checkpoint_copies": 1}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value, replace=False):
    path = Path(path)
    require(replace or not path.exists(), "Refuse existing acceptance artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlinked acceptance output forbidden")
    pending = path.with_name(path.name + ".partial")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    pending.replace(path)


class Ledger:
    def __init__(self):
        self.files = {}

    def pin(self, path, expected=None):
        path = regular_file(path)
        current = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        if isinstance(expected, str):
            require(current["sha256"] == expected, "Evidence hash differs: " + str(path))
        elif expected is not None:
            require(current["sha256"] == expected["sha256"] and
                    ("bytes" not in expected or current["bytes"] == expected["bytes"]),
                    "Evidence identity differs: " + str(path))
        require(self.files.setdefault(str(path), current) == current, "Evidence changed: " + str(path))
        return path

    def read(self, path, expected=None):
        return json.loads(self.pin(path, expected).read_text())

    def mapping(self, values, base=None):
        require(isinstance(values, dict) and bool(values), "Evidence read set is empty or malformed")
        for name, pin in values.items():
            path = Path(name)
            if base is not None:
                path = Path(base) / relative_path(name)
            else:
                require(path.is_absolute(), "Evidence file must be absolute")
            self.pin(path, pin)

    def final_check(self):
        for path, value in tuple(self.files.items()):
            self.pin(path, value)
        return dict(self.files)


def bound_report(data, value):
    require(value["protocol_sha256"] == data.protocol_sha256 and
            value["prepared_manifest_sha256"] == data.manifest_sha256, "Current report provenance differs")


def preparation_evidence(data, ledger):
    evidence = data.execution_protocol["preparation_evidence"]
    require(isinstance(evidence, dict) and len(evidence) >= 4, "Incomplete preparation evidence")
    found = set()
    for label, descriptor in evidence.items():
        path = Path(descriptor["file"])
        value = ledger.read(path, descriptor)
        require(value["status"] == "passed", "Preparation report has not passed: " + label)
        found.add(path.name)
        closed = False
        for key in PREPARATION_MAPS:
            if key in value:
                ledger.mapping(value[key])
                closed = True
        require(closed, "Preparation report lacks file identities: " + label)
        if path.name == "input_revalidation.json":
            require(value["prepared_manifest_sha256"] == data.manifest_sha256 and
                    value["source_counts"] == data.source_counts and value["prepared_outputs_rehashed"] == 24 and
                    value["all_final_hashes_unchanged"] is True, "Input revalidation evidence differs")
        elif path.name == "cpu_tests.json":
            require(isinstance(value.get("codepins"), dict) and value["codepins"], "CPU test code identities missing")
    require({"input_revalidation.json", "cpu_tests.json"} <= found, "Missing input revalidation or CPU tests")


def smoke_evidence(data, ledger):
    root = data.output_root / "smoke"
    report_path = root / "audit_runtime/metadata.json"
    report = ledger.read(report_path)
    bound_report(data, report)
    require(report["status"] == "completed" and report["all_new_checkpoint_replays_passed"] is True and
            report["checkpoints_replayed"] == 4 and report["optimizer_updates_performed"] == 0 and
            report["fixed_absolute_tolerance"] == 1e-5 and report["target_examples_read"] == 0 and
            report["source_test_examples_read"] == 0 and report["old_weights_loaded"] is False,
            "Current independent smoke replay failed or has wrong scope")
    require(len(report["pretraining"]) == 3 and
            [r["condition"] for r in report["pretraining"]] == ["EP", "ES", "EE"], "Smoke condition roster differs")
    for row in report["pretraining"]:
        maximum = row["max_absolute_nll_difference"]
        require(type(maximum) in (int, float) and math.isfinite(maximum) and 0 <= maximum <= 1e-5 and
                row["state_before_sha256"] == row["state_after_sha256"] == row["checkpoint"]["state_sha256"],
                "LM smoke replay/state identity differs")
    source = report["source_sft"]
    require(set(source["roles"]) == {"train", "validation"} and source["complete_head_retained"] is True and
            source["state_before_sha256"] == source["state_after_sha256"] == source["checkpoint"]["state_sha256"],
            "Complete smoke classifier replay differs")
    for row in source["roles"].values():
        maximum = row["max_absolute_log_probability_difference"]
        require(type(maximum) in (int, float) and math.isfinite(maximum) and 0 <= maximum <= 1e-5 and
                row["rows"] == 64 and row["decision_differences"] == 0, "Source smoke predictions differ")
    ledger.mapping(report["artifact_inputs"])
    ledger.mapping(report["data_and_code_inputs"])
    require(set(report["outputs"]) == {"EP.probe.npz", "ES.probe.npz", "EE.probe.npz",
            "sft.train.predictions.npz", "sft.validation.predictions.npz"}, "Independent smoke output roster differs")
    ledger.mapping(report["outputs"], report_path.parent)
    initial_states, lm_sizes, lm_records = [], [], {}
    for condition in ("EP", "ES", "EE"):
        directory = root / "pretrain" / condition / "pt0"
        metadata = ledger.read(directory / "metadata.json")
        bound_report(data, metadata)
        require(metadata["status"] == "completed" and metadata["kind"] == "pretraining" and
                metadata["job"] == {"condition": condition, "pt_seed": 0, "smoke": True} and
                metadata["run_kind"] == "smoke" and metadata["updates"] == 2 and
                metadata["fresh_random_initialization"] is True and metadata["old_weights_loaded"] is False,
                "Current pretraining smoke incomplete")
        checkpoint = metadata["checkpoint"]
        require(checkpoint["file"] == "model.pt" and checkpoint["kind"] == "causal_lm" and
                checkpoint["state_sha256"] != metadata["initial_state_sha256"], "Smoke LM identity/update differs")
        ledger.pin(directory / "model.pt", checkpoint)
        ledger.mapping(metadata["inputs"])
        require(report["pretraining"][len(lm_sizes)]["checkpoint"] == checkpoint, "Audited LM checkpoint differs")
        initial_states.append(metadata["initial_state_sha256"])
        lm_sizes.append(checkpoint["bytes"])
        lm_records[condition] = metadata
    require(len(set(initial_states)) == 1 and report["paired_initial_state_sha256"] == initial_states[0],
            "Three smoke initial states are not paired")
    directory = root / "source/EP/pt0/ft0"
    metadata = ledger.read(directory / "metadata.json")
    bound_report(data, metadata)
    require(metadata["status"] == "completed" and metadata["kind"] == "source_sft" and
            metadata["job"] == {"condition": "EP", "pt_seed": 0, "ft_seed": 0, "smoke": True} and
            metadata["run_kind"] == "smoke" and metadata["updates"] == 2 and
            metadata["source_counts"] == {"train": 64, "validation": 64} and
            metadata["sample_presentations"] == 64 and metadata["old_weights_loaded"] is False and
            metadata["new_parent_backbone_exactly_inherited"] is True,
            "Current SFT smoke incomplete or wrong parent")
    parent = root / "pretrain/EP/pt0"
    require(metadata["pretraining_checkpoint"] == {**lm_records["EP"]["checkpoint"], "file": str(parent / "model.pt")} and
            metadata["new_parent_metadata"] == {"file": str(parent / "metadata.json"),
                                                 "sha256": ledger.files[str(parent / "metadata.json")]["sha256"]},
            "SFT did not use the exact new smoke EP parent")
    cp = metadata["checkpoint"]
    require(cp["file"] == "best.pt" and cp["kind"] == "source_classifier" and cp == source["checkpoint"],
            "Audited complete source checkpoint differs")
    ledger.pin(directory / "best.pt", cp)
    ledger.mapping(metadata["inputs"])
    return {"pretraining_checkpoint_bytes": lm_sizes, "source_classifier_checkpoint_bytes": cp["bytes"],
            "smoke_training_updates": 8, "independent_replay_updates": 0}


def surface_evidence(data, ledger):
    path = data.protocol_path.parent / "verification/surface_audit.json"
    report = ledger.read(path)
    bound_report(data, report)
    require(report["status"] == "passed" and report["candidates"] == 5 and report["prediction_artifacts"] == 10 and
            report["checkpoint_artifacts"] == 5 and report["source_counts"] == data.source_counts and
            report["source_only"] is True and report["target_examples_read"] is False and
            report["all_read_files_final_rehashed"] is True, "Current source surface audit did not pass")
    ledger.mapping(report["files"])
    ledger.mapping(report["verified_read_files"])
    selection = Path(report["selection_file"])
    require(selection == data.output_root / "surface/selection.json", "Surface selection path differs")
    ledger.pin(selection, report["selection_sha256"])
    ledger.pin(data.output_root / relative_path(report["checkpoint"]), report["checkpoint_sha256"])
    return report


def storage_projection(policy, current_free_bytes, smoke, max_row_id_length, source_counts):
    require(policy == STORAGE_POLICY, "Frozen storage policy differs")
    require(type(current_free_bytes) is int and current_free_bytes >= 0 and type(max_row_id_length) is int and
            max_row_id_length > 0 and source_counts == {"train": 8044, "validation": 20276}, "Storage basis unknown")
    lm = max(smoke["pretraining_checkpoint_bytes"])
    classifier = smoke["source_classifier_checkpoint_bytes"]
    require(all(type(v) is int and v > 0 for v in smoke["pretraining_checkpoint_bytes"]) and
            type(classifier) is int and classifier > 0, "Unknown actual checkpoint size")
    lm_projected = math.ceil(lm * policy["checkpoint_size_multiplier"])
    classifier_projected = math.ceil(classifier * policy["checkpoint_size_multiplier"])
    checkpoints = 9 * lm_projected + 27 * classifier_projected
    prediction_rows = (source_counts["train"] + source_counts["validation"]) * 5 * 27
    prediction_bytes = prediction_rows * (16 + 8 + 8 + 4 * max_row_id_length) + 270 * policy["prediction_header_bytes_per_file"]
    atomic_bytes = max(lm_projected, classifier_projected) * policy["maximum_atomic_checkpoint_copies"]
    logs = policy["metadata_and_logs_allowance_gib"] * GIB
    future = checkpoints + prediction_bytes + atomic_bytes + logs
    projected_free = current_free_bytes - future
    return {"current_free_bytes": current_free_bytes, "measured_smoke_checkpoint_bytes": smoke,
            "future_pretraining_checkpoints": 9, "future_source_checkpoints": 27,
            "projected_pretraining_bytes_each": lm_projected, "projected_source_bytes_each": classifier_projected,
            "projected_full_checkpoint_bytes": checkpoints, "prediction_files": 270, "prediction_rows": prediction_rows,
            "maximum_row_id_length": max_row_id_length, "predictions_uncompressed_upper_bytes": prediction_bytes,
            "atomic_checkpoint_temporary_bytes": atomic_bytes, "metadata_and_logs_allowance_bytes": logs,
            "projected_future_increment_bytes": future, "projected_peak_free_bytes": projected_free,
            "projected_peak_free_gib": projected_free / GIB, "minimum_projected_peak_free_gib": 8,
            "existing_smoke_bytes_charged_twice": False, "no_old_artifacts_deleted": True}


def accept_training(protocol_path, protocol_sha256):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Training acceptance must be CPU-only")
    protocol_path = Path(protocol_path).absolute()
    root = protocol_path.parent
    output = root / "training_acceptance.json"
    storage_path = root / "verification/storage_acceptance.json"
    status_path = root / "verification/training_acceptance_status.json"
    require(not any(p.exists() for p in (output, storage_path, status_path)), "Preserve existing acceptance attempt")
    status = {"status": "running", "started_at_utc": now(), "protocol_sha256": protocol_sha256}
    atomic_json(status_path, status)
    try:
        data = ReleaseData(protocol_path, protocol_sha256, mode="prepare")
        require(data.source_counts == {"train": 8044, "validation": 20276}, "Full source counts differ")
        ledger = Ledger()
        ledger.mapping(data.verify_current_inputs())
        require(len(data.manifest["outputs"]) == 24, "Prepared output inventory differs")
        for relative, pin in data.manifest["outputs"].items():
            ledger.pin(data.root / relative_path(relative), pin)
        preparation_evidence(data, ledger)
        smoke = smoke_evidence(data, ledger)
        surface_evidence(data, ledger)
        maximum_length = 0
        for role in ("train", "validation"):
            arrays, rows = data.load_source(role, smoke=False)
            maximum_length = max(maximum_length, max(len(row["row_id"]) for row in rows))
            del arrays, rows
        ledger.mapping(data.verify_current_inputs())
        ledger.final_check()
        free_bytes = shutil.disk_usage(root).free
        storage = storage_projection(data.execution_protocol["storage_policy"], free_bytes, smoke, maximum_length, data.source_counts)
        require(storage["projected_peak_free_gib"] >= 8, "Insufficient projected peak storage reserve")
        storage.update(status="passed", created_at_utc=now(), protocol_sha256=protocol_sha256,
                       prepared_manifest_sha256=data.manifest_sha256, source_counts=data.source_counts)
        atomic_json(storage_path, storage)
        ledger.pin(storage_path)
        pins = ledger.final_check()
        data.verify_current_inputs()
        accepted = {"schema_version": 1, "status": "accepted_for_source_training", "all_checks_passed": True,
                    "completed_at_utc": now(), "protocol_sha256": protocol_sha256,
                    "prepared_manifest_sha256": data.manifest_sha256, "source_counts": data.source_counts,
                    "full_budget_training_enabled": True, "target_scoring_enabled": False,
                    "source_test_scoring_enabled": False, "old_weights_allowed": False,
                    "verified_files": pins, "verified_file_count": len(pins), "all_final_hashes_unchanged": True,
                    "current_smoke_complete_checkpoints": 4, "current_smoke_updates": 8,
                    "surface_candidates": 5, "surface_predictions": 10,
                    "full_pretraining_runs_authorized": 9, "full_source_fits_authorized": 27,
                    "model_deserialization_performed": False, "gpu_inference_performed": False,
                    "storage_acceptance": {"file": str(storage_path), **pins[str(storage_path)]}}
        atomic_json(output, accepted)
        status.update(status="completed", completed_at_utc=now(), acceptance_sha256=file_sha256(output))
        atomic_json(status_path, status, replace=True)
        print(json.dumps({"status": accepted["status"], "verified_files": len(pins),
                          "projected_peak_free_gib": storage["projected_peak_free_gib"]}), flush=True)
        return accepted
    except BaseException as error:
        status.update(status="failed", failed_at_utc=now(), error=f"{type(error).__name__}: {error}")
        atomic_json(status_path, status, replace=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args(argv)
    accept_training(args.protocol, args.protocol_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
