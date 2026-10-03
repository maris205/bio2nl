"""CPU acceptance producer integration and failure checks on isolated fixtures."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).absolute().parent
PACKAGE = ROOT.parents[1] / "new_data_full_training_v1"
sys.path.insert(0, str(PACKAGE))
spec = importlib.util.spec_from_file_location("accept_training_tested", ROOT / "accept_training.py")
accept = importlib.util.module_from_spec(spec)
spec.loader.exec_module(accept)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def pin(path):
    return {"sha256": accept.file_sha256(path), "bytes": path.stat().st_size}


class Fixture:
    def __init__(self, root):
        self.protocol_path = write(root / "execution_protocol.json", {"schema": "isolated fixture"})
        self.protocol_sha256 = accept.file_sha256(self.protocol_path)
        self.manifest_sha256 = "b" * 64
        self.source_counts = {"train": 8044, "validation": 20276}
        self.output_root = root / "results"
        self.root = root / "prepared"
        self.root.mkdir()
        self.manifest = {"outputs": {}}
        for index in range(24):
            path = write(self.root / f"input{index}.json", {"index": index})
            self.manifest["outputs"][path.name] = pin(path)
        self.base = {str(self.protocol_path): pin(self.protocol_path)}
        self.execution_protocol = {"storage_policy": copy.deepcopy(accept.STORAGE_POLICY), "preparation_evidence": {}}
        code = write(root / "frozen_code.json", {"kind": "synthetic code identity"})
        for name in ("input_revalidation", "cpu_tests", "review_one", "review_two"):
            value = {"status": "passed", "verified_files": {str(code): pin(code)}}
            if name == "input_revalidation":
                value.update(prepared_manifest_sha256=self.manifest_sha256, source_counts=self.source_counts,
                             prepared_outputs_rehashed=24, all_final_hashes_unchanged=True)
            elif name == "cpu_tests":
                value["codepins"] = {str(code): pin(code)}
            path = write(root / "verification" / f"{name}.json", value)
            self.execution_protocol["preparation_evidence"][name] = {"file": str(path), **pin(path)}
        self.make_smoke()
        self.make_surface()

    def shared(self):
        return {"protocol_sha256": self.protocol_sha256, "prepared_manifest_sha256": self.manifest_sha256}

    def verify_current_inputs(self):
        return self.base.copy()

    def load_source(self, role, smoke=False):
        return {}, [{"row_id": "a" * 64}, {"row_id": "b" * 64}]

    def make_smoke(self):
        root = self.output_root / "smoke"
        artifact_inputs, initial, records, pretraining = {}, "c" * 64, {}, []
        for condition in ("EP", "ES", "EE"):
            directory = root / "pretrain" / condition / "pt0"
            checkpoint_path = write(directory / "model.pt", {"model": condition})
            cp = {"file": "model.pt", **pin(checkpoint_path), "state_sha256": condition * 32, "kind": "causal_lm"}
            metadata = {**self.shared(), "status": "completed", "kind": "pretraining", "run_kind": "smoke",
                        "job": {"condition": condition, "pt_seed": 0, "smoke": True}, "updates": 2,
                        "fresh_random_initialization": True, "old_weights_loaded": False,
                        "initial_state_sha256": initial, "checkpoint": cp, "inputs": self.base}
            path = write(directory / "metadata.json", metadata)
            artifact_inputs.update({str(p): pin(p) for p in (checkpoint_path, path)})
            records[condition] = metadata
            pretraining.append({"condition": condition, "max_absolute_nll_difference": 0., "checkpoint": cp,
                                "state_before_sha256": cp["state_sha256"], "state_after_sha256": cp["state_sha256"]})
        parent = root / "pretrain/EP/pt0"
        directory = root / "source/EP/pt0/ft0"
        cp_path = write(directory / "best.pt", {"classifier": "new EP"})
        cp = {"file": "best.pt", **pin(cp_path), "kind": "source_classifier", "state_sha256": "d" * 64}
        metadata = {**self.shared(), "status": "completed", "kind": "source_sft", "run_kind": "smoke",
                    "job": {"condition": "EP", "pt_seed": 0, "ft_seed": 0, "smoke": True}, "updates": 2,
                    "source_counts": {"train": 64, "validation": 64}, "sample_presentations": 64,
                    "old_weights_loaded": False, "new_parent_backbone_exactly_inherited": True,
                    "pretraining_checkpoint": {**records["EP"]["checkpoint"], "file": str(parent / "model.pt")},
                    "new_parent_metadata": {"file": str(parent / "metadata.json"),
                                            "sha256": accept.file_sha256(parent / "metadata.json")},
                    "checkpoint": cp, "inputs": self.base}
        path = write(directory / "metadata.json", metadata)
        artifact_inputs.update({str(p): pin(p) for p in (cp_path, path)})
        report_dir = root / "audit_runtime"
        outputs = {}
        for name in ("EP.probe.npz", "ES.probe.npz", "EE.probe.npz", "sft.train.predictions.npz", "sft.validation.predictions.npz"):
            p = write(report_dir / name, {"synthetic": name}); outputs[name] = pin(p)
        self.smoke_path = report_dir / "metadata.json"
        self.smoke_report = {**self.shared(), "status": "completed", "all_new_checkpoint_replays_passed": True,
            "checkpoints_replayed": 4, "optimizer_updates_performed": 0, "fixed_absolute_tolerance": 1e-5,
            "target_examples_read": 0, "source_test_examples_read": 0, "old_weights_loaded": False,
            "pretraining": pretraining, "paired_initial_state_sha256": initial,
            "source_sft": {"checkpoint": cp, "complete_head_retained": True, "state_before_sha256": cp["state_sha256"],
                           "state_after_sha256": cp["state_sha256"], "roles": {role: {"rows": 64,
                             "max_absolute_log_probability_difference": 0., "decision_differences": 0}
                             for role in ("train", "validation")}}, "outputs": outputs,
            "artifact_inputs": artifact_inputs, "data_and_code_inputs": self.base}
        write(self.smoke_path, self.smoke_report)

    def make_surface(self):
        root = self.output_root / "surface"
        selection = write(root / "selection.json", {"fixture": "selected"})
        cp = write(root / "lambda0.checkpoint.json", {"fixture": "head"})
        self.surface_path = self.protocol_path.parent / "verification/surface_audit.json"
        self.surface_report = {**self.shared(), "status": "passed", "candidates": 5, "prediction_artifacts": 10,
            "checkpoint_artifacts": 5, "source_counts": self.source_counts, "source_only": True,
            "target_examples_read": False, "all_read_files_final_rehashed": True,
            "files": {str(p): accept.file_sha256(p) for p in (selection, cp)}, "verified_read_files": self.base,
            "selection_file": str(selection), "selection_sha256": accept.file_sha256(selection),
            "checkpoint": "surface/lambda0.checkpoint.json", "checkpoint_sha256": accept.file_sha256(cp)}
        write(self.surface_path, self.surface_report)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = Fixture(self.root)

    def run_acceptance(self, free_gib=64):
        with mock.patch.object(accept, "ReleaseData", return_value=self.fixture), \
             mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
             mock.patch.object(accept.shutil, "disk_usage", return_value=mock.Mock(free=free_gib * accept.GIB)):
            return accept.accept_training(self.fixture.protocol_path, self.fixture.protocol_sha256)

    def test_complete_cpu_producer_closes_inputs_and_preserves_scope(self):
        result = self.run_acceptance()
        self.assertEqual(result["status"], "accepted_for_source_training")
        self.assertTrue(result["full_budget_training_enabled"])
        self.assertFalse(result["target_scoring_enabled"])
        self.assertFalse(result["model_deserialization_performed"])
        self.assertIn(str(self.fixture.smoke_path), result["verified_files"])
        self.assertIn(str(self.fixture.surface_path), result["verified_files"])
        self.assertIn(str(self.root / "verification/storage_acceptance.json"), result["verified_files"])
        self.assertNotIn(str(self.root / "training_acceptance.json"), result["verified_files"])
        self.assertNotIn(str(self.root / "verification/training_acceptance_status.json"), result["verified_files"])
        for path, identity in result["verified_files"].items():
            self.assertEqual(pin(Path(path)), identity)
        with self.assertRaisesRegex(ValueError, "Preserve existing"):
            self.run_acceptance()

    def test_insufficient_disk_records_failure_without_gate(self):
        with self.assertRaisesRegex(ValueError, "Insufficient projected"):
            self.run_acceptance(free_gib=8)
        self.assertFalse((self.root / "training_acceptance.json").exists())
        self.assertEqual(json.loads((self.root / "verification/training_acceptance_status.json").read_text())["status"], "failed")

    def test_replay_above_frozen_threshold_rejected(self):
        self.fixture.smoke_report["pretraining"][0]["max_absolute_nll_difference"] = 1.01e-5
        write(self.fixture.smoke_path, self.fixture.smoke_report)
        with self.assertRaisesRegex(ValueError, "LM smoke"):
            self.run_acceptance()

    def test_nonfinite_replay_rejected(self):
        self.fixture.smoke_report["pretraining"][0]["max_absolute_nll_difference"] = float("nan")
        write(self.fixture.smoke_path, self.fixture.smoke_report)
        with self.assertRaisesRegex(ValueError, "LM smoke"):
            self.run_acceptance()

    def test_changed_source_decisions_rejected(self):
        self.fixture.smoke_report["source_sft"]["roles"]["train"]["decision_differences"] = 1
        write(self.fixture.smoke_path, self.fixture.smoke_report)
        with self.assertRaisesRegex(ValueError, "Source smoke predictions"):
            self.run_acceptance()

    def test_changed_audit_artifact_rejected(self):
        (self.fixture.output_root / "smoke/pretrain/EP/pt0/model.pt").write_text("changed checkpoint")
        with self.assertRaisesRegex(ValueError, "Evidence identity"):
            self.run_acceptance()

    def test_stale_surface_data_rejected(self):
        self.fixture.surface_report["prepared_manifest_sha256"] = "z" * 64
        write(self.fixture.surface_path, self.fixture.surface_report)
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.run_acceptance()

    def test_prior_failure_or_running_preparation_cannot_pass(self):
        descriptor = self.fixture.execution_protocol["preparation_evidence"]["review_one"]
        path = Path(descriptor["file"])
        value = json.loads(path.read_text()); value["status"] = "running"
        write(path, value); descriptor.update(pin(path))
        with self.assertRaisesRegex(ValueError, "Preparation report has not passed"):
            self.run_acceptance()

    def test_missing_prepared_output_rejected(self):
        (self.fixture.root / "input23.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_acceptance()

    def test_ledger_rejects_final_mutation(self):
        ledger = accept.Ledger()
        ledger.pin(self.fixture.protocol_path)
        self.fixture.protocol_path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "Evidence identity"):
            ledger.final_check()

    def test_projection_counts_future_only_and_unicode_rows(self):
        smoke = {"pretraining_checkpoint_bytes": [100, 101, 99], "source_classifier_checkpoint_bytes": 80}
        result = accept.storage_projection(accept.STORAGE_POLICY, 64 * accept.GIB, smoke, 64, self.fixture.source_counts)
        self.assertEqual(result["projected_full_checkpoint_bytes"], 9 * 107 + 27 * 84)
        self.assertEqual(result["prediction_rows"], (8044 + 20276) * 5 * 27)
        self.assertEqual(result["predictions_uncompressed_upper_bytes"], result["prediction_rows"] * 288 + 270 * 4096)
        self.assertEqual(result["atomic_checkpoint_temporary_bytes"], 107)
        self.assertFalse(result["existing_smoke_bytes_charged_twice"])
        self.assertEqual(result["projected_peak_free_bytes"], 64 * accept.GIB - result["projected_future_increment_bytes"])

    def test_unknown_storage_policy_or_checkpoint_size_rejected(self):
        smoke = {"pretraining_checkpoint_bytes": [100, 101, 99], "source_classifier_checkpoint_bytes": 80}
        changed = {**accept.STORAGE_POLICY, "checkpoint_size_multiplier": 1.}
        with self.assertRaisesRegex(ValueError, "policy differs"):
            accept.storage_projection(changed, 64 * accept.GIB, smoke, 64, self.fixture.source_counts)
        smoke["source_classifier_checkpoint_bytes"] = 0
        with self.assertRaisesRegex(ValueError, "Unknown actual"):
            accept.storage_projection(accept.STORAGE_POLICY, 64 * accept.GIB, smoke, 64, self.fixture.source_counts)


if __name__ == "__main__":
    unittest.main()
