"""CPU integration and fault checks; no production inputs or GPU launches."""
from __future__ import annotations

import copy
import gzip
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

PACKAGE = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PACKAGE))
import runtime_data as rd
import runtime
import model

torch.set_num_threads(1)


def tiny_design():
    return {"pretraining": {"architecture": {"model_type": "gpt2", "n_layer": 1, "n_head": 2, "n_embd": 16,
                                             "n_inner": 32, "attn_pdrop": 0.0, "embd_pdrop": 0.0, "resid_pdrop": 0.0},
                            "context_length": 16},
            "tokenizer": {"vocab_size": 32, "eos_token_id": 1, "pad_token_id": 0}}


def arrays_rows(split="train", count=64):
    ids = np.zeros((count, 512), dtype=np.int64)
    masks = np.zeros_like(ids)
    labels = np.arange(count, dtype=np.int64) % 2
    rows = []
    for index in range(count):
        sequence = [4, 2, 5, 1]
        ids[index, :4] = sequence
        masks[index, :4] = 1
        rows.append({"row_index": index, "row_id": f"{split}{index}", "label": int(labels[index]),
                     "ids_a": [4], "ids_b": [5], "metadata": {"split": split, "block_id": f"b{index // 4}"}})
    return {"input_ids": ids, "attention_mask": masks, "labels": labels}, rows


class Fixture:
    def __init__(self, root):
        self.root = root
        self.raw = root / "raw"
        self.prepared = root / "prepared"
        self.raw.mkdir()
        self.prepared.mkdir()
        self.token = self.raw / "tokenizers/mixed_bpe/tokenizer.json"
        self.token.parent.mkdir(parents=True)
        self.token.write_text("{}\n")
        raw_base = {"status": "accepted_new_raw_release_training_disabled", "release_id": "2026-10-01-portable-v2",
                    "training_enabled": False, "training_gate_pass": False, "target_scoring_enabled": False}
        self.raw_acceptance = self.raw / "portable_validation/acceptance.json"
        self.write(self.raw_acceptance, raw_base)
        self.raw_manifest = self.raw / "manifest.json"
        self.write(self.raw_manifest, {**raw_base, "acceptance": self.pin(self.raw_acceptance)})
        source, outputs, smoke = {}, {}, {}
        for split in ("train", "validation"):
            arrays, rows = arrays_rows(split)
            npz = self.prepared / f"source/{split}.npz"
            npz.parent.mkdir(exist_ok=True)
            np.savez_compressed(npz, **arrays)
            rowfile = self.prepared / f"source/{split}.rows.jsonl.gz"
            with gzip.open(rowfile, "wt") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            source[split] = {"npz": str(npz.relative_to(self.prepared)), "rows": str(rowfile.relative_to(self.prepared)), "count": 64,
                             "npz_sha256": rd.file_sha256(npz), "rows_sha256": rd.file_sha256(rowfile)}
            outputs.update({str(p.relative_to(self.prepared)): self.pin(p) for p in (npz, rowfile)})
            names = [r["row_id"] for r in rows]
            smoke[split] = {"rows": 64, "indices": list(range(64)), "row_ids": names, "row_ids_sha256": rd.canonical_hash(names)}
        self.manifest = {"status": "built_pending_independent_audit", "raw_release_id": "2026-10-01-portable-v2",
                         "raw_acceptance_sha256": rd.file_sha256(self.raw_acceptance), "raw_manifest_sha256": rd.file_sha256(self.raw_manifest),
                         "build_protocol_sha256": "a" * 64, "source_counts": {"train": 64, "validation": 64}, "source": source,
                         "outputs": outputs, "condition_streams": {k: list(v) for k, v in rd.CONDITION_STREAMS.items()},
                         "streams": dict.fromkeys(("english_common", "english_extra", "protein", "shuffled"), {}),
                         "tokenizer": {"file": str(self.token), "sha256": rd.file_sha256(self.token)},
                         "source_prefix64": smoke, "smoke_source": copy.deepcopy(smoke)}
        self.manifest_path = self.prepared / "manifest.json"
        self.write(self.manifest_path, self.manifest)
        self.acceptance = {"status": "source_only_smoke_inputs_accepted", "prepared_manifest_sha256": rd.file_sha256(self.manifest_path),
                           "data_protocol_sha256": "a" * 64, "raw_release_manifest_sha256": rd.file_sha256(self.raw_manifest),
                           "raw_release_acceptance_sha256": rd.file_sha256(self.raw_acceptance), "source_counts": self.manifest["source_counts"],
                           "full_training_enabled": False, "target_scoring_enabled": False, "source_test_scoring_enabled": False,
                           "all_checks_passed": True}
        self.acceptance_path = root / "data_acceptance.json"
        self.write(self.acceptance_path, self.acceptance)
        self.protocol = {"schema_version": 1, "status": "frozen_for_new_data_full_training", "scope": "source_only_new_data_full_training",
                         "bounded_smoke_training_enabled": True, "full_budget_training_enabled": True,
                         "source_training_authorized": True, "old_weights_allowed": False,
                         "training_acceptance_file": str(root / "training_acceptance.json"),
                         "target_scoring_enabled": False, "source_test_scoring_enabled": False,
                         "environment": {"python": sys.version, "packages": {n: importlib.metadata.version(n) for n in
                         ("numpy", "torch", "transformers", "tokenizers", "scikit-learn")}},
                         "runtime_code": {str(PACKAGE / name): self.pin(PACKAGE / name) for name in rd.RUNTIME_FILES},
                         "raw_release": {"root": str(self.raw), "manifest": self.pin(self.raw_manifest), "acceptance": self.pin(self.raw_acceptance)},
                         "prepared": {"root": str(self.prepared), "manifest": self.pin(self.manifest_path), "acceptance": self.pin(self.acceptance_path)},
                         "design": {"tokenizer": {"sha256": rd.file_sha256(self.token), "vocab_size": 32000, "pad_token_id": 0,
                                     "eos_token_id": 1, "pair_separator_id": 2, "unk_token_id": 3}}, "output_root": str(root / "results")}
        self.protocol_path = root / "execution_protocol.json"
        self.make_prior_smoke()
        self.write(self.protocol_path, self.protocol)

    @staticmethod
    def pin(path):
        return {"file": str(path), "sha256": rd.file_sha256(path), "bytes": path.stat().st_size}

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def refresh(self):
        self.write(self.manifest_path, self.manifest)
        self.acceptance["prepared_manifest_sha256"] = rd.file_sha256(self.manifest_path)
        self.write(self.acceptance_path, self.acceptance)
        self.protocol["prepared"]["manifest"] = self.pin(self.manifest_path)
        self.protocol["prepared"]["acceptance"] = self.pin(self.acceptance_path)
        self.make_prior_smoke()
        self.write(self.protocol_path, self.protocol)

    def reader(self, mode="prepare"):
        return rd.ReleaseData(self.protocol_path, rd.file_sha256(self.protocol_path), mode=mode)

    def make_prior_smoke(self):
        prior = self.root / "prior"
        prior.mkdir(exist_ok=True)
        execution = {"status": "frozen_for_new_data_smoke", "prepared": {"manifest": {"sha256": rd.file_sha256(self.manifest_path)}}}
        review_protocol = {"status": "frozen_for_reload_review_v2", "optimizer_updates_authorized": 0,
                           "fixed_absolute_tolerance": 1e-5}
        self.write(prior / "execution_protocol.json", execution)
        self.write(prior / "execution_status.json", {"status": "failed"})
        self.write(prior / "review_protocol.json", review_protocol)
        common = {"status": "completed", "all_new_checkpoint_replays_passed": True, "checkpoints_replayed": 4,
                  "optimizer_updates_performed": 0, "fixed_absolute_tolerance": 1e-5,
                  "original_queue_remains_failed": True}
        self.write(prior / "review_result.json", {**common,
            "parent_execution_protocol_sha256": rd.file_sha256(prior / "execution_protocol.json"),
            "review_protocol_sha256": rd.file_sha256(prior / "review_protocol.json")})
        self.write(prior / "metadata.json", {**common,
            "protocol_sha256": rd.file_sha256(prior / "execution_protocol.json"),
            "prepared_manifest_sha256": rd.file_sha256(self.manifest_path)})
        self.write(prior / "completion_review.json", {"status": "independent_completed_reload_review_passed",
            "checkpoints_replayed": 4, "optimizer_updates_in_this_review": 0, "original_failed_queue_preserved": True,
            "review_protocol_sha256": rd.file_sha256(prior / "review_protocol.json")})
        self.protocol["prior_smoke_evidence"] = {p.stem: self.pin(p) for p in prior.iterdir()}

    def accept_training(self):
        evidence = self.root / "smoke_audit.json"
        self.write(evidence, {"status": "passed"})
        shared = {"protocol_sha256": rd.file_sha256(self.protocol_path),
                  "prepared_manifest_sha256": rd.file_sha256(self.manifest_path)}
        smoke = self.root / "results/smoke/audit_runtime/metadata.json"
        surface = self.root / "verification/surface_audit.json"
        storage = self.root / "verification/storage_acceptance.json"
        self.write(smoke, {**shared, "status": "completed", "all_new_checkpoint_replays_passed": True,
                           "checkpoints_replayed": 4})
        self.write(surface, {**shared, "status": "passed", "candidates": 5, "prediction_artifacts": 10})
        self.write(storage, {**shared, "status": "passed", "projected_peak_free_gib": 12.})
        self.training_acceptance = {"status": "accepted_for_source_training", "all_checks_passed": True,
            "full_budget_training_enabled": True, "target_scoring_enabled": False,
            "source_test_scoring_enabled": False, **shared,
            "verified_files": {str(p): self.pin(p) for p in (self.protocol_path, evidence, smoke, surface, storage)}}
        self.training_acceptance_path = self.root / "training_acceptance.json"
        self.write(self.training_acceptance_path, self.training_acceptance)
        return self.training_acceptance


class DataGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(Path(self.temp.name))

    def test_valid_new_source_load_and_prefix(self):
        data = self.fixture.reader()
        arrays, rows = data.load_source("train", smoke=True)
        self.assertEqual(len(rows), 64)
        self.assertFalse(arrays["input_ids"].flags.writeable)
        self.assertTrue(data.verify_current_inputs())

    def test_wrong_execution_pin(self):
        with self.assertRaisesRegex(ValueError, "Execution protocol SHA"):
            rd.ReleaseData(self.fixture.protocol_path, "0" * 64)

    def test_disabled_full_training_authority_rejected(self):
        self.fixture.protocol["full_budget_training_enabled"] = False
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, "Execution scope"):
            self.fixture.reader()

    def test_derived_audit_must_pass(self):
        self.fixture.acceptance["all_checks_passed"] = False
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, "Derived acceptance"):
            self.fixture.reader()

    def test_wrong_new_data_lineage_rejected(self):
        self.fixture.manifest["raw_manifest_sha256"] = "1" * 64
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, "Prepared raw lineage"):
            self.fixture.reader()

    def test_derived_gate_protocol_binding(self):
        self.fixture.acceptance["data_protocol_sha256"] = "2" * 64
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, "Independent source data gate"):
            self.fixture.reader()

    def test_source_count_comes_from_new_gate(self):
        self.fixture.acceptance["source_counts"] = {"train": 8002, "validation": 20338}
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, "Accepted source counts"):
            self.fixture.reader()

    def test_runtime_inventory_required(self):
        del self.fixture.protocol["runtime_code"][str(PACKAGE / "runtime.py")]
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, "Incomplete runtime code"):
            self.fixture.reader()

    def test_forbidden_source_roles(self):
        data = self.fixture.reader()
        for split in ("test", "source_test", "target", "confirmation"):
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, "Only source train"):
                data.load_source(split)

    def test_frozen_prefix_mutation_rejected(self):
        self.fixture.manifest["source_prefix64"]["train"]["row_ids"][0] = "old-row"
        self.fixture.refresh()
        with self.assertRaisesRegex(ValueError, "Frozen source prefix"):
            self.fixture.reader().load_source("train", smoke=True)

    def test_mutated_source_bytes_rejected(self):
        data = self.fixture.reader()
        path = self.fixture.prepared / "source/train.npz"
        with path.open("ab") as handle:
            handle.write(b"x")
        with self.assertRaisesRegex(ValueError, "Input bytes"):
            data.load_source("train")

    def test_source_role_cannot_escape(self):
        data = self.fixture.reader()
        with self.assertRaisesRegex(ValueError, "Noncanonical"):
            data.output_path("../manifest.json")

    def test_final_gate_rehash(self):
        data = self.fixture.reader()
        self.fixture.acceptance_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Input bytes"):
            data.verify_current_inputs()

    def test_symlink_root_rejected(self):
        link = Path(self.temp.name) / "link.json"
        link.symlink_to(self.fixture.protocol_path)
        with self.assertRaisesRegex(ValueError, "Symlinked"):
            rd.ReleaseData(link, rd.file_sha256(self.fixture.protocol_path))

    def test_full_training_requires_new_acceptance(self):
        with self.assertRaises(FileNotFoundError):
            self.fixture.reader("full")
        self.assertFalse(self.fixture.reader("smoke").full_training_accepted)
        self.fixture.accept_training()
        self.assertTrue(self.fixture.reader("full").full_training_accepted)

    def test_full_gate_protocol_mismatch_rejected(self):
        value = self.fixture.accept_training()
        value["protocol_sha256"] = "c" * 64
        self.fixture.write(self.fixture.training_acceptance_path, value)
        with self.assertRaisesRegex(ValueError, "protocol/data binding"):
            self.fixture.reader("full")

    def test_full_gate_cannot_enable_target(self):
        value = self.fixture.accept_training()
        value["target_scoring_enabled"] = True
        self.fixture.write(self.fixture.training_acceptance_path, value)
        with self.assertRaisesRegex(ValueError, "has not passed"):
            self.fixture.reader("full")

    def test_full_gate_rejects_missing_evidence(self):
        value = self.fixture.accept_training()
        value["verified_files"] = {}
        self.fixture.write(self.fixture.training_acceptance_path, value)
        with self.assertRaisesRegex(ValueError, "evidence missing"):
            self.fixture.reader("full")

    def test_full_gate_evidence_rehashed(self):
        self.fixture.accept_training()
        data = self.fixture.reader("full")
        (self.fixture.root / "smoke_audit.json").write_text("{bad}")
        with self.assertRaisesRegex(ValueError, "Input bytes"):
            data.verify_current_inputs()

    def test_full_gate_itself_rehashed(self):
        self.fixture.accept_training()
        data = self.fixture.reader("full")
        self.fixture.training_acceptance_path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "Input bytes"):
            data.verify_current_inputs()

    def test_old_raw_gate_stays_disabled_after_full_acceptance(self):
        self.fixture.accept_training()
        self.fixture.reader("full")
        raw = json.loads(self.fixture.raw_manifest.read_text())
        self.assertFalse(raw["training_enabled"])
        self.assertFalse(raw["training_gate_pass"])

    def test_missing_mandatory_smoke_evidence_rejected(self):
        value = self.fixture.accept_training()
        del value["verified_files"][str(self.fixture.root / "results/smoke/audit_runtime/metadata.json")]
        self.fixture.write(self.fixture.training_acceptance_path, value)
        with self.assertRaisesRegex(ValueError, "mandatory evidence missing: smoke"):
            self.fixture.reader("full")

    def test_insufficient_storage_gate_rejected(self):
        value = self.fixture.accept_training()
        path = self.fixture.root / "verification/storage_acceptance.json"
        record = json.loads(path.read_text()); record["projected_peak_free_gib"] = 7.99
        self.fixture.write(path, record)
        value["verified_files"][str(path)] = self.fixture.pin(path)
        self.fixture.write(self.fixture.training_acceptance_path, value)
        with self.assertRaisesRegex(ValueError, "storage reserve"):
            self.fixture.reader("full")

    def test_incomplete_current_smoke_rejected(self):
        value = self.fixture.accept_training()
        path = self.fixture.root / "results/smoke/audit_runtime/metadata.json"
        record = json.loads(path.read_text()); record["checkpoints_replayed"] = 3
        self.fixture.write(path, record)
        value["verified_files"][str(path)] = self.fixture.pin(path)
        self.fixture.write(self.fixture.training_acceptance_path, value)
        with self.assertRaisesRegex(ValueError, "Current code smoke"):
            self.fixture.reader("full")

    def test_prior_smoke_cannot_claim_failed_queue_passed(self):
        path = self.fixture.root / "prior/execution_status.json"
        self.fixture.write(path, {"status": "completed"})
        self.fixture.protocol["prior_smoke_evidence"]["execution_status"] = self.fixture.pin(path)
        self.fixture.write(self.fixture.protocol_path, self.fixture.protocol)
        with self.assertRaisesRegex(ValueError, "failed queue must stay failed"):
            self.fixture.reader()

    def test_prior_smoke_does_not_allow_checkpoint_descriptors(self):
        value = self.fixture.protocol["prior_smoke_evidence"]
        value["metadata"]["file"] = str(self.fixture.root / "previous/model.pt")
        self.fixture.write(self.fixture.protocol_path, self.fixture.protocol)
        with self.assertRaisesRegex(ValueError, "Only the six"):
            self.fixture.reader()


class ContractTests(unittest.TestCase):
    def test_only_fixed_jobs(self):
        self.assertEqual(runtime.resolve_job("EP", 0), {"condition": "EP", "pt_seed": 0, "smoke": True})
        for args in (("EP", 0, None, "resume"), ("EP", 1), ("ES", 0, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                runtime.resolve_job(*args)

    def test_frozen_schedules(self):
        self.assertEqual([3e-4 * runtime.cosine_multiplier(i) for i in range(2)], [0, 3e-4 / 21])
        self.assertEqual([2e-5 * runtime.linear_multiplier(i, 2) for i in range(2)], [2e-5, 1e-5])

    def test_schedule_duplicate_block_rejected(self):
        schedule = np.zeros((2048, 16, 2), dtype=np.int64)
        schedule[:, 8:, 0] = 1
        for side in (0, 1):
            schedule[:, side * 8:(side + 1) * 8, 1] = np.arange(16384).reshape(-1, 8)
        rd.validate_schedule(schedule)
        schedule[0, 0, 1] = 1
        with self.assertRaisesRegex(ValueError, "omitted or repeated"):
            rd.validate_schedule(schedule)

    def test_source_endpoint_and_mask_rejected(self):
        arrays, rows = arrays_rows()
        rd.validate_source_arrays(arrays, rows, "train", 64)
        arrays["attention_mask"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "attention mask"):
            rd.validate_source_arrays(arrays, rows, "train", 64)

    def test_gather_keeps_each_stream_assignment(self):
        left = np.arange(24).reshape(3, 8)
        right = left + 100
        selected = runtime.gather_blocks(left, right, np.array([[1, 2], [0, 1]], dtype=np.int64))
        np.testing.assert_array_equal(selected.numpy(), np.stack([right[2], left[1]]))

    def test_all_full_matrix_jobs_have_distinct_paths(self):
        data = mock.Mock(execution_protocol={"output_root": "/tmp/new_full_runtime_results"})
        paths = set()
        for condition in ("EP", "ES", "EE"):
            for seed in range(3):
                job = runtime.resolve_job(condition, seed, run_kind="full")
                self.assertFalse(job["smoke"])
                paths.add(runtime.output_directory(data, condition, seed, run_kind="full"))
                for ft_seed in range(3):
                    job = runtime.resolve_job(condition, seed, ft_seed, "full")
                    self.assertFalse(job["smoke"])
                    paths.add(runtime.output_directory(data, condition, seed, ft_seed, "full"))
        self.assertEqual(len(paths), 36)
        self.assertNotIn(runtime.output_directory(data, "EP", 0, run_kind="smoke"), paths)

    def test_full_data_budget_retains_last_twelve_rows(self):
        budget = runtime.sft_budget(8044, "full")
        self.assertEqual(budget, {"epochs": 5, "steps_per_epoch": 252, "updates": 1260,
                                  "sample_presentations": 40220})
        generator = torch.Generator().manual_seed(2)
        presentations, steps = 0, 0
        for _ in range(budget["epochs"]):
            order = torch.randperm(8044, generator=generator).numpy()
            batches = [order[i:i+32] for i in range(0, len(order), 32)]
            self.assertEqual(len(batches[-1]), 12)
            np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(8044))
            steps += len(batches)
            presentations += sum(len(batch) for batch in batches)
        self.assertEqual((steps, presentations), (1260, 40220))
        self.assertEqual(runtime.sft_budget(64, "smoke")["updates"], 2)
        with self.assertRaisesRegex(ValueError, "count differs"):
            runtime.sft_budget(8002, "full")

    def test_full_scheduler_finite_and_budget_bound(self):
        rates = np.array([3e-4 * runtime.cosine_multiplier(i) for i in range(1024)])
        self.assertEqual(rates[0], 0.)
        self.assertEqual(rates[21], 3e-4)
        self.assertTrue(np.isfinite(rates).all())
        self.assertGreater(rates[-1], 0.)
        self.assertEqual(runtime.cosine_multiplier(1024), 0.)
        source_rates = [2e-5 * runtime.linear_multiplier(i, 1260) for i in range(1260)]
        self.assertEqual(source_rates[0], 2e-5)
        self.assertGreater(source_rates[-1], 0.)
        self.assertEqual(runtime.linear_multiplier(1260, 1260), 0.)

    def test_storage_reserve_fails_closed(self):
        state = {"weight": torch.zeros(2, 2)}
        with mock.patch.object(runtime.shutil, "disk_usage", return_value=mock.Mock(free=8 << 30)):
            with self.assertRaisesRegex(ValueError, "8 GiB"):
                runtime.require_checkpoint_space(Path("/tmp"), state)

    def test_prepare_mode_cannot_enter_training(self):
        data = mock.Mock(mode="prepare", full_training_accepted=False)
        data.source_context.return_value = {"protocol": {}}
        with self.assertRaisesRegex(ValueError, "separate acceptance"):
            runtime._entry(data, Path("/tmp/unused"), {"condition": "EP", "pt_seed": 0}, "full", "pretraining")


class TinyModelTests(unittest.TestCase):
    def test_fresh_paired_initializations_preserve_rng(self):
        before = torch.random.get_rng_state().clone()
        one = model.create_pretraining_model(tiny_design(), 0)
        two = model.create_pretraining_model(tiny_design(), 0)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))
        self.assertEqual(model.state_digest(one.state_dict()), model.state_digest(two.state_dict()))

    def test_classifier_parent_is_exact_and_trainable(self):
        lm = model.create_pretraining_model(tiny_design(), 0)
        before = model.state_digest(lm.transformer.state_dict())
        classifier = runtime.create_trainable_classifier(lm, 0)
        self.assertEqual(model.state_digest(classifier.backbone.state_dict()), before)
        self.assertTrue(all(p.requires_grad for p in classifier.parameters()))
        self.assertIsNone(classifier.score.bias)

    def test_complete_classifier_updates_and_safe_reload(self):
        classifier = runtime.create_trainable_classifier(model.create_pretraining_model(tiny_design(), 0), 0)
        before_head = model.state_digest({"score.weight": classifier.score.weight})
        before_backbone = model.state_digest(classifier.backbone.state_dict())
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=.01)
        ids = torch.tensor([[4, 2, 5, 1], [6, 2, 7, 1]])
        masks = torch.ones_like(ids)
        for _ in range(2):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(classifier(ids, masks), torch.tensor([0, 1])).backward()
            runtime.checked_gradient_norm(classifier, 1.)
            optimizer.step()
        self.assertNotEqual(model.state_digest({"score.weight": classifier.score.weight}), before_head)
        self.assertNotEqual(model.state_digest(classifier.backbone.state_dict()), before_backbone)
        job = {"condition": "EP", "pt_seed": 0, "ft_seed": 0, "smoke": True}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            cp = runtime.save_checkpoint(classifier, path, "source_classifier", {"job": job, "protocol_sha256": "a" * 64,
                                         "prepared_manifest_sha256": "b" * 64, "epoch": 1})
            kwargs = {"expected_sha256": cp["sha256"], "expected_kind": "source_classifier", "expected_job": job,
                      "expected_state_sha256": cp["state_sha256"], "expected_protocol_sha256": "a" * 64,
                      "expected_manifest_sha256": "b" * 64, "expected_epoch": 1}
            reloaded, _ = model.load_checkpoint(path, **kwargs)
            self.assertEqual(model.state_digest(reloaded.state_dict()), cp["state_sha256"])
            classifier.eval()
            with torch.inference_mode():
                torch.testing.assert_close(classifier(ids, masks), reloaded(ids, masks), rtol=0, atol=0)
            kwargs["expected_manifest_sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "manifest"):
                model.load_checkpoint(path, **kwargs)

    def test_gradient_nonfinite_rejected(self):
        parameter = torch.nn.Linear(2, 1)
        parameter.weight.grad = torch.full_like(parameter.weight, float("nan"))
        with self.assertRaises(RuntimeError):
            runtime.checked_gradient_norm(parameter, 1.)


if __name__ == "__main__":
    unittest.main()
