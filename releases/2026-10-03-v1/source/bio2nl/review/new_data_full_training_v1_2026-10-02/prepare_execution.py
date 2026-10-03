"""Freeze the user-authorized fresh 9+27 source-only experiment and serial plan."""
from datetime import datetime, timezone
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

R = Path(__file__).absolute().parent
W = R.parents[2]
P = W / "bio2nl/new_data_full_training_v1"
PRIOR = W / "bio2nl/review/new_data_training_smoke_v1_2026-10-01"
D = W / "data_rebuild/2026-10-01-portable-v2"
A = W / "release_staging/m1_m3_2026-09-30_v1"
SNAP = R / "code_snapshot"


def identity(path):
    path = Path(path).absolute()
    assert path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
    before = path.stat()
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    return {"sha256": value, "bytes": after.st_size}


def descriptor(path):
    return {"file": str(path), **identity(path)}


def exclusive(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main():
    assert sys.flags.optimize == 0
    assert not SNAP.exists() and not (R / "protocol.json").exists() and not (R / "results").exists()
    evidence_names = ("input_revalidation", "cpu_tests", "runtime_independent_review",
                      "queue_independent_review", "audits_independent_review",
                      "surface_independent_review", "producer_independent_review")
    evidence = {name: descriptor(R / "verification" / (name + ".json")) for name in evidence_names}
    for item in evidence.values():
        record = json.loads(Path(item["file"]).read_text())
        assert record["status"] == "passed"
        for field in ("verified_files", "codepins", "reviewed_files"):
            for path, pin in record.get(field, {}).items():
                assert identity(path) == {k: pin[k] for k in ("sha256", "bytes")}, path
    prepared_root = PRIOR / "prepared"
    prepared_pin = descriptor(prepared_root / "manifest.json")
    assert prepared_pin["sha256"] == "2ea23ffd2793ed2142156f2942cc4bfb77767a0c3e87ba668a77c34a3a16c010"
    prepared = json.loads((prepared_root / "manifest.json").read_text())
    assert prepared["source_counts"] == {"train": 8044, "validation": 20276}
    for relative, pin in prepared["outputs"].items():
        assert identity(prepared_root / relative) == pin
    raw_pin = descriptor(D / "manifest.json")
    raw_gate = descriptor(D / "portable_validation/acceptance.json")
    assert raw_pin["sha256"] == "c65ad951dced3285856f8edf49618e3d6b5329fc4ad006520dab748d35b407bf"
    assert raw_gate["sha256"] == "dc35bd6dd36566fef25120434419ab2b358f204678b5d83f7daa89903f5be3f2"
    derived_gate = descriptor(PRIOR / "data_acceptance.json")
    assert derived_gate["sha256"] == "395752ab8a836d1d82c829d60c37b189071e5b7b6a023bbad3bc29b3d5a97043"
    archive = descriptor(A / "archive_manifest.json")
    assert archive["sha256"] == "31db85eb9e88e0b12d5ed68cd0ea08e276bd509d2565fd11723ba192007aa44f"
    member = "bio2nl/review/joint_pretraining_v1_2026-09-29/protocol.json"
    archive_entries = {x["path"]: x for x in json.loads((A / "archive_manifest.json").read_text())["files"]}
    numerical = descriptor(A / member)
    assert {k: numerical[k] for k in ("sha256", "bytes")} == {k: archive_entries[member][k] for k in ("sha256", "bytes")}
    old = json.loads((A / member).read_text())
    design = {key: copy.deepcopy(old[key]) for key in
              ("pretraining", "source_sft", "surface_reference", "statistics", "evaluation", "retention",
               "constant_references", "conditions", "primary_contrast", "comparison_caveats")}
    design["tokenizer"] = {key: old["tokenizer"][key] for key in
                           ("vocab_size", "pad_token_id", "eos_token_id", "pair_separator_id", "unk_token_id", "sha256")}
    design["tokenizer"].update(source_release_id=D.name, rebuilt_in_raw_release=True,
                                retrained_for_this_experiment=False, new_blind_target_claimed=False)
    assert design["tokenizer"]["sha256"] == prepared["tokenizer"]["sha256"]
    design["source_sft"].update(new_full_source_counts=prepared["source_counts"], steps_per_epoch=252,
                                updates_per_fit=1260, presentations_per_fit=40220,
                                candidate_updates_per_fit=1260, candidate_presentations_per_fit=40220)
    assert design["pretraining"]["expected_runs"] == 9 and design["source_sft"]["expected_fits"] == 27
    assert design["pretraining"]["seeds"] == design["source_sft"]["fine_tuning_seeds"] == [0, 1, 2]
    prior_names = {"execution_protocol": PRIOR / "execution_protocol.json",
                   "original_failed_execution": PRIOR / "execution_status.json",
                   "corrected_review_protocol": PRIOR / "reload_review_v2/review_protocol.json",
                   "corrected_review_result": PRIOR / "reload_review_v2/review_result.json",
                   "corrected_reload_audit": PRIOR / "reload_review_v2/results/metadata.json",
                   "corrected_completion_review": PRIOR / "reload_review_v2/completion_review.json"}
    prior_evidence = {name: descriptor(path) for name, path in prior_names.items()}
    sources = {path.name: path for path in sorted(P.glob("*.py"))}
    assert {"runtime.py", "runtime_data.py", "model.py", "fit_surface.py", "audit_surface.py",
            "surface_features.py", "audit_runtime.py", "audit_source.py"} <= set(sources)
    sources.update({"run_queue.py": R / "run_queue.py", "accept_training.py": R / "accept_training.py"})
    source_pins = {name: descriptor(path) for name, path in sources.items()}
    assert source_pins["model.py"]["sha256"] == "b2362edf8850fe20fbd23cc958a71f92ac2a187de4316f53f8750d3bca857592"
    payloads = {name: path.read_bytes() for name, path in sources.items()}
    for name, payload in payloads.items():
        assert hashlib.sha256(payload).hexdigest() == source_pins[name]["sha256"]
        compile(payload, name, "exec")
    SNAP.mkdir()
    for name, payload in payloads.items():
        with (SNAP / name).open("xb") as stream:
            stream.write(payload)
    code = {str(SNAP / name): identity(SNAP / name) for name in sources}
    gpu_policy = {"max_samples": 16, "interval_seconds": 1, "memory_used_mib_below": 500,
                  "utilization_percent_at_most": 10, "requires_no_compute_processes": True,
                  "fresh_before_each_gpu_job": True}
    protocol = {
        "schema_version": 1, "status": "frozen_for_new_data_full_training",
        "scope": "source_only_new_data_full_training", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_training_authorized": True, "bounded_smoke_training_enabled": True,
        "full_budget_training_enabled": True, "target_scoring_enabled": False,
        "source_test_scoring_enabled": False, "old_weights_allowed": False,
        "remote_publication_enabled": False, "new_blind_confirmation_claimed": False,
        "user_authorization": "2026-10-02 user approved the next step after the explicit full retraining proposal.",
        "output_root": str(R / "results"), "training_acceptance_file": str(R / "training_acceptance.json"),
        "environment": {"python": sys.version, "packages": {name: importlib.metadata.version(name)
                         for name in ("numpy", "torch", "transformers", "tokenizers", "scikit-learn", "scipy", "pyarrow")}},
        "runtime_code": code, "preparation_evidence": evidence, "prior_smoke_evidence": prior_evidence,
        "raw_release": {"root": str(D), "manifest": raw_pin, "acceptance": raw_gate},
        "prepared": {"root": str(prepared_root), "manifest": prepared_pin, "acceptance": derived_gate},
        "design": design, "numerical_design_metadata_source": numerical, "numerical_design_archive_manifest": archive,
        "preparation_code": descriptor(Path(__file__).absolute()),
        "statistical_plan": descriptor(R / "STATISTICAL_PLAN.md"),
        "execution_protocol_document": descriptor(R / "EXECUTION_PROTOCOL.md"),
        "source_prefixes": prepared["source_prefix64"], "gpu_availability_policy": gpu_policy,
        "local_device": 0, "serial_only": True, "required_free_disk_gib": 8,
        "storage_policy": {"minimum_projected_peak_free_gib": 8, "checkpoint_size_multiplier": 1.05,
                           "metadata_and_logs_allowance_gib": 2, "prediction_header_bytes_per_file": 4096,
                           "maximum_atomic_checkpoint_copies": 1},
        "current_code_smoke": {"pretraining_jobs": 3, "source_sft_jobs": 1, "optimizer_updates_total": 8,
                               "checkpoint_replays": 4, "lm_rows_each": 2, "source_rows_each_role": 64,
                               "absolute_replay_tolerance": 1e-5, "lm_loss_layout": "FP32_channel_CE_B_V_T"},
        "formal_budget": {"pretraining_runs": 9, "source_sft_fits": 27, "surface_cpu_fits": 5,
                          "pretraining_updates_each": 1024, "source_sft_updates_each": 1260,
                          "source_sft_presentations_each": 40220, "source_epoch_prediction_artifacts": 270,
                          "source_epoch_prediction_rows": 3823200, "complete_retained_models": 36},
    }
    for name, source in sources.items():
        assert identity(source) == {k: source_pins[name][k] for k in ("sha256", "bytes")}
    for item in [*evidence.values(), *prior_evidence.values(), raw_pin, raw_gate, prepared_pin, derived_gate, numerical, archive]:
        assert identity(item["file"]) == {k: item[k] for k in ("sha256", "bytes")}
    exclusive(R / "protocol.json", protocol)
    protocol_pin = descriptor(R / "protocol.json")
    base = ["--protocol", str(R / "protocol.json"), "--protocol-sha256", protocol_pin["sha256"]]
    jobs = []

    def job(name, script, args, relative_result, result_status, gpu=False, true=(), false=()):
        jobs.append({"id": name, "command": [sys.executable, "-B", str(SNAP / script), *base, *args],
                     "cwd": str(R / "workers" / name), "log": str(R / "logs" / (name + ".log")),
                     "gpu": gpu, "result_file": str(R / relative_result), "result_status": result_status,
                     "required_true_flags": list(true), "required_false_flags": list(false),
                     "additional_peak_disk_gib": 2 if gpu else 0.25})

    job("surface_fit", "fit_surface.py", [], "results/surface/status.json", "completed")
    job("surface_audit", "audit_surface.py", [], "verification/surface_audit.json", "passed")
    for condition in ("EP", "ES", "EE"):
        job("smoke_pt_" + condition, "runtime.py",
            ["--job", "pretrain", "--condition", condition, "--pt-seed", "0", "--run-kind", "smoke"],
            f"results/smoke/pretrain/{condition}/pt0/metadata.json", "completed", True,
            true=("source_only",), false=("old_weights_loaded",))
    job("smoke_sft_EP", "runtime.py",
        ["--job", "sft", "--condition", "EP", "--pt-seed", "0", "--ft-seed", "0", "--run-kind", "smoke"],
        "results/smoke/source/EP/pt0/ft0/metadata.json", "completed", True,
        true=("source_only",), false=("old_weights_loaded",))
    job("smoke_audit", "audit_runtime.py", [], "results/smoke/audit_runtime/metadata.json", "completed", True,
        true=("all_new_checkpoint_replays_passed",), false=("old_weights_loaded",))
    job("accept_training", "accept_training.py", [], "training_acceptance.json", "accepted_for_source_training",
        true=("all_checks_passed", "full_budget_training_enabled"),
        false=("target_scoring_enabled", "source_test_scoring_enabled"))
    for seed in range(3):
        for condition in ("EP", "ES", "EE"):
            job(f"pt_{condition}_s{seed}", "runtime.py",
                ["--job", "pretrain", "--condition", condition, "--pt-seed", str(seed), "--run-kind", "full"],
                f"results/full/pretrain/{condition}/pt{seed}/metadata.json", "completed", True,
                true=("source_only",), false=("old_weights_loaded",))
    for seed in range(3):
        for condition in ("EP", "ES", "EE"):
            for ft in range(3):
                job(f"sft_{condition}_p{seed}_f{ft}", "runtime.py",
                    ["--job", "sft", "--condition", condition, "--pt-seed", str(seed), "--ft-seed", str(ft), "--run-kind", "full"],
                    f"results/full/source/{condition}/pt{seed}/ft{ft}/metadata.json", "completed", True,
                    true=("source_only", "only_source_validation_selected_checkpoint"), false=("old_weights_loaded",))
    job("audit_source", "audit_source.py", [], "results/report/audit.json", "passed",
        true=("all_read_files_final_rehashed",), false=("target_scoring_enabled", "source_test_scoring_enabled"))
    assert len(jobs) == 45 and sum(j["gpu"] for j in jobs) == 41
    pins = {**code, str(R / "protocol.json"): identity(R / "protocol.json")}
    for item in [*evidence.values(), *prior_evidence.values(), raw_pin, raw_gate, prepared_pin, derived_gate,
                 numerical, archive, protocol["preparation_code"], protocol["statistical_plan"],
                 protocol["execution_protocol_document"]]:
        pins[item["file"]] = {k: item[k] for k in ("sha256", "bytes")}
    plan = {"schema_version": 1, "scope": "source_only_new_data_full_training",
            "review_root": str(R), "status_file": str(R / "execution_status.json"),
            "workspace_lock": str(W / ".bio2nl_gpu_training_queue.lock"), "pins": pins, "jobs": jobs,
            "gpu_availability_policy": gpu_policy, "required_free_disk_gib": 8,
            "full_budget_training_enabled": True, "target_scoring_enabled": False, "source_test_scoring_enabled": False}
    exclusive(R / "execution_plan.json", plan)
    exclusive(R / "execution_freeze.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": protocol_pin, "plan": descriptor(R / "execution_plan.json"), "source_pins": source_pins,
              "stages": 45, "full_pretraining_runs": 9, "full_source_fits": 27,
              "formal_optimizer_updates": 43236, "additional_technical_smoke_updates": 8,
              "source_test_or_target_scoring_authorized": False})
    (R / "logs").mkdir()
    (R / "workers").mkdir()
    print(json.dumps({"status": "frozen_not_launched", "protocol": protocol_pin,
                      "plan": descriptor(R / "execution_plan.json"), "stages": len(jobs)}, indent=2))


if __name__ == "__main__":
    main()
