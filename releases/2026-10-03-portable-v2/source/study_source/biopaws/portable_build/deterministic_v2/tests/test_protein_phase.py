import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

HERE = Path(__file__).parents[1]
WORKSPACE = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


patches = load("v2_patches", HERE / "patches.py")
phase = load("v2_phase", HERE / "protein_phase.py")
previous = load("v1_test_helpers", HERE.parent / "deterministic_v1/tests/test_replay_pairs.py")
canonical = load("canonical_patch", WORKSPACE / "bio2nl/portable_data/deterministic_v1/patches.py")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def transformed(relative, raw):
    if relative == "data_v2/build_protein.py":
        return canonical.patch_source("code/biopaws/" + relative, raw)[0]
    if relative == "data_v2/build_pairs.py":
        return patches.patch_pair_source(raw)[0]
    if relative == "data_v2_remote/build_remote.py":
        return patches.patch_remote_source(raw)[0]
    return raw


@pytest.fixture
def bound(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    builders = root / "execution_code/biopaws"
    bootstrap_files = {}
    files = {}
    for relative in phase.BUILDERS:
        source = root / "code/biopaws" / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes((WORKSPACE / "biopaws" / relative).read_bytes())
        snapshot = builders / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(transformed(relative, source.read_bytes()))
        name = "code/biopaws/" + relative
        bootstrap_files[name] = phase.identity(source)
        files[name] = {"source": str(source), "source_sha256": phase.digest(source),
            "source_bytes": source.stat().st_size, "snapshot": str(snapshot),
            **phase.identity(snapshot), "transform": phase.TRANSFORMS.get(relative, "identity")}
    for name in phase.INITIAL_INPUTS:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((name + " fixture\n").encode())
        bootstrap_files[name] = phase.identity(path)
    bootstrap = root / "portable_validation/bootstrap.json"
    write_json(bootstrap, {"schema_version": 1, "output_root": str(root),
        "status": "raw_assets_seeded_not_a_data_acceptance", "training_enabled": False,
        "target_scoring_enabled": False, "prepared_data_copied": False,
        "work_products_copied": False, "files": bootstrap_files})
    manifest = tmp_path / "builders.json"
    write_json(manifest, {"schema_version": 1, "release_id": phase.RELEASE_ID,
        "status": "execution_code_overlay_not_data_acceptance", "training_enabled": False,
        "target_scoring_enabled": False, "release_root": str(root),
        "builder_root": str(root / "execution_code"), "bootstrap_sha256": phase.digest(bootstrap), "files": files})
    return root, builders, tmp_path / "logs", manifest


def test_source_edits_are_exact_and_reject_reapplication():
    for relative, patch in (("data_v2/build_pairs.py", patches.patch_pair_source),
                            ("data_v2_remote/build_remote.py", patches.patch_remote_source)):
        raw = (WORKSPACE / "biopaws" / relative).read_bytes()
        result, report = patch(raw)
        assert {"sha256": hashlib.sha256(result).hexdigest(), "bytes": len(result)} == phase.BUILDERS[relative]
        assert report["original_file_modified"] is False and report["scientific_thresholds_changed"] is False
        assert report["release_id"] == phase.RELEASE_ID and report["original_algorithm_release_id"] == "2026-09-25-v2"
        with pytest.raises(ValueError, match="Source SHA"):
            patch(result)
        with pytest.raises(ValueError, match="Source SHA"):
            patch(raw + b"\n")
        if relative.endswith("build_pairs.py"):
            reverted = result.decode().replace(patches.ORDER_INSERT, "", 1).replace(patches.PAIR_ID_AFTER, patches.PAIR_ID_BEFORE, 1)
        else:
            reverted = result.decode().replace(patches.REMOTE_ID_AFTER, patches.REMOTE_ID_BEFORE, 1)
        assert reverted.encode() == raw


def run_pairs(builders, table, rows, root, seed):
    for name in ("data/sequences", "data/pairs", "work/protein"):
        (root / name).mkdir(parents=True)
    (root / "data/sequences/protein_canonical_sequences.tsv.gz").write_bytes(table.read_bytes())
    candidates = root / "work/protein/positive_candidates.jsonl"
    previous.write_candidates(candidates, rows)
    result = subprocess.run([sys.executable, "-s", "-B", str(builders / "data_v2/build_pairs.py"),
        "--root", str(root), "--seed", "20260925"], env=dict(os.environ, PYTHONHASHSEED=str(seed), CUDA_VISIBLE_DEVICES=""),
        capture_output=True, text=True)
    return result


def test_actual_original_algorithm_matches_v1_canonical_replay(bound, tmp_path):
    _, builders, _, _ = bound
    fixture = tmp_path / "fixture"; fixture.mkdir()
    table, source, rows = previous.inputs.__wrapped__(fixture)
    # Preserve a representable score beyond an abbreviated printed decimal.
    rows[0]["identity"] = .6000000000000001
    previous.write_candidates(source, rows)
    expected_sorted = tmp_path / "expected_sorted.jsonl"
    order = previous.adapter.canonical_sort(source, expected_sorted, previous.adapter.canonical_table(table),
        previous.digest(source), buffer_bytes=80, fanin=2)
    roots = [tmp_path / "actual_a", tmp_path / "actual_b", tmp_path / "v1_control"]
    permutations = [rows, random.Random(39).sample(rows, len(rows))]
    for root, values, seed in zip(roots, permutations, (0, 12345)):
        completed = run_pairs(builders, table, values, root, seed)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        sorted_file = root / "work/protein/positive_candidates_sorted.jsonl"
        assert sorted_file.read_bytes() == expected_sorted.read_bytes()
        metadata = json.loads((root / "metadata/protein_candidate_order.json").read_text())
        assert metadata["canonical_sorted_jsonl_sha256"] == order["sorted_candidate_sha256"]
        assert metadata["records"] == 27 and metadata["score_values_changed"] is False
        assert metadata["python_hash_seed"] == str(seed)
        assert metadata["hash_randomization"] == int(seed != 0)
        assert metadata["input_candidate_sha256"] == phase.digest(root / "work/protein/positive_candidates.jsonl")
        assert metadata["sorted_candidate_bytes"] == expected_sorted.stat().st_size
        manifest = json.loads((root / "metadata/protein_pair_manifest.json").read_text())
        assert manifest["release_id"] == phase.RELEASE_ID and manifest["original_algorithm_release_id"] == "2026-09-25-v2"
    control = run_pairs(previous.BUILDER.parent.parent, table,
        [json.loads(line) for line in expected_sorted.read_text().splitlines()], roots[2], 0)
    assert control.returncode == 0, control.stdout + control.stderr
    for name in previous.adapter.PAIR_FILES:
        decoded = [gzip.decompress((root / name).read_bytes()) for root in roots]
        assert decoded[0] == decoded[1] == decoded[2]
    a = json.loads((roots[0] / "metadata/protein_candidate_order.json").read_text())
    b = json.loads((roots[1] / "metadata/protein_candidate_order.json").read_text())
    assert a["hash_probe"] != b["hash_probe"]
    pair_stage = phase.stage_plan(roots[0], builders, sys.executable)[6]
    phase.validate_pair_metadata(roots[0], phase.inventory(roots[0], pair_stage["outputs"]))


@pytest.mark.parametrize("change", ["duplicate", "conflicting_duplicate", "extra_field", "nan", "bad_split", "bad_score"])
def test_candidate_anomalies_fail_without_emitting_pairs(bound, tmp_path, change):
    _, builders, _, _ = bound
    fixture = tmp_path / "fixture"; fixture.mkdir()
    table, source, rows = previous.inputs.__wrapped__(fixture)
    if change == "duplicate": rows.append(dict(rows[0]))
    elif change == "conflicting_duplicate": rows.append({**rows[0], "identity": .7})
    elif change == "extra_field": rows[0]["surprise"] = 1
    elif change == "nan": rows[0]["identity"] = float("nan")
    elif change == "bad_split": rows[0]["split"] = "unknown"
    elif change == "bad_score": rows[0]["identity"] = .99
    root = tmp_path / "run"
    result = run_pairs(builders, table, rows, root, 0)
    assert result.returncode != 0
    assert not list((root / "data/pairs").iterdir())
    assert not (root / "metadata/protein_pair_manifest.json").exists()


@pytest.mark.parametrize("change", ["manifest_hash", "root", "snapshot", "source", "bootstrap", "raw", "transform", "extra_code", "code_symlink"])
def test_explicit_code_bootstrap_and_root_pins_fail_closed(bound, change):
    root, builders, logs, manifest = bound
    sha = phase.digest(manifest)
    doc = phase.read_json(manifest)
    if change == "manifest_hash": sha = "0" * 64
    elif change == "root": doc["release_root"] += "wrong"
    elif change == "snapshot": (builders / "data_v2/build_pairs.py").write_text("raise Exception()")
    elif change == "source": (root / "code/biopaws/data_v2/build_pairs.py").write_text("bad")
    elif change == "bootstrap": (root / "portable_validation/bootstrap.json").write_text("{}")
    elif change == "raw": (root / phase.INITIAL_INPUTS[0]).write_text("changed")
    elif change == "transform": doc["files"]["code/biopaws/data_v2/build_pairs.py"]["transform"] = "identity"
    elif change == "extra_code": (builders / "injected.py").write_text("bad")
    elif change == "code_symlink": (builders / "__pycache__").symlink_to(root, target_is_directory=True)
    if change in ("root", "transform"):
        write_json(manifest, doc); sha = phase.digest(manifest)
    with pytest.raises(ValueError):
        phase.preflight(root, builders, logs, phase.stage_plan(root, builders, sys.executable), manifest, sha)
    assert not logs.exists()


@pytest.mark.parametrize("change", ["output", "log", "marker", "parent_symlink"])
def test_fresh_output_and_no_symlink_paths(bound, change):
    root, builders, logs, manifest = bound
    if change == "output":
        target = root / "data/pairs/unrelated.txt"; target.parent.mkdir(parents=True); target.write_text("keep")
    elif change == "log": logs.mkdir(); (logs / "old.log").write_text("keep")
    elif change == "marker": (root / ".protein_phase_started.json").write_text("keep")
    elif change == "parent_symlink":
        (root / "work").mkdir(); (root / "work/protein").symlink_to(builders, target_is_directory=True)
    with pytest.raises(ValueError):
        phase.preflight(root, builders, logs, phase.stage_plan(root, builders, sys.executable), manifest, phase.digest(manifest))


@pytest.mark.parametrize("failure", ["exit", "missing_output", "mutated_input", "mutated_code", "late_target"])
def test_serial_queue_stops_first_failure_and_preserves_log(bound, failure):
    root, builders, logs, manifest = bound
    stages = [{"name": "tiny_a", "script": "data_v2/build_protein.py", "command": ["fixture", "a"],
               "inputs": [phase.INITIAL_INPUTS[0]], "outputs": ["metadata/tiny_a.json"]},
              {"name": "tiny_b", "script": "data_v2/build_protein.py", "command": ["fixture", "b"],
               "inputs": ["metadata/tiny_a.json"], "outputs": ["metadata/tiny_b.json"]}]
    sha = phase.digest(manifest)
    binding = phase.preflight(root, builders, logs, stages, manifest, sha)
    calls = []
    def worker(command, **kwargs):
        calls.append(command)
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "" and kwargs["env"]["PYTHONHASHSEED"] == "0"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "16" and "PYTHONPATH" not in kwargs["env"]
        kwargs["stdout"].write("fixture log\n")
        if failure == "exit": return subprocess.CompletedProcess(command, 9)
        if failure != "missing_output": (root / "metadata/tiny_a.json").write_text("{}")
        if failure == "mutated_input": (root / phase.INITIAL_INPUTS[0]).write_text("changed")
        if failure == "mutated_code": (builders / "data_v2/build_protein.py").write_text("changed")
        if failure == "late_target": (root / "metadata/tiny_b.json").write_text("existing")
        return subprocess.CompletedProcess(command, 0)
    with pytest.raises(ValueError):
        phase.execute_stages(root, builders, logs, stages, manifest, sha, binding, run_command=worker)
    status = phase.read_json(logs / "protein_phase_status.json")
    assert len(calls) == 1 and status["status"] == "failed"
    assert status["builder_manifest_sha256"] == sha and status["bootstrap_sha256"] == binding["bootstrap_sha256"]
    assert not status["training_enabled"] and not status["complete_release_gate_passed"]
    assert (logs / "01_tiny_a.log").read_text() == "fixture log\n"


def test_success_status_is_not_release_gate(bound):
    root, builders, logs, manifest = bound
    stages = [{"name": "fixture", "script": "data_v2/build_protein.py", "command": ["fixture"],
               "inputs": [phase.INITIAL_INPUTS[0]], "outputs": ["metadata/tiny.json"]}]
    sha = phase.digest(manifest)
    binding = phase.preflight(root, builders, logs, stages, manifest, sha)
    def worker(command, **kwargs):
        (root / "metadata/tiny.json").write_text("{}")
        return subprocess.CompletedProcess(command, 0)
    status = phase.execute_stages(root, builders, logs, stages, manifest, sha, binding, run_command=worker)
    assert status["status"] == "completed" and status["stages"][0]["returncode"] == 0
    assert status["stages"][0]["outputs"]["metadata/tiny.json"] == phase.identity(root / "metadata/tiny.json")
    assert status["complete_release_gate_passed"] is False and status["training_enabled"] is False


@pytest.mark.parametrize("change", [None, "stale_size", "stale_sha", "duplicate", "code", "old_id", "source_omitted"])
def test_remote_descriptors_are_current_without_legacy_exception(bound, change):
    root, builders, _, _ = bound
    source = root / "metadata/remote_sources.json"
    validation = root / "validation/remote_validation.json"
    write_json(validation, {"status": "passed_local_data_checks"})
    manifest = {"release_id": phase.RELEASE_ID, "original_algorithm_release_id": "2026-09-25-v2",
        "source_record": "metadata/remote_sources.json", "files": [{"file": "metadata/remote_sources.json", **phase.identity(source)}],
        "code": {"file": str(builders / "data_v2_remote/build_remote.py"),
                 "sha256": phase.digest(builders / "data_v2_remote/build_remote.py")}}
    if change == "stale_size": manifest["files"][0]["bytes"] += 1
    elif change == "stale_sha": manifest["files"][0]["sha256"] = "0" * 64
    elif change == "duplicate": manifest["files"].append(dict(manifest["files"][0]))
    elif change == "code": manifest["code"]["sha256"] = "0" * 64
    elif change == "old_id": manifest["release_id"] = "2026-09-25-v2"
    elif change == "source_omitted": manifest["files"] = []
    write_json(root / "metadata/remote_manifest.json", manifest)
    if change:
        with pytest.raises(ValueError): phase.verify_current_remote_manifest(root, builders)
    else:
        result = phase.verify_current_remote_manifest(root, builders)
        assert result["file_count"] == 1 and result["training_enabled"] is False
