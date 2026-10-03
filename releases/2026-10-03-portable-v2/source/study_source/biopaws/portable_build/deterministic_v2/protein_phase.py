#!/usr/bin/env python3
"""Nine CPU protein stages in a fresh release, using an explicit source overlay.

The full release verifier is a separate operation. Stage success never enables
training or claims that the old accepted release was byte-for-byte reproduced.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

RELEASE_ID = "2026-10-01-portable-v2"
BUILDERS = {'data_v2/build_pairs.py': {'sha256': '0afe6d2b08685f530c2e70b775eb84d787db268e0a1461485f693ec53d095626', 'bytes': 18924}, 'data_v2/build_protein.py': {'sha256': '9251f1b0db4814b06f49e449358eb970a9db661a3a02be30a9ea0dbbebd0f2ce', 'bytes': 12078}, 'data_v2/cross_dataset_homology_audit.py': {'sha256': 'c391e7d3f222f3c829dcbc470585891bea7e2db9acd6f6fe4d98710b0055111d', 'bytes': 5591}, 'data_v2/exclude_remote_homologs.py': {'sha256': 'c063a323bd28671a1e6e5998ff119b9ca88b7ac38193b1ffa9c39fedb64b3a8a', 'bytes': 5610}, 'data_v2/fetch_sources.py': {'sha256': '58f4220207b0aa204b4c635224ef0bd52727ddcdc97c652a06e5bca4aa4384ab', 'bytes': 2439}, 'data_v2/screen_remote.py': {'sha256': 'c308059594c1c50b531c08110773bc9f452f5312e8f4babff92e82830f39f6fe', 'bytes': 3369}, 'data_v2_remote/build_remote.py': {'sha256': '6958fe3f26c54dca0b52d479732c5d161a8e9d6c06ae84e116ee218f8e77e9d6', 'bytes': 27740}, 'data_v2_remote/download_scope.py': {'sha256': '9e52692bf7a9c34c793b2c309b901008d93d780d25f482fb4ce7feacbf7bc115', 'bytes': 6336}}

INITIAL_INPUTS = (
    "raw/protein/uniprot_sprot_2026_03.fasta.gz",
    "raw/protein/reldate_2026_03.txt",
    "raw/protein/release_verified_download.json",
    "tools/protein/mmseqs-linux-avx2.tar.gz",
    "tools/protein/mmseqs/bin/mmseqs",
    "raw/remote/astral-scopedom-seqres-gd-all-2.08-stable.fa",
    "raw/remote/dir.cla.scope.2.08-stable.txt",
    "metadata/remote_sources.json",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def safe_path(value, kind=None):
    path = Path(os.path.abspath(value))
    require(not any(p.is_symlink() for p in (path, *path.parents)), f"Symlinked path: {path}")
    if kind == "file": require(path.is_file(), f"Missing regular file: {path}")
    if kind == "dir": require(path.is_dir(), f"Missing directory: {path}")
    return path

def digest(path):
    with safe_path(path, "file").open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def identity(path):
    path = safe_path(path, "file")
    return {"sha256": digest(path), "bytes": path.stat().st_size}

def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result

def read_json(path):
    return json.loads(safe_path(path, "file").read_bytes(), object_pairs_hook=unique_keys)

def atomic_json(path, value):
    path = safe_path(path)
    temporary = safe_path(path.with_name(path.name + ".tmp"))
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)

def member(root, relative, kind="file"):
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and ".." not in rel.parts and str(rel) == relative,
            "Unsafe member: " + relative)
    path = safe_path(root / relative, kind)
    require(path.is_relative_to(root), "Member escaped root")
    return path

def inventory(root, patterns):
    result = {}
    for pattern in patterns:
        require(not PurePosixPath(pattern).is_absolute() and ".." not in PurePosixPath(pattern).parts,
                "Unsafe output pattern")
        paths = sorted(root.glob(pattern))
        require(paths, "Required artifact pattern has no matches: " + pattern)
        for path in paths:
            name = path.relative_to(root).as_posix()
            result[name] = identity(member(root, name))
    return result

def stage_plan(root, builders, python):
    def stage(name, script, arguments, inputs, outputs):
        return {"name": name, "script": script,
                "command": [str(python), "-s", "-B", str(builders / script), *arguments],
                "inputs": list(inputs), "outputs": list(outputs)}
    root_arg = ["--root", str(root)]
    remote_arg = ["--release-root", str(root), "--workers", "16"]
    return [
        stage("normalize", "data_v2/build_protein.py", [*root_arg, "normalize"],
              INITIAL_INPUTS[:5], ["data/sequences/protein_canonical_base.tsv.gz",
              "data/sequences/protein_source_accessions.tsv.gz", "data/sequences/protein_canonical_all.fasta",
              "data/sequences/protein_candidates_40_250.fasta", "metadata/protein_sources.json"]),
        stage("cluster", "data_v2/build_protein.py", [*root_arg, "cluster"],
              ["data/sequences/protein_canonical_all.fasta"],
              ["work/protein/full_cluster.tsv", "metadata/protein_full_cluster_command.json"]),
        stage("search", "data_v2/build_protein.py", [*root_arg, "search"],
              ["data/sequences/protein_candidates_40_250.fasta"],
              ["work/protein/short_alignments.tsv", "metadata/protein_short_search_command.json"]),
        stage("graph_split", "data_v2/build_protein.py", [*root_arg, "graph_split"],
              ["data/sequences/protein_canonical_base.tsv.gz", "data/sequences/protein_source_accessions.tsv.gz",
               "work/protein/full_cluster.tsv", "work/protein/short_alignments.tsv"],
              ["data/sequences/protein_canonical_sequences.tsv.gz", "data/sequences/protein_accession_splits.tsv.gz",
               "work/protein/positive_candidates.jsonl", "metadata/protein_split_policy.json"]),
        stage("remote_prepare", "data_v2_remote/build_remote.py", ["prepare", *remote_arg],
              INITIAL_INPUTS[5:7], ["data/sequences/remote_sequences.parquet", "data/sequences/remote_sequences.tsv.gz",
              "data/sequences/remote_domains.fasta", "metadata/remote_superfamily_splits.tsv",
              "metadata/remote_preparation.json", "metadata/remote_build_config.json"]),
        stage("remote_exclusion", "data_v2/exclude_remote_homologs.py", [*root_arg, "--threads", "16"],
              ["data/sequences/remote_domains.fasta", "data/sequences/protein_canonical_base.tsv.gz",
               "data/sequences/protein_canonical_all.fasta", "data/sequences/protein_canonical_sequences.tsv.gz",
               "data/sequences/protein_accession_splits.tsv.gz"],
              ["data/sequences/protein_canonical_base.tsv.gz", "data/sequences/protein_canonical_sequences.tsv.gz",
               "data/sequences/protein_accession_splits.tsv.gz", "work/protein/remote_exclusion/remote_vs_swissprot.tsv",
               "metadata/protein_remote_pretraining_exclusion.json"]),
        stage("protein_pairs", "data_v2/build_pairs.py", [*root_arg, "--seed", "20260925"],
              ["data/sequences/protein_canonical_sequences.tsv.gz", "work/protein/positive_candidates.jsonl"],
              ["data/pairs/protein_sequence_similarity_*.tsv.gz", "validation/protein_endpoint_degrees.tsv",
               "validation/protein_pair_baselines.csv", "validation/protein_pair_acceptance.json",
               "metadata/protein_pair_construction.json", "metadata/protein_pair_manifest.json",
               "work/protein/positive_candidates_sorted.jsonl", "metadata/protein_candidate_order.json"]),
        stage("remote_pairs", "data_v2_remote/build_remote.py", ["pairs", *remote_arg],
              ["data/sequences/remote_sequences.parquet", "metadata/remote_sources.json"],
              ["data/pairs/remote_pairs.parquet", "data/pairs/remote_*.csv.gz",
               "metadata/remote_positive_search.tsv", "metadata/remote_candidate_positive_edges.jsonl.gz",
               "metadata/remote_manifest.json", "validation/remote_endpoint_degrees.tsv",
               "validation/remote_baselines.csv", "validation/remote_validation.json"]),
        stage("cross_dataset_audit", "data_v2/cross_dataset_homology_audit.py", root_arg,
              ["work/protein/remote_exclusion/remote_vs_swissprot.tsv", "data/sequences/protein_canonical_sequences.tsv.gz",
               "raw/remote/astral-scopedom-seqres-gd-all-2.08-stable.fa",
               "data/pairs/protein_sequence_similarity_all.tsv.gz", "data/pairs/remote_pairs.parquet"],
              ["validation/cross_dataset_homology.json"]),
    ]


TRANSFORMS = {"data_v2/build_protein.py": "canonical_columns_deterministic_v1",
              "data_v2/build_pairs.py": "protein_deterministic_raw_v2",
              "data_v2_remote/build_remote.py": "remote_release_identity_v2"}
MUTATED_INPUTS = {"remote_exclusion": {"data/sequences/protein_canonical_base.tsv.gz",
    "data/sequences/protein_canonical_sequences.tsv.gz", "data/sequences/protein_accession_splits.tsv.gz"}}


def validate_binding(root, builders, manifest_path, manifest_sha256, *, verify_raw=True):
    """Bind the new copied sources and bootstrap without accepting arbitrary code."""
    require(digest(manifest_path) == manifest_sha256, "Builder manifest SHA mismatch")
    manifest = read_json(manifest_path)
    require(manifest["schema_version"] == 1 and manifest["release_id"] == RELEASE_ID,
            "Wrong builder manifest version/release")
    require(manifest["status"] == "execution_code_overlay_not_data_acceptance" and
            manifest["training_enabled"] is False and manifest["target_scoring_enabled"] is False,
            "Builder overlay cannot be a data/training gate")
    require(manifest["release_root"] == str(root) and manifest["builder_root"] == str(root / "execution_code"),
            "Builder manifest root mismatch")
    require(builders == root / "execution_code/biopaws", "Unexpected executable builder root")
    bootstrap_path = member(root, "portable_validation/bootstrap.json")
    require(digest(bootstrap_path) == manifest["bootstrap_sha256"], "Bootstrap SHA mismatch")
    bootstrap = read_json(bootstrap_path)
    require(bootstrap["schema_version"] == 1 and bootstrap["output_root"] == str(root) and
            bootstrap["status"] == "raw_assets_seeded_not_a_data_acceptance",
            "Bootstrap belongs to another root or stage")
    require(bootstrap["training_enabled"] is False and bootstrap["target_scoring_enabled"] is False and
            bootstrap["prepared_data_copied"] is False and bootstrap["work_products_copied"] is False,
            "Bootstrap must seed raw assets only")
    subset = {k.removeprefix("code/biopaws/"): v for k, v in manifest["files"].items()
              if k.startswith("code/biopaws/")}
    require(set(subset) == set(BUILDERS), "Missing/extra protein execution source descriptor")
    nodes = list(builders.rglob("*"))
    require(not any(p.is_symlink() for p in nodes), "Symlink in execution source tree")
    actual_files = {p.relative_to(builders).as_posix() for p in nodes if not p.is_dir()}
    require(actual_files == set(BUILDERS), "Unexpected execution source file")
    for relative, expected in BUILDERS.items():
        entry = subset[relative]
        source_name = "code/biopaws/" + relative
        source = member(root, source_name)
        snapshot = member(builders, relative)
        require(entry["source"] == str(source) and entry["snapshot"] == str(snapshot),
                "Source/snapshot path mismatch: " + relative)
        original = identity(source)
        require(original == {k: bootstrap["files"][source_name][k] for k in ("sha256", "bytes")} and
                original == {"sha256": entry["source_sha256"], "bytes": entry["source_bytes"]},
                "Original source/descriptor mismatch: " + relative)
        require(identity(snapshot) == expected == {k: entry[k] for k in ("sha256", "bytes")},
                "Execution source/descriptor mismatch: " + relative)
        require(entry["transform"] == TRANSFORMS.get(relative, "identity"), "Unexpected transform: " + relative)
        if relative not in TRANSFORMS:
            require(original == expected, "Unchanged helper differs from original")
    if verify_raw:
        for relative in INITIAL_INPUTS:
            expected = bootstrap["files"][relative]
            require(identity(member(root, relative)) == {k: expected[k] for k in ("sha256", "bytes")},
                    "Bootstrap raw input differs: " + relative)
    return manifest


def preflight(root, builders, logs, stages, manifest_path, manifest_sha256):
    binding = validate_binding(root, builders, manifest_path, manifest_sha256)
    require(not logs.is_relative_to(root / "execution_code") and not logs.is_relative_to(root / "code")
            and logs != root and not root.is_relative_to(logs), "Unsafe protein log root")
    for name in ("data/sequences", "data/pairs", "work/protein", "metadata", "validation"):
        target = member(root, name, None)
        if name in ("data/sequences", "data/pairs", "work/protein"):
            require(not target.exists() or (target.is_dir() and not any(target.iterdir())),
                    "Protein output directory must be new or empty: " + name)
    for stage in stages:
        for pattern in stage["outputs"]:
            require(not list(root.glob(pattern)), "Refusing existing stage artifact: " + pattern)
    require(not (root / ".protein_phase_started.json").exists(), "Protein phase already started")
    safe_path(root / ".protein_phase_started.json")
    require(not logs.exists() or (logs.is_dir() and not any(logs.iterdir())), "Fresh/empty protein log root required")
    for name in ("data/sequences", "data/pairs", "work/protein", "metadata", "validation"):
        (root / name).mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return binding


def stage_output_guard(root, stage):
    """Only remote exclusion may rewrite its three explicit current-run inputs."""
    allowed = MUTATED_INPUTS.get(stage["name"], set())
    for pattern in stage["outputs"]:
        # Also examine non-existent target ancestors, where glob is empty.
        safe_path(root / pattern)
        for path in root.glob(pattern):
            relative = path.relative_to(root).as_posix()
            member(root, relative)
            require(relative in allowed, "Refusing preexisting stage target: " + relative)


def validate_pair_metadata(root, outputs):
    manifest = read_json(root / "metadata/protein_pair_manifest.json")
    order = read_json(root / "metadata/protein_candidate_order.json")
    acceptance = read_json(root / "validation/protein_pair_acceptance.json")
    require(manifest["release_id"] == RELEASE_ID and
            manifest["original_algorithm_release_id"] == "2026-09-25-v2", "Wrong current pair identity")
    require(manifest["builder_sha256"] == BUILDERS["data_v2/build_pairs.py"]["sha256"], "Pair builder differs")
    require(manifest["acceptance"] == acceptance and acceptance["acceptance_pass"] is True,
            "Pair local acceptance failed")
    expected = {f"protein_sequence_similarity_{s}.tsv.gz": outputs[f"data/pairs/protein_sequence_similarity_{s}.tsv.gz"]["sha256"]
                for s in ("train", "validation", "test", "all")}
    require(manifest["data_files"] == expected, "Current pair descriptors differ")
    order_name = "metadata/protein_candidate_order.json"
    require(manifest["candidate_order_record"] == {"file": order_name, "sha256": outputs[order_name]["sha256"]},
            "Candidate order provenance differs")
    sorted_name = "work/protein/positive_candidates_sorted.jsonl"
    require(order["release_id"] == RELEASE_ID and order["seed"] == 20260925 and
            order["builder_sha256"] == manifest["builder_sha256"] and
            order["sorted_candidate_file"] == sorted_name and
            order["canonical_sorted_jsonl_sha256"] == outputs[sorted_name]["sha256"] and
            order["sorted_candidate_bytes"] == outputs[sorted_name]["bytes"] and
            order["input_candidate_sha256"] == digest(root / "work/protein/positive_candidates.jsonl"),
            "Candidate ordering identities differ")
    require(order["python_hash_seed"] == "0" and order["hash_randomization"] == 0,
            "Pair worker did not retain fixed hash seed")


def verify_current_remote_manifest(root, builders):
    """Every current descriptor must match its current file, without exceptions."""
    path = member(root, "metadata/remote_manifest.json")
    before = identity(path)
    manifest = read_json(path)
    require(manifest["release_id"] == RELEASE_ID and manifest["original_algorithm_release_id"] == "2026-09-25-v2",
            "Wrong current remote identity")
    require(manifest["source_record"] == "metadata/remote_sources.json", "Wrong remote source descriptor")
    entries = {}
    for entry in manifest["files"]:
        name = entry["file"]
        require(name not in entries, "Duplicate remote descriptor")
        require(identity(member(root, name)) == {k: entry[k] for k in ("sha256", "bytes")},
                "Current remote descriptor differs: " + name)
        entries[name] = {k: entry[k] for k in ("sha256", "bytes")}
    require("metadata/remote_sources.json" in entries, "Remote source descriptor absent")
    code = member(builders, "data_v2_remote/build_remote.py")
    require(manifest["code"]["file"] == str(code) and manifest["code"]["sha256"] == digest(code),
            "Remote code descriptor differs")
    for name, expected in entries.items():
        require(identity(member(root, name)) == expected, "Remote file changed during validation")
    require(identity(path) == before, "Remote manifest changed during validation")
    require(read_json(root / "validation/remote_validation.json")["status"] == "passed_local_data_checks",
            "Remote local validation failed")
    return {"status": "current_remote_references_verified_not_data_acceptance", "manifest": before,
            "files": entries, "file_count": len(entries), "training_enabled": False}


def execute_stages(root, builders, logs, stages, manifest_path, manifest_sha256, binding,
                   run_command=subprocess.run):
    status_path = logs / "protein_phase_status.json"
    status = {"schema_version": 1, "status": "running", "started_at_utc": utc_now(), "pid": os.getpid(),
        "release_id": RELEASE_ID, "release_root": str(root), "builder_root": str(builders),
        "builder_manifest_file": str(manifest_path), "builder_manifest_sha256": manifest_sha256,
        "bootstrap_sha256": binding["bootstrap_sha256"], "runner": {"file": str(Path(__file__).absolute()), **identity(Path(__file__))},
        "cpu_only": True, "threads": 16, "python_hash_seed": "0", "builders": BUILDERS,
        "stage_count": len(stages), "stages": [], "training_enabled": False, "target_scoring_enabled": False,
        "complete_release_gate_passed": False}
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="16", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONHASHSEED="0",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    atomic_json(status_path, status)
    try:
        for number, stage in enumerate(stages, 1):
            started = time.monotonic()
            validate_binding(root, builders, manifest_path, manifest_sha256, verify_raw=False)
            stage_output_guard(root, stage)
            log_path = logs / f"{number:02d}_{stage['name']}.log"
            record = {"name": stage["name"], "status": "running", "started_at_utc": utc_now(),
                      "command": stage["command"], "inputs": inventory(root, stage["inputs"]),
                      "builder_sha256": digest(member(builders, stage["script"])), "log": log_path.name}
            status["stages"].append(record)
            atomic_json(status_path, status)
            print(f"[{utc_now()}] Starting {number}/{len(stages)} {stage['name']}", flush=True)
            with log_path.open("x") as log:
                result = run_command(stage["command"], cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, check=False)
            record.update(returncode=result.returncode, ended_at_utc=utc_now(),
                          wall_seconds=time.monotonic() - started, log_sha256=digest(log_path))
            require(result.returncode == 0, f"Stage {stage['name']} returned {result.returncode}; see {log_path}")
            validate_binding(root, builders, manifest_path, manifest_sha256, verify_raw=False)
            # Raw and earlier outputs cannot be rewritten except those declared above.
            for name, expected in record["inputs"].items():
                if name not in MUTATED_INPUTS.get(stage["name"], set()):
                    require(identity(member(root, name)) == expected, "Stage modified an immutable input: " + name)
            record["outputs"] = inventory(root, stage["outputs"])
            if stage["name"] == "protein_pairs":
                validate_pair_metadata(root, record["outputs"])
            if stage["name"] == "remote_pairs":
                record["current_remote_reference_review"] = verify_current_remote_manifest(root, builders)
            record["status"] = "completed"
            atomic_json(logs / f"{number:02d}_{stage['name']}.json", record)
            atomic_json(status_path, status)
        validate_binding(root, builders, manifest_path, manifest_sha256)
        require(identity(Path(__file__)) == {k: status["runner"][k] for k in ("sha256", "bytes")},
                "Runner changed during execution")
        status["status"] = "completed"
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if status["stages"] and status["stages"][-1]["status"] == "running":
            status["stages"][-1].update(status="failed", ended_at_utc=utc_now())
        raise
    finally:
        status["ended_at_utc"] = utc_now()
        atomic_json(status_path, status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--builder-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--builder-manifest", type=Path, required=True)
    parser.add_argument("--builder-manifest-sha256", required=True)
    args = parser.parse_args(argv)
    root = safe_path(args.release_root, "dir")
    builders = safe_path(args.builder_root, "dir")
    logs = safe_path(args.log_root)
    manifest_path = safe_path(args.builder_manifest, "file")
    stages = stage_plan(root, builders, sys.executable)
    binding = preflight(root, builders, logs, stages, manifest_path, args.builder_manifest_sha256)
    with (root / ".protein_phase_started.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "started_at_utc": utc_now(), "release_id": RELEASE_ID,
                   "builder_manifest_sha256": args.builder_manifest_sha256}, stream)
    execute_stages(root, builders, logs, stages, manifest_path, args.builder_manifest_sha256, binding)


if __name__ == "__main__":
    main()
