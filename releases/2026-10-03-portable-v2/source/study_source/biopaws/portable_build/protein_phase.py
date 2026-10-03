#!/usr/bin/env python3
"""Run the accepted v2 protein builders in a new, explicitly named CPU build.

This wrapper does not change the scientific builders, acquire sources, resume an
old build, or certify the complete release. The top-level build owns raw-asset
verification and the final cross-component gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


BUILDERS = {
    "data_v2/build_protein.py": "592c9a50b26bfcf1e6502363a6b1477c0f8a00a99bb5437e84c75c86a398af85",
    "data_v2/build_pairs.py": "783a46065459e641d8d27209970e085654ece8c756e78b73f53b3c72f491767b",
    "data_v2/exclude_remote_homologs.py": "c063a323bd28671a1e6e5998ff119b9ca88b7ac38193b1ffa9c39fedb64b3a8a",
    "data_v2/cross_dataset_homology_audit.py": "c391e7d3f222f3c829dcbc470585891bea7e2db9acd6f6fe4d98710b0055111d",
    "data_v2_remote/build_remote.py": "20c00994ca244eb3843345e479b7dcfc8fbe7c0dc0ecbba1e3a722bb15f369e7",
}

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


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def contained_file(root, relative):
    path = root / relative
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Missing or non-contained regular input: {relative}")
    for ancestor in path.parents:
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError(f"Symlinked input ancestor: {relative}")
    return path


def inventory(root, patterns):
    result = {}
    for pattern in patterns:
        paths = sorted(root.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"Required artifact pattern has no matches: {pattern}")
        for path in paths:
            relative = path.relative_to(root).as_posix()
            path = contained_file(root, relative)
            result[relative] = {"sha256": digest(path), "bytes": path.stat().st_size}
    return result


def stage_plan(root, builders, python):
    def stage(name, script, arguments, inputs, outputs):
        return {"name": name, "script": script,
                "command": [str(python), "-B", str(builders / script), *arguments],
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
               "metadata/protein_pair_construction.json", "metadata/protein_pair_manifest.json"]),
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


def preflight(root, builders, logs, stages):
    for relative, expected in BUILDERS.items():
        path = contained_file(builders, relative)
        if digest(path) != expected:
            raise ValueError(f"Accepted builder hash mismatch: {relative}")
    inventory(root, INITIAL_INPUTS)
    for relative in ("data/sequences", "data/pairs", "work/protein", "metadata", "validation"):
        candidate = root / relative
        for path in [candidate, *candidate.parents]:
            if path == root:
                break
            if path.is_symlink():
                raise ValueError(f"Output ancestor must not be a symlink: {path}")
    for relative in ("data/sequences", "data/pairs", "work/protein"):
        path = root / relative
        if path.is_symlink() or (path.exists() and any(path.iterdir())):
            raise FileExistsError(f"Protein output directory must be new or empty: {path}")
    # Input corpora and NLP outputs may already exist. Protein metadata may not.
    for stage in stages:
        for pattern in stage["outputs"]:
            if list(root.glob(pattern)):
                raise FileExistsError(f"Refusing to overwrite existing stage artifact: {pattern}")
    if logs.is_symlink() or (logs.exists() and any(logs.iterdir())):
        raise FileExistsError(f"Use a new or empty protein log directory: {logs}")
    if logs == root or logs == builders or logs.is_relative_to(builders):
        raise ValueError("Log directory must be separate from the builder snapshot")
    for relative in ("data/sequences", "data/pairs", "work/protein", "metadata", "validation"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)


def execute_stages(root, builders, logs, stages, run_command=subprocess.run):
    status_path = logs / "protein_phase_status.json"
    status = {"status": "running", "started_at_utc": utc_now(), "pid": os.getpid(),
              "release_root": str(root), "builder_root": str(builders), "cpu_only": True,
              "threads": 16, "builder_sha256": BUILDERS, "stage_count": len(stages), "stages": []}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="16", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false")
    # Never inherit a caller's import-path injection into the frozen builders.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONSTARTUP", None)
    atomic_json(status_path, status)
    try:
        for number, stage in enumerate(stages, 1):
            started = time.monotonic()
            log_path = logs / f"{number:02d}_{stage['name']}.log"
            current_code_hash = digest(contained_file(builders, stage["script"]))
            if stage["script"] in BUILDERS and current_code_hash != BUILDERS[stage["script"]]:
                raise ValueError(f"Builder changed during execution: {stage['script']}")
            record = {"name": stage["name"], "status": "running", "started_at_utc": utc_now(),
                      "command": stage["command"], "inputs": inventory(root, stage["inputs"]),
                      "builder_sha256": current_code_hash,
                      "log": log_path.name}
            status["stages"].append(record)
            atomic_json(status_path, status)
            print(f"[{utc_now()}] Starting {number}/{len(stages)} {stage['name']}", flush=True)
            with log_path.open("x") as log:
                result = run_command(stage["command"], cwd=root, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, check=False)
            record.update(returncode=result.returncode, ended_at_utc=utc_now(),
                          wall_seconds=time.monotonic() - started, log_sha256=digest(log_path))
            if result.returncode:
                record["status"] = "failed"
                raise RuntimeError(f"Stage {stage['name']} returned {result.returncode}; see {log_path}")
            record["outputs"] = inventory(root, stage["outputs"])
            record["status"] = "completed"
            atomic_json(logs / f"{number:02d}_{stage['name']}.json", record)
            atomic_json(status_path, status)
        status["status"] = "completed"
        status["complete_release_gate_passed"] = False
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
    parser.add_argument("--builder-root", type=Path, required=True,
                        help="Pinned snapshot directory containing data_v2 and data_v2_remote")
    parser.add_argument("--log-root", type=Path, required=True)
    args = parser.parse_args(argv)
    root, builders, logs = (p.resolve() for p in (args.release_root, args.builder_root, args.log_root))
    if not root.is_dir() or not builders.is_dir():
        raise ValueError("Release and builder roots must already exist")
    stages = stage_plan(root, builders, sys.executable)
    preflight(root, builders, logs, stages)
    # An exclusive marker rejects concurrent entry; retain it as historical evidence.
    marker = root / ".protein_phase_started.json"
    with marker.open("x") as stream:
        json.dump({"pid": os.getpid(), "started_at_utc": utc_now()}, stream)
    execute_stages(root, builders, logs, stages)


if __name__ == "__main__":
    main()
