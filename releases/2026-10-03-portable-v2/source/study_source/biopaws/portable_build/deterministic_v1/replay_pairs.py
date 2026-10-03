#!/usr/bin/env python3
"""Replay the frozen protein pair builder after external candidate ordering.

This is an intermediate-input replay, never an accepted data release. The
scientific builder is executed unchanged, with its original seed and rules.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import heapq
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

SEED = 20260925
BUILDER_SHA = "783a46065459e641d8d27209970e085654ece8c756e78b73f53b3c72f491767b"
DEPENDENCY_SHA = "592c9a50b26bfcf1e6502363a6b1477c0f8a00a99bb5437e84c75c86a398af85"
FIELDS = {"a", "b", "identity", "query_coverage", "target_coverage", "evalue",
          "alignment_length", "split", "cluster_id"}
PAIR_FILES = [f"data/pairs/protein_sequence_similarity_{s}.tsv.gz"
              for s in ("train", "validation", "test", "all")]
OUTPUT_FILES = PAIR_FILES + ["validation/protein_endpoint_degrees.tsv",
    "validation/protein_pair_baselines.csv", "validation/protein_pair_acceptance.json",
    "metadata/protein_pair_construction.json", "metadata/original_builder_manifest.json"]

def require(ok, message):
    if not ok:
        raise ValueError(message)

def now():
    return datetime.now(timezone.utc).isoformat()

def safe_path(value, exists=True):
    path = Path(os.path.abspath(value))
    require(not any(p.is_symlink() for p in [path, *path.parents]), f"Symlinked path: {path}")
    if exists:
        require(path.is_file(), f"Missing regular input: {path}")
    return path

def identity(path):
    path = safe_path(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return {"path": str(path), "sha256": h.hexdigest(), "bytes": path.stat().st_size}

def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)

def reject_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")

def unique_keys(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"Duplicate JSON key: {key}")
        value[key] = item
    return value

def canonical_table(path):
    """Validate the frozen table without retaining sequences twice in memory."""
    index = {}
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), "Bad table header")
        require({"accession", "sequence", "sequence_sha256", "cluster_id", "split"} <= set(reader.fieldnames), "Missing canonical columns")
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()), "Malformed canonical row")
            accession = row["accession"]
            require(accession and accession not in index, "Duplicate/empty canonical accession")
            require(row["split"] in {"train", "validation", "test"} and row["cluster_id"], "Invalid canonical assignment")
            sequence = row["sequence"]
            require(sequence and set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"), "Invalid canonical sequence")
            require(hashlib.sha256(sequence.encode()).hexdigest() == row["sequence_sha256"], "Canonical sequence hash mismatch")
            index[accession] = (row["split"], row["cluster_id"])
    require(index, "Empty canonical table")
    return index

def candidate_line(raw, index):
    require(raw.strip(), "Blank candidate line")
    value = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    require(isinstance(value, dict) and set(value) == FIELDS, "Unexpected candidate fields")
    a, b = value["a"], value["b"]
    require(all(isinstance(x, str) and x and x.isascii() and not any(c.isspace() or ord(c) < 32 for c in x)
                for x in (a, b)), "Invalid candidate identifier")
    require(a < b and a in index and b in index, "Candidate endpoints must be canonical a < b")
    require(index[a] == index[b] == (value["split"], value["cluster_id"]), "Candidate/table assignment differs")
    for key in ("identity", "query_coverage", "target_coverage", "evalue"):
        require(type(value[key]) in (int, float) and math.isfinite(value[key]), f"Invalid candidate {key}")
    require(.4 <= value["identity"] <= .9 and
            .8 <= value["query_coverage"] <= 1 and .8 <= value["target_coverage"] <= 1 and
            0 <= value["evalue"] <= .001, "Candidate violates frozen positive definition")
    require(type(value["alignment_length"]) is int and value["alignment_length"] > 0, "Invalid alignment length")
    # JSON key/whitespace representation is canonical; numeric values are neither
    # rounded nor rescored. Parsing again gives the same values as the old loader.
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode() + b"\n"
    return a.encode() + b"\t" + b.encode() + b"\t" + payload

def merge_runs(paths, output):
    from contextlib import ExitStack
    with ExitStack() as stack, output.open("xb") as stream:
        readers = [stack.enter_context(path.open("rb")) for path in paths]
        for line in heapq.merge(*readers):
            stream.write(line)

def canonical_sort(source, destination, index, expected_sha256, buffer_bytes=64 << 20, fanin=32):
    require(buffer_bytes >= 1 and fanin >= 2, "Invalid external-sort bounds")
    require(not destination.exists(), "Refuse existing sorted candidates")
    h = hashlib.sha256(); records = total_bytes = 0
    with tempfile.TemporaryDirectory(prefix="candidate-sort-", dir=destination.parent) as tmp:
        work = Path(tmp); runs = []; chunk = []; size = 0
        def flush():
            nonlocal chunk, size
            if chunk:
                chunk.sort(); path = work / f"chunk-{len(runs):06d}"
                with path.open("xb") as out:
                    out.writelines(chunk)
                runs.append(path); chunk = []; size = 0
        with source.open("rb") as stream:
            while True:
                raw = stream.readline((1 << 20) + 1)
                if not raw:
                    break
                require(len(raw) <= 1 << 20, "Candidate record exceeds 1 MiB")
                h.update(raw); total_bytes += len(raw)
                line = candidate_line(raw, index)
                if chunk and size + len(line) > buffer_bytes:
                    flush()
                chunk.append(line); size += len(line); records += 1
        flush()
        require(records > 0, "Empty candidate set")
        require(h.hexdigest() == expected_sha256, "Candidate input SHA mismatch")
        generation = 0
        while len(runs) > fanin:
            next_runs = []
            for offset in range(0, len(runs), fanin):
                out = work / f"merge-{generation}-{offset}"
                merge_runs(runs[offset:offset + fanin], out); next_runs.append(out)
            for path in runs:
                path.unlink()
            runs = next_runs; generation += 1
        merged = work / "merged"
        merge_runs(runs, merged)
        previous = None; count = 0; content = hashlib.sha256()
        with merged.open("rb") as stream, destination.open("xb") as out:
            for line in stream:
                a, b, payload = line.split(b"\t", 2)
                key = (a, b)
                require(key != previous, "Duplicate candidate edge; no records may be silently discarded")
                require(previous is None or previous < key, "Candidate sort order violation")
                out.write(payload); content.update(payload); previous = key; count += 1
        require(count == records, "External sort changed record count")
    return {"input_sha256": h.hexdigest(), "input_bytes": total_bytes,
            "records": records, "sorted_candidate_sha256": content.hexdigest(),
            "canonical_content_sha256": content.hexdigest(), "key": ["a", "b"],
            "ordering": "ascending ASCII accession tuple before original fixed-seed shuffle",
            "duplicate_policy": "reject", "score_values_changed": False,
            "sort_buffer_bytes": buffer_bytes, "merge_fanin": fanin,
            "memory_bound_scope": "chunk encoded bytes plus Python overhead, one record, and canonical accession index; not a process RSS cap"}

def output_identity(root, relative):
    result = identity(root / relative); result.pop("path")
    if relative in PAIR_FILES:
        h = hashlib.sha256(); lines = 0
        with gzip.open(root / relative, "rb") as stream:
            for line in stream:
                h.update(line); lines += 1
        require(lines >= 2, "Empty pair output")
        result.update(decoded_sha256=h.hexdigest(), records=lines - 1)
    return result

def replay_pairs(builder, positive_candidates, candidates_sha256, canonical_sequences,
                 canonical_sha256, output_root, release_id, protocol, protocol_sha256,
                 sort_buffer_bytes=64 << 20, merge_fanin=32):
    builder, candidates, table, protocol = map(safe_path, (builder, positive_candidates, canonical_sequences, protocol))
    root = safe_path(output_root, exists=False)
    require(not root.exists(), "Output root must be fresh and absent")
    require(release_id and release_id != "2026-09-25-v2", "Replay must have a new release identity")
    code = {"adapter": identity(__file__), "builder": identity(builder),
            "builder_dependency": identity(builder.with_name("build_protein.py"))}
    require(code["builder"]["sha256"] == BUILDER_SHA and code["builder_dependency"]["sha256"] == DEPENDENCY_SHA, "Frozen builder identity differs")
    require(not root.is_relative_to(builder.parent), "Output root overlaps frozen builder directory")
    protocol_id = identity(protocol)
    require(protocol_id["sha256"] == protocol_sha256, "Protocol SHA mismatch")
    contract = json.loads(protocol.read_text())
    require(contract["schema_version"] == 1 and contract["release_id"] == release_id and contract["seed"] == SEED, "Protocol identity/seed mismatch")
    matches = [(name, run) for name, run in contract["runs"].items() if run["root"] == str(root)]
    require(len(matches) == 1, "Output root not uniquely bound in protocol")
    run_id, run = matches[0]
    worker_hash_seed = str(run["python_hash_seed"])
    require(worker_hash_seed.isdigit() and 0 <= int(worker_hash_seed) < 2**32,
            "Protocol requires a fixed Python hash seed")
    require(os.environ.get("PYTHONHASHSEED") == worker_hash_seed,
            "Adapter environment hash seed differs from protocol run")
    inputs = {"positive_candidates": identity(candidates), "canonical_sequences": identity(table)}
    require(inputs["positive_candidates"]["sha256"] == candidates_sha256 and inputs["canonical_sequences"]["sha256"] == canonical_sha256, "Actual input SHA mismatch")
    for name, record in inputs.items():
        expected = run["inputs"][name]
        require({"file": record["path"], "sha256": record["sha256"], "bytes": record["bytes"]} == expected, "Protocol input identity mismatch: " + name)
    for record in code.values():
        require(run["code"].get(record["path"]) == {"sha256": record["sha256"], "bytes": record["bytes"]}, "Protocol code identity mismatch")
    root.mkdir(parents=True, exist_ok=False)
    for relative in ("work/protein", "data/sequences", "data/pairs", "validation", "metadata"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "started_at_utc": now(), "release_id": release_id,
              "protocol_run_id": run_id, "protocol": protocol_id}
    write_json(root / "replay_status.json", status)
    try:
        local_table = root / "data/sequences/protein_canonical_sequences.tsv.gz"
        with table.open("rb") as src, local_table.open("xb") as dst:
            shutil.copyfileobj(src, dst, 8 << 20)
        require(identity(local_table)["sha256"] == canonical_sha256, "Copied table SHA mismatch")
        index = canonical_table(local_table); inputs["canonical_sequences"]["records"] = len(index)
        sorted_path = root / "work/protein/positive_candidates.jsonl"
        status["stage"] = "canonical_candidate_sort"; write_json(root / "replay_status.json", status)
        ordering = canonical_sort(candidates, sorted_path, index, candidates_sha256, sort_buffer_bytes, merge_fanin)
        del index
        inputs["positive_candidates"]["records"] = ordering["records"]
        ordering["sorted_candidate_file"] = sorted_path.relative_to(root).as_posix()
        ordering["local_table"] = {"file": local_table.relative_to(root).as_posix(),
            "sha256": canonical_sha256, "bytes": inputs["canonical_sequences"]["bytes"],
            "records": inputs["canonical_sequences"]["records"]}
        for record in [*code.values(), protocol_id, *inputs.values()]:
            require(identity(record["path"])["sha256"] == record["sha256"], "Input/code changed before builder launch")
        worker_runtime_file = root / "worker_runtime.json"
        worker_bootstrap = (
            "import runpy,sys,os,json,platform; from pathlib import Path; from importlib.metadata import version; "
            "p=Path(sys.argv[1]); root=Path(sys.argv[3]); "
            "runtime={'python':sys.version,'python_implementation':platform.python_implementation(),"
            "'python_hash_seed':os.environ.get('PYTHONHASHSEED'),'hash_randomization':sys.flags.hash_randomization,"
            "'hash_probe':hash('deterministic_pair_worker_probe_v1'),'hash_probe_text':'deterministic_pair_worker_probe_v1',"
            "'libraries':{name:version(name) for name in ('numpy','scipy','biopython','scikit-learn','pandas')}}; "
            "(root/'worker_runtime.json').write_text(json.dumps(runtime,sort_keys=True)+'\\n'); "
            "sys.path.insert(0,str(p.parent)); sys.argv=[str(p),*sys.argv[2:]]; runpy.run_path(str(p),run_name='__main__')")
        command = [sys.executable, "-P", "-s", "-B", "-c", worker_bootstrap,
            str(builder), "--root", str(root), "--seed", str(SEED)]
        # -I would ignore PYTHONHASHSEED. Keep safe-path/no-user-site flags and
        # remove Python environment injection while retaining the bound seed.
        env = {key: value for key, value in os.environ.items() if not key.startswith("PYTHON")}
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED=worker_hash_seed,
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        status.update(stage="frozen_pair_builder", command=command); write_json(root / "replay_status.json", status)
        with (root / "builder.log").open("x") as log:
            result = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        require(result.returncode == 0, f"Frozen builder failed ({result.returncode}); see builder.log")
        runtime = json.loads(worker_runtime_file.read_text())
        require(runtime["python_hash_seed"] == worker_hash_seed and
                runtime["hash_randomization"] == int(int(worker_hash_seed) != 0),
                "Actual worker hash-seed runtime differs from protocol")
        original = root / "metadata/protein_pair_manifest.json"
        original.rename(root / "metadata/original_builder_manifest.json")
        acceptance_file = "validation/protein_pair_acceptance.json"
        acceptance = json.loads((root / acceptance_file).read_text())
        require(acceptance["acceptance_pass"] is True and not acceptance["failures"], "Pair acceptance failed")
        legacy = json.loads((root / "metadata/original_builder_manifest.json").read_text())
        require(legacy["release_id"] == "2026-09-25-v2" and legacy["builder_sha256"] == BUILDER_SHA, "Unexpected original manifest identity")
        outputs = {relative: output_identity(root, relative) for relative in OUTPUT_FILES}
        for relative in PAIR_FILES:
            require(legacy["data_files"][Path(relative).name] == outputs[relative]["sha256"], "Original builder output binding differs")
        require(legacy["acceptance"] == acceptance, "Original builder acceptance differs")
        for record in [*code.values(), protocol_id, *inputs.values()]:
            require(identity(record["path"])["sha256"] == record["sha256"], "Input/code changed during replay")
        require(identity(sorted_path)["sha256"] == ordering["sorted_candidate_sha256"] and identity(local_table)["sha256"] == canonical_sha256, "Local replay inputs changed")
        manifest = {"schema_version": 1, "status": "completed_intermediate_input_replay",
            "release_id": release_id, "protocol_run_id": run_id, "protocol": protocol_id,
            "scope": "protein_pair_replay_from_pinned_intermediate_inputs", "seed": SEED,
            "training_enabled": False, "data_release_gate_passed": False,
            "complete_raw_reconstruction": False, "inputs": inputs, "implementation": code,
            "canonicalization": ordering, "outputs": outputs,
            "legacy_builder_manifest": {"file": "metadata/original_builder_manifest.json",
                "sha256": outputs["metadata/original_builder_manifest.json"]["sha256"],
                "embedded_release_id": "2026-09-25-v2",
                "interpretation": "Original builder hardcoded historical label only; not the current replay identity or accepted release."},
            "acceptance": {"file": acceptance_file, "sha256": outputs[acceptance_file]["sha256"], "acceptance_pass": True},
            "worker_runtime": {"file": "worker_runtime.json", **output_identity(root, "worker_runtime.json")},
            "environment": {**runtime,
                "stdlib_rng": "random.Random; unchanged fixed seed; sorted inputs before shuffle"},
            "started_at_utc": status["started_at_utc"], "completed_at_utc": now(),
            "ordered_pair_identity": "decoded_sha256 covers exact TSV bytes including order; gzip headers/runtime metadata may differ"}
        write_json(root / "replay_manifest.json", manifest)
        status.update(status="completed", stage="completed", replay_manifest_sha256=identity(root / "replay_manifest.json")["sha256"])
        return manifest
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        status["completed_at_utc"] = now(); write_json(root / "replay_status.json", status)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("builder", "positive-candidates", "canonical-sequences", "output-root", "protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("candidates-sha256", "canonical-sha256", "release-id", "protocol-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--sort-buffer-mib", type=int, default=64)
    args = vars(parser.parse_args(argv)); args["sort_buffer_bytes"] = args.pop("sort_buffer_mib") << 20
    result = replay_pairs(**args)
    print(json.dumps({"status": result["status"], "release_id": result["release_id"], "candidate_records": result["canonicalization"]["records"]}))

if __name__ == "__main__":
    main()
