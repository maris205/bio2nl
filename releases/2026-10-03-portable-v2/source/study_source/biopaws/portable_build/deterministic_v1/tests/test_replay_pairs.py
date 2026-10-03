import csv
import gzip
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

MODULE = Path(__file__).parents[1] / "replay_pairs.py"
spec = importlib.util.spec_from_file_location("deterministic_pairs_adapter", MODULE)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
BUILDER = Path(__file__).parents[3] / "data_v2/build_pairs.py"
RELEASE = "test-intermediate-pair-replay-v1"

@pytest.fixture(autouse=True)
def pinned_adapter_hash_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

@pytest.fixture
def inputs(tmp_path):
    rows = []; candidates = []
    alphabet = "ACDEFHIKLMNPQRSTVWY"
    for number, (split, cluster) in enumerate(itertools.product(("train", "validation", "test"), range(3))):
        x, y = alphabet[2 * number:2 * number + 2]
        sequences = [x * 38 + "GG", x * 22 + y * 16 + "GG", x * 22 + y * 8 + x * 8 + "GG"]
        ids = [f"{split}_{cluster}_{i}" for i in range(3)]
        component = "component_" + ids[0]
        for accession, sequence in zip(ids, sequences):
            rows.append({"accession": accession, "sequence": sequence,
                "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                "cluster_id": component, "split": split})
        # Overlapping positive edges exercise the original greedy disjoint-edge
        # choice; this is not an identity transformation on pair inputs.
        for a, b in itertools.combinations(ids, 2):
            candidates.append({"a": a, "b": b, "identity": .6,
                "query_coverage": 1., "target_coverage": .9, "evalue": 1e-12,
                "alignment_length": 40, "split": split, "cluster_id": component})
    table = tmp_path / "canonical.tsv.gz"
    with gzip.open(table, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    source = tmp_path / "candidates.jsonl"
    write_candidates(source, candidates)
    return table, source, candidates

def write_candidates(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

def protocol_for(tmp_path, table, sources, roots):
    code = {str(p): {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in (MODULE, BUILDER, BUILDER.with_name("build_protein.py"))}
    protocol = {"schema_version": 1, "release_id": RELEASE, "seed": adapter.SEED, "runs": {}}
    for i, (source, root) in enumerate(zip(sources, roots)):
        protocol["runs"][f"order_{i}"] = {"root": str(root), "code": code,
            "python_hash_seed": (0, 12345, 1259)[i],
            "inputs": {name: {"file": str(p), "sha256": digest(p), "bytes": p.stat().st_size}
                       for name, p in (("positive_candidates", source), ("canonical_sequences", table))}}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    return path

def command(table, source, root, protocol):
    return [sys.executable, "-B", str(MODULE), "--builder", str(BUILDER),
        "--positive-candidates", str(source), "--candidates-sha256", digest(source),
        "--canonical-sequences", str(table), "--canonical-sha256", digest(table),
        "--output-root", str(root), "--release-id", RELEASE,
        "--protocol", str(protocol), "--protocol-sha256", digest(protocol), "--sort-buffer-mib", "1"]

def run_direct(table, source, root, protocol, **kwargs):
    return adapter.replay_pairs(BUILDER, source, digest(source), table, digest(table),
        root, RELEASE, protocol, digest(protocol), **kwargs)

def test_original_algorithm_is_permutation_and_hashseed_invariant(tmp_path, inputs):
    table, source, candidates = inputs
    variants = [candidates, list(reversed(candidates)), random.Random(915).sample(candidates, len(candidates))]
    sources = []
    for i, rows in enumerate(variants):
        path = tmp_path / f"source_{i}.jsonl"; write_candidates(path, rows); sources.append(path)
    roots = [tmp_path / f"run_{i}" for i in range(3)]
    protocol = protocol_for(tmp_path, table, sources, roots)
    manifests = []
    for seed, src, root in zip(("0", "12345", "1259"), sources, roots):
        result = subprocess.run(command(table, src, root, protocol), env=dict(os.environ, PYTHONHASHSEED=seed),
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr + (root / "builder.log").read_text() if (root / "builder.log").exists() else result.stderr
        manifests.append(json.loads((root / "replay_manifest.json").read_text()))
    assert len({m["inputs"]["positive_candidates"]["sha256"] for m in manifests}) == 3
    assert len({m["canonicalization"]["sorted_candidate_sha256"] for m in manifests}) == 1
    assert [m["environment"]["python_hash_seed"] for m in manifests] == ["0", "12345", "1259"]
    assert [m["environment"]["hash_randomization"] for m in manifests] == [0, 1, 1]
    assert len({m["environment"]["hash_probe"] for m in manifests}) == 3
    assert all(m["canonicalization"]["records"] == 27 for m in manifests)
    for path in adapter.PAIR_FILES:
        payloads = [gzip.decompress((root / path).read_bytes()) for root in roots]
        assert payloads[0] == payloads[1] == payloads[2]
        assert len({m["outputs"][path]["decoded_sha256"] for m in manifests}) == 1
    assert all(m["outputs"][adapter.PAIR_FILES[-1]]["records"] == 18 for m in manifests)
    for root, manifest in zip(roots, manifests):
        assert manifest["release_id"] == RELEASE
        assert not manifest["training_enabled"] and not manifest["data_release_gate_passed"]
        assert not manifest["complete_raw_reconstruction"]
        assert not (root / "metadata/protein_pair_manifest.json").exists()
        legacy = json.loads((root / "metadata/original_builder_manifest.json").read_text())
        assert legacy["release_id"] == "2026-09-25-v2"
        assert manifest["legacy_builder_manifest"]["sha256"] == digest(root / "metadata/original_builder_manifest.json")
        assert legacy["acceptance"]["acceptance_pass"] is True
        assert set(manifest["outputs"]) == set(adapter.OUTPUT_FILES)
        assert manifest["implementation"]["builder"]["sha256"] == adapter.BUILDER_SHA
    # Control proves the scientific implementation is exactly the original main,
    # and ordering is the sole input change before invoking that main.
    direct = tmp_path / "direct"
    (direct / "data/sequences").mkdir(parents=True); (direct / "work/protein").mkdir(parents=True)
    (direct / "data/pairs").mkdir(parents=True)
    (direct / "data/sequences/protein_canonical_sequences.tsv.gz").write_bytes(table.read_bytes())
    (direct / "work/protein/positive_candidates.jsonl").write_bytes((roots[0] / "work/protein/positive_candidates.jsonl").read_bytes())
    result = subprocess.run([sys.executable, "-B", str(BUILDER), "--root", str(direct), "--seed", str(adapter.SEED)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for path in adapter.PAIR_FILES:
        assert gzip.decompress((direct / path).read_bytes()) == gzip.decompress((roots[0] / path).read_bytes())

def test_external_merge_preserves_values_across_chunks_and_json_key_order(tmp_path, inputs):
    table, source, rows = inputs
    rows[0]["identity"] = .6000000000000001
    write_candidates(source, [dict(reversed(list(row.items()))) for row in reversed(rows)])
    dest = tmp_path / "sorted.jsonl"
    report = adapter.canonical_sort(source, dest, adapter.canonical_table(table), digest(source), buffer_bytes=80, fanin=2)
    actual = [json.loads(line) for line in dest.read_text().splitlines()]
    assert actual == sorted(rows, key=lambda r: (r["a"], r["b"]))
    assert report["records"] == 27 and report["score_values_changed"] is False
    assert report["sorted_candidate_sha256"] == digest(dest)

@pytest.mark.parametrize("change", ["duplicate", "conflicting_duplicate", "nan", "unknown_endpoint", "bad_score", "extra_field", "split_mismatch", "blank", "duplicate_json_key"])
def test_bad_candidate_content_fails_without_silent_drop(tmp_path, inputs, change):
    table, source, rows = inputs
    if change == "duplicate":rows.append(dict(rows[0]))
    elif change == "conflicting_duplicate":rows.append({**rows[0], "identity": .7})
    elif change == "nan":rows[0]["identity"] = float("nan")
    elif change == "unknown_endpoint":rows[0]["b"] = "zz_missing"
    elif change == "bad_score":rows[0]["identity"] = .95
    elif change == "extra_field":rows[0]["ignored_score"] = .5
    elif change == "split_mismatch":rows[0]["split"] = "test"
    write_candidates(source, rows)
    if change == "blank":source.write_text(source.read_text() + "\n")
    if change == "duplicate_json_key":source.write_text(source.read_text().replace('"identity": 0.6', '"identity": 0.6, "identity": 0.7', 1))
    with pytest.raises(ValueError):
        adapter.canonical_sort(source, tmp_path / "sorted", adapter.canonical_table(table), digest(source), buffer_bytes=80, fanin=2)

@pytest.mark.parametrize("change", ["root", "input_path", "input_sha", "code_sha", "seed", "release", "hash_seed"])
def test_protocol_scope_cannot_be_bypassed(tmp_path, inputs, change):
    table, source, rows = inputs; root = tmp_path / "run"
    protocol = protocol_for(tmp_path, table, [source], [root]); data = json.loads(protocol.read_text())
    run = data["runs"]["order_0"]
    if change == "root":run["root"] = str(tmp_path / "other")
    if change == "input_path":run["inputs"]["positive_candidates"]["file"] = str(tmp_path / "other")
    if change == "input_sha":run["inputs"]["positive_candidates"]["sha256"] = "0" * 64
    if change == "code_sha":run["code"][str(MODULE)]["sha256"] = "0" * 64
    if change == "seed":data["seed"] += 1
    if change == "release":data["release_id"] = "2026-09-25-v2"
    if change == "hash_seed":run["python_hash_seed"] = 12345
    protocol.write_text(json.dumps(data))
    with pytest.raises(ValueError):run_direct(table, source, root, protocol)
    assert not root.exists()

def test_fresh_root_and_symlink_guards(tmp_path, inputs):
    table, source, rows = inputs; root = tmp_path / "run"
    protocol = protocol_for(tmp_path, table, [source], [root])
    root.mkdir()
    with pytest.raises(ValueError, match="fresh"):run_direct(table, source, root, protocol)
    link = tmp_path / "input-link"; link.symlink_to(source)
    with pytest.raises(ValueError, match="Symlinked"):
        adapter.replay_pairs(BUILDER, link, digest(source), table, digest(table), tmp_path / "fresh", RELEASE, protocol, digest(protocol))

def test_duplicate_canonical_accessions_fail(tmp_path, inputs):
    table, source, rows = inputs
    text = gzip.decompress(table.read_bytes()).decode(); text += text.splitlines()[1] + "\n"
    with gzip.open(table, "wt") as out:out.write(text)
    with pytest.raises(ValueError, match="Duplicate/empty canonical accession"):
        adapter.canonical_table(table)

def test_failed_builder_is_not_skipped_or_accepted(tmp_path, inputs, monkeypatch):
    table, source, rows = inputs; root = tmp_path / "run"
    protocol = protocol_for(tmp_path, table, [source], [root])
    monkeypatch.setattr(adapter.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a[0], 9))
    with pytest.raises(ValueError, match="Frozen builder failed"):
        run_direct(table, source, root, protocol, sort_buffer_bytes=80, merge_fanin=2)
    assert json.loads((root / "replay_status.json").read_text())["status"] == "failed"
    assert not (root / "replay_manifest.json").exists()
