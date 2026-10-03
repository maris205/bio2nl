#!/usr/bin/env python3
"""Prepare one audited, model-independent supervised sample of accepted v2 data.

Only declared sentence fields are model inputs. All row/group/sequence identifiers
remain in a nested audit-only metadata object. Validation and test are never sampled.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import unicodedata

SPLITS = ("train", "validation", "test")
DEFAULT_SEED = 20260925
TASKS = {
    "protein_sequence_similarity": {"kind": "protein", "group": "block_id", "inputs": ["sentence1", "sentence2"], "labels": {"0": "dissimilar_under_operational_filters", "1": "similar_under_operational_filters"}},
    "remote": {"kind": "remote", "group": "rewiring_block_id", "inputs": ["sentence1", "sentence2"], "labels": {"0": "different_SCOPe_superfamily", "1": "same_superfamily_different_family"}},
    "pawsx_en": {"kind": "nlp", "group": "input_group_id", "inputs": ["sentence1", "sentence2"], "labels": {"0": "not_paraphrase", "1": "paraphrase"}},
    "cola": {"kind": "nlp", "group": "input_group_id", "inputs": ["sentence"], "labels": {"0": "unacceptable", "1": "acceptable"}},
    "rte": {"kind": "nlp", "group": "input_group_id", "inputs": ["sentence1", "sentence2"], "labels": {"0": "entailment", "1": "not_entailment"}},
    **{f"dyck_L{n}_t3": {"kind": "dyck", "group": "pair_group_id", "inputs": ["sentence"], "labels": {"0": "invalid_typed_Dyck", "1": "valid_typed_Dyck"}} for n in (40, 60)},
    **{f"longrange_d{n}": {"kind": "longrange", "group": "template_group_id", "inputs": ["sentence"], "labels": {"0": "subject_verb_disagreement", "1": "subject_verb_agreement"}} for n in (8, 16, 24)},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def source_file(root: Path, task: str, split: str) -> Path:
    kind = TASKS[task]["kind"]
    if kind == "protein":
        return root / "data/pairs" / f"protein_sequence_similarity_{split}.tsv.gz"
    if kind == "remote":
        return root / "data/pairs" / f"remote_{split}.csv.gz"
    branch = "nlp_clean" if kind == "nlp" else "synthetic"
    return root / "data" / branch / task / f"{split}.jsonl"


def read_rows(path: Path) -> list[dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t" if ".tsv" in path.name else ","))
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    for row in rows:
        row["label"] = int(row["label"])
        if row["label"] not in (0, 1):
            raise ValueError(f"Nonbinary label in {path}")
        row.setdefault("row_id", row.get("pair_id"))
        if not row["row_id"]:
            raise ValueError(f"Missing row ID in {path}")
    return rows


def select_groups(rows: list[dict], group_key: str, target: int, seed: int, task: str) -> tuple[list[dict], dict]:
    """Uniform deterministic group order, nearest prefix to requested row budget.

    Unequal groups are never split or greedily skipped by size. Thus selected row
    counts may exceed the budget slightly, and we do not claim row-uniform sampling.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row[group_key]].append(row)
    ordered = sorted(groups, key=lambda key: (stable_hash(seed, task, key), key))
    if len(rows) <= target:
        selected = set(ordered)
    else:
        count = 0
        cutoff = 0
        for index, key in enumerate(ordered):
            after = count + len(groups[key])
            if after >= target:
                cutoff = index + int(after - target <= target - count)
                break
            count = after
        selected = set(ordered[:cutoff])
    result = [row for row in rows if row[group_key] in selected]
    if not result:
        raise ValueError("Whole-group budget selects no rows; increase target")
    return result, {"method": "seeded_sha256_group_order_nearest_prefix", "group_key": group_key, "source_groups": len(groups), "selected_groups": len(selected), "target_rows": target, "selected_rows": len(result), "whole_groups": True}


def select_train(rows: list[dict], task: str, target: int, seed: int) -> tuple[list[dict], dict]:
    spec = TASKS[task]
    if spec["kind"] == "longrange" and len(rows) > target:
        if target % 8:
            raise ValueError("Longrange target must be divisible by 8 for paired four-cell design")
        strata = defaultdict(list)
        for row in rows:
            strata[(row["head_number"], row["nearest_distractor_number"])].append(row)
        if len(strata) != 4:
            raise ValueError(f"Incomplete longrange design: {sorted(strata)}")
        selected_ids = set()
        reports = {}
        for stratum, subset in sorted(strata.items()):
            selected, report = select_groups(subset, spec["group"], target // 4, seed, task + ":" + ":".join(stratum))
            selected_ids.update(row["row_id"] for row in selected)
            reports["/".join(stratum)] = report
        result = [row for row in rows if row["row_id"] in selected_ids]
        return result, {"method": "four_fixed_head_nearest_number_strata_whole_templates", "target_rows": target, "selected_rows": len(result), "strata": reports, "whole_groups": True}
    if spec["kind"] == "nlp":
        # RTE is directional: reversed premises/hypotheses may be distinct
        # semantic examples but share an unordered split-isolation group.
        # Retain all of them when below budget; if larger, keep whole groups.
        if len(rows) > target and len({row[spec["group"]] for row in rows}) != len(rows):
            return select_groups(rows, spec["group"], target, seed, task)
        ids = {row["row_id"] for row in sorted(rows, key=lambda row: (stable_hash(seed, task, row["row_id"]), row["row_id"]))[:target]}
        result = [row for row in rows if row["row_id"] in ids]
        return result, {"method": "seeded_sha256_uniform_row_priority" if len(rows) > target else "all_available_training_rows", "target_rows": target, "selected_rows": len(result)}
    return select_groups(rows, spec["group"], target, seed, task)


def distribution(values: list[int]) -> dict:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {"min": ordered[0], "median": ordered[(len(ordered) - 1) // 2], "p95": ordered[math.ceil(.95 * len(ordered)) - 1], "max": ordered[-1], "mean": sum(ordered) / len(ordered)}


def audit_split(rows: list[dict], task: str, split: str) -> dict:
    spec = TASKS[task]
    kind = spec["kind"]
    if not rows or len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError(f"Empty split or duplicate row IDs: {task}/{split}")
    if any(row["split"] != split for row in rows):
        raise ValueError(f"Changed split assignment: {task}/{split}")
    labels = Counter(row["label"] for row in rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row[spec["group"]]].append(row)
    info = {"rows": len(rows), "labels": dict(sorted(labels.items())), "groups": len(groups)}
    if kind != "nlp":
        if labels[0] != labels[1]:
            raise ValueError(f"Unequal classes: {task}/{split}")
        for group in groups.values():
            count = Counter(row["label"] for row in group)
            if count[0] != count[1]:
                raise ValueError(f"Unbalanced group: {task}/{split}")
            if kind in ("dyck", "longrange") and len(group) != 2:
                raise ValueError(f"Incomplete synthetic contrast: {task}/{split}")
        info["balanced_classes_and_complete_groups"] = True
    if kind in ("protein", "remote"):
        degrees = defaultdict(Counter)
        pair_keys = set()
        for row in rows:
            hashes = [row[f"sequence_sha256_{role}"] for role in ("a", "b")]
            if hashes[0] == hashes[1]:
                raise ValueError(f"Self pair: {task}/{split}")
            pair_key = tuple(sorted(hashes))
            if pair_key in pair_keys:
                raise ValueError(f"Duplicate unordered pair: {task}/{split}")
            pair_keys.add(pair_key)
            for role, seq_hash in zip(("a", "b"), hashes):
                text = row["sentence1" if role == "a" else "sentence2"]
                if hashlib.sha256(text.encode()).hexdigest() != seq_hash:
                    raise ValueError(f"Sequence hash mismatch: {task}/{split}")
                if not set(text) <= set("ACDEFGHIKLMNPQRSTVWY"):
                    raise ValueError(f"Invalid protein symbols: {task}/{split}")
                degrees[(role, seq_hash)][row["label"]] += 1
        if any(count[0] != count[1] for count in degrees.values()):
            raise ValueError(f"Unequal per-role endpoint class degree: {task}/{split}")
        info["endpoint_role_degree_balance"] = True
        info["endpoint_roles_checked"] = len(degrees)
    inputs = [[row.get("sentence", row.get("sentence1"))] if len(spec["inputs"]) == 1 else [row["sentence1"], row["sentence2"]] for row in rows]
    if any(not isinstance(text, str) or not text for item in inputs for text in item):
        raise ValueError(f"Missing model input: {task}/{split}")
    info["input_characters"] = {"combined": distribution([sum(map(len, item)) for item in inputs]), "side1": distribution([len(item[0]) for item in inputs])}
    if len(spec["inputs"]) == 2:
        info["input_characters"]["side2"] = distribution([len(item[1]) for item in inputs])
    info["pure_byte_input_plus_two_special_tokens_exceed_512"] = sum(sum(len(s.encode()) for s in item) + 2 > 512 for item in inputs)
    if kind == "longrange":
        cells = Counter((r["head_number"], r["nearest_distractor_number"]) for r in rows)
        info["design_cells"] = {"/".join(k): v for k, v in sorted(cells.items())}
        info["nearest_noun_rule_accuracy"] = sum(r["head_number"] == r["nearest_distractor_number"] for r in rows) / len(rows)
        if len(set(cells.values())) != 1:
            raise ValueError(f"Unbalanced longrange design cells: {task}/{split}")
    return info


def isolation_audit(splits: dict[str, list[dict]], task: str) -> dict:
    spec = TASKS[task]
    kind = spec["kind"]
    dimensions = {"row_id": lambda r: [r["row_id"]], spec["group"]: lambda r: [r[spec["group"]]]}
    if kind in ("protein", "remote"):
        dimensions["sequence_sha256"] = lambda r: [r["sequence_sha256_a"], r["sequence_sha256_b"]]
        key = "cluster" if kind == "protein" else "superfamily"
        dimensions[key] = lambda r: [r[f"{key}_a"], r[f"{key}_b"]]
    elif kind == "dyck":
        dimensions["shape_id"] = lambda r: [r["shape_id"]]
        dimensions["sentence_exact"] = lambda r: [r["sentence"]]
    elif kind == "longrange":
        dimensions["sentence_exact"] = lambda r: [r["sentence"]]
    else:
        dimensions["normalized_input"] = lambda r: [stable_hash(*sorted(norm(r[k]) for k in ("sentence1", "sentence2") if r.get(k) is not None))]
    report = {}
    for dimension, values in dimensions.items():
        sets = {s: {v for row in rows for v in values(row)} for s, rows in splits.items()}
        overlaps = {f"{a}/{b}": len(sets[a] & sets[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
        if any(overlaps.values()):
            raise ValueError(f"Cross-split {dimension} overlap: {task}: {overlaps}")
        report[dimension] = {"unique_counts": {s: len(v) for s, v in sets.items()}, "cross_split_overlap": overlaps}
    return report


def prepared_row(row: dict, task: str) -> dict:
    spec = TASKS[task]
    result = {"row_id": row["row_id"], "label": row["label"]}
    for key in spec["inputs"]:
        result[key] = row.get(key, row.get("sentence1")) if key == "sentence" else row[key]
    keys = {spec["group"], "split", "official_split", "shape_id", "head_number", "nearest_distractor_number", "verb_root"}
    if spec["kind"] in ("protein", "remote"):
        keys.update(f"{key}_{role}" for role in ("a", "b") for key in ("sequence_sha256", "accession", "sequence_id", "cluster", "superfamily", "family"))
    result["metadata"] = {key: row[key] for key in sorted(keys) if key in row}
    return result


def write_json(path: Path, content) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "prepared")
    parser.add_argument("--train-target", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    root = args.release.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing prepared data: {output}")
    gate_path = root / "validation/integrated_release_acceptance.json"
    gate = json.loads(gate_path.read_text())
    release_manifest_path = root / "manifest.json"
    release_manifest = json.loads(release_manifest_path.read_text())
    if not gate.get("training_gate_pass") or not all(gate["gates"].values()) or not release_manifest.get("training_gate_pass"):
        raise SystemExit("Integrated release gate must pass")
    pinned = {item["file"]: item["sha256"] for item in release_manifest["files"]}
    if pinned["validation/integrated_release_acceptance.json"] != sha256(gate_path):
        raise SystemExit("Acceptance gate differs from frozen manifest")
    manifest = {"schema_version": 1, "release_id": release_manifest["release_id"], "release_manifest_sha256": sha256(release_manifest_path), "release_gate_sha256": sha256(gate_path), "builder_sha256": sha256(Path(__file__)), "seed": args.seed, "train_target": args.train_target, "model_independent_selection": True, "validation_test_subsampled": False, "output_row_schema": {"model_input_fields": "task-specific whitelist only", "row_id": "source-stable ID", "label": "unchanged integer 0 or 1", "metadata": "audit-only; never concatenate, tokenize, or give to model"}, "tasks": {}}
    output.mkdir(parents=True)
    for task, spec in TASKS.items():
        splits = {}
        report = {"kind": spec["kind"], "model_input_fields": spec["inputs"], "label_mapping": spec["labels"], "labels_remapped": False, "primary_metric": "matthews_correlation" if task == "cola" else "accuracy", "secondary_metrics": ["accuracy", "auroc_label_1"], "splits": {}}
        for split in SPLITS:
            path = source_file(root, task, split)
            relative = str(path.relative_to(root))
            actual_hash = sha256(path)
            if pinned.get(relative) != actual_hash:
                raise ValueError(f"Source hash mismatch: {relative}")
            rows = read_rows(path)
            source_count = len(rows)
            # Audit source first, so a sampled split cannot conceal source defects.
            source_audit = audit_split(rows, task, split)
            splits[split] = rows
            report["splits"][split] = {"source_file": relative, "source_sha256": actual_hash, "source_rows": source_count, "source_audit": source_audit}
        report["source_isolation"] = isolation_audit(splits, task)
        splits["train"], report["training_selection"] = select_train(splits["train"], task, args.train_target, args.seed)
        report["prepared_isolation"] = isolation_audit(splits, task)
        train_labels = Counter(row["label"] for row in splits["train"])
        majority_label = max((0, 1), key=lambda label: (train_labels[label], -label))
        report["majority_baseline"] = {"fit_split": "selected_train_only", "label": majority_label, "tie_break": "label_0", "accuracy": {split: sum(row["label"] == majority_label for row in rows) / len(rows) for split, rows in splits.items()}, "matthews_correlation": 0.0, "auroc_label_1": 0.5, "note": "Test labels used only to score this fixed train-derived classifier; never to choose majority label."}
        for split, rows in splits.items():
            split_report = report["splits"][split]
            split_report["prepared_audit"] = audit_split(rows, task, split)
            target = output / task / f"{split}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(prepared_row(row, task), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            ids = [row["row_id"] for row in rows]
            ids_path = output / task / f"{split}.row_ids.txt"
            ids_path.write_text("".join(row_id + "\n" for row_id in ids))
            split_report.update({"file": str(target.relative_to(output)), "rows": len(rows), "sha256": sha256(target), "selected_row_ids_file": str(ids_path.relative_to(output)), "selected_row_ids_sha256": sha256(ids_path)})
        manifest["tasks"][task] = report
        print(task, {s: len(rows) for s, rows in splits.items()}, flush=True)
    manifest["status"] = "accepted_for_fixed_downstream_evaluation"
    write_json(output / "manifest.json", manifest)
    write_json(output / "acceptance.json", {"status": "passed", "tasks": len(TASKS), "source_hashes_verified": True, "split_isolation_verified": True, "whole_group_selection_verified": True, "endpoint_role_degree_balance_verified": True, "metadata_excluded_from_model_input_whitelist": True, "manifest_sha256": sha256(output / "manifest.json")})
    print(f"Accepted prepared data: {output}", flush=True)


if __name__ == "__main__":
    main()
