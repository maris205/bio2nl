#!/usr/bin/env python3
"""Rebuild pinned NLP and count-matched typed-Dyck tasks. No training or model imports."""
import argparse
from collections import Counter, defaultdict
import datetime as dt
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import unicodedata
import urllib.request

HERE = Path(__file__).resolve().parent
OPEN = "([{"
CLOSE = ")]}"
SPLITS = ("train", "validation", "test")
EXPECTED = {"pawsx_en": (49401, 2000, 2000), "cola": (8551, 1043, 1063), "rte": (2490, 277, 3000)}


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(value):
    return digest_bytes(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {"path": str(path), "sha256": digest_file(path), "rows": len(rows), "bytes": path.stat().st_size}


def source_code():
    return {p.name: digest_file(p) for p in sorted(HERE.iterdir()) if p.suffix in (".py", ".json", ".md")}


def acquire(source, spec, root, offline):
    target = root / "metadata/nlp_raw" / source["task"] / Path(spec["path"]).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        cached = (Path.home() / ".cache/huggingface/hub" /
                  ("datasets--" + source["repo_id"].replace("/", "--")) /
                  "snapshots" / source["revision"] / spec["path"])
        if cached.is_file() and digest_file(cached) == spec["sha256"]:
            shutil.copyfile(cached, target)
        elif offline:
            raise FileNotFoundError(f"No checksum-matched raw source: {spec['url']}")
        else:
            temporary = target.with_suffix(".download")
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "bio2nl-data-rebuild/2"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as out:
                shutil.copyfileobj(response, out)
            if digest_file(temporary) != spec["sha256"]:
                raise ValueError(f"Downloaded checksum mismatch: {spec['url']}")
            temporary.replace(target)
    if digest_file(target) != spec["sha256"] or target.stat().st_size != spec["size"]:
        raise ValueError(f"Raw source integrity failure: {target}")
    return target


def overlap_report(by_split):
    """Report official overlap without silently changing the official benchmark."""
    ordered, unordered, endpoints = {}, {}, {}
    for split, rows in by_split.items():
        ordered[split], unordered[split], endpoints[split] = defaultdict(list), defaultdict(list), set()
        for row in rows:
            texts = [row["sentence1"]] + ([row["sentence2"]] if row["sentence2"] is not None else [])
            ordered[split][stable_hash(texts)].append(row["row_id"])
            unordered[split][stable_hash(sorted(texts))].append(row["row_id"])
            endpoints[split].update(stable_hash(s) for s in texts)
    result = {"within_split_duplicate_input_groups": {}, "cross_split": {}}
    for split in SPLITS:
        result["within_split_duplicate_input_groups"][split] = [ids for ids in ordered[split].values() if len(ids) > 1]
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            shared = set(ordered[a]) & set(ordered[b])
            shared_unordered = set(unordered[a]) & set(unordered[b])
            result["cross_split"][f"{a}__{b}"] = {
                "exact_ordered_input_groups": len(shared), "unordered_input_groups": len(shared_unordered),
                "shared_sentence_hashes": len(endpoints[a] & endpoints[b]),
                "ordered_row_ids": [{a: ordered[a][k], b: ordered[b][k]} for k in sorted(shared)],
                "unordered_row_ids": [{a: unordered[a][k], b: unordered[b][k]} for k in sorted(shared_unordered)]}
    return result


def build_nlp(root, offline):
    import pyarrow.parquet as pq
    lock = json.loads((HERE / "nlp_sources_lock.json").read_text())
    manifest = {"schema_version": 2, "kind": "official_nlp_splits", "sources": [],
                "code_sha256": source_code(), "normalization": "none; original Unicode and row order preserved",
                "test_policy": "No test-label fitting/tuning; hidden GLUE test labels become null with upstream_label=-1",
                "label_maps": {"pawsx_en": {"0": "not_paraphrase", "1": "paraphrase"},
                               "cola": {"0": "unacceptable", "1": "acceptable"},
                               "rte": {"0": "entailment", "1": "not_entailment"}},
                "packages": {"pyarrow": importlib.metadata.version("pyarrow")}}
    report = {"construction_acceptance_pass": True, "official_splits_preserved": True,
              "test_labels_used_for_tuning": False, "tasks": {}, "warnings": []}
    overlaps = {}
    for source in lock["sources"]:
        task, by_split, entry = source["task"], {}, dict(source)
        entry["outputs"] = []
        for spec in source["files"]:
            raw = acquire(source, spec, root, offline)
            split = spec["split"]
            rows = []
            for index, original in enumerate(pq.read_table(raw).to_pylist()):
                upstream_id = original.get("id", original.get("idx"))
                label = int(original["label"])
                rows.append({"task": task, "split": split,
                             "row_id": task + ":" + split + ":" + str(index) + ":" + stable_hash(
                                 [source["repo_id"], source["revision"], source["config"], split, upstream_id, index])[:16],
                             "source_row_index": index, "upstream_id": upstream_id,
                             "source_revision": source["revision"], "source_raw_sha256": spec["sha256"],
                             "sentence1": original.get("sentence1", original.get("sentence")),
                             "sentence2": original.get("sentence2"), "label": label if label >= 0 else None,
                             "has_label": label >= 0, "upstream_label": label})
            if len(rows) != EXPECTED[task][SPLITS.index(split)]:
                raise ValueError(f"Unexpected official row count: {task}/{split}")
            if any(not isinstance(r["sentence1"], str) or r["label"] not in (None, 0, 1) for r in rows):
                raise ValueError(f"Invalid schema: {task}/{split}")
            if len({r["row_id"] for r in rows}) != len(rows):
                raise ValueError(f"Duplicate row IDs: {task}/{split}")
            by_split[split] = rows
            artifact = write_rows(root / "data/nlp" / task / f"{split}.jsonl", rows)
            artifact["path"] = str(Path(artifact["path"]).relative_to(root))
            artifact["raw_path"] = str(raw.relative_to(root))
            entry["outputs"].append(artifact)
        train_labels = Counter(r["label"] for r in by_split["train"])
        majority_label = min(train_labels, key=lambda k: (-train_labels[k], k))
        stats = {}
        for split, rows in by_split.items():
            labels = Counter(r["label"] for r in rows if r["has_label"])
            labeled_n = sum(labels.values())
            stats[split] = {"rows": len(rows), "label_counts": dict(labels), "labeled_rows": labeled_n,
                            "train_majority_label": majority_label,
                            "train_majority_accuracy": labels[majority_label] / labeled_n if labeled_n else None,
                            "split_majority_fraction_descriptive_only": max(labels.values()) / labeled_n if labeled_n else None}
        overlaps[task] = overlap_report(by_split)
        stats["cross_split_exact_input_groups"] = sum(x["exact_ordered_input_groups"] for x in overlaps[task]["cross_split"].values())
        stats["cross_split_unordered_input_groups"] = sum(x["unordered_input_groups"] for x in overlaps[task]["cross_split"].values())
        stats["within_split_duplicate_input_groups"] = {s: len(v) for s, v in overlaps[task]["within_split_duplicate_input_groups"].items()}
        if stats["cross_split_exact_input_groups"] or stats["cross_split_unordered_input_groups"]:
            report["warnings"].append(f"{task}: official cross-split duplicate inputs retained and enumerated; strict input-disjoint acceptance is not passed")
        report["tasks"][task] = stats
        manifest["sources"].append(entry)
        print(f"Rebuilt {task}: " + "/".join(str(len(by_split[s])) for s in SPLITS), flush=True)
    report["strict_cross_split_input_disjoint_acceptance_pass"] = not report["warnings"]
    report["warnings"].append("GLUE CoLA/RTE official test labels are hidden; local scored evaluation uses validation, never fabricated test labels")
    report["warnings"].append("Official shared sentences and duplicate rows are retained for benchmark comparability; this release does not claim entity-disjoint NLP splits")
    write_json(root / "metadata/nlp_sources_manifest.json", manifest)
    write_json(root / "validation/nlp_acceptance_report.json", report)
    write_json(root / "validation/nlp_official_overlap_details.json", overlaps)
    shutil.copyfile(HERE / "nlp_sources_lock.json", root / "metadata/nlp_sources_lock.json")
    for card in HERE.glob("nlp_source_card_*.md"):
        shutil.copyfile(card, root / "metadata" / card.name)
    return report


def canonical_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def input_keys(row):
    texts = [canonical_text(row["sentence1"])]
    if row["sentence2"] is not None:
        texts.append(canonical_text(row["sentence2"]))
    # Entailment is directional: reversed RTE pairs share a split group but are
    # separate semantic inputs and need not have the same label.
    semantic = sorted(texts) if row["task"] == "pawsx_en" else texts
    return stable_hash(sorted(texts)), stable_hash(semantic)


def build_clean_nlp(root, seed):
    exclusions, conflicts, manifest = [], {}, {"schema_version": 2, "outputs": [], "code_sha256": source_code(),
        "seed": seed, "canonicalization": "Unicode NFKC, casefold, whitespace collapse; output original text unchanged",
        "grouping": "unordered normalized text pair for PAWS/RTE, normalized sentence for CoLA",
        "label_conflicts": "PAWS unordered pair; RTE ordered semantic input; CoLA sentence; remove every labeled occurrence of conflicting semantic input",
        "protocols": {"pawsx_en": "official test priority, then official validation, then train; deduplicate and remove conflicts",
                      "cola": "clean official validation becomes final test; clean official train grouped into 90/10 train/inner-validation",
                      "rte": "clean official validation becomes final test; clean official train grouped into 90/10 train/inner-validation"},
        "test_labels_used_for_tuning": False,
        "labels_used_for_predefined_conflict_filtering": True,
        "split_assignment_label_source": "official train only for CoLA/RTE stratification; no final-test labels drive assignment",
        "hidden_official_tests": "retained under data/nlp but unused for scored clean local protocol"}
    report = {"acceptance_pass": True, "tasks": {}, "labels_used_for_model_or_threshold_tuning": False,
              "limitations": ["New cleaned protocols differ from official benchmark leaderboards",
                              "Input-group isolation does not guarantee isolated individual sentences, paraphrases, or semantics",
                              "CoLA/RTE final test is the cleaned public official validation split, not the hidden official test",
                              "Contradictory-label filtering uses a fixed quality rule across labeled source rows; it is not model selection"]}
    for task in EXPECTED:
        official = {s: [json.loads(line) for line in (root / "data/nlp" / task / f"{s}.jsonl").read_text().splitlines()]
                    for s in SPLITS}
        labeled = [r for rows in official.values() for r in rows if r["has_label"]]
        semantics = defaultdict(list)
        for row in labeled:
            group, semantic = input_keys(row)
            row["input_group_id"], row["semantic_input_id"] = group, semantic
            semantics[semantic].append(row)
        bad = {key for key, rows in semantics.items() if len({r["label"] for r in rows}) > 1}
        conflicts[task] = [{"semantic_input_id": key,
                            "rows": [{"row_id": r["row_id"], "official_split": r["split"], "label": r["label"]}
                                     for r in semantics[key]]} for key in sorted(bad)]
        for key in bad:
            for row in semantics[key]:
                exclusions.append({"task": task, "row_id": row["row_id"], "official_split": row["split"],
                                   "reason": "contradictory_labels_for_same_semantic_input", "input_group_id": row["input_group_id"]})

        def filter_rows(rows, forbidden_groups, accepted_semantics):
            kept = []
            for row in rows:
                semantic, group = row["semantic_input_id"], row["input_group_id"]
                if semantic in bad:
                    continue
                reason = ("input_group_reserved_by_higher_priority_evaluation_split" if group in forbidden_groups else
                          "duplicate_semantic_input" if semantic in accepted_semantics else None)
                if reason:
                    exclusions.append({"task": task, "row_id": row["row_id"], "official_split": row["split"],
                                       "reason": reason, "input_group_id": group})
                    continue
                accepted_semantics.add(semantic)
                kept.append(row)
            return kept

        if task == "pawsx_en":
            seen_semantics, reserved, clean = set(), set(), {}
            for split in ("test", "validation", "train"):
                clean[split] = filter_rows(official[split], reserved, seen_semantics)
                reserved.update(r["input_group_id"] for r in clean[split])
        else:
            test = filter_rows(official["validation"], set(), set())
            train_pool = filter_rows(official["train"], {r["input_group_id"] for r in test}, set())
            groups = defaultdict(list)
            for row in train_pool:
                groups[row["input_group_id"]].append(row)
            strata = defaultdict(list)
            for key, rows in groups.items():
                strata[tuple(sorted(Counter(r["label"] for r in rows).items()))].append(key)
            validation_groups = set()
            for keys in strata.values():
                keys.sort(key=lambda key: stable_hash([seed, task, key]))
                n_validation = round(len(keys) * 0.1)
                validation_groups.update(keys[:n_validation])
            clean = {"test": test,
                     "validation": [r for r in train_pool if r["input_group_id"] in validation_groups],
                     "train": [r for r in train_pool if r["input_group_id"] not in validation_groups]}
        owner = {}
        task_report = {"source_labeled_rows": len(labeled), "conflicting_semantic_input_groups": len(bad),
                       "recommended_primary_metric": "matthews_correlation_coefficient" if task == "cola" else "accuracy",
                       "additional_metrics": ["accuracy", "auroc", "matthews_correlation_coefficient"],
                       "normalization_changes_model_inputs": False, "cross_split_input_group_overlaps": 0}
        majority_label = Counter(r["label"] for r in clean["train"]).most_common(1)[0][0]
        for split in SPLITS:
            rows = []
            for row in clean[split]:
                group = row["input_group_id"]
                if group in owner and owner[group] != split:
                    raise ValueError(f"Cross-split group in clean {task}")
                owner[group] = split
                rows.append({**row, "official_split": row["split"], "split": split,
                             "clean_protocol": "nlp_input_disjoint_v2", "clean_split_seed": seed})
            if not rows or any(r["label"] not in (0, 1) for r in rows):
                raise ValueError(f"Missing clean labels: {task}/{split}")
            if len({r["semantic_input_id"] for r in rows}) != len(rows):
                raise ValueError(f"Duplicate clean semantic input: {task}/{split}")
            artifact = write_rows(root / "data/nlp_clean" / task / f"{split}.jsonl", rows)
            artifact["path"] = str(Path(artifact["path"]).relative_to(root))
            manifest["outputs"].append(artifact)
            labels = Counter(r["label"] for r in rows)
            task_report[split] = {"rows": len(rows), "input_groups": len({r["input_group_id"] for r in rows}),
                                  "label_counts": dict(labels), "train_majority_label": majority_label,
                                  "train_majority_accuracy": labels[majority_label] / len(rows),
                                  "split_majority_fraction_descriptive_only": max(labels.values()) / len(rows),
                                  "official_split_counts": dict(Counter(r["official_split"] for r in rows))}
        task_report["excluded_labeled_rows"] = sum(x["task"] == task for x in exclusions)
        if task_report["excluded_labeled_rows"] + sum(task_report[s]["rows"] for s in SPLITS) != len(labeled):
            raise ValueError(f"Clean row accounting does not close for {task}")
        report["tasks"][task] = task_report
        print(f"Clean {task}: " + "/".join(str(task_report[s]["rows"]) for s in SPLITS), flush=True)
    write_json(root / "metadata/nlp_clean_manifest.json", manifest)
    write_json(root / "validation/nlp_clean_acceptance_report.json", report)
    write_json(root / "validation/nlp_clean_label_conflicts.json", conflicts)
    write_rows(root / "validation/nlp_clean_exclusions.jsonl", exclusions)
    return report


@lru_cache(None)
def completions(opens, closes):
    if opens < 0 or closes < opens:
        return 0
    if opens == 0:
        return 1
    return completions(opens - 1, closes) + (completions(opens, closes - 1) if closes > opens else 0)


def sample_positive(length, rng):
    """Uniform Catalan shape, independent uniform type per matched pair, then eligibility filtering."""
    opens = closes = length // 2
    seq, stack, pairs = [], [], []
    while closes:
        n_open = completions(opens - 1, closes) if opens else 0
        n_close = completions(opens, closes - 1) if closes > opens else 0
        if rng.randrange(n_open + n_close) < n_open:
            bracket_type = rng.randrange(3)
            stack.append((len(seq), bracket_type))
            seq.append(OPEN[bracket_type])
            opens -= 1
        else:
            start, bracket_type = stack.pop()
            pairs.append((start, len(seq), bracket_type))
            seq.append(CLOSE[bracket_type])
            closes -= 1
    return "".join(seq), pairs


def stack_accepts(seq):
    stack = []
    for char in seq:
        if char in OPEN:
            stack.append(OPEN.index(char))
        elif char in CLOSE and stack and stack.pop() == CLOSE.index(char):
            pass
        else:
            return False
    return not stack


def counter_accepts(seq, typed=True, prefix=False):
    balance = [0] * (3 if typed else 1)
    for char in seq:
        index = OPEN.index(char) if char in OPEN else CLOSE.index(char)
        balance[index if typed else 0] += 1 if char in OPEN else -1
        if prefix and min(balance) < 0:
            return False
    return not any(balance)


def adjacent_types_accept(seq):
    return all(OPEN.index(a) == CLOSE.index(b) for a, b in zip(seq, seq[1:]) if a in OPEN and b in CLOSE)


def make_negative(positive, pairs, rng):
    length = len(positive)
    candidates = [(inner, outer) for outer in pairs for inner in pairs
                  if outer[0] < inner[0] < inner[1] < outer[1] and inner[2] != outer[2]
                  and all(4 <= p[1] < length - 4 and p[1] - p[0] >= 5
                          and positive[p[1] - 1] in CLOSE for p in (inner, outer))]
    if not candidates:
        return None
    inner, outer = rng.choice(candidates)
    negative = list(positive)
    negative[inner[1]], negative[outer[1]] = negative[outer[1]], negative[inner[1]]
    return "".join(negative), [inner[1], outer[1]]


def validate_synthetic(by_task):
    report = {"acceptance_pass": True, "tasks": {}, "checks": {
        "balanced_labels": True, "exact_sequence_disjoint": True, "shape_disjoint_within_task": True,
        "pair_counts_matched": True, "pair_prefix_suffix_4_matched": True,
        "all_typed_prefix_counters_nonnegative": True, "all_untyped_dyck_valid": True,
        "all_adjacent_leaf_pairs_valid": True, "stack_labels_exact": True}}
    globally_seen = set()
    for task, by_split in by_task.items():
        shape_owner = {}
        splits = {}
        for split, rows in by_split.items():
            labels = Counter(r["label"] for r in rows)
            assert labels[0] == labels[1]
            groups = defaultdict(list)
            for row in rows:
                seq = row["sequence"]
                assert seq not in globally_seen
                globally_seen.add(seq)
                assert len(seq) == row["length"]
                assert stack_accepts(seq) == bool(row["label"])
                assert counter_accepts(seq, typed=True, prefix=True)
                assert counter_accepts(seq, typed=False, prefix=True)
                assert adjacent_types_accept(seq)
                shape = row["shape_id"]
                assert shape not in shape_owner or shape_owner[shape] == split
                shape_owner[shape] = split
                groups[row["pair_group_id"]].append(row)
            for group in groups.values():
                assert len(group) == 2 and {r["label"] for r in group} == {0, 1}
                a, b = (r["sequence"] for r in group)
                assert Counter(a) == Counter(b)
                assert a[:4] == b[:4] and a[-4:] == b[-4:]
            baselines = {}
            rules = {"typed_global_count": lambda s: counter_accepts(s),
                     "typed_prefix_counters": lambda s: counter_accepts(s, prefix=True),
                     "untyped_stack": lambda s: counter_accepts(s, typed=False, prefix=True),
                     "adjacent_leaf_type_check": adjacent_types_accept,
                     "typed_stack_oracle": stack_accepts}
            for name, rule in rules.items():
                baselines[name] = sum(int(rule(r["sequence"])) == r["label"] for r in rows) / len(rows)
            assert baselines["typed_stack_oracle"] == 1.0
            assert all(v == 0.5 for k, v in baselines.items() if k != "typed_stack_oracle")
            splits[split] = {"rows": len(rows), "labels": dict(labels), "unique_sequences": len(rows),
                             "unique_shapes": len(groups), "baselines_accuracy": baselines}
        report["tasks"][task] = splits
    return report


def build_synthetic(root, seed, sizes):
    all_tasks, artifacts, attempts_by_task = {}, [], {}
    for length in (40, 60):
        task = f"dyck_L{length}_t3"
        rng = random.Random(seed + length)
        seen_sequences, seen_shapes, by_split = set(), set(), {}
        attempts = Counter()
        for split, count in zip(SPLITS, sizes):
            rows = []
            while len(rows) < count:
                attempts["sampled_shapes"] += 1
                positive, pairs = sample_positive(length, rng)
                negative_result = make_negative(positive, pairs, rng)
                if negative_result is None:
                    attempts["no_eligible_nested_swap"] += 1
                    continue
                negative, swap_positions = negative_result
                shape = "".join("(" if c in OPEN else ")" for c in positive)
                shape_id = stable_hash(shape)
                if shape_id in seen_shapes or positive in seen_sequences or negative in seen_sequences:
                    attempts["duplicate_shape_or_sequence"] += 1
                    continue
                seen_shapes.add(shape_id)
                seen_sequences.update([positive, negative])
                group_id = f"{task}:" + stable_hash([positive, negative])[:24]
                for sequence, label in ((positive, 1), (negative, 0)):
                    rows.append({"task": task, "split": split, "row_id": f"{task}:" + stable_hash(sequence),
                                 "sentence": " ".join(sequence), "sequence": sequence, "label": label,
                                 "length": length, "bracket_types": 3, "pair_group_id": group_id,
                                 "shape_id": shape_id, "mutation_positions": swap_positions,
                                 "generator_seed": seed + length})
            rng.shuffle(rows)
            by_split[split] = rows
            artifact = write_rows(root / "data/synthetic" / task / f"{split}.jsonl", rows)
            artifact["path"] = str(Path(artifact["path"]).relative_to(root))
            artifacts.append(artifact)
        all_tasks[task] = by_split
        attempts_by_task[task] = dict(attempts)
        print(f"Rebuilt {task}: {sizes}", flush=True)
    report = validate_synthetic(all_tasks)
    report["generation_attempts"] = attempts_by_task
    report["limitations"] = ["Positive distribution is uniform Catalan sampling conditioned on an eligible nested swap; not all Dyck strings are represented",
                              "Count/prefix/adjacency controls and stack oracle do not prove absence of every local or learned shortcut",
                              "Each mutation pair shares a shape within its split; shapes are disjoint across splits",
                              "L40 and L60 are exact bracket-token lengths, without the spaces in sentence; model tokenizer length may differ"]
    write_json(root / "validation/synthetic_dyck_acceptance_report.json", report)
    write_json(root / "metadata/synthetic_dyck_manifest.json", {
        "schema_version": 2, "generator": "typed_dyck_nested_closer_swap_v2", "base_seed": seed,
        "code_sha256": source_code(), "outputs": artifacts, "split_sizes": dict(zip(SPLITS, sizes)),
        "alphabet": OPEN + CLOSE, "positive_label": 1, "negative_label": 0,
        "negative_construction": "Swap different-type closing symbols of nested non-leaf nodes; preserve first/last four tokens and each type count; typed-prefix counts stay nonnegative",
        "split_unit": "shape and positive-negative mutation group, before output shuffling",
        "serialization": "sequence=raw brackets; sentence=one ASCII space between brackets, no wrappers",
        "training_input_fields": ["sentence"], "label_field": "label",
        "metadata_must_not_be_model_features": ["row_id", "pair_group_id", "shape_id", "mutation_positions", "generator_seed"]})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "nlp", "clean", "synthetic"), default="all")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--synthetic-sizes", default="16000,2000,2000")
    args = parser.parse_args()
    sizes = tuple(int(x) for x in args.synthetic_sizes.split(","))
    if len(sizes) != 3 or any(x <= 0 or x % 2 for x in sizes):
        parser.error("synthetic sizes must be three positive even integers")
    root = args.release_root.resolve()
    (root / "metadata").mkdir(parents=True, exist_ok=True)
    if args.mode in ("all", "nlp"):
        build_nlp(root, args.offline)
    if args.mode in ("all", "clean"):
        build_clean_nlp(root, args.seed)
    if args.mode in ("all", "synthetic"):
        build_synthetic(root, args.seed, sizes)
    shutil.copyfile(HERE / "README.md", root / "metadata/nlp_synthetic_README.md")
    print("Data build finished; consult both acceptance reports for limitations.")


if __name__ == "__main__":
    main()
