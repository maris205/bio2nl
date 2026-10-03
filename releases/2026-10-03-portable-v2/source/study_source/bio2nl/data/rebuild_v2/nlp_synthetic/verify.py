#!/usr/bin/env python3
"""Independently verify saved NLP rows and Dyck strings, without training models."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import unicodedata


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def reduction_accepts(sequence):
    """An independent repeated-reduction oracle, not the generator's stack code."""
    while True:
        reduced = sequence.replace("()", "").replace("[]", "").replace("{}", "")
        if reduced == sequence:
            return not reduced
        sequence = reduced


def main():
    import pyarrow.parquet as pq
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release-root", type=Path, required=True)
    a = p.parse_args()
    root = a.release_root.resolve()
    result = {"passed": True, "official_rows_checked": 0, "clean_rows_checked": 0,
              "synthetic_rows_checked": 0, "independent_oracle": "iterative elimination of (), [], {}"}
    manifest = json.loads((root / "metadata/nlp_sources_manifest.json").read_text())
    for source in manifest["sources"]:
        for artifact, raw_spec in zip(source["outputs"], source["files"]):
            target = root / artifact["path"]
            raw = root / artifact["raw_path"]
            require(sha(target) == artifact["sha256"], f"Output checksum {target}")
            require(sha(raw) == raw_spec["sha256"], f"Raw checksum {raw}")
            observed, original = rows(target), pq.read_table(raw).to_pylist()
            require(len(observed) == len(original), f"Row count {target}")
            for index, (r, upstream) in enumerate(zip(observed, original)):
                require(r["source_row_index"] == index, "Source row index")
                require(r["upstream_id"] == upstream.get("id", upstream.get("idx")), "Source ID")
                require(r["sentence1"] == upstream.get("sentence1", upstream.get("sentence")), "Sentence1 changed")
                require(r["sentence2"] == upstream.get("sentence2"), "Sentence2 changed")
                require(r["upstream_label"] == upstream["label"], "Source label changed")
                require(r["label"] == (upstream["label"] if upstream["label"] >= 0 else None), "Label conversion")
                require(r["source_revision"] == source["revision"], "Source revision")
            result["official_rows_checked"] += len(observed)
    clean_manifest = json.loads((root / "metadata/nlp_clean_manifest.json").read_text())
    owners = {}
    clean_ids = set()
    for artifact in clean_manifest["outputs"]:
        target = root / artifact["path"]
        require(sha(target) == artifact["sha256"], f"Clean checksum {target}")
        source_rows = {}
        for r in rows(target):
            task, split = r["task"], r["split"]
            official = root / "data/nlp" / task / f"{r['official_split']}.jsonl"
            if str(official) not in source_rows:
                source_rows[str(official)] = {x["row_id"]: x for x in rows(official)}
            original = source_rows[str(official)][r["row_id"]]
            require(all(r[k] == original[k] for k in ("sentence1", "sentence2", "label", "upstream_id", "source_row_index")), "Clean source mismatch")
            texts = [normalized(r["sentence1"])] + ([normalized(r["sentence2"])] if r["sentence2"] is not None else [])
            key = (task, tuple(sorted(texts)))
            require(key not in owners or owners[key] == split, "Clean input group crossed splits")
            owners[key] = split
            require(r["row_id"] not in clean_ids, "Clean row repeated")
            clean_ids.add(r["row_id"])
            result["clean_rows_checked"] += 1
    excluded = rows(root / "validation/nlp_clean_exclusions.jsonl")
    excluded_ids = [r["row_id"] for r in excluded]
    require(len(excluded_ids) == len(set(excluded_ids)), "Exclusion row counted twice")
    require(not (set(excluded_ids) & clean_ids), "Excluded row also retained")
    labeled_ids = set()
    for task in ("pawsx_en", "cola", "rte"):
        for split in ("train", "validation", "test"):
            labeled_ids.update(r["row_id"] for r in rows(root / "data/nlp" / task / f"{split}.jsonl") if r["has_label"])
    require(labeled_ids == clean_ids | set(excluded_ids), "Retained/excluded accounting mismatch")
    synthetic_manifest = json.loads((root / "metadata/synthetic_dyck_manifest.json").read_text())
    all_strings, shape_owners = set(), {}
    for artifact in synthetic_manifest["outputs"]:
        target = root / artifact["path"]
        require(sha(target) == artifact["sha256"], f"Synthetic checksum {target}")
        data = rows(target)
        labels = Counter(r["label"] for r in data)
        require(labels[0] == labels[1], "Unbalanced synthetic labels")
        groups = defaultdict(list)
        for r in data:
            seq = r["sequence"]
            require(seq not in all_strings, "Repeated synthetic string")
            all_strings.add(seq)
            require(len(seq) == r["length"] and r["sentence"] == " ".join(seq), "Synthetic serialization")
            require(reduction_accepts(seq) == bool(r["label"]), "Independent oracle disagreement")
            for opening, closing in zip("([{", ")]}"):
                balance = 0
                for char in seq:
                    balance += (char == opening) - (char == closing)
                    require(balance >= 0, "Typed negative prefix counter")
                require(balance == 0, "Unequal typed totals")
            shape = "".join("(" if c in "([{ " else ")" for c in seq)
            key = (r["task"], shape)
            require(key not in shape_owners or shape_owners[key] == r["split"], "Shape crossed splits")
            shape_owners[key] = r["split"]
            groups[r["pair_group_id"]].append(r)
        for group in groups.values():
            require(len(group) == 2 and {r["label"] for r in group} == {0, 1}, "Mutation group incomplete")
            left, right = (r["sequence"] for r in group)
            require(Counter(left) == Counter(right), "Mutation changed counts")
            require(left[:4] == right[:4] and left[-4:] == right[-4:], "Mutation changed boundaries")
        result["synthetic_rows_checked"] += len(data)
    result["excluded_labeled_rows_checked"] = len(excluded_ids)
    output = root / "validation/nlp_synthetic_independent_verification.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
