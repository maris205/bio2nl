#!/usr/bin/env python3
"""Build and audit SCOPe 2.08 remote domain pairs from official raw data.

Stages prepare and pairs are separate so the canonical domain FASTA is available
early for the independently run pretraining-overlap exclusion. CPU only.
"""
from __future__ import annotations

import os
for thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[thread_var] = "1"

import argparse
import csv
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import random
from collections import Counter, defaultdict
from pathlib import Path

import Bio
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

AA = "ACDEFGHIKLMNPQRSTVWY"
SPLITS = ("train", "validation", "test")
CONFIG = {
    "release": "SCOPe-2.08-stable", "seed": 20260925,
    "allowed_classes": "abcdefg", "canonical_alphabet": AA,
    "min_domain_length": 40, "max_domain_length": 500,
    "split_unit": "SCOPe superfamily sunid", "split_fractions": [0.70, 0.15, 0.15],
    "split_stratification": "class code; seeded shuffle of sorted superfamilies",
    "positive": "same superfamily; different nonzero classified family",
    "negative": "different superfamily; not a claim of proven nonhomology",
    "alignment": "end-to-end global Needleman-Wunsch affine-gap; Biopython PairwiseAligner",
    "matrix": "BLOSUM62", "gap_open": -10.0, "gap_extend": -0.5,
    "terminal_gaps_penalized": True,
    "remote_identity_definition": "identical residues / non-gap paired residues",
    "remote_identity_strict_upper_bound": 0.25,
    "minimum_both_paired_residue_coverage": 0.60,
    "same_alignment_filters_for_positive_and_negative": True,
    "max_candidate_domains_per_family": 24,
    "max_positive_candidates_per_superfamily": 5000,
    "max_positive_edges_per_superfamily": 100,
    "max_positive_degree_per_domain": 3,
    "negative_construction": "two positive edges from different superfamilies -> two cross-rewired negative edges",
    "negative_match_pool": 128, "negative_alignment_attempts_per_edge": 64,
    "negative_candidate_ranking": "similar corresponding endpoint lengths and composition, fixed a priori",
    "target_pairs": {"train": 14000, "validation": 3000, "test": 3000},
    "minimum_pairs_for_acceptance": {"train": 2000, "validation": 400, "test": 400},
    "minimum_superfamilies_per_split": 10,
    "forbid_self_pairs": True, "forbid_duplicate_unordered_pairs": True,
    "require_per_endpoint_per_role_equal_label_degree": True,
    "pretraining_exclusion": "separate required global release gate; not certified by this builder",
    "cross_superfamily_homology": "SCOPe SF-disjointness does not eliminate all cross-SF sequence similarity; external MMseqs diagnostic required",
}


def digest(path):
    obj = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            obj.update(chunk)
    return obj.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def stable_seed(label):
    return int(text_hash(str(CONFIG["seed"]) + ":" + str(label))[:16], 16)


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fasta(path):
    header, sequence = None, []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence)
                header, sequence = line[1:].strip(), []
            else:
                sequence.append(line.strip())
    if header is not None:
        yield header, "".join(sequence)


def prepare(root):
    raw = root / "raw/remote"
    classification_path = raw / "dir.cla.scope.2.08-stable.txt"
    fasta_path = raw / "astral-scopedom-seqres-gd-all-2.08-stable.fa"
    classification = {}
    with classification_path.open() as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            sid, pdb, boundaries, sccs, domain_sunid, hierarchy = line.rstrip().split("\t")
            levels = dict(item.split("=") for item in hierarchy.split(","))
            classification[sid] = {
                "sequence_id": sid, "pdb_id": pdb, "domain_boundaries": boundaries,
                "sccs": sccs, "domain_sunid": domain_sunid,
                "class_code": sccs.split(".")[0], "fold_id": levels.get("cf", ""),
                "superfamily_id": levels.get("sf", ""), "family_id": levels.get("fa", ""),
            }
    groups, rejects = defaultdict(list), Counter()
    raw_domains = 0
    for header, original in fasta(fasta_path):
        raw_domains += 1
        sid = header.split()[0]
        record = classification.get(sid)
        if record is None:
            rejects["no_classification"] += 1
            continue
        if record["class_code"] not in CONFIG["allowed_classes"]:
            rejects["excluded_class"] += 1
            continue
        if not record["superfamily_id"] or not record["family_id"] or record["sccs"].split(".")[-1] == "0":
            rejects["unknown_family_or_superfamily"] += 1
            continue
        seq = original.upper()
        if not set(seq) <= set(AA):
            rejects["noncanonical_residues_or_gaps"] += 1
            continue
        if not CONFIG["min_domain_length"] <= len(seq) <= CONFIG["max_domain_length"]:
            rejects["length_outside_range"] += 1
            continue
        sequence_hash = text_hash(seq)
        groups[sequence_hash].append({**record, "sequence": seq, "length": len(seq),
                                      "sequence_sha256": sequence_hash})
    canonical = []
    for sequence_hash, records in sorted(groups.items()):
        assignments = {(r["superfamily_id"], r["family_id"]) for r in records}
        if len(assignments) != 1:
            rejects["ambiguous_exact_duplicate_classification"] += len(records)
            continue
        records.sort(key=lambda r: r["sequence_id"])
        row = dict(records[0])
        row["source_sids_json"] = json.dumps([r["sequence_id"] for r in records], separators=(",", ":"))
        row["source_pdbs_json"] = json.dumps(sorted({r["pdb_id"] for r in records}), separators=(",", ":"))
        row["alias_count"] = len(records)
        canonical.append(row)
        rejects["collapsed_exact_duplicate_aliases"] += len(records)-1
    sf_by_class = defaultdict(set)
    for row in canonical:
        sf_by_class[row["class_code"]].add(row["superfamily_id"])
    sf_split = {}
    for cls, values in sorted(sf_by_class.items()):
        ids = sorted(values, key=int)
        random.Random(stable_seed("split:" + cls)).shuffle(ids)
        train_end = int(len(ids) * 0.70)
        validation_end = train_end + int(len(ids) * 0.15)
        for index, sf in enumerate(ids):
            sf_split[sf] = "train" if index < train_end else "validation" if index < validation_end else "test"
    for row in canonical:
        row["split"] = sf_split[row["superfamily_id"]]
    canonical.sort(key=lambda row: row["sequence_id"])
    dest = root / "data/sequences"
    dest.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(canonical)
    table.to_parquet(dest / "remote_sequences.parquet", index=False)
    table.to_csv(dest / "remote_sequences.tsv.gz", sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    with (dest / "remote_domains.fasta").open("w") as handle:
        for row in canonical:
            handle.write(f">{row['sequence_id']} sf={row['superfamily_id']} family={row['family_id']} split={row['split']}\n{row['sequence']}\n")
    sf_rows = []
    for sf, subframe in table.groupby("superfamily_id", sort=True):
        sf_rows.append({"superfamily_id": sf, "split": sf_split[sf], "n_domains": len(subframe),
                        "n_families": subframe.family_id.nunique(), "class_code": subframe.class_code.iloc[0]})
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sf_rows).to_csv(metadata / "remote_superfamily_splits.tsv", sep="\t", index=False)
    stats = {"status": "canonical_domains_prepared", "raw_classified_domains": len(classification),
             "raw_fasta_domains": raw_domains, "canonical_unique_domains": len(table),
             "canonical_superfamilies": len(sf_split), "rejections_and_collapses": dict(rejects),
             "split_domain_counts": dict(Counter(row["split"] for row in canonical)),
             "eligible_cross_family_superfamilies": dict(Counter(r["split"] for r in sf_rows if r["n_families"] >= 2)),
             "raw_sha256": {str(path.relative_to(root)): digest(path) for path in (classification_path, fasta_path)},
             "config": CONFIG, "biopython_version": Bio.__version__}
    dump_json(metadata / "remote_preparation.json", stats)
    dump_json(metadata / "remote_build_config.json", CONFIG)
    print(json.dumps(stats, ensure_ascii=False), flush=True)


_ALIGNER = None


def global_alignment(a, b):
    global _ALIGNER
    if _ALIGNER is None:
        _ALIGNER = PairwiseAligner(mode="global")
        _ALIGNER.substitution_matrix = substitution_matrices.load(CONFIG["matrix"])
        _ALIGNER.open_gap_score = CONFIG["gap_open"]
        _ALIGNER.extend_gap_score = CONFIG["gap_extend"]
    alignment = _ALIGNER.align(a, b)[0]
    coords = alignment.coordinates
    matches = paired = columns = 0
    cigar = []
    for k in range(coords.shape[1]-1):
        a0, a1 = map(int, coords[0, k:k+2])
        b0, b1 = map(int, coords[1, k:k+2])
        na, nb = a1-a0, b1-b0
        columns += max(na, nb)
        if na and nb:
            assert na == nb
            matches += sum(x == y for x, y in zip(a[a0:a1], b[b0:b1]))
            paired += na
            cigar.append(f"{na}M")
        elif na:
            cigar.append(f"{na}D")
        else:
            cigar.append(f"{nb}I")
    return {"alignment_score": float(alignment.score), "identical_residues": matches,
            "paired_residues": paired, "alignment_columns": columns,
            "identity_paired": matches / paired if paired else 0.0,
            "identity_all_columns": matches / columns if columns else 0.0,
            "coverage_a": paired / len(a), "coverage_b": paired / len(b),
            "alignment_cigar_a_to_b": "".join(cigar)}


def alignment_passes(result):
    return (result["identity_paired"] < CONFIG["remote_identity_strict_upper_bound"]
            and min(result["coverage_a"], result["coverage_b"]) >= CONFIG["minimum_both_paired_residue_coverage"])


def positive_worker(task):
    sf, records = task
    rng = random.Random(stable_seed("positive:" + sf))
    families = defaultdict(list)
    for row in records:
        families[row["family_id"]].append(row)
    for fam in families:
        families[fam].sort(key=lambda row: row["sequence_id"])
        rng.shuffle(families[fam])
        families[fam] = families[fam][:CONFIG["max_candidate_domains_per_family"]]
    family_ids = sorted(families)
    seen, degree, edges = set(), Counter(), []
    attempts = alignments = 0
    max_attempts = CONFIG["max_positive_candidates_per_superfamily"]
    while attempts < max_attempts and len(edges) < CONFIG["max_positive_edges_per_superfamily"]:
        attempts += 1
        fa, fb = rng.sample(family_ids, 2)
        a, b = rng.choice(families[fa]), rng.choice(families[fb])
        ia, ib = a["sequence_id"], b["sequence_id"]
        key = tuple(sorted((ia, ib)))
        if key in seen:
            continue
        seen.add(key)
        if max(degree[ia], degree[ib]) >= CONFIG["max_positive_degree_per_domain"]:
            continue
        if min(a["length"], b["length"]) / max(a["length"], b["length"]) < CONFIG["minimum_both_paired_residue_coverage"]:
            continue
        result = global_alignment(a["sequence"], b["sequence"])
        alignments += 1
        if not alignment_passes(result):
            continue
        edges.append({"a": ia, "b": ib, "sf": sf, **result})
        degree[ia] += 1
        degree[ib] += 1
    return edges, {"superfamily_id": sf, "sample_attempts": attempts, "unique_candidate_pairs": len(seen),
                   "alignments": alignments, "remote_positive_edges": len(edges)}


def aa_vector(seq):
    counts = Counter(seq)
    return np.array([counts[x] / len(seq) for x in AA], dtype=float)


def kmer_jaccard(a, b, k=3):
    aa = {a[i:i+k] for i in range(len(a)-k+1)}
    bb = {b[i:i+k] for i in range(len(b)-k+1)}
    return len(aa & bb) / len(aa | bb) if aa or bb else 0.0


def make_record(edge, label, split, nodes, block_id):
    a, b = nodes[edge["a"]], nodes[edge["b"]]
    hashes = sorted((a["sequence_sha256"], b["sequence_sha256"]))
    vector_a, vector_b = aa_vector(a["sequence"]), aa_vector(b["sequence"])
    result = {
        "pair_id": text_hash("remote-v2:" + ":".join(hashes)), "split": split, "label": label,
        "rewiring_block_id": block_id, "sequence_id_a": edge["a"], "sequence_id_b": edge["b"],
        "sequence_sha256_a": a["sequence_sha256"], "sequence_sha256_b": b["sequence_sha256"],
        "sentence1": a["sequence"], "sentence2": b["sequence"],
        "superfamily_a": a["superfamily_id"], "superfamily_b": b["superfamily_id"],
        "family_a": a["family_id"], "family_b": b["family_id"],
        "length_a": a["length"], "length_b": b["length"],
        "composition_l1": float(np.abs(vector_a-vector_b).sum()),
        "composition_l2": float(np.linalg.norm(vector_a-vector_b)),
        "kmer3_jaccard": kmer_jaccard(a["sequence"], b["sequence"]),
        "label_basis": "same_SCOPe_SF_different_family" if label else "different_SCOPe_SF",
    }
    result.update({key: value for key, value in edge.items() if key not in ("a", "b", "sf")})
    return result


def balanced_pairs(edges, split, nodes):
    rng = random.Random(stable_seed("rewire:" + split))
    edges = list(edges)
    rng.shuffle(edges)
    lengths = np.array([[math.log(nodes[e["a"]]["length"]), math.log(nodes[e["b"]]["length"])] for e in edges])
    if not len(edges):
        return [], {"positive_candidates": 0, "accepted_blocks": 0}
    tree = cKDTree(lengths)
    vectors = {key: aa_vector(row["sequence"]) for key, row in nodes.items() if row["split"] == split}
    available, used = set(range(len(edges))), set()
    positive_pair_ids = {tuple(sorted((edge["a"], edge["b"]))) for edge in edges}
    rows = []
    negative_alignments = rejected = 0
    for i, first in enumerate(edges):
        if i not in available:
            continue
        if len(rows) >= CONFIG["target_pairs"][split]:
            break
        _, nearest = tree.query(lengths[i], k=min(CONFIG["negative_match_pool"], len(edges)))
        candidates = [int(j) for j in np.atleast_1d(nearest) if int(j) in available and int(j) != i and edges[int(j)]["sf"] != first["sf"]]
        def candidate_distance(j):
            second = edges[j]
            length_distance = float(np.abs(lengths[i]-lengths[j]).sum())
            composition_distance = float(np.abs(vectors[first["a"]]-vectors[second["a"]]).sum() + np.abs(vectors[first["b"]]-vectors[second["b"]]).sum())
            return length_distance + composition_distance
        candidates.sort(key=lambda j: (candidate_distance(j), j))
        accepted = False
        for j in candidates[:CONFIG["negative_alignment_attempts_per_edge"]]:
            second = edges[j]
            negative_ids = [(first["a"], second["b"]), (second["a"], first["b"])]
            keys = [tuple(sorted(pair)) for pair in negative_ids]
            if keys[0] == keys[1] or any(key in used or key in positive_pair_ids for key in keys):
                continue
            negative = []
            for ia, ib in negative_ids:
                result = global_alignment(nodes[ia]["sequence"], nodes[ib]["sequence"])
                negative_alignments += 1
                if not alignment_passes(result):
                    break
                negative.append({"a": ia, "b": ib, **result})
            if len(negative) != 2:
                rejected += 1
                continue
            block_id = text_hash("block:" + ":".join(sorted([first["a"], first["b"], second["a"], second["b"]])))
            rows.extend([make_record(first, 1, split, nodes, block_id), make_record(second, 1, split, nodes, block_id),
                         make_record(negative[0], 0, split, nodes, block_id), make_record(negative[1], 0, split, nodes, block_id)])
            used.update(keys)
            available.remove(i)
            available.remove(j)
            accepted = True
            break
        if not accepted:
            # Keep this edge available as a future partner; no partial blocks are emitted.
            continue
    rng.shuffle(rows)
    return rows, {"positive_candidates": len(edges), "accepted_blocks": len(rows)//4,
                  "negative_alignments": negative_alignments, "rejected_negative_blocks": rejected,
                  "target_pairs": CONFIG["target_pairs"][split], "actual_pairs": len(rows),
                  "target_met": len(rows) >= CONFIG["target_pairs"][split]}


def validate(root, table, nodes, pairing_stats):
    failures = []
    if table.empty:
        failures.append("no_pairs")
    if table.pair_id.duplicated().any():
        failures.append("duplicate_unordered_pair")
    if (table.sequence_sha256_a == table.sequence_sha256_b).any():
        failures.append("self_pair")
    positive = table[table.label == 1]
    negative = table[table.label == 0]
    if not ((positive.superfamily_a == positive.superfamily_b) & (positive.family_a != positive.family_b)).all():
        failures.append("invalid_positive_annotation")
    if not (negative.superfamily_a != negative.superfamily_b).all():
        failures.append("invalid_negative_annotation")
    if not ((table.identity_paired < .25) & (table.coverage_a >= .60) & (table.coverage_b >= .60)).all():
        failures.append("alignment_filter_violation")
    degree = Counter()
    split_hashes, split_sfs = {}, {}
    degree_rows, split_stats = [], {}
    for split in SPLITS:
        frame = table[table.split == split]
        hashes = set(frame.sequence_sha256_a) | set(frame.sequence_sha256_b)
        sf_ids = set(frame.superfamily_a) | set(frame.superfamily_b)
        split_hashes[split], split_sfs[split] = hashes, sf_ids
        for row in frame.itertuples():
            degree[(split, row.sequence_id_a, "a", row.label)] += 1
            degree[(split, row.sequence_id_b, "b", row.label)] += 1
        counts = frame.label.value_counts().to_dict()
        if counts.get(0, 0) != counts.get(1, 0):
            failures.append(f"class_imbalance:{split}")
        if len(frame) < CONFIG["minimum_pairs_for_acceptance"][split]:
            failures.append(f"insufficient_pair_count:{split}")
        if len(sf_ids) < CONFIG["minimum_superfamilies_per_split"]:
            failures.append(f"insufficient_superfamily_diversity:{split}")
        split_stats[split] = {"pairs": len(frame), "positive_pairs": int(counts.get(1, 0)),
                              "negative_pairs": int(counts.get(0, 0)), "unique_sequences": len(hashes),
                              "superfamilies": len(sf_ids), "families": len(set(frame.family_a) | set(frame.family_b))}
    for split, sid, role in sorted({key[:3] for key in degree}):
        pos, neg = degree[(split, sid, role, 1)], degree[(split, sid, role, 0)]
        degree_rows.append({"split": split, "sequence_id": sid, "role": role, "positive_degree": pos, "negative_degree": neg})
        if pos != neg:
            failures.append(f"endpoint_label_degree_mismatch:{split}:{sid}:{role}")
    overlaps = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i+1:]:
            overlaps[f"{a}__{b}"] = {"exact_sequence_hash_overlap": len(split_hashes[a] & split_hashes[b]),
                                       "superfamily_overlap": len(split_sfs[a] & split_sfs[b])}
            if any(overlaps[f"{a}__{b}"].values()):
                failures.append(f"split_overlap:{a}:{b}")
    pd.DataFrame(degree_rows).to_csv(root / "validation/remote_endpoint_degrees.tsv", sep="\t", index=False)

    def metric_row(method, split, y, probability, predicted):
        return {"method": method, "split": split, "n": len(y), "accuracy": float(accuracy_score(y, predicted)),
                "auroc": float(roc_auc_score(y, probability)), "mcc": float(matthews_corrcoef(y, predicted))}
    baseline_rows = []
    train = table[table.split == "train"]
    known_negative_b = set(train[train.label == 0].sequence_sha256_b)
    fixed_scores = {
        "negative_absolute_length_difference": -np.abs(table.length_a-table.length_b).to_numpy(),
        "negative_composition_l2": -table.composition_l2.to_numpy(),
        "3mer_jaccard": table.kmer3_jaccard.to_numpy(),
        "global_paired_residue_identity": table.identity_paired.to_numpy(),
    }
    for name, scores in fixed_scores.items():
        for split in SPLITS:
            mask = (table.split == split).to_numpy()
            baseline_rows.append({"method": name, "split": split, "n": int(mask.sum()),
                                  "accuracy": None, "auroc": float(roc_auc_score(table.loc[mask, "label"], scores[mask])), "mcc": None})
    for split in SPLITS:
        frame = table[table.split == split]
        predicted = (~frame.sequence_sha256_b.isin(known_negative_b)).astype(int)
        baseline_rows.append(metric_row("legacy_negative_sentence2_identity_lookup", split, frame.label, predicted, predicted))
    features = pd.DataFrame({
        "length_a": table.length_a, "length_b": table.length_b,
        "length_difference": np.abs(table.length_a-table.length_b),
        "length_ratio": np.minimum(table.length_a, table.length_b)/np.maximum(table.length_a, table.length_b),
        "composition_l1": table.composition_l1, "composition_l2": table.composition_l2,
        "kmer3_jaccard": table.kmer3_jaccard,
    })
    models = {"length_logistic_regression": ["length_a", "length_b", "length_difference", "length_ratio"],
              "composition_logistic_regression": ["composition_l1", "composition_l2"],
              "length_composition_3mer_logistic_regression": list(features)}
    train_mask = (table.split == "train").to_numpy()
    for name, columns in models.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=CONFIG["seed"]))
        model.fit(features.loc[train_mask, columns], table.loc[train_mask, "label"])
        for split in SPLITS:
            mask = (table.split == split).to_numpy()
            x, y = features.loc[mask, columns], table.loc[mask, "label"]
            baseline_rows.append(metric_row(name, split, y, model.predict_proba(x)[:, 1], model.predict(x)))
    pd.DataFrame(baseline_rows).to_csv(root / "validation/remote_baselines.csv", index=False)
    result = {"status": "passed_local_data_checks" if not failures else "blocked",
              "failures": failures, "counts": split_stats, "split_overlaps": overlaps,
              "pairing": pairing_stats, "endpoint_label_degree_balance": not any("degree_mismatch" in x for x in failures),
              "zero_gap_input_sequences": all(set(row["sequence"]) <= set(AA) for row in nodes.values()),
              "local_checks_do_not_certify": ["all cross-superfamily homology absent", "pretraining overlap excluded", "all simple biological similarity signal absent"],
              "external_gates_pending": ["MMseqs cross-superfamily and pretraining overlap diagnostics"],
              "baseline_policy": "Report held-out baselines; do not tune data to force all legitimate similarity signals to chance."}
    dump_json(root / "validation/remote_validation.json", result)
    return result


def build_pairs(root, workers):
    sequence_table = pd.read_parquet(root / "data/sequences/remote_sequences.parquet")
    nodes = {row["sequence_id"]: row for row in sequence_table.to_dict("records")}
    tasks = []
    for sf, group in sequence_table.groupby("superfamily_id", sort=True):
        if group.family_id.nunique() >= 2:
            tasks.append((sf, group.to_dict("records")))
    edges, stats = [], []
    with mp.Pool(processes=min(workers, 16)) as pool:
        for index, (new_edges, summary) in enumerate(pool.imap(positive_worker, tasks), 1):
            edges.extend(new_edges)
            stats.append(summary)
            if index % 25 == 0 or index == len(tasks):
                print(f"positive search {index}/{len(tasks)} SFs, {len(edges)} edges", flush=True)
    pd.DataFrame(stats).to_csv(root / "metadata/remote_positive_search.tsv", sep="\t", index=False)
    with gzip.open(root / "metadata/remote_candidate_positive_edges.jsonl.gz", "wt") as handle:
        for edge in edges:
            handle.write(json.dumps(edge, sort_keys=True) + "\n")
    rows, pairing_stats = [], {}
    for split in SPLITS:
        subset = [edge for edge in edges if nodes[edge["a"]]["split"] == split]
        result, summary = balanced_pairs(subset, split, nodes)
        rows.extend(result)
        pairing_stats[split] = summary
        print(split, json.dumps(summary), flush=True)
    table = pd.DataFrame(rows)
    pairs_dir = root / "data/pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    (root / "validation").mkdir(parents=True, exist_ok=True)
    table.to_parquet(pairs_dir / "remote_pairs.parquet", index=False)
    for split in SPLITS:
        table[table.split == split].to_csv(pairs_dir / f"remote_{split}.csv.gz", index=False,
                                           compression={"method": "gzip", "mtime": 0})
    validation = validate(root, table, nodes, pairing_stats)
    paths = sorted([*pairs_dir.glob("remote_*"), *(root / "data/sequences").glob("remote_*"),
                    *(root / "validation").glob("remote_*"), *(root / "metadata").glob("remote_*")])
    manifest = {"release_id": "2026-09-25-v2", "task": "remote_domain_homology_SCOPe2.08",
                "status": validation["status"], "config": CONFIG, "counts": validation["counts"],
                "code": {"file": str(Path(__file__)), "sha256": digest(Path(__file__)), "biopython": Bio.__version__},
                "files": [{"file": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest(path)}
                          for path in paths if path.name != "remote_manifest.json"],
                "source_record": "metadata/remote_sources.json", "license": "CC-BY-4.0",
                "global_release_ready": False,
                "remaining_release_gate": "integrated pretraining and cross-dataset homology exclusion audit"}
    dump_json(root / "metadata/remote_manifest.json", manifest)
    print(json.dumps({"status": validation["status"], "counts": validation["counts"], "failures": validation["failures"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "pairs"))
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    root = args.release_root.resolve()
    if args.stage == "prepare":
        prepare(root)
    else:
        build_pairs(root, args.workers)


if __name__ == "__main__":
    main()
