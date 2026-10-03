import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

spec = importlib.util.spec_from_file_location("independent_cross_dataset", Path(__file__).with_name("cross_dataset.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader(); writer.writerows(rows)


def hit(domain, accession, identity=.3, coverage=80, evalue=.001):
    return "\t".join(map(str, [domain, accession, identity, 100, 1, coverage, 100, 1, 100, 100, 0, evalue, 50])) + "\n"


@pytest.fixture
def fixture(tmp_path):
    paths = {key: tmp_path / name for key, name in audit.FILES.items()}
    for p in paths.values(): p.parent.mkdir(parents=True, exist_ok=True)
    canonical = [{"accession": a, "split": split, "pretrain_eligible": eligible, "sequence": seq,
                  "sequence_sha256": hashlib.sha256(seq.encode()).hexdigest()}
                 for a, split, eligible, seq in (("A", "train", "0", "AAAA"), ("B", "train", "1", "CCCC"),
                   ("C", "validation", "0", "GGGG"), ("D", "validation", "1", "TTTT"),
                   ("E", "test", "0", "VVVV"), ("F", "test", "0", "WWWW"), ("G", "train", "0", "AAAA"))]
    table(paths["canonical"], canonical)
    paths["fasta"].write_text(">d1 details\nAA\nAA\n>d2\nCCCC\n>d3\naaaa\n>d4\nYYYY\n>d5\nGGGG\n")
    remote = [{"sequence_id_a": "d1", "sequence_id_b": "d4", "split": "train", "label": y} for y in (0, 1)]
    pq.write_table(pa.Table.from_pylist(remote), paths["remote_pairs"])
    paths["hits"].write_text(hit("d1", "A") + hit("d1", "A") + hit("d1", "B", .7) + hit("d5", "C", .31)
        + hit("d3", "E", .4, 80, 0) + hit("d4", "E", .2999999999)
        + hit("d4", "F", .8, 79) + hit("d2", "F", .9, 100, .0010000001))
    pairs = [{"accession_a": a, "accession_b": b, "split": split, "label": str(y)}
             for a, b, split, y in (("A", "B", "train", 0), ("B", "A", "train", 1), ("G", "B", "train", 1),
                 ("C", "D", "validation", 0), ("D", "C", "validation", 1),
                 ("E", "F", "test", 0), ("F", "E", "test", 1))]
    table(paths["protein_pairs"], pairs)
    historical = {**audit.THRESHOLDS, "interpretation": audit.INTERPRETATION,
                  "limitations": list(audit.LIMITATIONS), "remote_scop_domains_in_raw_fasta": 999999,
                  "protein_table_sha256": "historical SHA is ignored", "search_output_sha256": "ignored"}
    return tmp_path, paths, canonical, remote, pairs, historical


def test_threshold_boundaries_duplicate_hits_and_both_roles_are_counted_once(fixture):
    root, paths, canonical, remote, pairs, historical = fixture
    before = {k: audit.identity(p) for k, p in paths.items()}
    result = audit.recompute_cross_dataset(root, historical)
    assert result == {"status": "completed", **audit.THRESHOLDS,
        "search_output_sha256": before["hits"]["sha256"], "protein_table_sha256": before["canonical"]["sha256"],
        "remote_scop_domains_in_raw_fasta": 5, "domains_with_qualifying_swissprot_hits": 3,
        "unique_qualifying_swissprot_accessions": 4, "excluded_from_protein_pretraining": 4,
        "remaining_eligible_swissprot_accessions_with_qualifying_hits": 1,
        "exact_domain_sequence_overlap": 4, "exact_domain_accessions_with_swissprot": 6,
        "pair_task_overlap": {"sequence_similarity": {"unique_accessions": 7,
            "matches_any_SCOPe_domain_threshold": 4,
            "matching_accessions_by_protein_cluster_split": {"train": 2, "validation": 1, "test": 1},
            "pair_rows_touching_matching_accession_by_split_and_label": {
                "train": {"0": 1, "1": 2}, "validation": {"0": 1, "1": 1}, "test": {"0": 1, "1": 1}},
            "exact_domain_sequence_accessions": 4},
            "SCOPe_remote": {"unique_domains": 2, "exact_domain_sequence_accessions": 1,
                             "threshold_hit_swissprot_accessions": 2}},
        "interpretation": audit.INTERPRETATION, "limitations": audit.LIMITATIONS}
    assert {k: audit.identity(p) for k, p in paths.items()} == before
    # The caller's report and its nested explanation list are not mutated.
    assert historical["remote_scop_domains_in_raw_fasta"] == 999999
    assert result["limitations"] is not historical["limitations"]


@pytest.mark.parametrize("change", ["duplicate_canonical", "canonical_hash", "canonical_split", "canonical_eligibility",
    "duplicate_domain", "empty_domain", "fasta_without_header", "unknown_search_domain", "unknown_search_accession",
    "nonfinite_identity", "nonfinite_evalue", "nonfinite_bits", "zero_qlen", "wrong_columns",
    "unknown_remote_domain", "remote_split", "remote_label", "protein_split", "protein_label", "unknown_protein_accession",
    "cross_split_pair", "threshold", "explanation", "symlink"])
def test_invalid_sources_and_metadata_fail_closed(fixture, change):
    root, paths, canonical, remote, pairs, historical = fixture
    if change == "duplicate_canonical": table(paths["canonical"], canonical + [canonical[0]])
    elif change.startswith("canonical_"):
        field, value = {"canonical_hash": ("sequence_sha256", "0" * 64), "canonical_split": ("split", "unknown"),
                        "canonical_eligibility": ("pretrain_eligible", "2")}[change]
        canonical[0][field] = value; table(paths["canonical"], canonical)
    elif change == "duplicate_domain": paths["fasta"].write_text(paths["fasta"].read_text() + ">d1\nAAAA\n")
    elif change == "empty_domain": paths["fasta"].write_text(paths["fasta"].read_text() + ">empty\n")
    elif change == "fasta_without_header": paths["fasta"].write_text("AAAA\n" + paths["fasta"].read_text())
    elif change == "unknown_search_domain": paths["hits"].write_text(hit("unknown", "A"))
    elif change == "unknown_search_accession": paths["hits"].write_text(hit("d1", "unknown"))
    elif change in ("nonfinite_identity", "nonfinite_evalue", "nonfinite_bits", "zero_qlen", "wrong_columns"):
        cells = hit("d1", "A").rstrip().split("\t")
        if change == "wrong_columns": cells.pop()
        else:
            index = {"nonfinite_identity": 2, "nonfinite_evalue": 11, "nonfinite_bits": 12, "zero_qlen": 6}[change]
            cells[index] = "0" if change == "zero_qlen" else "nan"
        paths["hits"].write_text("\t".join(cells) + "\n")
    elif change in ("unknown_remote_domain", "remote_split", "remote_label"):
        key, value = {"unknown_remote_domain": ("sequence_id_a", "unknown"), "remote_split": ("split", "unknown"),
                      "remote_label": ("label", 2)}[change]
        remote[0][key] = value; pq.write_table(pa.Table.from_pylist(remote), paths["remote_pairs"])
    elif change in ("protein_split", "protein_label", "unknown_protein_accession", "cross_split_pair"):
        key, value = {"protein_split": ("split", "unknown"), "protein_label": ("label", "2"),
                      "unknown_protein_accession": ("accession_a", "unknown"), "cross_split_pair": ("accession_a", "C")}[change]
        pairs[0][key] = value; table(paths["protein_pairs"], pairs)
    elif change == "threshold": historical["identity_threshold"] = .29
    elif change == "explanation": historical["limitations"] = []
    elif change == "symlink":
        moved = root / "moved_fasta"; paths["fasta"].rename(moved); paths["fasta"].symlink_to(moved)
    with pytest.raises(ValueError): audit.recompute_cross_dataset(root, historical)


def test_no_hits_and_no_exact_match_preserve_zero_and_empty_dict_schema(fixture):
    root, paths, canonical, remote, pairs, historical = fixture
    paths["hits"].write_text("")
    paths["fasta"].write_text(">d1\nPQRS\n>d4\nMNPQ\n")
    result = audit.recompute_cross_dataset(root, historical)
    assert result["domains_with_qualifying_swissprot_hits"] == result["exact_domain_sequence_overlap"] == 0
    task = result["pair_task_overlap"]["sequence_similarity"]
    assert task["matching_accessions_by_protein_cluster_split"] == {}
    assert task["pair_rows_touching_matching_accession_by_split_and_label"] == {s: {"0": 0, "1": 0} for s in audit.SPLITS}
