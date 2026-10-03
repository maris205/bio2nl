"""Independently recompute cross-task overlap from current physical inputs.

No search, data construction or writes. All report numbers are recomputed;
historical input supplies only validated, fixed explanatory prose/thresholds.
"""
from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

SPLITS = ("train", "validation", "test")
THRESHOLDS = {"identity_threshold": .30, "minimum_remote_domain_query_coverage": .80,
              "evalue_max": .001}
INTERPRETATION = "All Swiss-Prot matches to any SCOPe domain are excluded from the protein pretraining corpus. Overlap with separate supervised pair-task source tables is measured and disclosed; the pair tasks are trained/evaluated separately. MMseqs2 is heuristic and the fixed thresholds do not certify absence of all remote evolutionary relationships."
LIMITATIONS = ["Sequence comparison uses MMseqs2 heuristic prefilter and may miss remote relationships",
              "A reported overlap is an audit finding, not an automatic exclusion from a separately run benchmark protocol."]
FILES = {
    "canonical": "data/sequences/protein_canonical_sequences.tsv.gz",
    "fasta": "raw/remote/astral-scopedom-seqres-gd-all-2.08-stable.fa",
    "remote_pairs": "data/pairs/remote_pairs.parquet",
    "hits": "work/protein/remote_exclusion/remote_vs_swissprot.tsv",
    "protein_pairs": "data/pairs/protein_sequence_similarity_all.tsv.gz",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_file(path):
    path = Path(os.path.abspath(path))
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlinked input: " + str(path))
    require(path.is_file(), "Missing regular input: " + str(path))
    return path


def identity(path):
    path = safe_file(path)
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": sha, "bytes": path.stat().st_size}


def tsv_rows(path, required):
    with gzip.open(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames)
                and set(required) <= set(reader.fieldnames), "Malformed TSV header: " + str(path))
        for number, row in enumerate(reader, 2):
            require(None not in row and all(v is not None for v in row.values()),
                    f"Malformed TSV row {number}: {path}")
            yield row


def canonical_index(path):
    byaccession, eligible, bysha = {}, set(), {}
    for row in tsv_rows(path, ("accession", "split", "pretrain_eligible", "sequence", "sequence_sha256")):
        accession = row["accession"]
        require(accession and accession not in byaccession, "Duplicate/empty canonical accession")
        require(row["split"] in SPLITS and row["pretrain_eligible"] in ("0", "1"), "Invalid canonical split/eligibility")
        sequence = row["sequence"]
        require(sequence and hashlib.sha256(sequence.encode()).hexdigest() == row["sequence_sha256"],
                "Canonical sequence SHA mismatch")
        byaccession[accession] = row["split"]
        if row["pretrain_eligible"] == "1":
            eligible.add(accession)
        bysha.setdefault(row["sequence_sha256"], []).append(accession)
    require(byaccession, "Empty canonical input")
    return byaccession, eligible, bysha


def fasta_overlap(path, bysha):
    """Keep IDs and exact-overlap accessions, never the complete sequence corpus."""
    domains, exact_domains, exact_accessions = set(), set(), set()
    exact_associations = 0
    header, sequence = None, []
    def finish():
        nonlocal exact_associations
        if header is None:
            return
        text = "".join(sequence).upper()
        require(text, "Empty FASTA sequence")
        matching = bysha.get(hashlib.sha256(text.encode()).hexdigest())
        if matching:
            exact_domains.add(header)
            exact_accessions.update(matching)
            exact_associations += len(matching)
    with path.open() as stream:
        for line in stream:
            if line.startswith(">"):
                finish()
                parts = line[1:].split()
                require(parts, "Empty FASTA identifier")
                header = parts[0]
                require(header not in domains, "Duplicate FASTA identifier")
                domains.add(header)
                sequence = []
            else:
                require(header is not None or not line.strip(), "FASTA sequence before header")
                sequence.append(line.strip())
        finish()
    require(domains, "Empty FASTA input")
    return domains, exact_domains, exact_accessions, exact_associations


def remote_endpoints(path, domains):
    import pyarrow.parquet as pq
    reader = pq.ParquetFile(path)
    columns = ("sequence_id_a", "sequence_id_b", "split", "label")
    require(set(columns) <= set(reader.schema_arrow.names), "Missing remote endpoint/role columns")
    endpoints = set()
    for batch in reader.iter_batches(batch_size=65536, columns=list(columns), use_threads=False):
        for row in batch.to_pylist():
            require(row["split"] in SPLITS and type(row["label"]) is int and row["label"] in (0, 1),
                    "Invalid remote split/label")
            for name in columns[:2]:
                endpoint = row[name]
                require(isinstance(endpoint, str) and endpoint in domains, "Unknown remote pair domain")
                endpoints.add(endpoint)
    return endpoints


def qualifying_hits(path, domains, accessions, remote_ids):
    hit_domains, hit_accessions, remote_hit_accessions = set(), set(), set()
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            cells = line.rstrip("\r\n").split("\t")
            require(len(cells) == 13, f"Expected 13 MMseqs fields at line {number}")
            query, target = cells[:2]
            require(query in domains and target in accessions, "Unknown search domain/accession")
            try:
                fraction, evalue, bits = float(cells[2]), float(cells[11]), float(cells[12])
                alignment, qstart, qend, qlen, tstart, tend, tlen, nident = map(int, cells[3:11])
            except (ValueError, OverflowError) as exc:
                raise ValueError(f"Invalid search number at line {number}") from exc
            require(all(math.isfinite(x) for x in (fraction, evalue, bits)), "Nonfinite search value")
            require(0 <= fraction <= 1 and evalue >= 0 and alignment > 0 and
                    1 <= qstart <= qend <= qlen and 1 <= tstart <= tend <= tlen and
                    0 <= nident <= alignment, "Invalid search score/coordinates")
            coverage = (qend - qstart + 1) / qlen
            if fraction >= .30 and coverage >= .80 and evalue <= .001:
                hit_domains.add(query)
                hit_accessions.add(target)
                if query in remote_ids:
                    remote_hit_accessions.add(target)
    return hit_domains, hit_accessions, remote_hit_accessions


def protein_pair_overlap(path, byaccession, hits, exact_accessions):
    endpoints = set()
    touches = {split: {"0": 0, "1": 0} for split in SPLITS}
    for row in tsv_rows(path, ("accession_a", "accession_b", "split", "label")):
        split, label = row["split"], row["label"]
        require(split in SPLITS and label in ("0", "1"), "Invalid protein pair split/label")
        a, b = row["accession_a"], row["accession_b"]
        require(a in byaccession and b in byaccession, "Unknown protein pair accession")
        require(byaccession[a] == byaccession[b] == split, "Protein pair/canonical split mismatch")
        endpoints.update((a, b))
        # A row counts once even when both endpoint roles qualify.
        if a in hits or b in hits:
            touches[split][label] += 1
    matching = endpoints & hits
    return {"unique_accessions": len(endpoints), "matches_any_SCOPe_domain_threshold": len(matching),
        "matching_accessions_by_protein_cluster_split": dict(Counter(byaccession[x] for x in matching)),
        "pair_rows_touching_matching_accession_by_split_and_label": touches,
        "exact_domain_sequence_accessions": len(endpoints & exact_accessions)}


def recompute_cross_dataset(root, historical_report):
    """Return the original report schema with independently recomputed numbers.

    `historical_report` is a dict or a JSON file path. Its old hashes and all
    numbers other than the three fixed threshold constants are ignored.
    The remote parquet is required for this full-release verifier.
    """
    if not isinstance(historical_report, dict):
        historical_report = json.loads(safe_file(historical_report).read_bytes())
    require(all(type(historical_report.get(k)) in (int, float) and historical_report[k] == v
                for k, v in THRESHOLDS.items()), "Historical threshold constants differ")
    require(historical_report.get("interpretation") == INTERPRETATION and
            historical_report.get("limitations") == LIMITATIONS, "Historical fixed explanatory text differs")
    root = Path(os.path.abspath(root))
    paths = {key: safe_file(root / name) for key, name in FILES.items()}
    before = {key: identity(path) for key, path in paths.items()}
    accessions, eligible, bysha = canonical_index(paths["canonical"])
    domains, exact_domains, exact_accessions, exact_associations = fasta_overlap(paths["fasta"], bysha)
    del bysha
    remote_ids = remote_endpoints(paths["remote_pairs"], domains)
    hit_domains, excluded, remote_hits = qualifying_hits(paths["hits"], domains, accessions, remote_ids)
    protein = protein_pair_overlap(paths["protein_pairs"], accessions, excluded, exact_accessions)
    report = {"status": "completed", **THRESHOLDS,
        "search_output_sha256": before["hits"]["sha256"],
        "protein_table_sha256": before["canonical"]["sha256"],
        "remote_scop_domains_in_raw_fasta": len(domains),
        "domains_with_qualifying_swissprot_hits": len(hit_domains),
        "unique_qualifying_swissprot_accessions": len(excluded),
        "excluded_from_protein_pretraining": len(excluded & accessions.keys()),
        "remaining_eligible_swissprot_accessions_with_qualifying_hits": len(excluded & eligible),
        "exact_domain_sequence_overlap": len(exact_domains),
        "exact_domain_accessions_with_swissprot": exact_associations,
        "pair_task_overlap": {"sequence_similarity": protein,
            "SCOPe_remote": {"unique_domains": len(remote_ids),
                "exact_domain_sequence_accessions": len(remote_ids & exact_domains),
                "threshold_hit_swissprot_accessions": len(remote_hits)}},
        "interpretation": historical_report["interpretation"], "limitations": list(historical_report["limitations"])}
    for key, path in paths.items():
        require(identity(path) == before[key], "Input changed during independent audit: " + key)
    return report
