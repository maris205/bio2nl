"""Build fresh document/cluster/chromosome splits and matched AA controls.

No legacy prepared text files are read. Only pinned raw downloads, rebuilt NLP
splits, and the BioPAWS v2 canonical cluster table are accepted.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import re
import unicodedata
from contextlib import ExitStack
from common import digest, fasta_records, text_hash, write_json
from fetch_corpora import ASSEMBLY, PREFIX, OWT_REV

SPLITS = ("train", "validation", "test")
AA = "ACDEFGHIKLMNPQRSTVWY"
SEED = 20260925


def gz_writer(path, stack):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = stack.enter_context(path.open("wb"))
    gz = stack.enter_context(gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0))
    return stack.enter_context(io.TextIOWrapper(gz, encoding="utf-8"))


def emit(root, condition, records):
    records.sort(key=lambda r: text_hash(str(SEED) + ":" + r["record_id"]))
    stats = {s: dict(records=0, utf8_bytes=0, characters=0) for s in SPLITS}
    hashes = {s: set() for s in SPLITS}
    with ExitStack() as stack:
        writers = {s: gz_writer(root / "data/corpora" / condition / (s + ".jsonl.gz"), stack) for s in SPLITS}
        for record in records:
            split = record["split"]
            record["text_sha256"] = text_hash(record["text"])
            writers[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            stats[split]["records"] += 1
            stats[split]["utf8_bytes"] += len(record["text"].encode("utf-8"))
            stats[split]["characters"] += len(record["text"])
            hashes[split].add(record["text_sha256"])
    overlap = {f"{a}/{b}": len(hashes[a] & hashes[b]) for i, a in enumerate(SPLITS) for b in SPLITS[i+1:]}
    for s in SPLITS:
        path = root / "data/corpora" / condition / (s + ".jsonl.gz")
        stats[s].update(file=str(path.relative_to(root)), sha256=digest(path), unique_texts=len(hashes[s]))
    report = dict(condition=condition, splits=stats, exact_text_cross_split_overlap=overlap)
    write_json(root / "metadata" / f"corpora_{condition}.json", report)
    assert not any(overlap.values()), report
    return report


def words(text):
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())


def shingle_hash(tokens):
    return hashlib.blake2b(" ".join(tokens).encode(), digest_size=16).digest()


def benchmark_shingles(root):
    source_files = sorted((root / "data/nlp").glob("*/*.jsonl"))
    if len(source_files) != 9:
        raise RuntimeError("Expected all nine rebuilt official NLP split files before English decontamination")
    forbidden = set()
    for path in source_files:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                for key in ("sentence1", "sentence2"):
                    tokens = words(row.get(key) or "")
                    forbidden.update(shingle_hash(tokens[i:i+8]) for i in range(len(tokens)-7))
    return forbidden, {str(p.relative_to(root)): digest(p) for p in source_files}


def english(root):
    import pyarrow.parquet as pq
    source = root / "raw/corpora/openwebtext-00000.parquet"
    forbidden, benchmark_files = benchmark_shingles(root)
    records, seen, rejected = [], set(), Counter()
    index = 0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=512):
        for value in batch.column("text").to_pylist():
            row_index = index; index += 1
            text = value.replace("\r\n", "\n").strip()
            normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
            group = text_hash(normalized)
            if len(text.encode("utf-8")) < 200 or len(text.encode("utf-8")) > 1_000_000:
                rejected["outside_200_to_1000000_utf8_bytes"] += 1; continue
            if group in seen:
                rejected["normalized_exact_duplicate"] += 1; continue
            seen.add(group)
            tokens = words(text)
            if any(shingle_hash(tokens[i:i+8]) in forbidden for i in range(len(tokens)-7)):
                rejected["shares_eight_word_span_with_any_nlp_split"] += 1; continue
            bucket = int(group[:16], 16) % 1000
            split = "train" if bucket < 950 else "validation" if bucket < 975 else "test"
            records.append(dict(record_id=f"owt:{OWT_REV}:shard0:{row_index}", source="openwebtext",
                                source_row_index=row_index, normalized_group_sha256=group,
                                split=split, text=text))
    report = emit(root, "english", records)
    report.update(source_rows=index, rejected=dict(rejected), split_rule="normalized document SHA256 mod 1000; 950/25/25",
                  benchmark_exclusion_files=benchmark_files, forbidden_eight_word_spans=len(forbidden),
                  limitation="Exact normalized documents and exact 8-word overlap excluded; semantic/near-document independence not proven")
    write_json(root / "validation/corpora_english.json", report)
    print("english", {s: report["splits"][s]["records"] for s in SPLITS}, flush=True)


def dna(root):
    raw = root / "raw/corpora"
    chromosome = {}
    for line in (raw / (PREFIX + "_assembly_report.txt")).read_text().splitlines():
        if line.startswith("#") or not line.strip(): continue
        fields = line.split("\t")
        if fields[1] == "assembled-molecule": chromosome[fields[6]] = fields[2]
    assignments = {"2L": "train", "2R": "train", "3L": "train", "3R": "train", "X": "validation", "4": "test", "Y": "test"}
    records, seen, rejected, contigs = [], set(), Counter(), []
    with gzip.open(raw / (PREFIX + "_genomic.fna.gz"), "rt") as stream:
        sequences = [(h, seq.upper()) for h, seq in fasta_records(stream) if chromosome.get(h.split()[0]) in assignments]
    # Test/validation win exact (including reverse-complement) duplicate collisions.
    priority = {"test": 0, "validation": 1, "train": 2}
    sequences.sort(key=lambda hs: (priority[assignments[chromosome[hs[0].split()[0]]]], hs[0]))
    complement = str.maketrans("ACGT", "TGCA")
    for header, sequence in sequences:
        accession = header.split()[0]; chrom = chromosome[accession]; split = assignments[chrom]
        contigs.append(dict(accession=accession, chromosome=chrom, split=split, length=len(sequence)))
        for start in range(0, len(sequence)-4096+1, 4096):
            text = sequence[start:start+4096]
            if set(text) - set("ACGT"):
                rejected["non_acgt_window"] += 1; continue
            group = text_hash(min(text, text.translate(complement)[::-1]))
            if group in seen:
                rejected["exact_or_reverse_complement_duplicate"] += 1; continue
            seen.add(group)
            records.append(dict(record_id=f"{ASSEMBLY}:{accession}:{start}:{start+4096}",
                                source="ncbi_refseq", assembly=ASSEMBLY, accession=accession,
                                chromosome=chrom, start=start, end=start+4096, coordinate_system="0-based half-open",
                                canonical_orientation_sha256=group, split=split, text=text))
    report = emit(root, "dna", records)
    report.update(chromosomes=contigs, rejected=dict(rejected), window_size=4096, stride=4096,
                  split_rule=assignments, limitation="Chromosome-disjoint, exact/reverse-complement-window deduplicated; shorter repeats/homologous segments can remain")
    write_json(root / "validation/corpora_dna.json", report)
    print("dna", {s: report["splits"][s]["records"] for s in SPLITS}, flush=True)


def proteins(root):
    source = root / "data/sequences/protein_canonical_sequences.tsv.gz"
    records, rejected, aa_counts = [], Counter(), Counter()
    with gzip.open(source, "rt") as stream:
        rows = csv.DictReader(stream, delimiter="\t")
        for row in rows:
            if "pretrain_eligible" not in row:
                raise RuntimeError("Protein table must include the completed SCOPe homology exclusion")
            if row["pretrain_eligible"].lower() not in ("true", "1", "yes"):
                rejected["remote_homology_exclusion"] += 1; continue
            seq = row["sequence"]
            assert not set(seq) - set(AA) and row["split"] in SPLITS
            record = dict(record_id=f"swissprot:{row['accession']}:{row['sequence_version']}", source="swissprot",
                          accession=row["accession"], sequence_version=row["sequence_version"],
                          cluster_id=row["cluster_id"], split=row["split"], text=seq)
            records.append(record)
            if row["split"] == "train": aa_counts.update(seq)
    report = emit(root, "protein", records)
    report.update(source_table_sha256=digest(source), rejected=dict(rejected),
                  serialization="Canonical uppercase sequence, no added spaces or domain tags; EOS added at tokenization",
                  iid_background_training_residue_counts=dict(aa_counts))
    write_json(root / "validation/corpora_protein.json", report)
    # Controls inherit source record IDs and split groups; background comes only from train.
    weights = [aa_counts[c] for c in AA]
    all_generated = {name: set() for name in ("shuffled", "randomaa")}
    for condition in ("shuffled", "randomaa"):
        control, duplicates = [], 0
        for row in records:
            rng = random.Random(text_hash(f"{SEED}:{condition}:{row['record_id']}"))
            if condition == "shuffled":
                symbols = list(row["text"]); rng.shuffle(symbols); text = "".join(symbols)
                assert Counter(text) == Counter(row["text"])
            else:
                text = "".join(rng.choices(AA, weights=weights, k=len(row["text"])))
            h = text_hash(text)
            if h in all_generated[condition]:
                duplicates += 1
                # Avoid silently changing splits for very short/low-complexity controls.
                continue
            all_generated[condition].add(h)
            control.append({**row, "record_id":condition + ":" + row["record_id"],
                            "parent_record_id":row["record_id"], "text":text, "source":condition})
        result = emit(root, condition, control)
        result.update(seed=SEED, removed_generated_exact_duplicates=duplicates,
                      background="Protein training residues only" if condition == "randomaa" else "Same per-record residue multiset",
                      parent_table_sha256=digest(source))
        write_json(root / "validation" / f"corpora_{condition}.json", result)
    print("protein and controls", {s: report["splits"][s]["records"] for s in SPLITS}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--condition", choices=("english", "dna", "protein", "all"), default="all")
    a = p.parse_args()
    for name, fn in [("english", english), ("dna", dna), ("protein", proteins)]:
        if a.condition in (name, "all"): fn(a.root)
