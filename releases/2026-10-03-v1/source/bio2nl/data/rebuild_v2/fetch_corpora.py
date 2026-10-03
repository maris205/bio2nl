"""Acquire a fixed OpenWebText shard and an accession-versioned reference genome."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from common import digest, download, timestamp, write_json

OWT_REV = "79d93d786212f7344586290adb811d4ae6a1762c"
OWT_FILE = "plain_text/train-00000-of-00080.parquet"
OWT_SHA = "caed9f4b7053d7cd4d1a13ce9ec9224d84a3bba1f11579193562a7e31ebe656e"
ASSEMBLY = "GCF_000001215.4"
PREFIX = ASSEMBLY + "_Release_6_plus_ISO1_MT"
NCBI_BASE = "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/215/" + PREFIX + "/"


def acquire_english(root):
    url = f"https://huggingface.co/datasets/Skylion007/openwebtext/resolve/{OWT_REV}/{OWT_FILE}"
    path = download(url, root / "raw/corpora/openwebtext-00000.parquet", expected_sha256=OWT_SHA)
    source = dict(source_id="openwebtext", repository="Skylion007/openwebtext",
                  revision=OWT_REV, upstream_file=OWT_FILE, url=url,
                  file=str(path.relative_to(root)), sha256=digest(path), bytes=path.stat().st_size,
                  acquired_at=timestamp(), scope="One explicitly selected source shard; not the full 8M-document corpus",
                  license="Repository card: CC0-1.0; individual crawled texts retain their original rights")
    write_json(root / "metadata/corpora_english_source.json", source)
    return source


def acquire_dna(root):
    raw = root / "raw/corpora"
    md5_path = download(NCBI_BASE + "md5checksums.txt", raw / (PREFIX + "_md5checksums.txt"), direct=True)
    checksums = {line.split()[1].removeprefix("./"): line.split()[0]
                 for line in md5_path.read_text().splitlines() if line.strip()}
    files = []
    for suffix in ["_assembly_report.txt", "_genomic.fna.gz"]:
        name = PREFIX + suffix
        path = download(NCBI_BASE + name, raw / name, expected_md5=checksums[name], direct=True)
        files.append(dict(file=str(path.relative_to(root)), url=NCBI_BASE + name,
                          md5=checksums[name], sha256=digest(path), bytes=path.stat().st_size))
    source = dict(source_id="drosophila_reference", organism="Drosophila melanogaster",
                  assembly_accession=ASSEMBLY, assembly_name="Release 6 plus ISO1 MT",
                  taxonomy_id=7227, upstream="NCBI RefSeq", acquired_at=timestamp(), files=files,
                  checksum_manifest_sha256=digest(md5_path),
                  license_url="https://www.ncbi.nlm.nih.gov/home/about/policies/")
    write_json(root / "metadata/corpora_dna_source.json", source)
    return source


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(lambda fn: fn(a.root), [acquire_english, acquire_dna]):
            print(result["source_id"], "download and checksum complete", flush=True)
