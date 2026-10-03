#!/usr/bin/env python3
"""Download the authors' fixed SCOPe 2.08 stable archive; verify published MD5.

The official download page links the Zenodo concept DOI 5829560. We pin the
actual version record 5829561, not the moving concept or periodic updates.
No models, credentials, or GPU libraries are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

RECORD = "5829561"
FILES = {
    "dir.cla.scope.2.08-stable.txt": "97232c4552f4e77f0194f671011578ae",
    "scopeseq-2.08.tgz": "d595ba9998e3029ede93f8c7f612e228",
}


def digest(path, algorithm="sha256"):
    obj = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            obj.update(chunk)
    return obj.hexdigest()


def download(raw, metadata, name):
    info = next(f for f in metadata["files"] if f["key"] == name)
    assert info["checksum"] == "md5:" + FILES[name]
    path = raw / name
    url = info["links"]["self"]
    if path.exists():
        if digest(path, "md5") != FILES[name]:
            raise ValueError(f"Existing file checksum mismatch: {path}")
    else:
        partial = path.with_name(path.name + ".partial")
        for attempt in range(5):
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            try:
                with requests.get(url, headers=headers, stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    append = bool(offset and response.status_code == 206)
                    done = offset if append else 0
                    milestone = done // (50 * 1024 * 1024)
                    with partial.open("ab" if append else "wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            done += len(chunk)
                            if done // (50 * 1024 * 1024) > milestone:
                                milestone = done // (50 * 1024 * 1024)
                                print(f"{name}: {done}/{info['size']} bytes", flush=True)
                if partial.stat().st_size != info["size"]:
                    raise IOError(f"Size mismatch for {name}")
                if digest(partial, "md5") != FILES[name]:
                    raise ValueError(f"Published MD5 mismatch for {name}")
                partial.replace(path)
                break
            except (requests.RequestException, IOError) as exc:
                print(f"download attempt {attempt+1} {name}: {type(exc).__name__}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(2)
    result = {"file": str(path), "url": url, "bytes": path.stat().st_size,
              "md5": digest(path, "md5"), "sha256": digest(path)}
    print(f"verified {name}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    args = parser.parse_args()
    raw = args.release_root.resolve() / "raw/remote"
    raw.mkdir(parents=True, exist_ok=True)
    record_file = raw / "zenodo_record_5829561.json"
    if record_file.exists():
        record = json.loads(record_file.read_text())
    else:
        response = requests.get(f"https://zenodo.org/api/records/{RECORD}", timeout=45)
        response.raise_for_status()
        record = response.json()
        assert str(record["id"]) == RECORD
        record_file.write_text(json.dumps(record, indent=2) + "\n")
    assert record["metadata"]["license"]["id"] == "cc-by-4.0"
    with ThreadPoolExecutor(max_workers=2) as pool:
        downloaded = list(pool.map(lambda name: download(raw, record, name), FILES))
    archive = raw / "scopeseq-2.08.tgz"
    target = raw / "astral-scopedom-seqres-gd-all-2.08-stable.fa"
    with tarfile.open(archive, "r:gz") as tar:
        matches = [m for m in tar.getmembers()
                   if m.isfile() and Path(m.name).name == target.name]
        if len(matches) != 1:
            alternatives = [m.name for m in tar.getmembers() if "seqres-gd-all" in m.name]
            raise ValueError(f"Expected one stable full genetic-domain FASTA: {alternatives}")
        member = matches[0]
        if not target.exists():
            with tar.extractfile(member) as source, target.open("wb") as dest:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    dest.write(chunk)
    for item in downloaded:
        item["file"] = str(Path(item["file"]).relative_to(args.release_root.resolve()))
    sources = {
        "database": "SCOPe / ASTRAL", "release": "2.08-stable",
        "publication_date": record["metadata"]["publication_date"],
        "archive_version_doi": "10.5281/zenodo.5829561",
        "official_concept_archive_doi": "10.5281/zenodo.5829560",
        "official_download_page": "https://scop.berkeley.edu/downloads/ver=2.08",
        "official_about_page": "https://scop.berkeley.edu/about/ver=2.08",
        "license": "CC-BY-4.0", "license_evidence": "pinned authors' Zenodo record metadata.license.id",
        "copyright_notice": "SCOP and SCOPe authors; attribution required",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "files": downloaded,
        "extracted_fasta": {"file": str(target.relative_to(args.release_root.resolve())), "archive_member": member.name,
                            "bytes": target.stat().st_size, "sha256": digest(target)},
        "citations": ["10.1093/nar/gkab1054", "10.1093/nar/gkt1240", "10.1093/nar/gkh034"],
        "third_party_filtered_subset_used": False,
    }
    metadata_dir = args.release_root.resolve() / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "remote_sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    print(json.dumps({"status": "download_verified", "fasta": str(target)}), flush=True)


if __name__ == "__main__":
    main()
