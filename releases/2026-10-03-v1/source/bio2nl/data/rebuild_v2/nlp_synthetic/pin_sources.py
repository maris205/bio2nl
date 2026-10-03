#!/usr/bin/env python3
"""Pin public HF NLP source objects without using credentials or executing dataset code."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
SOURCES = [
    ("pawsx_en", "google-research-datasets/paws-x", "en", "4cd8187c404bda33cb1f62b49b001115862acf37"),
    ("cola", "nyu-mll/glue", "cola", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"),
    ("rte", "nyu-mll/glue", "rte", "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"),
]


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "bio2nl-public-data-rebuild/2"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    lock = {"schema_version": 1, "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "sources": []}
    for task, repo, config, revision in SOURCES:
        info_url = f"https://huggingface.co/api/datasets/{repo}/revision/{revision}"
        info = json.loads(fetch(info_url))
        if info["sha"] != revision:
            raise ValueError(f"Revision mismatch for {repo}")
        tree_url = f"https://huggingface.co/api/datasets/{repo}/tree/{revision}/{config}"
        entries = json.loads(fetch(tree_url))
        files = []
        for split in ("train", "validation", "test"):
            path = f"{config}/{split}-00000-of-00001.parquet"
            obj = next(x for x in entries if x["path"] == path)
            oid = obj.get("lfs", {}).get("oid")
            if not oid or len(oid) != 64:
                raise ValueError(f"No authoritative LFS SHA-256 for {path}")
            files.append({"split": split, "path": path, "sha256": oid, "size": obj["size"],
                          "url": f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"})
        card_url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/README.md"
        card = fetch(card_url)
        (HERE / f"nlp_source_card_{task}.md").write_bytes(card)
        lock["sources"].append({"task": task, "repo_id": repo, "config": config, "revision": revision,
                                "revision_api": info_url, "tree_api": tree_url, "files": files,
                                "card_url": card_url, "card_sha256": hashlib.sha256(card).hexdigest(),
                                "card_license_metadata": info.get("cardData", {}).get("license")})
        print(f"Verified {task}: {revision}", flush=True)
    target = HERE / "nlp_sources_lock.json"
    target.write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n")
    print(target)


if __name__ == "__main__":
    main()
