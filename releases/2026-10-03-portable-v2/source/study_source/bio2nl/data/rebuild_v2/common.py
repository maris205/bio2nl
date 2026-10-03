"""Small shared utilities; no dependency on legacy corpora or credentials."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def download(url, path, expected_sha256=None, expected_md5=None, direct=False):
    """Cache only checksum-verified files; never accept a leftover partial file."""
    import requests
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if expected_sha256 and digest(path) != expected_sha256:
            raise ValueError(f"Existing file does not match pinned SHA256: {path}")
        if expected_md5 and digest(path, "md5") != expected_md5:
            raise ValueError(f"Existing file does not match pinned MD5: {path}")
        return path
    tmp = path.with_suffix(path.suffix + ".partial")
    for attempt in range(3):
        try:
            session = requests.Session()
            session.trust_env = not direct
            with session.get(url, stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                with tmp.open("wb") as out:
                    for block in response.iter_content(1024 * 1024):
                        out.write(block)
            if expected_sha256 and digest(tmp) != expected_sha256:
                raise ValueError("Downloaded SHA256 differs from pinned source")
            if expected_md5 and digest(tmp, "md5") != expected_md5:
                raise ValueError("Downloaded MD5 differs from official checksum")
            tmp.replace(path)
            return path
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def fasta_records(stream):
    header, seq = None, []
    for line in stream:
        line = line.strip()
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq)
            header, seq = line[1:], []
        elif header is not None:
            seq.append(line)
    if header is not None:
        yield header, "".join(seq)


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
