#!/usr/bin/env python3
"""Fetch one explicitly requested, SHA-bound input; does not imply redistribution rights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.parse
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe",type=Path,required=True)
    parser.add_argument("--id",required=True)
    parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args()
    recipe=json.loads(args.recipe.read_text())
    matches=[r for r in recipe["sources"] if r["id"]==args.id]
    if len(matches)!=1:raise ValueError("Expected exactly one declared source id")
    spec=matches[0];url=urllib.parse.urlsplit(spec["url"])
    if url.scheme!="https" or url.username or url.password:raise ValueError("Recipe must use a credential-free HTTPS source URL")
    destination=args.destination.absolute()
    if destination.exists() or destination.is_symlink():raise FileExistsError(destination)
    if any(p.is_symlink() for p in destination.parents):raise ValueError("Destination ancestors cannot be symlinks")
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(prefix="verified-input-",dir=destination.parent,delete=False) as stream:
            temporary=Path(stream.name);digest=hashlib.sha256();size=0
            with urllib.request.urlopen(spec["url"],timeout=60) as response:
                if urllib.parse.urlsplit(response.geturl()).scheme!="https":
                    raise ValueError("Download redirected away from HTTPS")
                while data:=response.read(1024*1024):
                    size+=len(data)
                    if size>spec["bytes"]:raise ValueError("Source exceeds locked byte count; revision may have changed")
                    stream.write(data);digest.update(data)
        if size!=spec["bytes"] or digest.hexdigest()!=spec["sha256"]:
            raise ValueError("Downloaded bytes differ from fixed input; do not substitute current release")
        os.link(temporary,destination)
        print(json.dumps({"status":"downloaded_and_hash_verified","id":args.id,"bytes":size,"sha256":digest.hexdigest(),"redistribution_rights_granted":False}))
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


if __name__=="__main__":main()
