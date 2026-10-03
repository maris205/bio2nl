#!/usr/bin/env python3
"""Acquire pinned official sources, or validate a fully offline frozen snapshot.

Current URLs are only acquisition locations. Future upstream releases are never
silently substituted: if unavailable, use the preserved release raw/tools files.
"""
import argparse, json, os, tarfile, urllib.request
from datetime import datetime,timezone
from pathlib import Path
from build_protein import RELEASE,RAW_SHA256,TOOL_SHA256,BASE,sha,dump

def download(url,path):
    temp=path.with_suffix(path.suffix+'.part')
    with urllib.request.urlopen(url,timeout=120) as r,open(temp,'wb') as f:
        while block:=r.read(1<<20): f.write(block)
    temp.replace(path)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
    raw=root/'raw/protein';tools=root/'tools/protein';raw.mkdir(parents=True,exist_ok=True);tools.mkdir(parents=True,exist_ok=True)
    fasta=raw/f'uniprot_sprot_{RELEASE}.fasta.gz';rd=raw/f'reldate_{RELEASE}.txt';evidence=raw/'release_verified_download.json'
    if not fasta.exists():
        before=urllib.request.urlopen(BASE+'reldate.txt',timeout=60).read().decode()
        if f'Release {RELEASE}' not in before:raise RuntimeError('Current upstream release changed; restore the frozen published raw snapshot for '+RELEASE)
        download(BASE+'uniprot_sprot.fasta.gz',fasta)
        after=urllib.request.urlopen(BASE+'reldate.txt',timeout=60).read().decode()
        if before!=after:raise RuntimeError('Upstream release changed during download')
        rd.write_text(before)
        dump(evidence,{'release_before':before,'release_after':after,'utc_checked':datetime.now(timezone.utc).isoformat(),'download_url':BASE+'uniprot_sprot.fasta.gz'})
    assert sha(fasta)==RAW_SHA256,'Frozen source hash mismatch'
    assert rd.exists() and evidence.exists(),'Missing packaged release/download evidence'
    archive=tools/'mmseqs-linux-avx2.tar.gz'
    if not archive.exists():download('https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz',archive)
    assert sha(archive)==TOOL_SHA256,'Current MMseqs archive changed; restore the frozen tools snapshot'
    if not (tools/'mmseqs/bin/mmseqs').exists():
        with tarfile.open(archive) as tf:tf.extractall(tools,filter='data')
    print('Pinned sources and MMseqs archive verified; no current-release network request needed for existing snapshots.')
if __name__=='__main__':main()
