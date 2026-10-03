#!/usr/bin/env python3
"""Exclude Swiss-Prot proteins matching any pinned SCOPe domain from pretraining.

Run after remote FASTA extraction and before protein corpora/tokenization. Query
coverage refers to the entire SCOPe domain; this conservatively removes any
Swiss-Prot protein with >=30% identity over >=80% of a reference domain.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--threads",type=int,default=16);a=p.parse_args();r=a.root
    remote_fasta=r/"data/sequences/remote_domains.fasta"
    protein=r/"data/sequences/protein_canonical_base.tsv.gz"
    if not remote_fasta.exists() or not protein.exists():raise FileNotFoundError("Need remote preparation and canonical Swiss-Prot normalization first")
    mm=r/"tools/protein/mmseqs/bin/mmseqs"; work=r/"work/protein/remote_exclusion";work.mkdir(parents=True,exist_ok=True)
    out=work/"remote_vs_swissprot.tsv"
    command=[str(mm),"easy-search",str(remote_fasta),str(r/"data/sequences/protein_canonical_all.fasta"),str(out),str(work/"tmp"),
             "--min-seq-id","0.3","-c","0.8","--cov-mode","2","-e","0.001","--threads",str(a.threads),"-s","7.5",
             "--max-seqs","10000","--max-accept","10000","--alignment-mode","3","--format-output","query,target,fident,alnlen,qstart,qend,qlen,tstart,tend,tlen,nident,evalue,bits",
             "--remove-tmp-files","1"]
    env=dict(os.environ,OMP_NUM_THREADS=str(a.threads))
    with (work/"search.log").open("w") as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    homologs=set(); best={}
    with out.open() as f:
        for line in f:
            v=line.rstrip().split("\t")
            # q/t are full raw sequence IDs; MMseqs fident uses its aligned-residue denominator.
            identity=float(v[2]); qcov=(int(v[5])-int(v[4])+1)/int(v[6]); e=float(v[11])
            if identity>=0.3 and qcov>=0.8 and e<=0.001:
                homologs.add(v[1]);best[v[1]]=max(best.get(v[1],0.0),identity)
    n=0
    table_paths=[protein,r/"data/sequences/protein_canonical_sequences.tsv.gz"]
    for table in table_paths:
        if not table.exists():continue
        tmp=table.with_suffix(".tmp.gz")
        with gzip.open(table,"rt") as source, gzip.open(tmp,"wt",newline="") as dest:
            reader=csv.DictReader(source,delimiter="\t");fields=list(reader.fieldnames)
            for c in ("pretrain_eligible","pretrain_exclusion_reason"):
                if c not in fields:fields.append(c)
            writer=csv.DictWriter(dest,fieldnames=fields,delimiter="\t");writer.writeheader();table_n=0
            for row in reader:
                table_n+=1
                if row["accession"] in homologs:
                    row["pretrain_eligible"]="0";row["pretrain_exclusion_reason"]="SCOPe_2.08_domain_identity_ge_30pct_and_domain_query_coverage_ge_80pct"
                else:
                    row["pretrain_eligible"]="1";row["pretrain_exclusion_reason"]=""
                writer.writerow(row)
        tmp.replace(table)
        if table==protein:n=table_n
    report={"status":"completed","criterion":"MMseqs2 easy-search: any official SCOPe2.08 full-length domain query has >=30% aligned identity, >=80% query-domain coverage, E<=1e-3 against full UniProtKB Swiss-Prot canonical protein sequence.",
            "coverage_mode":"query/domain coverage; target protein may contain domain within a longer chain",
            "tool_version":subprocess.check_output([str(mm),"version"],text=True).strip(),"threads":a.threads,
            "remote_fasta_sha256":digest(remote_fasta),"protein_table_sha256_after":digest(protein),
            "matching_swissprot_accessions":len(homologs),"records_evaluated":n,"best_hit_identity_summary":{"max":max(best.values()) if best else None},
            "all_remote_domains_used":True,"domain_split_independent_exclusion":True,
            "command":command,"search_output_sha256":digest(out),"completed_utc":datetime.now(timezone.utc).isoformat()}
    meta=r/"metadata/protein_remote_pretraining_exclusion.json";meta.write_text(json.dumps(report,indent=2)+"\n")
    alias=r/"data/sequences/protein_accession_splits.tsv.gz"
    if alias.exists():
        canonical={}
        with gzip.open(protein,"rt") as f:
            for row in csv.DictReader(f,delimiter="\t"):canonical[row["accession"]]=row
        tmp=alias.with_suffix(".tmp.gz")
        with gzip.open(alias,"rt") as source,gzip.open(tmp,"wt",newline="") as dest:
            reader=csv.DictReader(source,delimiter="\t");fields=list(reader.fieldnames);writer=csv.DictWriter(dest,fieldnames=fields,delimiter="\t");writer.writeheader()
            for row in reader:
                base=canonical.get(row.get("canonical_accession", ""))
                row["pretrain_eligible"]=base["pretrain_eligible"] if base else "0"
                if base and base["pretrain_eligible"]=="0":row["pretrain_exclusion_reason"]=base["pretrain_exclusion_reason"]
                writer.writerow(row)
        tmp.replace(alias)
    print(json.dumps({"eligible":n-len(homologs),"excluded":len(homologs),"report":str(meta)}),flush=True)


if __name__=="__main__":main()
