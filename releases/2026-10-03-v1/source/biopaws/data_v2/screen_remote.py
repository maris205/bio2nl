#!/usr/bin/env python3
"""Exclude source proteins with detectable matches to any official SCOPe domain.

All domains are excluded conservatively, regardless of remote benchmark split.
Coverage applies to the domain query; a short domain in a long protein counts.
"""
import argparse,csv,json
from pathlib import Path
from collections import Counter
from build_protein import run,read_tsv,write_tsv,dump,sha,COLS
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/bio-trans/data_rebuild/2026-09-25-v2'));ap.add_argument('--domains',type=Path,required=True);a=ap.parse_args();r=a.root;w=r/'work/protein';m=r/'tools/protein/mmseqs/bin/mmseqs'
    out=w/'remote_to_swissprot.tsv'
    run(r,[m,'easy-search',a.domains,r/'data/sequences/protein_canonical_all.fasta',out,w/'remote_screen_tmp',
           '--min-seq-id','0.3','-c','0.8','--cov-mode','2','-e','0.001','--threads','16','-s','7.5',
           '--max-seqs','10000','--max-accept','10000','--alignment-mode','3','--format-output',','.join(COLS),
           '--remove-tmp-files','1'],'remote_screen')
    excluded={};hits=0
    with out.open() as f:
        for line in f:
            v=line.rstrip().split('\t');identity=int(v[10])/int(v[3]);qc=(int(v[12])-int(v[11])+1)/int(v[8])
            if identity>=.3 and qc>=.8 and float(v[6])<=.001:
                hits+=1
                if v[1] not in excluded or identity>excluded[v[1]]['domain_identity']:
                    excluded[v[1]]={'accession':v[1],'remote_domain':v[0],'domain_identity':identity,'domain_coverage':qc,'evalue':float(v[6])}
    records=read_tsv(r/'data/sequences/protein_canonical_sequences.tsv.gz')
    counts=Counter()
    for row in records:
        if row['accession'] in excluded:
            row['pretrain_eligible']='0';row['pretrain_exclusion_reason']='detectable_match_to_official_SCOPe_domain';counts[row['split']]+=1
            excluded[row['accession']]['sequence_sha256']=row['sequence_sha256'];excluded[row['accession']]['split']=row['split']
    write_tsv(r/'data/sequences/protein_canonical_sequences.tsv.gz',records,list(records[0]))
    aliases=read_tsv(r/'data/sequences/protein_accession_splits.tsv.gz')
    for row in aliases:
        if row['canonical_accession'] in excluded:row['pretrain_eligible']='0'
    write_tsv(r/'data/sequences/protein_accession_splits.tsv.gz',aliases,list(aliases[0]))
    fields=['accession','sequence_sha256','split','remote_domain','domain_identity','domain_coverage','evalue']
    write_tsv(r/'metadata/protein_remote_exclusions.tsv.gz',excluded.values(),fields)
    dump(r/'metadata/protein_remote_exclusion_policy.json',{'domain_fasta':str(a.domains),'domain_fasta_sha256':sha(a.domains),'screen':'MMseqs2 >=30% sequence identity, >=80% query/domain coverage, E<=1e-3; target/full-protein coverage unrestricted. All domains regardless of split.',
         'interpretation':'Exclude detected domains and related sequences; this heuristic search cannot certify absence of all evolutionary remote homology.',
         'excluded_canonical_by_split':dict(counts),'excluded_canonical_total':len(excluded),'qualifying_alignment_hits':hits,'final_canonical_table_sha256':sha(r/'data/sequences/protein_canonical_sequences.tsv.gz')})
    print('remote exclusions',len(excluded),dict(counts),flush=True)
if __name__=='__main__':main()
