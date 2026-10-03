#!/usr/bin/env python3
"""Report Swiss-Prot/SCOPe sequence overlap across pair tasks and pretraining."""
import argparse,csv,gzip,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path

def read_tsv(path):
    with gzip.open(path,'rt') as f:return list(csv.DictReader(f,delimiter='\t'))
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    hits=defaultdict(set);hit_score={}
    hitfile=r/'work/protein/remote_exclusion/remote_vs_swissprot.tsv'
    with hitfile.open() as f:
        for line in f:
            v=line.rstrip().split('\t')
            identity=float(v[2]);qcov=(int(v[5])-int(v[4])+1)/int(v[6]);eval_=float(v[11])
            if identity>=.3 and qcov>=.8 and eval_<=.001:
                hits[v[0]].add(v[1]);hit_score[(v[0],v[1])]=identity
    protein=read_tsv(r/'data/sequences/protein_canonical_sequences.tsv.gz')
    byid={x['accession']:x for x in protein}; excluded={t for values in hits.values() for t in values}
    raw_fasta=r/'raw/remote/astral-scopedom-seqres-gd-all-2.08-stable.fa'
    remote_seq={}
    header=None;seq=[]
    with raw_fasta.open() as f:
        for line in f:
            if line.startswith('>'):
                if header is not None:remote_seq[header]=''.join(seq).upper()
                header=line[1:].split()[0];seq=[]
            else:seq.append(line.strip())
        if header is not None:remote_seq[header]=''.join(seq).upper()
    exact_domain_to_protein=defaultdict(set)
    for row in protein:exact_domain_to_protein[row['sequence_sha256']].add(row['accession'])
    exact={}
    for did,seq in remote_seq.items():
        key=hashlib.sha256(seq.encode()).hexdigest()
        if key in exact_domain_to_protein:exact[did]=sorted(exact_domain_to_protein[key])
    pair_tables={
      'sequence_similarity':read_tsv(r/'data/pairs/protein_sequence_similarity_all.tsv.gz'),
    }
    remote_pairs_path=r/'data/pairs/remote_pairs.parquet'
    if remote_pairs_path.exists():
        import pandas as pd
        frame=pd.read_parquet(remote_pairs_path)
        pair_tables['SCOPe_remote']=frame.to_dict('records')
    task_summary={}
    for name,rows in pair_tables.items():
        endpoint=set()
        for row in rows:
            endpoint.update((row.get('accession_a',''),row.get('accession_b',''),row.get('sequence_id_a',''),row.get('sequence_id_b','')))
        if name=='sequence_similarity':
            acc={x for x in endpoint if x in byid}
            matching=acc&excluded
            splitcounts=Counter(byid[x]['split'] for x in matching)
            pair_occurrences={s:{str(y):sum(row['split']==s and row['label']==str(y) and
                (row['accession_a'] in matching or row['accession_b'] in matching) for row in rows) for y in (0,1)}
                for s in ('train','validation','test')}
            task_summary[name]={'unique_accessions':len(acc),'matches_any_SCOPe_domain_threshold':len(matching),
                'matching_accessions_by_protein_cluster_split':dict(splitcounts),
                'pair_rows_touching_matching_accession_by_split_and_label':pair_occurrences,
                'exact_domain_sequence_accessions':len(acc & set().union(*(set(x) for x in exact.values())))}
        else:
            remote_ids={x for x in endpoint if x in remote_seq}
            task_summary[name]={'unique_domains':len(remote_ids),'exact_domain_sequence_accessions':sum(x in exact for x in remote_ids),
                'threshold_hit_swissprot_accessions':len(set().union(*(hits.get(x,set()) for x in remote_ids))) if remote_ids else 0}
    eligible={x['accession'] for x in protein if x['pretrain_eligible']=='1'}
    report={'status':'completed','identity_threshold':.30,'minimum_remote_domain_query_coverage':.80,'evalue_max':.001,
      'search_output_sha256':sha(hitfile),'protein_table_sha256':sha(r/'data/sequences/protein_canonical_sequences.tsv.gz'),
      'remote_scop_domains_in_raw_fasta':len(remote_seq),'domains_with_qualifying_swissprot_hits':len(hits),
      'unique_qualifying_swissprot_accessions':len(excluded),'excluded_from_protein_pretraining':len(excluded & set(byid)),
      'remaining_eligible_swissprot_accessions_with_qualifying_hits':len(excluded & eligible),
      'exact_domain_sequence_overlap':len(exact),'exact_domain_accessions_with_swissprot':sum(len(x) for x in exact.values()),
      'pair_task_overlap':task_summary,
      'interpretation':'All Swiss-Prot matches to any SCOPe domain are excluded from the protein pretraining corpus. Overlap with separate supervised pair-task source tables is measured and disclosed; the pair tasks are trained/evaluated separately. MMseqs2 is heuristic and the fixed thresholds do not certify absence of all remote evolutionary relationships.',
      'limitations':['Sequence comparison uses MMseqs2 heuristic prefilter and may miss remote relationships','A reported overlap is an audit finding, not an automatic exclusion from a separately run benchmark protocol.']}
    path=r/'validation/cross_dataset_homology.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','domains_with_qualifying_swissprot_hits','unique_qualifying_swissprot_accessions','remaining_eligible_swissprot_accessions_with_qualifying_hits','pair_task_overlap')},indent=2),flush=True)

if __name__=='__main__':main()
