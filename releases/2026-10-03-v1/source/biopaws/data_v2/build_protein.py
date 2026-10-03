#!/usr/bin/env python3
"""Rebuild a versioned sequence-similarity benchmark from official Swiss-Prot.

All scientific thresholds are explicit. This is a sequence similarity task, not
a structural remote-homology ground truth dataset. MMseqs2 searches are heuristic.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse, csv, gzip, hashlib, json, re, subprocess, time, urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

RELEASE='2026_03'
RAW_SHA256='a9c3496aa727e61d25d9e1291c9ba28d2c2ad19c0e26569dc083d53ec4169536'
TOOL_SHA256='7d84a954cc8373ef93e27c64c3c15f3d1b7c07b2a993f437c9a5508fd1493b61'
BASE='https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/'
AA=set('ACDEFGHIKLMNPQRSTVWY')
COLS=['query','target','fident','alnlen','qcov','tcov','evalue','bits','qlen','tlen','nident','qstart','qend','tstart','tend']
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def seqsha(s): return hashlib.sha256(s.encode()).hexdigest()
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2)+'\n')
def fasta(path):
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt') as f:
        header=None; buf=[]
        for line in f:
            if line.startswith('>'):
                if header is not None: yield header,''.join(buf)
                header=line[1:].strip(); buf=[]
            else: buf.append(line.strip())
        if header is not None: yield header,''.join(buf)
def write_tsv(path,rows,fields):
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',extrasaction='ignore'); w.writeheader(); w.writerows(rows)
def read_tsv(path):
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt') as f: return list(csv.DictReader(f,delimiter='\t'))
def run(root,cmd,name):
    env=dict(os.environ,OMP_NUM_THREADS='16'); started=time.time()
    with open(root/'work/protein'/f'{name}.log','w') as log:
        subprocess.run(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
    dump(root/'metadata'/f'protein_{name}_command.json',{'argv':list(map(str,cmd)),'seconds':time.time()-started,'utc_completed':datetime.now(timezone.utc).isoformat()})
def normalize(root):
    raw=root/'raw/protein'/f'uniprot_sprot_{RELEASE}.fasta.gz'
    before=(root/'raw/protein'/f'reldate_{RELEASE}.txt').read_text()
    # Normalization/reproduction is offline. Network release checks belong to acquisition.
    evidence=json.loads((root/'raw/protein/release_verified_download.json').read_text())
    after=evidence['release_after']
    assert before==after and f'Release {RELEASE}' in before
    assert sha(raw)==RAW_SHA256, 'Frozen raw Swiss-Prot hash mismatch'
    assert sha(root/'tools/protein/mmseqs-linux-avx2.tar.gz')==TOOL_SHA256, 'Frozen MMseqs2 hash mismatch'
    canonical={}; aliases=[]; counts=Counter(); all_count=0
    for header,sequence in fasta(raw):
        all_count+=1
        accession,entry=header.split()[0].split('|')[1:]
        seq=sequence.upper(); digest=seqsha(seq)
        tax=re.search(r'\bOX=(\d+)',header); sv=re.search(r'\bSV=(\d+)',header)
        reasons=[]
        if not seq or not set(seq)<=AA: reasons.append('noncanonical_amino_acid')
        if '(Fragment)' in header or '(Fragments)' in header: reasons.append('annotated_fragment')
        row={'accession':accession,'entry_name':entry,'taxon_id':tax.group(1) if tax else '',
             'sequence_version':sv.group(1) if sv else '', 'sequence_sha256':digest,'sequence':seq,'length':len(seq)}
        if reasons:
            for reason in reasons: counts[reason]+=1
            aliases.append({**row,'canonical_accession':'','filter_reason':'|'.join(reasons)})
            continue
        if digest not in canonical: canonical[digest]=row
        aliases.append({**row,'canonical_accession':canonical[digest]['accession'],'filter_reason':''})
    records=sorted(canonical.values(),key=lambda r:r['accession'])
    fields=['accession','entry_name','taxon_id','sequence_version','sequence_sha256','sequence','length']
    write_tsv(root/'data/sequences/protein_canonical_base.tsv.gz',records,fields)
    write_tsv(root/'data/sequences/protein_source_accessions.tsv.gz',aliases,fields+['canonical_accession','filter_reason'])
    with open(root/'data/sequences/protein_canonical_all.fasta','w') as full, open(root/'data/sequences/protein_candidates_40_250.fasta','w') as short:
        for r in records:
            record=f">{r['accession']}\n{r['sequence']}\n"; full.write(record)
            if 40<=r['length']<=250: short.write(record)
    tool=root/'tools/protein/mmseqs/bin/mmseqs'
    version=subprocess.check_output([str(tool),'version'],text=True,env=dict(os.environ,OMP_NUM_THREADS='1')).strip()
    info={'source':'official UniProtKB/Swiss-Prot','release':RELEASE,'download_url':BASE+'uniprot_sprot.fasta.gz',
          'release_snapshot_note':'Current-release URL; release checked before/after and frozen by file SHA-256. URL itself is not an immutable archive.',
          'release_before':before,'release_after':after,'raw_sha256':sha(raw),'raw_bytes':raw.stat().st_size,
          'source_records':all_count,'filter_reason_counts':dict(counts),'canonical_unique_sequences':len(records),
          'benchmark_candidate_unique_sequences':sum(40<=r['length']<=250 for r in records),
          'normalization':'Full official sequence uppercased; reject non-20-AA and annotated fragments; exact SHA-256 dedup; retain every source accession/alias.',
          'mmseqs2_binary_version':version,'mmseqs2_tar_sha256':sha(root/'tools/protein/mmseqs-linux-avx2.tar.gz'),
          'mmseqs2_download_url':'https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz','created_utc':datetime.now(timezone.utc).isoformat()}
    dump(root/'metadata/protein_sources.json',info); print(json.dumps(info,indent=2),flush=True)
def cluster(root):
    m=root/'tools/protein/mmseqs/bin/mmseqs'; w=root/'work/protein'
    run(root,[m,'easy-cluster',root/'data/sequences/protein_canonical_all.fasta',w/'full',w/'cluster_tmp',
              '--min-seq-id','0.3','-c','0.8','--cov-mode','0','--cluster-mode','1','--linclust-version','1',
              '--threads','16','-s','7.5','--max-seqs','1000','--alignment-mode','3','--remove-tmp-files','1'], 'full_cluster')
def search(root):
    m=root/'tools/protein/mmseqs/bin/mmseqs'; w=root/'work/protein'; inp=root/'data/sequences/protein_candidates_40_250.fasta'
    run(root,[m,'easy-search',inp,inp,w/'short_alignments.tsv',w/'search_tmp',
              '--min-seq-id','0.3','-c','0.8','--cov-mode','0','-e','0.001','--threads','16','-s','7.5',
              '--max-seqs','10000','--max-accept','10000','--alignment-mode','3','--format-output',','.join(COLS),
              '--remove-tmp-files','1'], 'short_search')

class Union:
    def __init__(self,ids): self.p={x:x for x in ids}; self.size={x:1 for x in ids}
    def find(self,x):
        while self.p[x]!=x: self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return
        if self.size[a]<self.size[b]: a,b=b,a
        self.p[b]=a; self.size[a]+=self.size[b]

def graph_split(root):
    records=read_tsv(root/'data/sequences/protein_canonical_base.tsv.gz'); byid={r['accession']:r for r in records}; uf=Union(byid)
    with open(root/'work/protein/full_cluster.tsv') as f:
        for line in f:
            a,b=line.rstrip().split('\t'); uf.join(a,b)
    positive=[]; n_hits=0
    with open(root/'work/protein/short_alignments.tsv') as f:
        for line in f:
            v=line.rstrip().split('\t'); a,b=v[:2]
            # MMseqs2's nident output is zero for this pinned binary, so use
            # its explicit fident field. Recompute coverage from coordinates.
            identity=float(v[2]);qcov=(int(v[12])-int(v[11])+1)/int(v[8]);tcov=(int(v[14])-int(v[13])+1)/int(v[9]);evalue=float(v[6])
            if a==b:continue
            if identity>=.3 and min(qcov,tcov)>=.8 and evalue<=.001:
                uf.join(a,b); n_hits+=1
                if a<b and .4<=identity<=.9:
                    positive.append({'a':a,'b':b,'identity':identity,'query_coverage':qcov,'target_coverage':tcov,'evalue':evalue,'alignment_length':int(v[3])})
    groups=defaultdict(list)
    for x in byid: groups[uf.find(x)].append(x)
    group_list=sorted(groups.values(),key=lambda xs:(-len(xs),min(xs)))
    totals={s:0 for s in ['train','validation','test']}; fractions={'train':.7,'validation':.15,'test':.15}; assigned={}
    for members in group_list:
        s=min(totals,key=lambda s:totals[s]/fractions[s]); cluster_id='sp30_'+min(members)
        for x in members: assigned[x]=(cluster_id,s)
        totals[s]+=len(members)
    for r in records:
        r['cluster_id'],r['split']=assigned[r['accession']]
        # Preserve precomputed source-exclusion decisions (for example, any
        # Swiss-Prot sequence matching the frozen SCOPe domain library).
        r.setdefault('pretrain_eligible','1'); r.setdefault('pretrain_exclusion_reason','')
    fields=list(records[0]); write_tsv(root/'data/sequences/protein_canonical_sequences.tsv.gz',records,fields)
    # All aliases map to the same canonical split. Filtered source rows stay excluded.
    aliases=read_tsv(root/'data/sequences/protein_source_accessions.tsv.gz')
    by_accession={r['accession']:r for r in records}
    for r in aliases:
        c=r['canonical_accession']; r['cluster_id'],r['split']=assigned.get(c,('','excluded'))
        source=by_accession.get(c)
        r['pretrain_eligible']=source['pretrain_eligible'] if source else '0'
        r['pretrain_exclusion_reason']=source['pretrain_exclusion_reason'] if source else r.get('filter_reason','filtered_from_protein_corpus')
    write_tsv(root/'data/sequences/protein_accession_splits.tsv.gz',aliases,list(aliases[0]))
    with open(root/'work/protein/positive_candidates.jsonl','w') as f:
        for p in positive:
            p['split']=assigned[p['a']][1]; p['cluster_id']=assigned[p['a']][0]; f.write(json.dumps(p)+'\n')
    dump(root/'metadata/protein_split_policy.json',{'definition':'Connected components of full-library MMseqs2 clusters, union all subsequently detected short-sequence links >=30% identity, >=80% coverage on both ends, E<=1e-3; whole components greedily assigned 70/15/15 by sequence count before pairing.',
        'search_limitations':'MMseqs2 prefilter/search is heuristic, full-library clustering has max-seqs=1000. This is operational cluster isolation, not proof of absence of all evolutionary homology.',
        'canonical_counts':totals,'cluster_count':len(groups),'largest_cluster':max(map(len,groups.values())), 'short_detected_directed_nonself_links':n_hits,'positive_candidate_undirected_edges':len(positive),'positive_definition':'MMseqs2 identity 40-90%, both coverages >=80%, E<=1e-3; full canonical source strings are the inputs.'})
    print('graph_split',totals,'clusters',len(groups),'positive_edges',len(positive),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/bio-trans/data_rebuild/2026-09-25-v2')); ap.add_argument('stage',choices=['normalize','cluster','search','graph_split']); a=ap.parse_args()
    globals()[a.stage](a.root)
if __name__=='__main__':main()
