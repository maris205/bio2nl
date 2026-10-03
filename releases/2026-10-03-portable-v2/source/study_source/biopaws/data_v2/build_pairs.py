#!/usr/bin/env python3
"""Construct pairs after cluster split; preserve each endpoint's label degree.

Disjoint positive edges are rewired within exact B-length groups using a perfect
matching. Each retained accession occurs once in each label in the same column.
Both labels therefore have exactly equal endpoint/length marginals, and each A
has the same |length(A)-length(B)| under both labels. Evolutionary similarity and
composition remain legitimate signals in this sequence-similarity task.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse, csv, gzip, hashlib, json, random, time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from Bio.Align import PairwiseAligner, substitution_matrices
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from build_protein import read_tsv, write_tsv, dump, seqsha, sha

def aligner(mode):
    a=PairwiseAligner(); a.mode=mode; a.substitution_matrix=substitution_matrices.load('BLOSUM62'); a.open_gap_score=-11; a.extend_gap_score=-1
    return a
GLOBAL=aligner('global'); LOCAL=aligner('local')
def alignment_metrics(a,b,engine):
    al=engine.align(a,b)[0]; coords=al.coordinates; matches=0; cols=0
    for i in range(coords.shape[1]-1):
        a0,a1=int(coords[0,i]),int(coords[0,i+1]); b0,b1=int(coords[1,i]),int(coords[1,i+1])
        la,lb=a1-a0,b1-b0; cols+=max(la,lb)
        if la and lb: matches+=sum(x==y for x,y in zip(a[a0:a1],b[b0:b1]))
    return {'identity':matches/max(1,cols),'query_coverage':int(coords[0,-1]-coords[0,0])/len(a),
            'target_coverage':int(coords[1,-1]-coords[1,0])/len(b),'alignment_length':cols,'score':float(al.score)}

def validate_and_baselines(root,rows,byid):
    """Independent structural checks and train-fitted shortcut baselines."""
    import pandas as pd
    failures=[]; degree=Counter(); seqsplit=defaultdict(set); pair_ids=Counter(); block_labels=defaultdict(Counter); a_lengths=defaultdict(lambda:{0:[],1:[]})
    aa='ACDEFGHIKLMNPQRSTVWY'
    for r in rows:
        if r['sequence_sha256_a']==r['sequence_sha256_b']:failures.append('self_pair')
        if set(r['sentence1'])-set(aa) or set(r['sentence2'])-set(aa):failures.append('noncanonical_or_gap_input')
        pair_ids[r['pair_id']]+=1;block_labels[(r['split'],r['block_id'])][r['label']]+=1
        a_lengths[(r['split'],r['sequence_sha256_a'])][r['label']].append(len(r['sentence2']))
        seqsplit[r['sequence_sha256_a']].add(r['split']);seqsplit[r['sequence_sha256_b']].add(r['split'])
        degree[(r['split'],r['sequence_sha256_a'],'a',r['label'])]+=1
        degree[(r['split'],r['sequence_sha256_b'],'b',r['label'])]+=1
        if r['label']==1 and not (.4<=float(r['mmseqs_identity'])<=.9 and float(r['mmseqs_query_coverage'])>=.8 and float(r['mmseqs_target_coverage'])>=.8):failures.append('positive_definition_violation')
    if any(n!=1 for n in pair_ids.values()):failures.append('duplicate_unordered_pair')
    if any(len(s)>1 for s in seqsplit.values()):failures.append('sequence_cross_split_overlap')
    if any(x[0]!=x[1] for x in block_labels.values()):failures.append('rewiring_block_not_label_balanced')
    length_mismatch=sum(sorted(v[0])!=sorted(v[1]) for v in a_lengths.values())
    if length_mismatch:failures.append('length_difference_not_matched_per_A')
    ids={(s,seq,role) for s,seq,role,l in degree}
    degree_table=[]
    for s,seq,role in sorted(ids):
        pos,neg=degree[(s,seq,role,1)],degree[(s,seq,role,0)]
        degree_table.append({'split':s,'sequence_sha256':seq,'role':role,'positive_degree':pos,'negative_degree':neg})
        if pos!=neg:failures.append('endpoint_role_degree_mismatch')
    if degree_table:
        pd.DataFrame(degree_table).to_csv(root/'validation/protein_endpoint_degrees.tsv',sep='\t',index=False)
    features=[]
    for r in rows:
        x,y=r['sentence1'],r['sentence2'];cx=Counter(x);cy=Counter(y)
        vx=np.array([cx[c]/len(x) for c in aa]);vy=np.array([cy[c]/len(y) for c in aa])
        features.append({'length_a':len(x),'length_b':len(y),'length_difference':abs(len(x)-len(y)),
                         'composition_l1':float(np.abs(vx-vy).sum()),
                         'kmer3_jaccard':len({x[i:i+3] for i in range(len(x)-2)}&{y[i:i+3] for i in range(len(y)-2)})/max(1,len({x[i:i+3] for i in range(len(x)-2)}|{y[i:i+3] for i in range(len(y)-2)})),
                         'global_identity':float(r['global_identity'])})
    frame=pd.DataFrame(features);labels=np.array([r['label'] for r in rows]);splits=np.array([r['split'] for r in rows]);baseline=[]
    for feat,score in [('negative_length_difference',-frame.length_difference.to_numpy()),('negative_composition_l1',-frame.composition_l1.to_numpy()),('3mer_jaccard',frame.kmer3_jaccard.to_numpy()),('global_identity',frame.global_identity.to_numpy())]:
        for split in ('train','validation','test'):
            mask=splits==split
            baseline.append({'method':feat,'split':split,'n':int(mask.sum()),'accuracy':None,'auroc':float(roc_auc_score(labels[mask],score[mask])),'mcc':None})
    # Logistic probes fit only to train; feature interpretation is diagnostic.
    for name,cols in {'length_logistic':['length_a','length_b','length_difference'],
                      'composition_logistic':['composition_l1'],'length_composition_kmer_logistic':['length_a','length_b','length_difference','composition_l1','kmer3_jaccard']}.items():
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=20260925))
        train=splits=='train';model.fit(frame.loc[train,cols],labels[train])
        for split in ('train','validation','test'):
            mask=splits==split;pred=model.predict(frame.loc[mask,cols]);prob=model.predict_proba(frame.loc[mask,cols])[:,1]
            baseline.append({'method':name,'split':split,'n':int(mask.sum()),'accuracy':float(accuracy_score(labels[mask],pred)),
                             'auroc':float(roc_auc_score(labels[mask],prob)),'mcc':float(matthews_corrcoef(labels[mask],pred))})
    pd.DataFrame(baseline).to_csv(root/'validation/protein_pair_baselines.csv',index=False)
    counts={s:{str(l):sum(r['split']==s and r['label']==l for r in rows) for l in (0,1)} for s in ('train','validation','test')}
    for s,c in counts.items():
        if c['0']!=c['1']:failures.append('split_class_imbalance:'+s)
        if c['0']==0:failures.append('empty_split:'+s)
    result={'acceptance_pass':not failures,'failures':sorted(set(failures)),'pair_counts':counts,
            'unique_unordered_pairs':len(pair_ids),'unique_sequences':len(seqsplit),'endpoint_rows':len(degree_table),
            'sequence_cross_split_overlap':sum(len(x)>1 for x in seqsplit.values()),
            'endpoint_role_degree_balance':not any('degree_mismatch' in x for x in failures),
            'length_difference_matched_per_A':length_mismatch==0,'A_endpoints_checked_for_length_match':len(a_lengths),
            'A_endpoints_with_length_mismatch':length_mismatch,'gap_or_noncanonical_input_count':sum(bool(set(r['sentence1'])-set(aa) or set(r['sentence2'])-set(aa)) for r in rows),
            'task':'Operational sequence-similarity discrimination; negative thresholds screen alignment, not evolutionary proof.',
            'train_fitted_baselines':'validation/test receive no model fitting; report length/composition/kmer/alignment probes without tuning data to chance.',
            'protocol_limitations':['MMseqs2 heuristic search may miss remote similarity','pair split isolates operational detected sequence clusters, not every possible family relationship','degree balance is within full task split; training subsamples must select whole block_id cycles']}
    dump(root/'validation/protein_pair_acceptance.json',result)
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/bio-trans/data_rebuild/2026-09-25-v2')); ap.add_argument('--seed',type=int,default=20260925); a=ap.parse_args(); root=a.root
    started=time.time(); rows=read_tsv(root/'data/sequences/protein_canonical_sequences.tsv.gz'); byid={r['accession']:r for r in rows}
    candidates=[json.loads(l) for l in open(root/'work/protein/positive_candidates.jsonl')]
    rng=random.Random(a.seed); rng.shuffle(candidates); used=set(); buckets=defaultdict(list)
    for p in candidates:
        x,y=p['a'],p['b']
        if x in used or y in used:continue
        # Flip orientation independently; each endpoint keeps this column for both labels.
        if rng.random()<.5:
            p={**p,'a':y,'b':x,'query_coverage':p['target_coverage'],'target_coverage':p['query_coverage']}; x,y=y,x
        used.update([x,y]); buckets[(p['split'],len(byid[y]['sequence']))].append(p)
    output=[]; counts=Counter(); blocks=0; align_cache={}
    for (split,length),group in sorted(buckets.items()):
        # An inter-component perfect matching requires no component to occupy > half.
        while group:
            clusters=Counter(p['cluster_id'] for p in group); dominant,n=clusters.most_common(1)[0]
            if n<=len(group)/2:break
            j=next(i for i,p in enumerate(group) if p['cluster_id']==dominant); group.pop(j); counts['dropped_component_dominance']+=1
        n=len(group)
        if n<2:counts['dropped_singleton_length_bucket']+=n;continue
        prng=np.random.RandomState(int(seqsha(f'{a.seed}:{split}:{length}')[:8],16)); cost=prng.random_sample((n,n))
        for i,p in enumerate(group):
            for j,q in enumerate(group):
                if p['cluster_id']==q['cluster_id']:cost[i,j]=np.inf
        success=False
        for attempt in range(n+1):
            try: src,dst=linear_sum_assignment(cost)
            except ValueError:break
            invalid=[]
            for i,j in zip(src,dst):
                x,y=group[i]['a'],group[j]['b']; key=(x,y)
                if key not in align_cache:
                    sx,sy=byid[x]['sequence'],byid[y]['sequence']; align_cache[key]=(alignment_metrics(sx,sy,GLOBAL),alignment_metrics(sx,sy,LOCAL))
                gm,lm=align_cache[key]
                if gm['identity']>.25 or (lm['identity']>=.3 and min(lm['query_coverage'],lm['target_coverage'])>=.8):invalid.append((i,j))
            if not invalid:success=True;break
            for i,j in invalid:cost[i,j]=np.inf;counts['rejected_negative_alignment']+=1
        if not success:counts['dropped_infeasible_length_bucket_pairs']+=n;continue
        permutation={int(i):int(j) for i,j in zip(src,dst)}; visited=set(); block_by_index={}
        for i in range(n):
            if i in visited:continue
            block_id=f'{split}_cycle_{blocks:06d}';blocks+=1; j=i
            while j not in visited:visited.add(j);block_by_index[j]=block_id;j=permutation[j]
        for i,p in enumerate(group):
            for label,y in [(1,p['b']),(0,group[permutation[i]]['b'])]:
                x=p['a']; rx,ry=byid[x],byid[y]; sx,sy=rx['sequence'],ry['sequence']
                pair_id=seqsha('|'.join(sorted([rx['sequence_sha256'],ry['sequence_sha256']])))
                if label:
                    gm=alignment_metrics(sx,sy,GLOBAL);lm=alignment_metrics(sx,sy,LOCAL)
                    identity,qcov,tcov,evalue,alnlen=p['identity'],p['query_coverage'],p['target_coverage'],p['evalue'],p['alignment_length']
                else:
                    gm,lm=align_cache[(x,y)];identity=qcov=tcov=evalue=alnlen=''
                output.append({'pair_id':pair_id,'split':split,'block_id':block_by_index[i],'label':label,
                    'accession_a':x,'accession_b':y,'cluster_a':rx['cluster_id'],'cluster_b':ry['cluster_id'],
                    'sequence_sha256_a':rx['sequence_sha256'],'sequence_sha256_b':ry['sequence_sha256'],
                    'sentence1':sx,'sentence2':sy,'length_a':len(sx),'length_b':len(sy),
                    'mmseqs_identity':identity,'mmseqs_query_coverage':qcov,'mmseqs_target_coverage':tcov,'mmseqs_evalue':evalue,'mmseqs_alignment_length':alnlen,
                    'global_identity':gm['identity'],'global_alignment_length':gm['alignment_length'],'global_score':gm['score'],
                    'local_identity':lm['identity'],'local_query_coverage':lm['query_coverage'],'local_target_coverage':lm['target_coverage'],
                    'label_basis':'MMseqs2 >=40% <=90% identity, >=80% both coverage, E<=1e-3' if label else 'different operational 30%-identity component; global identity <=25%; no >=30% local identity with >=80% coverage on both ends'})
    rng.shuffle(output)
    assert output, 'No feasible data; do not publish an empty benchmark'
    for split in ['train','validation','test']:
        splitrows=[r for r in output if r['split']==split]
        write_tsv(root/'data/pairs'/f'protein_sequence_similarity_{split}.tsv.gz',splitrows,list(output[0]))
    write_tsv(root/'data/pairs/protein_sequence_similarity_all.tsv.gz',output,list(output[0]))
    root.joinpath('validation').mkdir(parents=True,exist_ok=True)
    acceptance=validate_and_baselines(root,output,byid)
    dump(root/'metadata/protein_pair_construction.json',{'seed':a.seed,'runtime_seconds':time.time()-started,'rows':len(output),
        'split_label_counts':{s:dict(Counter(str(r['label']) for r in output if r['split']==s)) for s in ['train','validation','test']},
        'block_count':blocks,'filter_counts':dict(counts),'endpoint_rule':'Each endpoint appears exactly once in each label in the same column; positive edges are disjoint; negatives are perfect-match rewiring within exact B-length bins.',
        'length_rule':'Per-A positive/negative B lengths equal; joint (lenA,lenB), absolute length difference, and all marginal length distributions exactly match by label.',
        'negative_alignment':'Biopython PairwiseAligner, BLOSUM62, affine gap open=-11, extend=-1. Global identity counts matches divided by all alignment columns including gaps. Local coverage is coordinate span/full length. Negative means screen-defined sequence dissimilarity, not established evolutionary nonhomology.',
        'task':'Sequence-similarity discrimination, not proof of remote structural transfer. Composition/kmer/alignment signal is expected and reported rather than erased.',
        'sampling':'For degree-preserving training subsets sample whole block_id permutation cycles, retaining both labels. Never claim arbitrary row subsets preserve endpoint degree.'})
    dump(root/'metadata/protein_pair_manifest.json',{'release_id':'2026-09-25-v2','acceptance':acceptance,
        'source_release':'UniProtKB/Swiss-Prot 2026_03','builder_sha256':sha(Path(__file__)),
        'data_files':{p.name:sha(p) for p in (root/'data/pairs').glob('protein_sequence_similarity_*.tsv.gz')},
        'duration_seconds':time.time()-started,'remote_pretraining_exclusion_required':True})
    print('pair construction complete',len(output),'seconds',time.time()-started,'acceptance',acceptance['acceptance_pass'],flush=True)
if __name__=='__main__':main()
