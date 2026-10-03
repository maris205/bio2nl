"""Locally reconstruct the previously scored QQP cohort from fixed raw bytes and indices."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from fetch import verify, sha

HERE = Path(__file__).resolve().parent

def norm(text):
    return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', text).casefold())) if isinstance(text,str) else ''

def update(d, row):
    d.update((json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n').encode())

def rebuild(raw, membership, tokenizer_path, output):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    if output.exists(): raise FileExistsError('Output exists')
    verify(raw, membership['raw'])
    if sha(tokenizer_path) != membership['tokenizer_sha256']: raise ValueError('Tokenizer identity changed')
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    rows=sorted(pq.read_table(raw).to_pylist(),key=lambda x:x['idx'])
    if len(rows)!=membership['raw_rows'] or [r['idx'] for r in rows]!=list(range(membership['raw_rows'])):
        raise ValueError('Original QQP indices differ')
    excluded=membership['excluded_upstream_idx']
    if excluded!=sorted(set(excluded)) or any(type(i) is not int or i<0 or i>=len(rows) for i in excluded):
        raise ValueError('Invalid membership projection')
    parents={}
    def find(a):
        parents.setdefault(a,a)
        while a!=parents[a]:
            parents[a]=parents[parents[a]];a=parents[a]
        return a
    for r in rows:
        if set(r)!={'idx','question1','question2','label'} or type(r['label']) is not int or r['label'] not in (0,1):
            raise ValueError('Raw QQP schema changed')
        a,b=norm(r['question1']),norm(r['question2'])
        if a and b:
            x,y=find(a),find(b);parents[max(x,y)]=min(x,y)
    text_digest, encoded_digest=hashlib.sha256(),hashlib.sha256()
    result=[];groups=set();labels=Counter();excluded=set(excluded)
    for r in rows:
        if r['idx'] in excluded:continue
        a,b=norm(r['question1']),norm(r['question2'])
        if not a or not b:raise ValueError('Retained empty endpoint')
        cid=hashlib.sha256(find(a).encode()).hexdigest()
        row={'id':f"qqp_validation:{r['idx']}",'upstream_idx':r['idx'],'label':r['label'],
             'text_a':r['question1'],'text_b':r['question2'],
             'endpoint_a_sha256':hashlib.sha256(a.encode()).hexdigest(),
             'endpoint_b_sha256':hashlib.sha256(b.encode()).hexdigest(),'component_id':cid}
        update(text_digest,row)
        content=[]
        for text in (r['question1'],r['question2']):
            ids=tokenizer.encode(text,add_special_tokens=False).ids
            if not 0<len(ids)<=255 or not all(4<=i<32000 for i in ids) or tokenizer.decode(ids,skip_special_tokens=False)!=text:
                raise ValueError('Target encoding mismatch')
            content.append(ids)
        update(encoded_digest,{'id':row['id'],'label':row['label'],'component_id':cid,'ids_a':content[0],'ids_b':content[1]})
        result.append(row);groups.add(cid);labels[str(r['label'])]+=1
    if len(result)!=membership['retained_rows'] or dict(labels)!=membership['expected_labels'] or len(groups)!=membership['expected_components']:
        raise ValueError('Target cohort counts changed')
    if text_digest.hexdigest()!=membership['expected_content_sha256'] or encoded_digest.hexdigest()!=membership['expected_encoded_content_sha256']:
        raise ValueError('Target scientific content differs from published study')
    output.mkdir(parents=True)
    # The text output must remain local; it is never part of an upload allowlist.
    with (output/'qqp_local_only.jsonl').open('x') as f:
        for row in result:f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n')
    report={'status':'fixed_cohort_reconstructed_exactly','rows':len(result),'labels':dict(labels),'components':len(groups),
            'content_sha256':text_digest.hexdigest(),'encoded_content_sha256':encoded_digest.hexdigest(),
            'qualification_algorithm_rerun':False,'historically_model_scored':True,'new_model_scoring':False,
            'membership_projection_canonical_sha256':hashlib.sha256(json.dumps(membership,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'text_redistribution_cleared':False}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(rebuild(a.raw,json.loads((HERE/'qqp_membership.json').read_text()),a.tokenizer,a.output)))

if __name__=='__main__':main()
