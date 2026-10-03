"""Small independent-auditor adversarial fixtures; no GPU/model/data builds."""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('independent_data_audit_test', Path(__file__).resolve().parents[1]/'audit_data.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class Tokenizer:
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    def encode(self,text,add_special_tokens=False):
        return SimpleNamespace(ids=[self.alphabet.index(c)+4 for c in text])
    def decode(self,ids,skip_special_tokens=False):
        return ''.join(self.alphabet[i-4] for i in ids)


def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj)+'\n')


def write_rows(path,values):
    path.parent.mkdir(parents=True,exist_ok=True)
    opener=gzip.open if str(path).endswith('.gz') else open
    with opener(path,'wt') as f:
        for row in values:f.write(json.dumps(row)+'\n')


def test_complete_group_nearest_prefix_and_tie_longer():
    counts={'a':6,'b':6,'c':6}
    groups, report=a.independently_selected_groups(counts,target=9)
    assert len(groups)==2 and report['selected_rows']==12
    assert a.independently_selected_groups(dict(reversed(list(counts.items()))),target=9)==(groups,report)
    assert a.independently_selected_groups(counts,target=100)[0]==set(counts)
    with pytest.raises(ValueError):a.independently_selected_groups(counts,target=1)
    with pytest.raises(ValueError):a.independently_selected_groups({'a':0})


@pytest.mark.parametrize('name',['../escape','a/../escape','/absolute','a//b','a\\b'])
def test_member_escape_rejected(tmp_path,name):
    with pytest.raises(ValueError):a.inside(tmp_path,name)


def test_readset_changed_bytes_and_symlink_fail(tmp_path):
    p=tmp_path/'file';p.write_bytes(b'old')
    s=a.ReadSet();s.pin(p)
    p.write_bytes(b'new')
    with pytest.raises(ValueError,match='changed'):s.finish()
    link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(ValueError,match='symlink'):a.identity(link)


def stream_fixture(tmp_path,budget=8):
    tok=Tokenizer();texts=['ACDE','FGHI']; index=[]; tokens=[]
    for i,text in enumerate(texts):
        full=tok.encode(text).ids;used=min(len(full),budget-len(tokens)-1)
        row={'parent_id':f'p{i}','source_record_index':i,'text_sha256':a.text_sha(text),
             'content_tokens_full':len(full),'content_tokens_used':used,'offset':len(tokens),
             'length':used+1,'eos_offset':len(tokens)+used,'truncated':used<len(full),
             'source_characters':len(text),'exposed_characters':used}
        index.append(row);tokens+=full[:used]+[1]
    path=tmp_path/'streams';path.mkdir()
    np.asarray(tokens,dtype='<u2').tofile(path/'protein.bin')
    write_rows(path/'protein.index.jsonl',index)
    descriptor={'bin':'streams/protein.bin','index':'streams/protein.index.jsonl','tokens':budget,
        'blocks':budget//512,'dtype':'<u2','records':2,'content_tokens':budget-2,'eos_tokens':2,
        'clipped_records':1,'exposed_characters':budget-2,'ordered_parent_ids_sha256':a.text_sha('["p0","p1"]'),
        'last_source_record_index':1}
    return tok,texts,index,descriptor


def test_stream_all_tokens_index_eos_and_summary(tmp_path):
    tok,texts,_,d=stream_fixture(tmp_path)
    r=a.StreamAudit(tmp_path,'protein',d,tok,budget=8)
    for i,text in enumerate(texts):r.take(f'p{i}',i,text)
    assert r.finish()['tokens']==8 and r.seen=={'p0','p1'} and r.full=={'p0'}


@pytest.mark.parametrize('fault',['token','eos','index_parent','offset','summary','extra_row','budget'])
def test_stream_faults(tmp_path,fault):
    tok,texts,index,d=stream_fixture(tmp_path)
    if fault in ('token','eos'):
        values=np.fromfile(tmp_path/d['bin'],dtype='<u2');values[0 if fault=='token' else -1]=30;values.tofile(tmp_path/d['bin'])
    if fault=='index_parent':index[1]['parent_id']='wrong'
    if fault=='offset':index[1]['offset']+=1
    if fault=='summary':d['content_tokens']+=1
    if fault=='extra_row':index.append(index[-1])
    if fault=='budget':(tmp_path/d['bin']).write_bytes(b'bad')
    write_rows(tmp_path/d['index'],index)
    with pytest.raises(ValueError):
        r=a.StreamAudit(tmp_path,'protein',d,tok,budget=8)
        for i,text in enumerate(texts):r.take(f'p{i}',i,text)
        r.finish()


def schedules(tmp_path):
    result={};(tmp_path/'schedules').mkdir()
    for seed in (0,1,2):
        data=np.empty((2,16,2),dtype=np.int64)
        for stream in (0,1):
            key=int(hashlib.sha256(f'joint-v1:schedule:{seed}:{stream}'.encode()).hexdigest(),16)
            values=np.random.default_rng(key).permutation(16).reshape(2,8)
            data[:,stream*8:(stream+1)*8,0]=stream
            data[:,stream*8:(stream+1)*8,1]=values
        name=f'schedules/seed{seed}.npy';np.save(tmp_path/name,data,allow_pickle=False)
        result[str(seed)]={'file':name,'shape':[2,16,2],'dtype':'int64'}
    return result


def test_schedule_complete_domain_and_order(tmp_path):
    s=schedules(tmp_path)
    assert len(a.audit_schedules(tmp_path,s,blocks=16))==3


@pytest.mark.parametrize('fault',['domain','duplicate','order','dtype','seed'])
def test_schedule_faults(tmp_path,fault):
    s=schedules(tmp_path);path=tmp_path/s['0']['file'];x=np.load(path)
    if fault=='domain':x[0,0,0]=1
    if fault=='duplicate':x[0,0,1]=x[0,1,1]
    if fault=='order':x[0,[0,1],1]=x[0,[1,0],1]
    if fault=='dtype':x=x.astype(np.int32)
    if fault=='seed':del s['2']
    np.save(path,x,allow_pickle=False)
    with pytest.raises(ValueError):a.audit_schedules(tmp_path,s,blocks=16)


def protein_text(n):
    alphabet=Tokenizer.alphabet
    suffix=''
    while n:
        suffix+=alphabet[n%20];n//=20
    return 'ACDEFGHIKLMNPQRSTVWY'*2+suffix+'A'


def source_fixture(tmp_path):
    tok=Tokenizer();rawpaths={};manifest={'source':{},'source_prefix64':{},'smoke_source':{}}
    canonical={};report={'policy':{'target_rows':66,'selection_seed':20260925,'expected_train_rows':66,
        'expected_train_groups':11,'expected_validation_rows':66,'expected_raw_train_rows':66},'roles':{}}
    identities={}
    for split,offset in [('train',0),('validation',1000)]:
        raw=[];selected=[];encoded=[];seqs=set();groups=set();clusters=set()
        for block in range(11):
            aa=[protein_text(offset+block*6+j+1) for j in range(3)]
            bb=[protein_text(offset+block*6+j+4) for j in range(3)]
            for j in range(3):
                for label in (0,1):
                    left,right=aa[j],bb[(j+(1-label))%3]
                    sa,sb=a.text_sha(left),a.text_sha(right);pid=a.text_sha('|'.join(sorted((sa,sb))))
                    meta={'block_id':f'{split}_b{block}','split':split,'sequence_sha256_a':sa,
                        'sequence_sha256_b':sb,'cluster_a':f'{split}_c{block}','cluster_b':f'{split}_c{block}',
                        'accession_a':f'a{offset}_{block}_{j}','accession_b':f'b{offset}_{block}_{j}_{label}'}
                    row={'pair_id':pid,'label':str(label),'sentence1':left,'sentence2':right,**meta};raw.append(row)
                    item={'row_id':pid,'label':label,'sentence1':left,'sentence2':right,'metadata':meta};selected.append(item)
                    ta,tb=tok.encode(left).ids,tok.encode(right).ids
                    encoded.append({'row_index':len(encoded),'row_id':pid,'label':label,'ids_a':ta,'ids_b':tb,
                        'metadata':meta,'raw_lengths':[len(left),len(right)],'precap_token_lengths':[len(ta),len(tb)]})
                    for sha,text in [(sa,left),(sb,right)]:canonical[sha]={'split':split,'cluster_id':f'{split}_c{block}','length':len(text)}
                    seqs.update((sa,sb));groups.add(meta['block_id']);clusters.add(meta['cluster_a'])
        rawpath=tmp_path/(split+'.tsv.gz')
        import csv
        with gzip.open(rawpath,'wt') as f:
            writer=csv.DictWriter(f,fieldnames=list(raw[0]),delimiter='\t');writer.writeheader();writer.writerows(raw)
        rawpaths[split]=rawpath
        write_rows(tmp_path/'source'/f'{split}.jsonl',selected)
        write_rows(tmp_path/'source'/f'{split}.rows.jsonl.gz',encoded)
        ids=np.zeros((66,512),dtype=np.int64);mask=ids.copy();labels=np.empty(66,dtype=np.int64)
        keys=[]
        for i,r in enumerate(encoded):
            joined=r['ids_a']+[2]+r['ids_b']+[1];ids[i,:len(joined)]=joined;mask[i,:len(joined)]=1;labels[i]=r['label'];keys.append(tuple(joined))
        np.savez_compressed(tmp_path/'source'/f'{split}.npz',input_ids=ids,attention_mask=mask,labels=labels)
        manifest['source'][split]={'npz':f'source/{split}.npz','rows':f'source/{split}.rows.jsonl.gz','labels':{'0':33,'1':33},
            'truncated_endpoints_by_label':{'0':0,'1':0},'same_label_encoded_pair_collisions':0,'conflicting_encoded_pair_labels':0,
            'identity_counts':{'row_ids':66,'groups':11,'sequences':66,'clusters':11,'inputs':66},'count':66,'shape':[66,512],
            'dtype':'int64','endpoint_role_label_balance':True,'within_block_role_label_balance':True}
        rowids=[r['row_id'] for r in selected];prefix={'indices':list(range(64)),'row_ids':rowids[:64],'rows':64,
            'row_ids_sha256':a.text_sha(json.dumps(rowids[:64],separators=(',',':'))),
            'purpose':'bounded technical smoke only; not full-group scientific training subset'}
        manifest['source_prefix64'][split]=prefix;manifest['smoke_source'][split]=copy.deepcopy(prefix)
        report['roles'][split]={'raw_rows':66,'raw_groups':11,'selected_rows':66,'selected_groups':11,
            'ordered_row_ids_sha256':a.text_sha(json.dumps(rowids,separators=(',',':')))}
        identities[split]={'row_ids':set(rowids),'groups':groups,'sequences':seqs,'clusters':clusters}
    report['cross_split_overlap']={k:len(identities['train'][k]&identities['validation'][k]) for k in identities['train']}
    manifest['source_selection']=report;write_json(tmp_path/'source_selection.json',report)
    return manifest,rawpaths,tok,canonical,report['policy']


def test_source_independent_rows_encoding_and_role_balance(tmp_path):
    m,raw,tok,c,p=source_fixture(tmp_path)
    details,overlap,prefix=a.audit_source(tmp_path,m,raw,tok,c,p)
    assert details['train']['rows']==66 and not any(overlap.values()) and len(prefix['train']['row_ids'])==64


@pytest.mark.parametrize('fault',['label','padding','eos','mask','row_order','prefix','canonical_split'])
def test_source_faults(tmp_path,fault):
    m,raw,tok,c,p=source_fixture(tmp_path)
    path=tmp_path/'source/train.npz'
    if fault in ('label','padding','eos','mask'):
        with np.load(path) as f:x={k:f[k] for k in f.files}
        if fault=='label':x['labels'][0]=1-x['labels'][0]
        if fault=='padding':x['input_ids'][0,-1]=5
        if fault=='eos':x['input_ids'][0,x['attention_mask'][0].sum()-1]=2
        if fault=='mask':x['attention_mask'][0,-1]=1
        np.savez_compressed(path,**x)
    if fault=='row_order':
        path=tmp_path/'source/train.jsonl';rows=list(a.json_rows(path));rows[0],rows[1]=rows[1],rows[0];write_rows(path,rows)
    if fault=='prefix':m['smoke_source']['train']['row_ids'][0]='wrong'
    if fault=='canonical_split':next(iter(c.values()))['split']='validation'
    with pytest.raises(ValueError):a.audit_source(tmp_path,m,raw,tok,c,p)


def test_duplicate_json_key_rejected():
    with pytest.raises(ValueError):a.loads('{"a":1,"a":2}')
    with pytest.raises(ValueError):a.loads('{"a":NaN}')


def test_output_inventory_is_closed_source_only():
    outputs=a.expected_output_names()
    assert len(outputs)==24 and not any('target' in n or 'source_test' in n for n in outputs)
