"""Synthetic tests for heldout integrity; these never read benchmark examples."""
import copy
import csv
import gzip
import hashlib
import json

import numpy as np
import pytest

import audit_data
import prepare_data


class Encoding:
    def __init__(self, ids):
        self.ids = ids


class TinyTokenizer:
    def __init__(self, ids=None, decoded='abc'):
        self.ids = [4, 5, 6] if ids is None else ids
        self.decoded = decoded

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return Encoding(self.ids)

    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens is False
        return self.decoded


def test_content_roundtrip():
    assert prepare_data.encode_content(TinyTokenizer(), 'abc') == [4, 5, 6]


@pytest.mark.parametrize('ids', [[3], [0], [1], [2], [32000], []])
def test_special_unknown_or_empty_ids_rejected(ids):
    with pytest.raises(ValueError):
        prepare_data.encode_content(TinyTokenizer(ids=ids), 'abc')


def test_roundtrip_failure_rejected():
    with pytest.raises(ValueError, match='roundtrip'):
        prepare_data.encode_content(TinyTokenizer(decoded='different'), 'abc')


def test_independent_caps_preserve_second_endpoint_and_eos():
    a, b = [4] * 300, [5] * 301
    ids, mask = prepare_data.pack(a, b)
    assert ids.shape == mask.shape == (512,)
    assert ids[255] == 2 and ids[-1] == 1
    assert np.all(ids[:255] == 4) and np.all(ids[256:-1] == 5)
    assert np.all(mask == 1)
    audit_data.compare_encoded(ids, mask, np.int64(1), a, b, 1)


def test_short_pair_right_padding_and_eos():
    ids, mask = prepare_data.pack([7], [8, 9])
    assert ids[:5].tolist() == [7, 2, 8, 9, 1]
    assert mask.sum() == 5 and np.all(ids[5:] == 0)
    audit_data.compare_encoded(ids, mask, np.int64(0), [7], [8, 9], 0)


@pytest.mark.parametrize('value', [True, False, 2, -1, '1', 1.0])
def test_label_coercion_forbidden(value):
    with pytest.raises(ValueError):
        prepare_data.validate_label(value)


def test_independent_auditor_detects_changed_mask():
    ids, mask = prepare_data.pack([4], [5])
    mask[-1] = 1
    with pytest.raises(ValueError, match='mask'):
        audit_data.compare_encoded(ids, mask, np.int64(0), [4], [5], 0)


def test_builder_checks_barrier_before_prepared_or_target_access(monkeypatch):
    def stop(root):
        raise ValueError('barrier missing')
    monkeypatch.setattr(prepare_data, 'check_source_gate', stop)
    monkeypatch.setattr(prepare_data, 'read', lambda path: pytest.fail('Read attempted before source gate'))
    monkeypatch.setattr(prepare_data, 'sha', lambda path: pytest.fail('Hash attempted before source gate'))
    with pytest.raises(ValueError, match='barrier missing'):
        prepare_data.build()


def test_auditor_checks_barrier_before_manifest_access(monkeypatch):
    def stop(root):
        raise ValueError('barrier missing')
    monkeypatch.setattr(audit_data, 'check_source_gate', stop)
    monkeypatch.setattr(audit_data, 'read', lambda path: pytest.fail('Read attempted before source gate'))
    with pytest.raises(ValueError, match='barrier missing'):
        audit_data.audit()


def target_fixture():
    raw = [dict(idx=2, question1='Beta', question2='GAMMA!', label=0),
           dict(idx=1, question1='Ⓐlpha', question2='Beta', label=1),
           dict(idx=3, question1='GAMMA!', question2='Delta', label=0),
           dict(idx=4, question1='', question2='Invalid', label=0)]
    group = hashlib.sha256(b'alpha').hexdigest()
    ledger = [dict(upstream_idx=i, component_id=group if i < 4 else None, retained=i in (1,2),
                   reasons=[] if i in (1,2) else ['fixed_historical_exclusion']) for i in (1,2,3,4)]
    expected = []
    for row in sorted(raw, key=lambda x:x['idx'])[:2]:
        a, b = prepare_data.normalized(row['question1']), prepare_data.normalized(row['question2'])
        expected.append(dict(id='qqp_validation:'+str(row['idx']), upstream_idx=row['idx'], label=row['label'],
                             text_a=row['question1'], text_b=row['question2'], endpoint_a_sha256=hashlib.sha256(a.encode()).hexdigest(),
                             endpoint_b_sha256=hashlib.sha256(b.encode()).hexdigest(), component_id=group))
    digest = hashlib.sha256(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in expected).encode()).hexdigest()
    return raw, ledger, expected, digest


@pytest.mark.parametrize('function',[prepare_data.reconstruct_target,audit_data.reconstruct_target_independent])
def test_fixed_raw_reconstruction_uses_excluded_rows_for_components(function):
    raw, ledger, expected, digest = target_fixture()
    assert function(raw,ledger,digest,4) == expected
    # Delta (excluded) belongs to the raw graph and must not change retained groups.
    assert ledger[2]['component_id'] == expected[0]['component_id']


@pytest.mark.parametrize('function',[prepare_data.reconstruct_target,audit_data.reconstruct_target_independent])
@pytest.mark.parametrize('fault',['member','component','order','label','text','duplicate','schema','digest'])
def test_fixed_target_tampering_rejected(function,fault):
    raw, ledger, _, digest = target_fixture()
    if fault=='member': ledger[2].update(retained=True,reasons=[])
    elif fault=='component': ledger[0]['component_id']='wrong'
    elif fault=='order': ledger.reverse()
    elif fault=='label': raw[0]['label']=1
    elif fault=='text': raw[0]['question2']='changed'
    elif fault=='duplicate': raw[0]['idx']=1
    elif fault=='schema': raw[0]['label']=True
    else: digest='0'*64
    with pytest.raises(ValueError): function(raw,ledger,digest,4)


class CharTokenizer:
    def encode(self,text,add_special_tokens=False):
        return Encoding([ord(c)+4 for c in text])
    def encode_batch(self,texts,add_special_tokens=False):
        return [self.encode(x,add_special_tokens) for x in texts]
    def decode(self,ids,skip_special_tokens=False):
        return ''.join(chr(x-4) for x in ids)


def source_fixture(tmp_path):
    path=tmp_path/'new_test.tsv.gz'
    rows=[]
    for i,(a,b,label) in enumerate([('ACD','EFG',0),('ACD','HIK',1),('LMN','EFG',1),('LMN','HIK',0)]):
        rows.append(dict(pair_id='new:'+str(i),label=str(label),sentence1=a,sentence2=b,split='test',block_id='block',
                         sequence_sha256_a=hashlib.sha256(a.encode()).hexdigest(),sequence_sha256_b=hashlib.sha256(b.encode()).hexdigest(),
                         cluster_a='cluster:'+a,cluster_b='cluster:'+b))
    with gzip.open(path,'wt',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    return path


def miniature_contract(monkeypatch):
    monkeypatch.setattr(prepare_data,'EXPECTED_COUNTS',{'source_test':4,'target':2})
    monkeypatch.setattr(audit_data,'COUNTS',{'source_test':4,'target':2})
    for module in (prepare_data,audit_data):
        monkeypatch.setattr(module,'EXPECTED_LABELS',{'source_test':{'0':2,'1':2},'target':{'0':1,'1':1}})
        monkeypatch.setattr(module,'TARGET_GROUPS',1)


def test_full_miniature_source_reencoding_and_metadata(tmp_path,monkeypatch):
    miniature_contract(monkeypatch)
    path=source_fixture(tmp_path)
    raw=prepare_data.source_rows(path)
    independent=audit_data.reconstruct_source(path)
    assert raw==independent
    folder=tmp_path/'prepared';folder.mkdir()
    entry=prepare_data.encode_role('source_test',raw,CharTokenizer(),folder)
    findings,rows=audit_data.verify_role(tmp_path,'source_test',entry,CharTokenizer(),independent)
    assert findings['count']==4 and findings['groups']==1
    assert entry['npz']=='prepared/source_test.npz' and entry['rows'].startswith('prepared/')
    assert len(audit_data.identities(rows)['sequences'])==4
    changed=copy.deepcopy(rows);changed[0]['metadata']['cluster_a']='leak'
    assert 'leak' in audit_data.identities(changed)['clusters']


@pytest.mark.parametrize('fault',['label','mask','row_order','metadata','tokens'])
def test_miniature_source_audit_detects_tampering(tmp_path,monkeypatch,fault):
    miniature_contract(monkeypatch);path=source_fixture(tmp_path);raw=prepare_data.source_rows(path)
    folder=tmp_path/'prepared';folder.mkdir();entry=prepare_data.encode_role('source_test',raw,CharTokenizer(),folder)
    if fault in ('label','mask','tokens'):
        with np.load(tmp_path/entry['npz']) as archive: arrays={k:archive[k] for k in archive.files}
        if fault=='label':arrays['labels'][0]=1
        if fault=='mask':arrays['attention_mask'][0,-1]=1
        if fault=='tokens':arrays['input_ids'][0,0]+=1
        np.savez_compressed(tmp_path/entry['npz'],**arrays)
    else:
        rows=list(prepare_data.records(tmp_path/entry['rows']))
        if fault=='row_order':rows.reverse()
        else: rows[0]['metadata']['cluster_a']='changed'
        with gzip.open(tmp_path/entry['rows'],'wt') as stream:
            for row in rows:stream.write(json.dumps(row)+'\n')
    with pytest.raises(ValueError):audit_data.verify_role(tmp_path,'source_test',entry,CharTokenizer(),raw)


def test_full_miniature_target_raw_to_fresh_encoding(tmp_path,monkeypatch):
    miniature_contract(monkeypatch)
    raw,ledger,expected,digest=target_fixture()
    rebuilt=prepare_data.reconstruct_target(raw,ledger,digest,4)
    independent=audit_data.reconstruct_target_independent(raw,ledger,digest,4)
    folder=tmp_path/'prepared';folder.mkdir();entry=prepare_data.encode_role('target',rebuilt,CharTokenizer(),folder)
    token_digest=hashlib.sha256()
    for row in expected:
        item={k:row[k] for k in ('id','label','component_id')}
        item.update(ids_a=[ord(c)+4 for c in row['text_a']],ids_b=[ord(c)+4 for c in row['text_b']])
        token_digest.update((json.dumps(item,ensure_ascii=False,sort_keys=True)+'\n').encode())
    findings,_=audit_data.verify_role(tmp_path,'target',entry,CharTokenizer(),independent,token_digest.hexdigest())
    assert findings['count']==2 and entry['previously_observed_evaluation_split'] is True


def test_actual_preparation_refuses_truncation(tmp_path,monkeypatch):
    miniature_contract(monkeypatch);path=source_fixture(tmp_path);raw=prepare_data.source_rows(path)
    for row in raw:
        for side,field in [('a','sentence1'),('b','sentence2')]:
            row[field]=row[field]*90
            row['metadata']['sequence_sha256_'+side]=hashlib.sha256(row[field].encode()).hexdigest()
    folder=tmp_path/'prepared';folder.mkdir()
    with pytest.raises(ValueError,match='truncation'):prepare_data.encode_role('source_test',raw,CharTokenizer(),folder)


def test_auditor_implementation_is_independent():
    import inspect
    source=inspect.getsource(audit_data)
    assert 'import prepare_data' not in source and 'from prepare_data' not in source
    assert 'prior_encoded_input' not in source
