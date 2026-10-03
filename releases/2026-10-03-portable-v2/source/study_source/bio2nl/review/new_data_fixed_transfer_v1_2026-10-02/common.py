"""Frozen new-source identities and inference-only gates for exploratory fixed transfer."""
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[2]
M2=HERE.parent/'new_data_full_training_v1_2026-10-02'
M2_RESULTS=M2/'results'
ARCHIVE=WORKSPACE/'release_staging/m1_m3_2026-09-30_v1'
M1=ARCHIVE/'bio2nl/review/confirmation_data_v1_2026-09-29'
NEW_RAW=WORKSPACE/'data_rebuild/2026-10-01-portable-v2'
NEW_PREPARED=HERE.parent/'new_data_training_smoke_v1_2026-10-01/prepared'
CONDITIONS=('EP','ES','EE')
ROLES=('source_test','target')
COUNTS={'source_test':20854,'target':39893}


def now():return datetime.now(timezone.utc).isoformat()
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def path_inside(root,name):
    require(not Path(name).is_absolute(),'Expected relative path')
    root=Path(root).resolve();p=(root/name).resolve();require(p.is_relative_to(root),'Artifact escapes namespace');return p
def check_hashes(root,bindings):
    for name,value in bindings.items():
        path=Path(name) if Path(name).is_absolute() else path_inside(root,name)
        expected=value.get('sha256') if isinstance(value,dict) else value
        require(sha(path)==expected,'Changed pinned artifact: '+str(path))
        if isinstance(value,dict) and 'bytes' in value:
            require(path.stat().st_size==value['bytes'],'Changed pinned size: '+str(path))

def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def check_source_gate(root=HERE):
    root=Path(root);barrier=read(root/'source_barrier.json');protocol=read(root/'protocol.json')
    require(barrier['status']=='frozen_sources_for_heldout_evaluation','Source barrier not accepted')
    require(protocol['source_barrier_sha256']==sha(root/'source_barrier.json'),'Protocol/source barrier mismatch')
    check_hashes(root,barrier['files'])
    selection_path=Path(protocol['source_selection']['file'])
    require(selection_path==M2_RESULTS/'source_selection.json','Unexpected source selection root')
    selection=read(selection_path)
    require(sha(selection_path)==protocol['source_selection']['sha256'],'Source selection changed')
    require(selection['status']=='frozen' and selection['conditions_filtered_by_score'] is False,'Unfrozen/filtered source selection')
    expected={(c,p,f) for c in CONDITIONS for p in range(3) for f in range(3)}
    jobs=selection['selected'];actual={(x['job']['condition'],x['job']['pt_seed'],x['job']['ft_seed']) for x in jobs}
    require(len(jobs)==27 and actual==expected and all(not x['job']['smoke'] for x in jobs),'Incomplete source classifier grid')
    require(canonical({k:v for k,v in selection.items() if k!='selection_sha256'})==selection['selection_sha256'],'Canonical source selection differs')
    require(selection['target_scoring_started'] is False and selection['target_scoring_enabled'] is False and selection['source_test_scoring_enabled'] is False,'Source barrier historical flags changed')
    for entry in jobs:
        for key in ('checkpoint','source_result'):
            spec=entry[key];path=path_inside(M2_RESULTS,spec['file'])
            require(str(path) in barrier['files'] and barrier['files'][str(path)]==spec['sha256'],'Source artifact not closed by barrier')
    queue=read(M2/'execution_status.json')
    require(queue['status']=='completed' and len(queue['jobs'])==45 and all(x['status']=='completed' and x['exit_code']==0 for x in queue['jobs']),'Source queue incomplete')
    source_audit=read(path_inside(M2_RESULTS,selection['source_audit']['file']))
    require(source_audit['status']=='passed' and source_audit['source_fits']==27 and source_audit['pretraining_runs']==9 and source_audit['prediction_artifacts_audited']==270 and source_audit['complete_checkpoints_audited']==36,'Source report gate failed')
    completion=read(barrier['completion_review']['file'])
    require(completion['status']=='passed','Fresh source completion review failed')
    target=read(M1/'confirmation_acceptance.json')
    require(target['status']=='accepted_for_local_confirmation' and target['confirmation_data_ready'] and target['retained_rows']==39893,'Fixed historical QQP metadata gate failed')
    require(protocol['training_enabled'] is False and protocol['calibration_enabled'] is False and protocol['target_selection_enabled'] is False and protocol['condition_filtering'] is False,'Training/calibration/filtering forbidden')
    require(protocol['source_test_scoring_enabled'] is True and protocol['target_scoring_enabled'] is True,'Current inference authorization missing')
    require(protocol['conditions']==list(CONDITIONS) and protocol['pretraining_seeds']==[0,1,2] and protocol['fine_tuning_seeds']==[0,1,2],'Fixed grid changed')
    require({k:v['rows'] for k,v in protocol['roles'].items()}==COUNTS,'Cohort contract changed')
    require(protocol['source_protocol']['file']==str(M2/'protocol.json') and protocol['source_protocol']['sha256']==selection['protocol_sha256'],'Source protocol identity changed')
    require(protocol['model_loader']['file']==str(M2/'code_snapshot/model.py'),'Unexpected classifier loader')
    require(read(NEW_RAW/'portable_validation/acceptance.json')['status']=='accepted_new_raw_release_training_disabled','New raw gate changed')
    require(read(NEW_PREPARED.parent/'data_acceptance.json')['status']=='source_only_smoke_inputs_accepted','Prior derived gate changed')
    return protocol


def check_gate(root=HERE,mode='full'):
    root=Path(root);require(mode in ('smoke','full'),'Unknown gate mode')
    protocol=check_source_gate(root)
    freeze=read(root/'execution_freeze.json');require(freeze['status']=='frozen_for_fixed_scoring','Scoring execution not frozen')
    check_hashes(root,freeze['files'])
    environment=read(root/'environment.json');require(environment['python']==sys.version,'Python changed')
    for name in ('torch','transformers','numpy','scipy','scikit-learn','tokenizers'):
        require(importlib.metadata.version(name)==environment[name],'Runtime changed: '+name)
    manifest=read(root/'prepared/manifest.json');check_hashes(root,manifest['inputs']);check_hashes(root,manifest['outputs'])
    audit=read(root/'verification/data_audit.json')
    require(audit['status']=='passed' and audit['prepared_manifest_sha256']==sha(root/'prepared/manifest.json'),'Data audit failed or stale')
    if mode=='full':
        accepted=read(root/'scoring_acceptance.json')
        require(accepted['status']=='accepted_for_fixed_scoring' and accepted['execution_freeze_sha256']==sha(root/'execution_freeze.json'),'Technical scoring acceptance failed')
        check_hashes(root,accepted['files'])
    return protocol


def gate_file_hashes(root=HERE,mode='full'):
    """Collect every static gate input for callers to rehash after inference."""
    root=Path(root).resolve();require(mode in ('smoke','full'),'Unknown gate mode')
    result={}
    def add(path,digest=None):
        path=Path(path);path=path if path.is_absolute() else path_inside(root,path)
        value=digest.get('sha256') if isinstance(digest,dict) else digest
        value=sha(path) if value is None else value
        previous=result.get(str(path));require(previous is None or previous==value,'Conflicting closure identities')
        result[str(path)]=value
    for name in ('source_barrier.json','protocol.json','execution_freeze.json','environment.json','prepared/manifest.json','verification/data_audit.json'):
        add(name)
    for key in ('source_barrier.json','execution_freeze.json'):
        for path,digest in read(root/key)['files'].items():add(path,digest)
    manifest=read(root/'prepared/manifest.json')
    for key in ('inputs','outputs'):
        for path,digest in manifest[key].items():add(path,digest)
    for path,digest in read(root/'verification/data_audit.json')['bindings'].items():add(path,digest)
    if mode=='full':
        add('scoring_acceptance.json')
        for path,digest in read(root/'scoring_acceptance.json')['files'].items():add(path,digest)
    check_hashes(root,result)
    return result


def load_role(root,role,manifest=None):
    root=Path(root);require(role in ROLES,'Unplanned scoring role')
    manifest=manifest or read(root/'prepared/manifest.json');item=manifest['roles'][role]
    require(item['count']==COUNTS[role],'Held-out cohort size changed')
    for field in ('npz','rows'):
        require(item[field].startswith('prepared/'),'Role artifact outside preparation')
        require(sha(path_inside(root,item[field]))==manifest['outputs'][item[field]],'Changed held-out input')
    with np.load(path_inside(root,item['npz']),allow_pickle=False) as stored:arrays={k:stored[k] for k in stored.files}
    with gzip.open(path_inside(root,item['rows']),'rt') as stream:rows=[json.loads(line) for line in stream]
    n=item['count'];require(len(rows)==n and len({r['row_id'] for r in rows})==n,'Cohort row identity differs')
    require(set(arrays)=={'input_ids','attention_mask','labels'} and all(x.dtype==np.int64 for x in arrays.values()),'Role tensor fields/dtypes differ')
    ids,mask,y=arrays['input_ids'],arrays['attention_mask'],arrays['labels']
    require(ids.shape==mask.shape==(n,512) and y.shape==(n,),'Role tensor shapes differ')
    require(np.array_equal(y,[r['label'] for r in rows]) and np.isin(y,[0,1]).all(),'Role labels differ')
    for i,row in enumerate(rows):
        a,b=row['ids_a'],row['ids_b'];require(0<len(a)<=255 and 0<len(b)<=255,'Endpoint cap differs')
        require(all(type(v)is int and 4<=v<32000 for v in a+b),'Special/unknown content ID')
        joined=a+[2]+b+[1];require(np.array_equal(ids[i,:len(joined)],joined) and not ids[i,len(joined):].any(),'Role encoding differs')
        require(np.all(mask[i,:len(joined)]==1) and not mask[i,len(joined):].any(),'Role padding mask differs')
        require(isinstance(row['group_id'],str) and bool(row['group_id']),'Missing group identity')
    return arrays,rows


def metrics(labels,logp):
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,matthews_corrcoef,roc_auc_score
    labels=np.asarray(labels);logp=np.asarray(logp,dtype=np.float64)
    require(logp.shape==(len(labels),2) and np.isfinite(logp).all(),'Invalid finite log probabilities')
    require(np.allclose(np.logaddexp(logp[:,0],logp[:,1]),0,atol=2e-6,rtol=0),'Unnormalized log probabilities')
    pred=logp.argmax(1)
    return dict(rows=len(labels),accuracy=float(accuracy_score(labels,pred)),balanced_accuracy=float(balanced_accuracy_score(labels,pred)),
        mcc=float(matthews_corrcoef(labels,pred)),auroc=float(roc_auc_score(labels,logp[:,1]-logp[:,0])) if len(set(labels))==2 else None,
        cross_entropy=float(-logp[np.arange(len(labels)),labels].mean()),predicted_positive_fraction=float(pred.mean()),
        confusion_matrix=confusion_matrix(labels,pred,labels=[0,1]).tolist())


def save_predictions(path,labels,logp,row_ids,group_ids):
    path=Path(path);require(not path.exists(),'Refuse to overwrite predictions');metrics(labels,logp)
    require(len(labels)==len(row_ids)==len(group_ids),'Prediction identity lengths differ')
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.npz.partial')
    with temp.open('xb') as stream:
        np.savez_compressed(stream,labels=np.asarray(labels,dtype=np.int64),log_probabilities=np.asarray(logp,dtype=np.float64),
            predictions=np.asarray(logp).argmax(1),row_ids=np.asarray(row_ids,dtype=str),group_ids=np.asarray(group_ids,dtype=str))
    temp.replace(path)
