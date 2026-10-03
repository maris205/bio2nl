"""Synthetic-only identity, metric, replay and complete54+6 reporter checks."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent

def module(name):
    spec=importlib.util.spec_from_file_location('_transfer_test_'+name,HERE/(name+'.py'))
    value=importlib.util.module_from_spec(spec);sys.modules[spec.name]=value;spec.loader.exec_module(value);return value

audit=module('audit_results')
smoke=module('audit_smoke')
common_actual=module('common')


def test_fixed_auc_direction_and_ties():
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    scores = np.array([0., .5, .5, 1.])
    decisions = np.array([0, 0, 1, 1], dtype=np.int64)
    actual = audit.independent_metrics(labels, decisions, scores)
    assert actual['auroc'] == .875
    assert audit.independent_metrics(labels, decisions, -scores)['auroc'] == .125
    assert audit.independent_metrics(labels, decisions, np.zeros(4))['auroc'] == .5

def test_argmax_tie_selects_zero_and_normalization_is_checked():
    labels = np.array([0, 1], dtype=np.int64)
    logp = np.full((2, 2), -np.log(2), dtype=np.float64)
    assert audit.independent_metrics(labels, np.zeros(2, dtype=np.int64), np.zeros(2), logp)['cross_entropy'] == np.log(2)
    with pytest.raises(ValueError, match='ties'):
        audit.independent_metrics(labels, np.ones(2, dtype=np.int64), np.zeros(2), logp)
    with pytest.raises(ValueError, match='normalized'):
        audit.independent_metrics(labels, np.zeros(2, dtype=np.int64), np.zeros(2), logp + 1)

def test_deterministic_constant_cross_entropy_is_null_not_epsilon():
    labels = np.array([0, 0, 0, 1], dtype=np.int64)
    for label, accuracy in ((0, .75), (1, .25)):
        metrics = audit.independent_metrics(labels, np.full(4, label, dtype=np.int64), np.zeros(4))
        assert metrics['accuracy'] == accuracy
        assert metrics['balanced_accuracy'] == metrics['auroc'] == .5
        assert metrics['mcc'] == 0
        assert metrics['cross_entropy'] is None
        corrupt = dict(metrics, cross_entropy=1e-12)
        with pytest.raises(ValueError, match='null'):
            audit.compare_metrics(corrupt, metrics)

def test_component_and_row_identity_reject_reordering(tmp_path):
    labels = np.array([0, 1], dtype=np.int64)
    rows = [{'row_id': 'a', 'group_id': 'component1'}, {'row_id': 'b', 'group_id': 'component2'}]
    logp = np.log(np.array([[.9, .1], [.1, .9]], dtype=np.float64))
    predictions = logp.argmax(1)
    metrics = audit.independent_metrics(labels, predictions, logp[:, 1] - logp[:, 0], logp)
    path = tmp_path / 'pred.npz'
    values = dict(labels=labels, log_probabilities=logp, predictions=predictions,
                  row_ids=np.array(['a', 'b']), group_ids=np.array(['component1', 'component2']))
    np.savez(path, **values)
    assert audit.audit_prediction(path, labels, rows, metrics)[0]['accuracy'] == 1
    for key in ('row_ids', 'group_ids'):
        changed = dict(values, **{key: values[key][::-1]})
        np.savez(path, **changed)
        with pytest.raises(ValueError, match='identity'):
            audit.audit_prediction(path, labels, rows, metrics)

def test_constant_artifact_decision_is_separate_from_tied_score(tmp_path):
    labels = np.array([0, 1], dtype=np.int64)
    rows = [{'row_id': str(i), 'group_id': 'g'} for i in range(2)]
    decisions = np.ones(2, dtype=np.int64)
    actual = audit.independent_metrics(labels, decisions, np.zeros(2))
    path = tmp_path / 'constant1.npz'
    np.savez(path, labels=labels, predictions=decisions, scores=np.zeros(2), row_ids=np.array(['0', '1']), group_ids=np.array(['g', 'g']))
    assert audit.audit_prediction(path, labels, rows, actual, constant_label=1)[0]['predicted_positive_fraction'] == 1
    with pytest.raises(ValueError, match='decisions'):
        audit.audit_prediction(path, labels, rows, actual, constant_label=0)

def test_surface_counter_features_handle_multiplicity_and_empty_bigrams():
    vector = audit.independent_surface_vector([4, 4, 5], [4, 5, 5])
    assert vector[3] == 1
    assert vector[4] == 2 / 3
    assert vector[5] == .8
    assert vector[6] == 1 / 3
    assert vector[7] == vector[8] == .5
    assert audit.independent_surface_vector([4], [5])[6:] == [1., 1., 1.]
    assert audit.independent_surface_vector([4], [4, 5])[6:] == [0., 0., 0.]
    np.testing.assert_array_equal(vector, audit.independent_surface_vector([4, 5, 5], [4, 4, 5]))

def fake_cells():
    cells = []
    for c, offset in (('EP', .62), ('ES', .61), ('EE', .60)):
        for p in range(3):
            for f in range(3):
                for role in ('source_test', 'target'):
                    cells.append({'condition': c, 'pt_seed': p, 'ft_seed': f, 'role': role,
                                  'metrics': {m: offset + .01 * p + .001 * f for m in audit.METRICS}})
    return cells

def test_nested_summary_uses_three_pretraining_means_not_nine_independent_seeds():
    rows = audit.nested_summary(fake_cells())
    selected = next(x for x in rows if x['condition'] == 'EP' and x['role'] == 'target' and x['metric'] == 'auroc')
    assert selected['pretraining_seeds'] == selected['fine_tuning_seeds_per_pretraining_seed'] == 3
    assert selected['mean'] == pytest.approx(.631)
    assert selected['pretraining_sample_sd'] == pytest.approx(.01)
    assert selected['pt0_ft_sample_sd'] == pytest.approx(.001)
    contrast = next(x for x in audit.paired_contrasts(fake_cells()) if x['contrast'] == 'EP-ES' and x['role'] == 'target' and x['metric'] == 'auroc')
    assert contrast['mean_difference'] == pytest.approx(.01)
    assert contrast['pretraining_sample_sd'] == pytest.approx(0, abs=1e-15)
    assert contrast['all_three_pt_differences_positive'] is True

def test_no_missing_or_duplicated_arm_in_summary():
    cells = fake_cells()
    with pytest.raises(ValueError, match='grid'):
        audit.nested_summary(cells[:-1])
    cells[-1] = cells[0]
    with pytest.raises(ValueError, match='grid'):
        audit.paired_contrasts(cells)

def test_direction_consistency_requires_each_pretraining_seed():
    cells = fake_cells()
    for row in cells:
        if row['condition'] == 'EP' and row['pt_seed'] == 2:
            row['metrics']['auroc'] -= .03
    result = next(x for x in audit.paired_contrasts(cells) if x['contrast'] == 'EP-ES' and x['role'] == 'target' and x['metric'] == 'auroc')
    assert result['all_three_pt_differences_positive'] is False
    assert result['pt2_difference'] == pytest.approx(-.02)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,allow_nan=False))
    return path


def report_fixture(tmp_path):
    root=tmp_path/'transfer';source=tmp_path/'source/results';root.mkdir(parents=True);source.mkdir(parents=True)
    role_rows={role:[{'row_id':role+str(i),'group_id':'component'+str(i//2),
                    'ids_a':[4,4+i,6],'ids_b':[4,5+i,7]}for i in range(4)]for role in audit.ROLES}
    labels=np.asarray([0,0,1,1],dtype=np.int64)
    state={'coefficients':[.01]*9,'mean':[0.]*9,'scale':[1.]*9,'bias':.1,'source_train_only':True,'dtype':'float64',
           'protocol_sha256':'a'*64,'prepared_manifest_sha256':'b'*64}
    surface=put(source/'surface/lambda0.json',state)
    surface_selection=put(source/'surface/selection.json',{'checkpoint':'surface/lambda0.json','checkpoint_sha256':sha(surface)})
    chosen=[]
    for c in audit.CONDITIONS:
        for p in range(3):
            for f in range(3):
                path=source/f'full/source/{c}/pt{p}/ft{f}/best.pt';path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(f'opaque-checkpoint-fixture-{c}{p}{f}'.encode())
                chosen.append({'job':{'condition':c,'pt_seed':p,'ft_seed':f,'smoke':False},'best_epoch':1,
                               'checkpoint':{'file':str(path.relative_to(source)),'sha256':sha(path),'state_sha256':sha(path)}})
    selection=put(source/'source_selection.json',{'status':'frozen','conditions_filtered_by_score':False,'selected':chosen,
                    'protocol_sha256':'a'*64,'prepared_manifest_sha256':'b'*64,
                    'surface_selection':{'file':'surface/selection.json','sha256':sha(surface_selection)}})
    protocol=put(root/'protocol.json',{'source_selection':{'file':str(selection),'sha256':sha(selection)}})
    manifest=put(root/'prepared/manifest.json',{'fixture_only':True})
    base={str(protocol):sha(protocol),str(manifest):sha(manifest),str(selection):sha(selection)}
    def record():
        return {'status':'completed','source_selection_sha256':sha(selection),'protocol_sha256':sha(protocol),
                'prepared_manifest_sha256':sha(manifest),'roles':{},'labels_used_for_training_or_selection':False,
                'all_input_hashes_unchanged':True,'file_sha256':dict(base)}
    def prediction(directory,role,logp=None,constant=None):
        rows=role_rows[role];values={'labels':labels,'row_ids':np.asarray([r['row_id']for r in rows]),
                                  'group_ids':np.asarray([r['group_id']for r in rows])}
        if constant is None:
            values.update(log_probabilities=logp,predictions=logp.argmax(1));metric=audit.independent_metrics(labels,values['predictions'],logp[:,1]-logp[:,0],logp)
        else:
            values.update(scores=np.zeros(4),predictions=np.full(4,constant,dtype=np.int64));metric=audit.independent_metrics(labels,values['predictions'],values['scores'])
        path=directory/f'{role}.predictions.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**values)
        return {'metrics':metric,'predictions':{'file':str(path.relative_to(root)),'sha256':sha(path)}}
    for selected in chosen:
        c,p,f=[selected['job'][k]for k in ('condition','pt_seed','ft_seed')]
        directory=root/f'results/neural/{c}__pt{p}__ft{f}';value=record();cp=selected['checkpoint']
        value.update(job=selected['job'],checkpoint={'file':str(source/cp['file']),'sha256':cp['sha256'],'state_sha256':cp['state_sha256']},
                     state_before_sha256=cp['state_sha256'],state_after_sha256=cp['state_sha256'],source_best_epoch=1,
                     model_eval=True,weights_frozen=True,batch_size=32,parameter_dtype='float32',autocast_dtype='bfloat16',tf32_enabled=False)
        quality=.60+.02*audit.CONDITIONS.index(c)+.005*p+.001*f
        probabilities=np.where(labels[:,None]==np.arange(2),quality,1-quality)
        for role in audit.ROLES:value['roles'][role]=prediction(directory,role,np.log(probabilities))
        put(directory/'metrics.json',value)
    for name in ('surface','constant0','constant1'):
        directory=root/'results/references'/name;value=record();constant=None if name=='surface'else int(name[-1])
        value['kind']='surface'if constant is None else 'constant'
        if constant is None:value['checkpoint']={'file':str(surface),'sha256':sha(surface)}
        else:value['constant_label']=constant
        for role in audit.ROLES:
            result=prediction(directory,role,audit.surface_logp(role_rows[role],state)if constant is None else None,constant)
            result['cross_entropy_status']='finite'if constant is None else 'infinite'
            if constant is not None:result['cross_entropy_reason']=audit.CE_REASON
            value['roles'][role]=result
        put(directory/'metrics.json',value)
    def require(value,message):
        if not value:raise ValueError(message)
    def inside(base,name):
        p=(Path(base)/name).resolve();require(not Path(name).is_absolute()and p.is_relative_to(Path(base).resolve()),'path escapes');return p
    def check_hashes(unused,bindings):
        for p,digest in bindings.items():require(sha(p)==digest,'Changed pinned artifact: '+str(p))
    fake=SimpleNamespace(check_gate=lambda root,mode:json.loads(protocol.read_text()),sha=sha,read=lambda p:json.loads(Path(p).read_text()),
                         path_inside=inside,gate_file_hashes=lambda root,mode:dict(base),COUNTS={'source_test':4,'target':4},
                         load_role=lambda root,role,manifest=None:({'labels':labels.copy()},copy.deepcopy(role_rows[role])),
                         check_hashes=check_hashes,atomic=lambda p,v:put(p,v),now=lambda:'synthetic-test-only')
    return root,fake


def test_full54plus6_report_integration(tmp_path,monkeypatch):
    root,fake=report_fixture(tmp_path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(audit,'import_file',lambda name,path:fake)
    result=audit.audit(root)
    assert result['prediction_artifacts_audited']==60 and result['prediction_rows_audited']==240
    assert result['complete_checkpoint_hashes_verified']==27 and result['maximum_metric_difference']==0
    assert result['maximum_surface_log_probability_difference']==0 and result['all_input_hashes_unchanged']
    summary=json.loads((root/'results/report/summary.json').read_text())
    assert len(summary['neural_cells'])==54 and len(summary['reference_cells'])==6
    assert len(summary['nested_metrics'])==len(summary['paired_contrasts'])==36
    assert summary['exploratory_after_prior_target_use']and not summary['new_blind_confirmation_claimed']
    assert '不是新的盲确认' in (root/'results/report/REPORT.md').read_text()
    with pytest.raises(ValueError,match='existing'):audit.audit(root)


@pytest.mark.parametrize('mutation',['changed_prediction','wrong_group','wrong_epoch','unclosed_inputs','missing_gate_pin','target_fit'])
def test_report_faults_fail_before_completed_report(tmp_path,monkeypatch,mutation):
    root,fake=report_fixture(tmp_path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');monkeypatch.setattr(audit,'import_file',lambda name,path:fake)
    meta=root/'results/neural/EP__pt0__ft0/metrics.json';value=json.loads(meta.read_text())
    pred=root/value['roles']['target']['predictions']['file']
    if mutation=='changed_prediction':pred.write_bytes(pred.read_bytes()+b'tampered')
    elif mutation=='wrong_group':
        with np.load(pred)as loaded:values={k:loaded[k]for k in loaded.files}
        values['group_ids']=np.asarray(['wrong']*4);np.savez_compressed(pred,**values)
        value['roles']['target']['predictions']['sha256']=sha(pred)
    elif mutation=='wrong_epoch':value['source_best_epoch']=2
    elif mutation=='unclosed_inputs':value['all_input_hashes_unchanged']=False
    elif mutation=='missing_gate_pin':value['file_sha256'].pop(str(root/'protocol.json'))
    else:value['labels_used_for_training_or_selection']=True
    put(meta,value)
    with pytest.raises(ValueError):audit.audit(root)
    assert not(root/'results/report/audit.json').exists()


def smoke_values():
    rows=[{'row_id':str(i),'group_id':str(i//2)}for i in range(64)];labels=np.arange(64,dtype=np.int64)%2
    logp=np.log(np.where(labels[:,None]==np.arange(2),.8,.2)).astype(np.float64)
    saved={'labels':labels,'predictions':logp.argmax(1),'log_probabilities':logp.copy(),
           'row_ids':np.asarray([r['row_id']for r in rows]),'group_ids':np.asarray([r['group_id']for r in rows])}
    return rows,labels,logp,saved


def test_smoke_exact_identity_and_replay():
    rows,y,logp,saved=smoke_values();assert smoke.verify_replay(logp,saved,y,rows)==0


@pytest.mark.parametrize('mutation',['rows','groups','labels','nonfinite','precision','decisions','shape'])
def test_smoke_rejects_identity_and_numerical_corruption(mutation):
    rows,y,logp,saved=smoke_values()
    if mutation=='rows':saved['row_ids']=saved['row_ids'][::-1]
    elif mutation=='groups':saved['group_ids']=np.asarray(['x']*64)
    elif mutation=='labels':saved['labels']=1-y
    elif mutation=='nonfinite':logp[0,0]=np.nan
    elif mutation=='precision':saved['log_probabilities']=saved['log_probabilities'].astype(np.float32)
    elif mutation=='decisions':saved['predictions']=1-saved['predictions']
    else:saved['row_ids']=saved['row_ids'][:,None]
    with pytest.raises(ValueError):smoke.verify_replay(logp,saved,y,rows)


def test_smoke_fixed1e5_threshold_is_not_relaxed():
    rows,y,logp,saved=smoke_values()
    logits=np.log(np.where(y[:,None]==np.arange(2),.8,.2));logits[0,0]+=1e-4
    replay=logits-np.logaddexp(logits[:,0],logits[:,1])[:,None]
    with pytest.raises(ValueError,match='fixed1e-5'):smoke.verify_replay(replay,saved,y,rows)


@pytest.mark.parametrize('mutation',['missing','duplicate','filtered','smoke'])
def test_smoke_selection_retains_all27(mutation):
    value={'status':'frozen','conditions_filtered_by_score':False,'selected':[{'job':{'condition':c,'pt_seed':p,'ft_seed':f,'smoke':False}}for c in audit.CONDITIONS for p in range(3)for f in range(3)]}
    assert smoke.exact_selected(value)['job']=={'condition':'EP','pt_seed':0,'ft_seed':0,'smoke':False}
    if mutation=='missing':value['selected'].pop()
    elif mutation=='duplicate':value['selected'][-1]=value['selected'][0]
    elif mutation=='filtered':value['conditions_filtered_by_score']=True
    else:value['selected'][0]['job']['smoke']=True
    with pytest.raises(ValueError):smoke.exact_selected(value)


def test_surface_invalid_content_tokens_rejected():
    with pytest.raises(ValueError,match='content'):audit.independent_surface_vector([1,4],[5])


def test_cpu_report_rejects_gpu_environment(tmp_path,monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(ValueError,match='CPU-only'):audit.audit(tmp_path)


def test_source_selection_current_canonical_includes_creation_timestamp(tmp_path,monkeypatch):
    # Exercise the real current source gate entirely against synthetic JSON files.
    c=common_actual;source=tmp_path/'source';m2r=source/'results';m1=tmp_path/'m1';raw=tmp_path/'raw';prepared=tmp_path/'derived/prepared';root=tmp_path/'transfer'
    for name,value in [('M2',source),('M2_RESULTS',m2r),('M1',m1),('NEW_RAW',raw),('NEW_PREPARED',prepared)]:monkeypatch.setattr(c,name,value)
    files={};entries=[]
    for condition in c.CONDITIONS:
        for p in range(3):
            for f in range(3):
                cp=put(m2r/f'full/source/{condition}/pt{p}/ft{f}/best.pt',{'fixture':True})
                result=put(cp.parent/'metadata.json',{'fixture':True})
                for path in(cp,result):files[str(path)]=sha(path)
                entries.append({'job':{'condition':condition,'pt_seed':p,'ft_seed':f,'smoke':False},
                                'checkpoint':{'file':str(cp.relative_to(m2r)),'sha256':sha(cp)},
                                'source_result':{'file':str(result.relative_to(m2r)),'sha256':sha(result)}})
    put(m2r/'report/audit.json',{'status':'passed','source_fits':27,'pretraining_runs':9,'prediction_artifacts_audited':270,'complete_checkpoints_audited':36})
    selection={'created_at_utc':'synthetic-current-source-time','status':'frozen','conditions_filtered_by_score':False,'selected':entries,
               'target_scoring_started':False,'target_scoring_enabled':False,'source_test_scoring_enabled':False,
               'source_audit':{'file':'report/audit.json'},'protocol_sha256':'a'*64}
    selection['selection_sha256']=c.canonical(selection)
    selectionpath=put(m2r/'source_selection.json',selection)
    put(source/'execution_status.json',{'status':'completed','jobs':[{'status':'completed','exit_code':0}]*45})
    completion=put(tmp_path/'completion.json',{'status':'passed'})
    put(m1/'confirmation_acceptance.json',{'status':'accepted_for_local_confirmation','confirmation_data_ready':True,'retained_rows':39893})
    put(raw/'portable_validation/acceptance.json',{'status':'accepted_new_raw_release_training_disabled'})
    put(prepared.parent/'data_acceptance.json',{'status':'source_only_smoke_inputs_accepted'})
    barrier=put(root/'source_barrier.json',{'status':'frozen_sources_for_heldout_evaluation','files':files,'completion_review':{'file':str(completion)}})
    protocol={'source_barrier_sha256':sha(barrier),'source_selection':{'file':str(selectionpath),'sha256':sha(selectionpath)},
              'training_enabled':False,'calibration_enabled':False,'target_selection_enabled':False,'condition_filtering':False,
              'source_test_scoring_enabled':True,'target_scoring_enabled':True,'conditions':list(c.CONDITIONS),
              'pretraining_seeds':[0,1,2],'fine_tuning_seeds':[0,1,2],'roles':{k:{'rows':v}for k,v in c.COUNTS.items()},
              'source_protocol':{'file':str(source/'protocol.json'),'sha256':'a'*64},'model_loader':{'file':str(source/'code_snapshot/model.py')}}
    put(root/'protocol.json',protocol)
    assert c.check_source_gate(root)==protocol
    selection['selection_sha256']=c.canonical({k:v for k,v in selection.items()if k not in('created_at_utc','selection_sha256')})
    put(selectionpath,selection);protocol['source_selection']['sha256']=sha(selectionpath);put(root/'protocol.json',protocol)
    with pytest.raises(ValueError,match='Canonical'):c.check_source_gate(root)


def test_smoke_requires_entire_gate_readset_not_only_nonempty():
    expected={'/fixture/protocol.json':'a'*64,'/fixture/checkpoint.pt':'b'*64}
    record={'all_input_hashes_unchanged':True,'file_sha256':dict(expected)}
    smoke.validate_input_closure(record,expected)
    record['file_sha256'].pop('/fixture/checkpoint.pt')
    with pytest.raises(ValueError,match='omits'):smoke.validate_input_closure(record,expected)
