"""Produce a NEW training acceptance only from this run's real CPU/GPU evidence."""
from pathlib import Path
import argparse
import math
import shutil
from support import require,load,pin,check,write,now

def validate_evidence(data):
    root=data.protocol_path.parent
    named={'cpu':root/'verification/cpu_checks.json','surface':root/'verification/surface_audit.json','smoke':data.output_root/'smoke/audit_runtime/metadata.json','storage':root/'verification/storage_acceptance.json'}
    evidence={str(data.protocol_path):pin(data.protocol_path)}
    def include(path,expected=None):
        path=Path(path).absolute()
        if expected is not None:check(path,expected)
        value=pin(path);previous=evidence.setdefault(str(path),value);require(previous==value,'Conflicting evidence identity')
    reports={}
    for name,path in named.items():
        value=load(path);reports[name]=value;include(path)
        require(value['protocol_sha256']==data.protocol_sha256,'Evidence is from another protocol: '+name)
        if name!='storage':require(value['prepared_manifest_sha256']==data.manifest_sha256,'Evidence prepared inputs differ')
    cpu,surface,smoke,storage=(reports[k] for k in ('cpu','surface','smoke','storage'))
    require(cpu['status']=='passed' and cpu['gpu_execution'] is False and cpu['fixture_only_model_checks'] is True,'Current CPU checks missing')
    require(cpu['checks']>=8 and cpu['source_counts']==data.source_counts,'CPU check coverage differs')
    for path,desc in cpu['inputs'].items():include(path,desc)
    require(surface['status']=='passed' and surface['candidates']==5 and surface['prediction_artifacts']==10 and surface['all_read_files_final_rehashed'] is True,'Independent current surface audit missing')
    require(surface['source_only'] is True and surface['target_examples_read'] is False,'Surface scope differs')
    for path,digest in surface['files'].items():
        desc=pin(path);require(desc['sha256']==digest,'Surface evidence changed');include(path,desc)
    require(smoke['status']=='completed' and smoke['all_new_checkpoint_replays_passed'] is True and smoke['checkpoints_replayed']==4,'Current four-checkpoint GPU smoke audit missing')
    require(smoke['fixed_absolute_tolerance']==1e-5 and smoke['optimizer_updates_performed']==0 and smoke['model_reload_and_forward_independent'] is True,'Smoke numerical contract differs')
    require(smoke['target_examples_read']==0 and smoke['source_test_examples_read']==0 and smoke['old_weights_loaded'] is False,'Smoke input scope differs')
    for field in ('artifact_inputs','data_and_code_inputs'):
        for path,desc in smoke[field].items():include(path,desc)
    for name,desc in smoke['outputs'].items():
        require(Path(name).name==name,'Invalid smoke output member');include(named['smoke'].parent/name,desc)
    expected_smoke=[]
    for condition in ('EP','ES','EE'):expected_smoke.append((data.output_root/f'smoke/pretrain/{condition}/pt0/metadata.json','pretraining',{'condition':condition,'pt_seed':0,'smoke':True}))
    expected_smoke.append((data.output_root/'smoke/source/EP/pt0/ft0/metadata.json','source_sft',{'condition':'EP','pt_seed':0,'ft_seed':0,'smoke':True}))
    for path,kind,job in expected_smoke:
        require(str(path) in evidence,'GPU audit did not pin required smoke metadata')
        m=load(path);require(m['status']=='completed' and m['kind']==kind and m['updates']==2 and m['job']==job and m['protocol_sha256']==data.protocol_sha256 and m['run_kind']=='smoke','Smoke training identity/budget differs')
        cp=m['checkpoint'];require(str(path.parent/cp['file']) in evidence,'GPU audit did not pin required complete checkpoint')
    require(storage['status']=='passed' and storage['remaining_artifact_budget_gib']==24 and storage['minimum_reserve_gib']==8,'Storage projection specification differs')
    require(math.isfinite(storage['projected_peak_free_gib']) and storage['projected_peak_free_gib']>=8,'Projected peak storage reserve too small')
    # Bind all runtime input/code pins and reject changes after examining the reports.
    for path,desc in data.verify_current_inputs().items():include(path,desc)
    for path,desc in evidence.items():check(path,desc)
    return evidence

def produce(protocol,sha256):
    from runtime_data import ReleaseData
    data=ReleaseData(protocol,sha256,mode='prepare');root=data.protocol_path.parent
    require(data.execution_protocol.get('cpu_validation_only') is False,'CPU-validation protocol cannot mint GPU/full acceptance')
    require(not (root/'training_acceptance.json').exists(),'Refuse existing training acceptance')
    path=root/'verification/storage_acceptance.json';free=shutil.disk_usage(root).free/(1<<30)
    require(free>=32,'Full run requires 24 GiB projected remaining artifacts plus 8 GiB reserve')
    write(path,{'status':'passed','protocol_sha256':sha256,'at_utc':now(),'free_gib':free,'remaining_artifact_budget_gib':24,'minimum_reserve_gib':8,'projected_peak_free_gib':free-24})
    evidence=validate_evidence(data)
    out={'schema_version':1,'status':'accepted_for_portable_source_training','protocol_sha256':sha256,'prepared_manifest_sha256':data.manifest_sha256,'verified_files':evidence,'full_budget_training_enabled':True,'target_scoring_enabled':False,'source_test_scoring_enabled':False,'historical_acceptance_reused':False,'accepted_at_utc':now()}
    write(root/'training_acceptance.json',out);return out
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True);a=p.parse_args();produce(a.protocol,a.protocol_sha256)
