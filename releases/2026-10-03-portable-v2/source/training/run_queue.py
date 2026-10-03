"""Serial explicit technical GPU smoke or full 9PT+27SFT execution. No automatic promotion."""
from pathlib import Path
import argparse
import fcntl
import os
import subprocess
import sys
from support import clean_path,require,pin,check,load,write,now,fresh_gpu,disk_check,require_gpu_environment

def job_plan(phase,protocol,sha256):
    common=['--protocol',str(protocol),'--protocol-sha256',sha256]
    def job(name,script,args=(),gpu=False):return {'name':name,'script':script,'args':[*common,*args],'gpu':gpu}
    if phase=='technical':
        result=[job('cpu_checks','cpu_check.py'),job('surface_fit','fit_surface.py'),job('surface_audit','audit_surface.py')]
        for condition in ('EP','ES','EE'):result.append(job('smoke_pt_'+condition,'runtime.py',['--job','pretrain','--condition',condition,'--pt-seed','0','--run-kind','smoke'],True))
        result += [job('smoke_source','runtime.py',['--job','sft','--condition','EP','--pt-seed','0','--ft-seed','0','--run-kind','smoke'],True),job('independent_smoke_reload','audit_runtime.py',gpu=True),job('training_acceptance','accept_training.py')]
        return result
    require(phase=='full','Unknown phase');result=[]
    for seed in range(3):
        for condition in ('EP','ES','EE'):result.append(job(f'pretrain_{condition}_pt{seed}','runtime.py',['--job','pretrain','--condition',condition,'--pt-seed',str(seed),'--run-kind','full'],True))
    for seed in range(3):
        for condition in ('EP','ES','EE'):
            for ft in range(3):result.append(job(f'source_{condition}_pt{seed}_ft{ft}','runtime.py',['--job','sft','--condition',condition,'--pt-seed',str(seed),'--ft-seed',str(ft),'--run-kind','full'],True))
    result.append(job('independent_source_audit','audit_source.py'));return result

def run(protocol,sha256,phase):
    from runtime_data import ReleaseData
    data=ReleaseData(protocol,sha256,mode='prepare' if phase=='technical' else 'full')
    require(not data.execution_protocol['cpu_validation_only'],'CPU-validation-only protocol cannot enter a GPU queue')
    require_gpu_environment(data.execution_protocol['environment'])
    root=data.protocol_path.parent;status_path=root/f'{phase}_queue_status.json';require(not status_path.exists(),'Refuse overwrite/resume; compile a fresh run after failure')
    lock=clean_path(data.execution_protocol['gpu_lock']);lock.parent.mkdir(parents=True,exist_ok=True)
    # O_NOFOLLOW protects final lock entry; ancestor symlinks were checked above.
    fd=os.open(lock,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    status={'status':'starting','phase':phase,'protocol_sha256':sha256,'started_at_utc':now(),'pid':os.getpid(),'jobs':[],'source_only':True,'target_scoring_enabled':False}
    completed_pins={}
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(status_path,status)
        if phase=='full':
            acceptance=root/'training_acceptance.json';completed_pins[str(acceptance)]=pin(acceptance)
        for spec in job_plan(phase,protocol,sha256):
            data.verify_current_inputs()
            for path,desc in completed_pins.items():check(path,desc)
            entry={**spec,'started_at_utc':now(),'status':'running'}
            if spec['gpu']:entry['gpu_before_launch']=fresh_gpu()
            entry['disk_before_launch']=disk_check(root,10)
            for path,desc in completed_pins.items():check(path,desc)
            status['jobs'].append(entry);status['status']='running';write(status_path,status,replace=True)
            log=root/'logs'/(phase+'_'+spec['name']+'.log');log.parent.mkdir(exist_ok=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if spec['gpu'] else '',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
            with log.open('x') as handle:
                child=subprocess.Popen([sys.executable,'-B',str(root/'code'/spec['script']),*spec['args']],cwd=root/'code',env=env,stdout=handle,stderr=subprocess.STDOUT)
                entry['pid']=child.pid;write(status_path,status,replace=True)
                try:code=child.wait()
                except BaseException:
                    child.terminate();child.wait(timeout=30);raise
            entry.update(exit_code=code,completed_at_utc=now(),log=str(log));require(code==0,'Stage failed: '+spec['name']+'; see '+str(log))
            # Never rehash all tensor weights here: producers/auditors do that. Pin every small
            # completed metadata/report/selection file so later stages cannot change outcomes.
            for path in sorted((root/'results').rglob('*.json')) if (root/'results').exists() else []:
                old=completed_pins.get(str(path))
                if old is not None:check(path,old)
                else:completed_pins[str(path)]=pin(path)
            for path,desc in completed_pins.items():check(path,desc)
            entry['status']='completed';write(status_path,status,replace=True)
        data.verify_current_inputs();status.update(status='completed',completed_at_utc=now(),completed_metadata_pins=completed_pins);write(status_path,status,replace=True)
    except BaseException as error:
        if status_path.exists():status.update(status='failed',error=repr(error),failed_at_utc=now());write(status_path,status,replace=True)
        raise
    finally:os.close(fd)
    return status
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True);p.add_argument('--phase',choices=['technical','full'],required=True)
    p.add_argument('--execute-gpu-smoke',action='store_true');p.add_argument('--execute-full-training',action='store_true');a=p.parse_args()
    require((a.phase=='technical' and a.execute_gpu_smoke and not a.execute_full_training) or (a.phase=='full' and a.execute_full_training and not a.execute_gpu_smoke),'Explicit phase-specific execution flag required')
    run(a.protocol,a.protocol_sha256,a.phase)
