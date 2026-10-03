"""Serial local fixed inference, closed stage evidence, fail on first error."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from common import HERE, CONDITIONS, atomic, now, require, check_gate, sha, read, check_hashes


def stages(mode):
    if mode=='smoke':return [('scoring_smoke',True,['score_models.py','--condition','EP','--pt-seed','0','--ft-seed','0','--smoke']),
        ('independent_smoke_replay',True,['audit_smoke.py']),('technical_acceptance',False,['accept_scoring.py'])]
    require(mode=='full','Unknown queue mode')
    return [(f'score_{c}_pt{p}_ft{f}',True,['score_models.py','--condition',c,'--pt-seed',str(p),'--ft-seed',str(f)])
        for c in CONDITIONS for p in range(3) for f in range(3)]+[
        ('fixed_references',False,['score_references.py']),('independent_final_report',False,['audit_results.py'])]


def expected_outputs(label,mode):
    if label=='scoring_smoke':return [('results/smoke/EP__pt0__ft0/metrics.json','completed')]
    if label=='independent_smoke_replay':return [('verification/smoke_audit.json','passed')]
    if label=='technical_acceptance':return [('scoring_acceptance.json','accepted_for_fixed_scoring')]
    if label=='fixed_references':return [(f'results/references/{name}/metrics.json','completed') for name in ('surface','constant0','constant1')]
    if label=='independent_final_report':return [('results/report/audit.json','passed')]
    require(mode=='full' and label.startswith('score_'),'Unknown queue output')
    _,c,p,f=label.split('_')
    return [(f'results/neural/{c}__{p}__{f}/metrics.json','completed')]


def finish_outputs(label,mode):
    out={}
    for name,status in expected_outputs(label,mode):
        path=HERE/name;value=read(path);require(value['status']==status,'Unaccepted stage output: '+name)
        out[name]=sha(path)
    return out


def check_completed(status):
    for row in status['stages']:
        if row['status']=='completed':
            check_hashes(HERE,row['outputs'])
            check_hashes(HERE,{row['log']:row['log_sha256']})


def available(query,processes):
    lines=query.strip().splitlines();require(len(lines)==1,'Expected exactly one local GPU')
    fields=[x.strip() for x in lines[0].split(',')]
    require(len(fields)==5 and fields[0]=='0','Unexpected GPU inventory')
    return int(fields[3])<500 and int(fields[2])<=10 and not processes.strip()


def gpu_check(name):
    samples=[]
    for attempt in range(16):
        query=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        ok=available(query.stdout,processes.stdout)
        samples.append({'checked_at_utc':now(),'gpu':query.stdout,'processes':processes.stdout,'available':ok})
        atomic(HERE/f'verification/gpu_checks/{name}.json',{'samples':samples})
        if ok:return samples[-1]
        if attempt<15:time.sleep(1)
    raise RuntimeError('GPU unavailable after bounded16sample check; stop queue')


def main(mode):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    lockpath=HERE.parents[2]/'.bio2nl_gpu_training_queue.lock'
    with lockpath.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        statuspath=HERE/f'{mode}_queue_status.json';require(not statuspath.exists(),'Queue already started; preserve existing status/results')
        check_gate(HERE,mode=mode)
        if mode=='full':
            prior=read(HERE/'smoke_queue_status.json')
            require(prior['status']=='completed' and len(prior['stages'])==3 and all(s['exit_code']==0 for s in prior['stages']),'Technical queue incomplete')
            check_completed(prior)
        expected=stages(mode);start=time.monotonic();freeze=read(HERE/'execution_freeze.json')
        status={'status':'running','mode':mode,'queue_pid':os.getpid(),'started_at_utc':now(),'expected_stages':len(expected),'stages':[],
            'execution_freeze_sha256':sha(HERE/'execution_freeze.json'),'training_enabled':False,'workspace_lock':str(lockpath)}
        active=None
        def stop_active():
            if active is not None and active.poll() is None:
                os.killpg(active.pid,signal.SIGTERM)
                try:active.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(active.pid,signal.SIGKILL);active.wait()
        def stopped(sig,frame):raise RuntimeError('Queue interrupted by signal '+str(sig))
        signal.signal(signal.SIGTERM,stopped);signal.signal(signal.SIGINT,stopped)
        atomic(statuspath,status);atomic(HERE/'execution_status.json',status)
        try:
            for label,gpu,args in expected:
                status['current_stage']=label;atomic(statuspath,status);atomic(HERE/'execution_status.json',status)
                require(sha(HERE/'execution_freeze.json')==status['execution_freeze_sha256'],'Execution freeze changed')
                check_hashes(HERE,freeze['files']);check_completed(status)
                require(shutil.disk_usage(HERE).free>=9*(1<<30),'Need8GiB reserve plus1GiB job allocation')
                sample=gpu_check(mode+'_'+label) if gpu else None
                check_hashes(HERE,freeze['files']);check_completed(status)
                require(shutil.disk_usage(HERE).free>=9*(1<<30),'Disk reserve changed before process launch')
                log=HERE/f'logs/{mode}/{label}.log';log.parent.mkdir(parents=True,exist_ok=True)
                worker_dir=HERE/f'workers/{mode}/{label}';worker_dir.mkdir(parents=True,exist_ok=False)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0' if gpu else '',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_DISABLED='true')
                command=[sys.executable,'-B',str(HERE/args[0]),*args[1:]]
                row={'name':label,'gpu':gpu,'command':command,'log':str(log.relative_to(HERE)),'started_at_utc':now(),'status':'running','gpu_prelaunch_sample':sample}
                status['stages'].append(row)
                with log.open('x') as stream:
                    active=subprocess.Popen(command,cwd=worker_dir,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    row['pid']=active.pid;atomic(statuspath,status);atomic(HERE/'execution_status.json',status)
                    print(label,'started PID',active.pid,flush=True)
                    rc=active.wait();active=None
                row.update(exit_code=rc,completed_at_utc=now(),status='process_exited' if rc==0 else 'failed')
                atomic(statuspath,status);atomic(HERE/'execution_status.json',status)
                require(rc==0,'Stage failed: '+label+'; inspect '+str(log))
                outputs=finish_outputs(label,mode)
                check_hashes(HERE,freeze['files']);check_completed(status)
                row.update(status='completed',outputs=outputs,log_sha256=sha(log))
                atomic(statuspath,status);atomic(HERE/'execution_status.json',status)
            status.update(status='completed',completed_at_utc=now(),elapsed_seconds=time.monotonic()-start)
        except BaseException as exc:
            stop_active()
            status.update(status='failed',failed_at_utc=now(),error=str(exc),elapsed_seconds=time.monotonic()-start)
            atomic(statuspath,status);atomic(HERE/'execution_status.json',status)
            raise
        atomic(statuspath,status);atomic(HERE/'execution_status.json',status)
        print(mode,'queue completed',len(expected),'stages',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','full'],required=True)
    main(parser.parse_args().mode)
