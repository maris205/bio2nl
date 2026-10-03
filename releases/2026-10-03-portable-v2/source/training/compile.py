"""Compile a fresh relocatable source-training run; never launch a GPU."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
from support import check,clean_path,environment,load,pin,require,sha,write,now,gpu_environment_consistent

HERE=Path(__file__).resolve().parent
ADAPTERS=('runtime_data.py','support.py','run_queue.py','accept_training.py','cpu_check.py','design.json','scientific_payload_lock.json','algorithm_lock.json')

def adapt(name,text):
    # Provenance namespace changes only. Optimization/model/inference arithmetic is untouched.
    changes=[]
    if 'raw_manifest_sha256' in text:
        count=text.count('raw_manifest_sha256');text=text.replace('raw_manifest_sha256','input_resource_lock_sha256');changes.append({'from':'raw_manifest_sha256','to':'input_resource_lock_sha256','occurrences':count})
    if 'data.raw_root' in text:
        count=text.count('data.raw_root');text=text.replace('data.raw_root','data.source_code_root');changes.append({'from':'data.raw_root','to':'data.source_code_root','occurrences':count})
    return text,changes

def compile_run(prepared_root,output_root,source_code_root,prepared_manifest_sha256,prepared_acceptance_sha256,gpu_lock,cpu_validation_only=False):
    prepared=clean_path(prepared_root,True);source=clean_path(source_code_root,True);out=clean_path(output_root);lock=clean_path(gpu_lock)
    require(prepared.is_dir() and source.is_dir(),'Input roots must be directories')
    require(not out.exists(),'New empty output root required')
    require(not any(out.is_relative_to(p) or p.is_relative_to(out) for p in (prepared,source,HERE)),'Output root overlaps immutable inputs/code')
    require(not lock.is_relative_to(out),'GPU lock must be shared outside this run root')
    manifest=prepared/'manifest.json';acceptance=prepared/'acceptance.json'
    require(sha(manifest)==prepared_manifest_sha256 and sha(acceptance)==prepared_acceptance_sha256,'Explicit preparation identity differs')
    algorithms=load(HERE/'algorithm_lock.json')['files']
    for name,descriptor in algorithms.items():check(source/name,descriptor)
    current=environment();historical=load(HERE/'historical_environment.json')
    require(current['python_version']=='3.12.3','Install the pinned Python 3.12.3 runtime')
    require(current['packages']==historical['packages'],'Training package versions differ from the recorded October 2 regime')
    require(cpu_validation_only or gpu_environment_consistent(current),'Inconsistent PyTorch distribution/binary: normal training compilation refused; --cpu-validation-only creates a permanently GPU-disabled validation run')
    out.mkdir(parents=True,exist_ok=False);(out/'code').mkdir();(out/'verification').mkdir()
    try:
        transformations={}
        for name in algorithms:
            content,changes=adapt(name,(source/name).read_text());(out/'code'/name).write_text(content);transformations[name]=changes
        for name in ADAPTERS:shutil.copyfile(HERE/name,out/'code'/name)
        code={str(f):pin(f) for f in sorted((out/'code').iterdir())}
        protocol={'schema_version':1,'status':'frozen_portable_source_training_v1','scope':'source_only_portable_reproduction','created_at_utc':now(),
                  'cpu_validation_only':cpu_validation_only,'bounded_smoke_training_enabled':not cpu_validation_only,'full_budget_training_enabled':not cpu_validation_only,'target_scoring_enabled':False,'source_test_scoring_enabled':False,'old_weights_allowed':False,
                  'prepared':{'root':str(prepared),'manifest':{'file':str(manifest),**pin(manifest)},'acceptance':{'file':str(acceptance),**pin(acceptance)}},
                  'source_code_root':str(source),'source_algorithm_files':{str(source/n):d for n,d in algorithms.items()},'source_adapter_files':{str(HERE/n):pin(HERE/n) for n in (*ADAPTERS,'compile.py','historical_environment.json')},
                  'provenance_only_substitutions':transformations,'runtime_code':code,'environment':current,'design':load(HERE/'design.json'),
                  'output_root':str(out/'results'),'gpu_lock':str(lock),'training_acceptance_file':str(out/'training_acceptance.json'),
                  'historical_acceptance_copied':False,'gpu_execution_performed_by_compiler':False,'full_training_reproduction_demonstrated':False}
        write(out/'protocol.json',protocol)
        # Validate the newly compiled gate in a clean interpreter. No model is constructed.
        import subprocess,os
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        command=[sys.executable,'-B','-c','from runtime_data import ReleaseData; import sys; d=ReleaseData(sys.argv[1],sys.argv[2]); print(d.manifest_sha256)',str(out/'protocol.json'),sha(out/'protocol.json')]
        result=subprocess.run(command,cwd=out/'code',env=env,text=True,capture_output=True,check=False)
        (out/'verification/compiler_preflight.log').write_text(result.stdout+result.stderr)
        require(result.returncode==0,'Portable prepared gate failed; see compiler_preflight.log')
        value={'status':'compiled_cpu_validation_only' if cpu_validation_only else 'compiled_not_trained','protocol_sha256':sha(out/'protocol.json'),'prepared_manifest_sha256':prepared_manifest_sha256,'environment':current,'code_files':len(code),'gpu_environment_consistent':gpu_environment_consistent(current),'gpu_execution':False,'model_inference':False,'acceptance_created':False,'old_gates_modified':False,'created_at_utc':now()}
        write(out/'verification/compiler_preflight.json',value);print(json.dumps(value));return value
    except BaseException as error:
        write(out/'compilation_failure.json',{'status':'failed','error':repr(error),'at_utc':now()});raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('prepared-root','output-root','source-code-root','prepared-manifest-sha256','prepared-acceptance-sha256','gpu-lock'):p.add_argument('--'+flag,required=True)
    p.add_argument('--cpu-validation-only',action='store_true',help='Compile CPU validation only; cannot authorize GPU work even after an environment change')
    a=p.parse_args();compile_run(a.prepared_root,a.output_root,a.source_code_root,a.prepared_manifest_sha256,a.prepared_acceptance_sha256,a.gpu_lock,a.cpu_validation_only)
if __name__=='__main__':main()
