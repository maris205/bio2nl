"""Portable compiler/queue helpers; no imports of historical runtime gates."""
from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone

PACKAGES=('numpy','torch','transformers','tokenizers','scipy','scikit-learn','pyarrow')

def require(value,message):
    if not value: raise ValueError(message)

def now(): return datetime.now(timezone.utc).isoformat()

def clean_path(path,existing=False):
    p=Path(path).expanduser().absolute()
    require('..' not in p.parts,'Parent traversal is forbidden')
    for item in [*reversed(p.parents),p]: require(not item.is_symlink(),'Symlinked path forbidden: '+str(item))
    if existing: require(p.exists(),'Missing path: '+str(p))
    return p

def sha(path):
    p=clean_path(path,True); before=p.stat();require(stat.S_ISREG(before.st_mode),'Regular file required')
    with p.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    after=p.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Input changed while hashing')
    return digest

def pin(path):
    p=clean_path(path,True);return {'bytes':p.stat().st_size,'sha256':sha(p)}

def check(path,expected):
    require(set(expected)>={'bytes','sha256'} and type(expected['bytes']) is int,'Invalid file descriptor')
    require(pin(path)=={k:expected[k] for k in ('bytes','sha256')},'Pinned file differs: '+str(path));return Path(path)

def load(path): return json.loads(Path(path).read_text())

def write(path,value,replace=False):
    p=clean_path(path);require(replace or not p.exists(),'Refuse existing output: '+str(p));p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.partial')
    with temp.open('x') as f: json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    os.replace(temp,p)

def environment():
    import torch
    return {'python':sys.version,'python_version':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in PACKAGES},'torch_build_version':torch.__version__,'torch_cuda_build':torch.version.cuda}

def fresh_gpu():
    """Read immediately before each GPU subprocess; reject occupied devices."""
    command=['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']
    result=subprocess.run(command,check=True,text=True,capture_output=True,timeout=20)
    lines=[s for s in result.stdout.splitlines() if s.strip()];require(len(lines)==1,'Exactly one visible physical GPU is required')
    parts=[s.strip() for s in lines[0].split(',')];require(len(parts)==5 and parts[0]=='0','Expected GPU index 0')
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],check=True,text=True,capture_output=True,timeout=20)
    require(not processes.stdout.strip(),'GPU compute processes are active')
    require(int(parts[3])<=512 and int(parts[4])<=5,'GPU not idle')
    return {'checked_at_utc':now(),'index':0,'name':parts[1],'total_mib':int(parts[2]),'used_mib':int(parts[3]),'utilization_percent':int(parts[4]),'compute_pids':[]}

def disk_check(root,required_gib=10):
    free=shutil.disk_usage(root).free;require(free>=required_gib*(1<<30),'Insufficient disk reserve')
    return {'free_bytes':free,'required_gib':required_gib,'checked_at_utc':now()}


def gpu_environment_consistent(env):
    return env['torch_build_version'].split('+',1)[0]==env['packages']['torch']

def require_gpu_environment(env):
    require(gpu_environment_consistent(env),'PyTorch distribution metadata and imported binary version disagree; resolve in an isolated environment before any GPU execution')
