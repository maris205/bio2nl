#!/usr/bin/env python3
"""Offline, CPU-only rebuilding from a separately accepted relocated release."""
import argparse
import json
import os
from pathlib import Path
import sys

PACKAGE=Path(__file__).resolve().parent
sys.path.insert(0,str(PACKAGE.parent))
from derived.common import Inputs, require, write_json, now
from derived.stages import STAGES


def main():
    require(__debug__, 'Optimized Python disables required historical assertions; run without -O')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--release',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['all',*STAGES],default='all')
    args=parser.parse_args()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Set CUDA_VISIBLE_DEVICES empty for CPU data construction')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    output=args.output.resolve(); archive=args.archive.resolve();release=args.release.resolve()
    require(not output.is_relative_to(archive) and not output.is_relative_to(release),'Output must be separate from input roots')
    require(not archive.is_relative_to(output) and not release.is_relative_to(output),'Output cannot contain input roots')
    stages=list(STAGES) if args.stage=='all' else [args.stage]
    if args.stage=='all':require(not output.exists(),'Fresh output root required for all stages')
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    for stage in stages:
        store=Inputs(archive,release,output)
        result=STAGES[stage](store);results.append(result)
        print(json.dumps(result,sort_keys=True),flush=True)
    if args.stage=='all':
        write_json(output/'completion.json',{'status':'passed','completed_at_utc':now(),'stages':results,
            'stage_manifest_sha256': {stage: __import__('hashlib').sha256((output/stage/'manifest.json').read_bytes()).hexdigest() for stage in stages},
            'model_loaded':False,'target_scoring':False,'new_blind_qualification':False})


if __name__=='__main__':main()
