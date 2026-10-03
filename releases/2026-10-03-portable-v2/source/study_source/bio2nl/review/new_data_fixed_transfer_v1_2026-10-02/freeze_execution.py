"""Freeze reviewed code/data before first technical or formal held-out scoring."""
import importlib.metadata
import sys
from common import HERE, check_source_gate, read, sha, atomic, require, now, check_hashes


def main():
    check_source_gate(HERE)
    require(not (HERE/'execution_freeze.json').exists(),'Execution already frozen')
    require(not (HERE/'results').exists(),'Scoring occurred before execution freeze')
    manifest=read(HERE/'prepared/manifest.json');check_hashes(HERE,manifest['inputs']);check_hashes(HERE,manifest['outputs'])
    audit=read(HERE/'verification/data_audit.json')
    require(audit['status']=='passed' and audit['prepared_manifest_sha256']==sha(HERE/'prepared/manifest.json'),'Data audit stale')
    check_hashes(HERE,audit['bindings'])
    for name in ('cpu_tests','code_review'):
        report=read(HERE/f'verification/{name}.json');require(report['status']=='passed',name+' failed')
        check_hashes(HERE,report['files'])
    env={'python':sys.version,**{name:importlib.metadata.version(name) for name in ('torch','transformers','numpy','scipy','scikit-learn','tokenizers')}}
    atomic(HERE/'environment.json',env)
    paths=list(HERE.glob('*.py'))+[HERE/n for n in ('TRANSFER_PROTOCOL.md','STATISTICAL_PLAN.md','DATA_REPORT.md','.gitignore','data_input_spec.json')]
    paths += [HERE/p for p in ('protocol.json','source_barrier.json','environment.json','prepared/manifest.json',
        'verification/data_audit.json','verification/cpu_tests.json','verification/code_review.json','verification/source_freeze_review.json')]
    atomic(HERE/'execution_freeze.json',{'status':'frozen_for_fixed_scoring','created_at_utc':now(),
        'files':{str(p.relative_to(HERE)):sha(p) for p in sorted(paths)},'training_enabled':False})
    print('Execution frozen',sha(HERE/'execution_freeze.json'),flush=True)

if __name__=='__main__':main()
