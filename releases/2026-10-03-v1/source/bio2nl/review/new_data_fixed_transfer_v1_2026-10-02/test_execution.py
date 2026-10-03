import json
import numpy as np
import pytest
import common
import run_queue


def test_fixed_full_grid_and_cpu_tail():
    stages=run_queue.stages('full')
    assert len(stages)==29 and len({x[0] for x in stages})==29
    assert all(x[1] for x in stages[:27]) and not any(x[1] for x in stages[27:])
    assert {tuple(x[2][1:]) for x in stages[:27]}=={('--condition',c,'--pt-seed',str(p),'--ft-seed',str(f)) for c in ('EP','ES','EE') for p in range(3) for f in range(3)}


def test_smoke_has_no_choice_or_training():
    stages=run_queue.stages('smoke')
    assert len(stages)==3 and stages[0][2]==['score_models.py','--condition','EP','--pt-seed','0','--ft-seed','0','--smoke']
    assert stages[-1][2]==['accept_scoring.py']


def test_missing_source_barrier_fails_before_any_input(tmp_path):
    with pytest.raises(FileNotFoundError):common.check_source_gate(tmp_path)


def test_hash_mutation_rejected(tmp_path):
    p=tmp_path/'data';p.write_text('before');pins={'data':common.sha(p)}
    common.check_hashes(tmp_path,pins);p.write_text('after')
    with pytest.raises(ValueError,match='Changed pinned'):common.check_hashes(tmp_path,pins)


def test_prediction_no_overwrite(tmp_path):
    p=tmp_path/'p.npz';labels=np.array([0,1]);logp=np.log([[.8,.2],[.3,.7]])
    common.save_predictions(p,labels,logp,['a','b'],['x','x'])
    with pytest.raises(ValueError,match='overwrite'):common.save_predictions(p,labels,logp,['a','b'],['x','x'])


def test_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError):common.path_inside(tmp_path,'../outside')
    with pytest.raises(ValueError):common.path_inside(tmp_path,'/outside')


@pytest.mark.parametrize('gpu,processes,wanted',[
    ('0, card, 10, 499, 32760','',True),
    ('0, card, 11, 0, 32760','',False),
    ('0, card, 0, 500, 32760','',False),
    ('0, card, 0, 0, 32760','123, python, 5',False)])
def test_gpu_availability_rejects_load_and_processes(gpu,processes,wanted):
    assert run_queue.available(gpu,processes) is wanted


def test_gpu_inventory_rejects_multiple_devices():
    with pytest.raises(ValueError,match='exactly one'):
        run_queue.available('0, card, 0, 0, 32760\n1, card, 0, 0, 32760','')


@pytest.mark.parametrize('which',['result','log'])
def test_completed_stage_change_stops_later_work(tmp_path,which):
    result=tmp_path/'result.json';result.write_text('{"status":"completed"}')
    log=tmp_path/'job.log';log.write_text('finished')
    s={'stages':[{'status':'completed','outputs':{str(result):common.sha(result)},'log':str(log),'log_sha256':common.sha(log)}]}
    run_queue.check_completed(s)
    (result if which=='result' else log).write_text('changed')
    with pytest.raises(ValueError,match='Changed pinned'):run_queue.check_completed(s)


def test_gpu_bounded_wait_records_busy_then_free(tmp_path,monkeypatch):
    from types import SimpleNamespace
    samples=iter(['0, card, 99, 0, 32760','','0, card, 0, 0, 32760',''])
    monkeypatch.setattr(run_queue,'HERE',tmp_path)
    monkeypatch.setattr(run_queue.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout=next(samples)))
    monkeypatch.setattr(run_queue.time,'sleep',lambda _:None)
    assert run_queue.gpu_check('sample')['available'] is True
    assert len(common.read(tmp_path/'verification/gpu_checks/sample.json')['samples'])==2


def test_gpu_stops_after_declared_wait_limit(tmp_path,monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(run_queue,'HERE',tmp_path)
    def query(cmd,**kw):return SimpleNamespace(stdout='0, card, 99, 900, 32760' if '--query-gpu' in cmd[1] else '99,other,900')
    monkeypatch.setattr(run_queue.subprocess,'run',query)
    monkeypatch.setattr(run_queue.time,'sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='bounded16'):run_queue.gpu_check('busy')
    assert len(common.read(tmp_path/'verification/gpu_checks/busy.json')['samples'])==16


def test_failed_process_preserves_log_and_never_starts_next(tmp_path,monkeypatch):
    import signal
    from types import SimpleNamespace
    first=tmp_path/'first.py';first.write_text('import sys; print("planned failure",flush=True); sys.exit(3)')
    second=tmp_path/'second.py';second.write_text('raise RuntimeError("must never run")')
    common.atomic(tmp_path/'execution_freeze.json',{'files':{}})
    monkeypatch.setattr(run_queue,'HERE',tmp_path)
    monkeypatch.setattr(run_queue,'check_gate',lambda *a,**kw:{})
    monkeypatch.setattr(run_queue,'stages',lambda _: [('failure',False,['first.py']),('never',False,['second.py'])])
    monkeypatch.setattr(run_queue.shutil,'disk_usage',lambda _:SimpleNamespace(free=50*(1<<30)))
    old={v:signal.getsignal(v) for v in (signal.SIGINT,signal.SIGTERM)}
    try:
        with pytest.raises(ValueError,match='Stage failed'):run_queue.main('smoke')
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)
    status=common.read(tmp_path/'smoke_queue_status.json')
    assert status['status']=='failed' and len(status['stages'])==1
    assert status['stages'][0]['exit_code']==3
    assert (tmp_path/'logs/smoke/failure.log').read_text().strip()=='planned failure'
    assert not (tmp_path/'workers/smoke/never').exists()
