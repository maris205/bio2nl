"""CPU-only synthetic and fail-closed adapter tests. No GPU jobs or real fits."""
from pathlib import Path
import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import support
import run_queue

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('portable_compile',HERE/'compile.py');compiler=importlib.util.module_from_spec(spec);spec.loader.exec_module(compiler)

class PureGates(unittest.TestCase):
    def test_plan_has_exact_scientific_roster(self):
        plan=run_queue.job_plan('full','p','a'*64);self.assertEqual(len(plan),37)
        pt=[r for r in plan if r['name'].startswith('pretrain_')];sft=[r for r in plan if r['name'].startswith('source_')]
        self.assertEqual(len(pt),9);self.assertEqual(len(sft),27);self.assertTrue(all(r['gpu'] for r in pt+sft));self.assertFalse(plan[-1]['gpu'])
        for row in sft:
            args=row['args'];condition=args[args.index('--condition')+1];seed=args[args.index('--pt-seed')+1]
            self.assertLess(next(i for i,r in enumerate(plan) if r['name']==f'pretrain_{condition}_pt{seed}'),plan.index(row))
    def test_technical_does_not_schedule_full_training(self):
        p=run_queue.job_plan('technical','p','a'*64);self.assertEqual(sum(r['gpu'] for r in p),5)
        self.assertFalse(any('full' in r['args'] for r in p));self.assertEqual(p[-1]['script'],'accept_training.py')
    def test_distribution_binary_mismatch_blocks_gpu(self):
        with self.assertRaises(ValueError):support.require_gpu_environment({'packages':{'torch':'2.9.1'},'torch_build_version':'2.3.0+cu121'})
    def test_matching_distribution_binary_allows_next_gpu_checks(self):
        support.require_gpu_environment({'packages':{'torch':'2.9.1'},'torch_build_version':'2.9.1+cu128'})
    def test_bad_phase_rejected(self):
        with self.assertRaises(ValueError):run_queue.job_plan('target','p','a'*64)
    def test_pin_rejects_mutated_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';p.write_bytes(b'ab');pin=support.pin(p);p.write_bytes(b'ac')
            with self.assertRaises(ValueError):support.check(p,pin)
    def test_symlink_ancestor_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'real').mkdir();(p/'link').symlink_to(p/'real',target_is_directory=True)
            with self.assertRaises(ValueError):support.clean_path(p/'link'/'file')
    def test_atomic_write_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x';support.write(p,{'a':1})
            with self.assertRaises(ValueError):support.write(p,{'a':2})
            self.assertEqual(support.load(p),{'a':1})
    def test_gpu_active_process_rejected(self):
        from subprocess import CompletedProcess
        with patch('support.subprocess.run',side_effect=[CompletedProcess([],0,'0, fixture, 32000, 0, 0\n',''),CompletedProcess([],0,'1234\n','')]):
            with self.assertRaises(ValueError):support.fresh_gpu()
    def test_gpu_busy_utilization_rejected(self):
        from subprocess import CompletedProcess
        with patch('support.subprocess.run',side_effect=[CompletedProcess([],0,'0, fixture, 32000, 1, 80\n',''),CompletedProcess([],0,'','')]):
            with self.assertRaises(ValueError):support.fresh_gpu()
    def test_disk_reserve_rejected(self):
        from collections import namedtuple
        disk=namedtuple('disk','total used free')
        with patch('support.shutil.disk_usage',return_value=disk(100,90,1)):
            with self.assertRaises(ValueError):support.disk_check(HERE)
    def test_adapter_changes_only_declared_provenance(self):
        old='x["raw_manifest_sha256"]; data.raw_root; loss=model(ids).loss\n';new,changes=compiler.adapt('runtime.py',old)
        self.assertEqual(new,'x["input_resource_lock_sha256"]; data.source_code_root; loss=model(ids).loss\n');self.assertEqual(len(changes),2)
    def test_compile_rejects_missing_roots_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):compiler.compile_run(Path(d)/'missing',Path(d)/'out',HERE,'a'*64,'b'*64,Path(d)/'gpu.lock')
            self.assertFalse((Path(d)/'out').exists())

class NumericalAndInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.folder=Path(cls.temp.name)
        source=Path(os.environ['BIO2NL_ALGORITHM_ROOT']).resolve()
        for name,descriptor in support.load(HERE/'algorithm_lock.json')['files'].items():
            support.check(source/name,descriptor);text,_=compiler.adapt(name,(source/name).read_text());(cls.folder/name).write_text(text)
        (cls.folder/'runtime_data.py').write_text((HERE/'runtime_data.py').read_text())
        sys.path.insert(0,str(cls.folder));import runtime,model,runtime_data
        cls.runtime=runtime;cls.model=model;cls.reader=runtime_data
    @classmethod
    def tearDownClass(cls):sys.path.remove(str(cls.folder));cls.temp.cleanup()
    def test_tiny_cpu_model_and_corrected_ce(self):
        import cpu_check
        self.assertEqual(cpu_check.numerical_checks(self.runtime,self.model),8)
    def test_budget_drift_rejected(self):
        design=support.load(HERE/'design.json');self.runtime.validate_numerical_design(design,{'train':8044,'validation':20276})
        design['pretraining']['updates_per_run']=1025
        with self.assertRaises(ValueError):self.runtime.validate_numerical_design(design,{'train':8044,'validation':20276})
    def test_wrong_sft_membership_rejected(self):
        with self.assertRaises(ValueError):self.runtime.sft_budget(8002,'full')
    def test_smoke_cannot_request_extra_seed(self):
        with self.assertRaises(ValueError):self.runtime.resolve_job('EP',1,run_kind='smoke')
    def test_smoke_cannot_cross_ft_roster(self):
        with self.assertRaises(ValueError):self.runtime.resolve_job('ES',0,0,run_kind='smoke')
    def test_invalid_pt_seed_rejected(self):
        with self.assertRaises(ValueError):self.runtime.resolve_job('EP',3,run_kind='full')
    def test_schedule_duplicates_rejected(self):
        schedule=np.zeros((2048,16,2),dtype=np.int64);schedule[:,8:,0]=1
        with self.assertRaises(ValueError):self.reader.validate_schedule(schedule)
    def test_heldout_role_refused_without_io(self):
        obj=object.__new__(self.reader.ReleaseData)
        with self.assertRaises(ValueError):obj.load_source('test')
    def test_cpu_validation_protocol_refuses_gpu_mode(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'protocol.json';support.write(p,{'schema_version':1,'status':'frozen_portable_source_training_v1','scope':'source_only_portable_reproduction','cpu_validation_only':True,'bounded_smoke_training_enabled':False,'full_budget_training_enabled':False,'target_scoring_enabled':False,'source_test_scoring_enabled':False,'old_weights_allowed':False})
            with self.assertRaisesRegex(ValueError,'CPU-validation protocol'):
                self.reader.ReleaseData(p,support.sha(p),mode='smoke')
    def test_old_protocol_scope_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'protocol.json';support.write(p,{'schema_version':1,'status':'frozen_for_new_data_full_training','scope':'source_only_new_data_full_training'})
            with self.assertRaises(ValueError):self.reader.ReleaseData(p,support.sha(p))
    def test_epoch_tie_favors_earlier(self):
        import audit_source
        history=[{'epoch':i,'validation':{'metrics':{'cross_entropy':.5}}} for i in range(1,6)]
        self.assertEqual(audit_source.select_epoch(history)['epoch'],1)
    def test_empty_evidence_cannot_authorize_full(self):
        from types import SimpleNamespace
        from accept_training import validate_evidence
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);support.write(root/'protocol.json',{});data=SimpleNamespace(protocol_path=root/'protocol.json',output_root=root/'results')
            with self.assertRaises(FileNotFoundError):validate_evidence(data)

if __name__=='__main__':unittest.main(verbosity=2)
