import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from derived.common import Inputs, inside, semantic, sha, write_json, write_rows, V2
from derived import algorithms as alg
from derived import stages


class Fixture:
    def __init__(self, base):
        self.archive=base/'archive';self.release=base/'release';self.output=base/'output'
        self.archive.mkdir();self.release.mkdir();self.output.mkdir()
        self.member='data/f.jsonl';self.old=self.archive/V2/self.member;self.new=self.release/self.member
        write_rows(self.old,[{'a':1},{'a':2}]);write_rows(self.new,[{'a':1},{'a':2}])
        write_json(self.archive/'archive_manifest.json',{'files':[{'path':V2+self.member,'bytes':self.old.stat().st_size,'sha256':sha(self.old)}]})
        self.gate={'status':'accepted_for_derived_reconstruction','training_enabled':False,'target_scoring_enabled':False,
            'files':{self.member:{'sha256':sha(self.new),'bytes':self.new.stat().st_size,'reference_sha256':sha(self.old),'comparison_mode':'ordered_json_records','equivalent':True}}}
        self.gatepath=self.release/'portable_validation/acceptance.json';write_json(self.gatepath,self.gate)
    def store(self):
        return Inputs(self.archive,self.release,self.output,sha(self.archive/'archive_manifest.json'))
    def update(self):
        self.gatepath.write_text(json.dumps(self.gate))


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name);self.f=Fixture(self.base)
    def tearDown(self):self.temp.cleanup()
    def test_valid_new_gate(self):self.assertEqual(self.f.store().release_file(self.f.member),self.f.new)
    def test_wrong_archive_pin(self):
        with self.assertRaisesRegex(ValueError,'trust anchor'):Inputs(self.f.archive,self.f.release,self.f.output,'0'*64)
    def test_gate_scope(self):
        self.f.gate['training_enabled']=True;self.f.update()
        with self.assertRaisesRegex(ValueError,'scope'):self.f.store()
    def test_gate_status(self):
        self.f.gate['status']='passed';self.f.update()
        with self.assertRaisesRegex(ValueError,'not accepted'):self.f.store()
    def test_changed_new_bytes(self):
        self.f.new.write_text('{"a":3}\n')
        with self.assertRaisesRegex(ValueError,'Changed new'):self.f.store().release_file(self.f.member)
    def test_wrong_old_binding(self):
        self.f.gate['files'][self.f.member]['reference_sha256']='0'*64;self.f.update()
        with self.assertRaisesRegex(ValueError,'historical binding'):self.f.store().release_file(self.f.member)
    def test_unaccepted_equivalence(self):
        self.f.gate['files'][self.f.member]['equivalent']=False;self.f.update()
        with self.assertRaisesRegex(ValueError,'semantic acceptance'):self.f.store().release_file(self.f.member)
    def test_unpinned_input(self):
        with self.assertRaisesRegex(ValueError,'not covered'):self.f.store().release_file('missing.json')
    def test_archive_changed(self):
        self.f.old.write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'Changed archive'):self.f.store().archived(V2+self.f.member)
    def test_escape(self):
        with self.assertRaisesRegex(ValueError,'escaping'):inside(self.base,'../bad')
    def test_symlink(self):
        (self.f.release/'link').symlink_to(self.f.new)
        with self.assertRaisesRegex(ValueError,'Symlinked'):inside(self.f.release,'link')
    def test_stage_fresh(self):
        store=self.f.store();stages.fresh(store,'x')
        with self.assertRaisesRegex(ValueError,'existing stage'):stages.fresh(store,'x')
    def test_no_test_in_source_builder(self):
        with self.assertRaisesRegex(ValueError,'Source-test'):alg.make_source('test',None,None,None,None,None)
    def test_input_rechecked_before_acceptance(self):
        store=self.f.store();store.release_file(self.f.member);stages.fresh(store,'x')
        self.f.new.write_text('{}\n')
        with self.assertRaisesRegex(ValueError,'Input changed'):store.finish('x',{},'start')


class ComparisonTests(unittest.TestCase):
    def setUp(self):self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def test_gzip_headers_not_data(self):
        a,b=self.base/'a.jsonl.gz',self.base/'b.jsonl.gz'
        for p,mtime in [(a,3),(b,9)]:
            with p.open('wb') as raw:
                with gzip.GzipFile(fileobj=raw,mode='wb',filename='different'+str(mtime),mtime=mtime) as f:f.write(b'{"x":1}\n')
        self.assertNotEqual(sha(a),sha(b));self.assertEqual(semantic(a),semantic(b))
    def test_row_order_matters(self):
        a,b=self.base/'a.jsonl',self.base/'b.jsonl';write_rows(a,[{'x':1},{'x':2}]);write_rows(b,[{'x':2},{'x':1}])
        self.assertNotEqual(semantic(a),semantic(b))
    def test_json_key_order_does_not(self):
        a,b=self.base/'a.json',self.base/'b.json';a.write_text('{"x":1,"y":2}');b.write_text('{"y":2,"x":1}')
        self.assertEqual(semantic(a),semantic(b))
    def test_array_dtype_matters(self):
        a,b=self.base/'a.npz',self.base/'b.npz';np.savez(a,x=np.array([1],dtype=np.int64));np.savez(b,x=np.array([1],dtype=np.int32))
        self.assertNotEqual(semantic(a),semantic(b))
    def test_array_value_matters(self):
        a,b=self.base/'a.npy',self.base/'b.npy';np.save(a,np.array([1]));np.save(b,np.array([2]));self.assertNotEqual(semantic(a),semantic(b))
    def test_no_overwrite(self):
        path=self.base/'x.json';write_json(path,{'a':1})
        with self.assertRaises(FileExistsError):write_json(path,{'a':2})
    def test_report_json_integer_keys_normalized(self):
        self.assertEqual(stages.compare_report({'labels': {0: 2, 1: 3}}, {'labels': {'0': 2, '1': 3}}), 1)
    def test_report_scientific_changes_not_ignored(self):
        with self.assertRaises(ValueError):stages.compare_report({'labels': {0: 2}}, {'labels': {'0': 3}})
    def test_named_arrays_all_compared(self):
        a,b=self.base/'a.npz',self.base/'b.npz';np.savez(a,x=np.array([1]),y=np.array([2]));np.savez(b,x=np.array([1]))
        self.assertNotEqual(semantic(a),semantic(b))


class AlgorithmTests(unittest.TestCase):
    def test_component_order_invariance(self):
        a,b=alg.Graph(),alg.Graph()
        for x,y in [('b','c'),('a','b')]:a.join(x,y)
        for x,y in [('a','b'),('b','c')]:b.join(x,y)
        self.assertEqual({x:a.root(x) for x in ('a','b','c')},{x:b.root(x) for x in ('a','b','c')})
    def test_lexical_normalization(self):self.assertEqual(alg.norm('Ａ_B?! Foo'),'a b foo')
    def test_short_phrase_is_full_pattern(self):self.assertEqual(alg.patterns('a b c'),{'a b c'})
    def test_shuffle_preserves_composition_and_identity_seed(self):
        from collections import Counter
        text='ACDEFGHIKLMNPQRSTVWY'*4
        value,seed=alg.shuffle_record('p',text)
        self.assertEqual(Counter(value),Counter(text));self.assertEqual((value,seed),alg.shuffle_record('p',text))
        self.assertNotEqual(seed,alg.shuffle_record('q',text)[1])
    def test_schedule_visits_each_stream_block_once(self):
        schedule=alg.make_schedule(2,blocks=24,half=4)
        self.assertEqual(schedule.shape,(6,8,2))
        for domain in (0,1):
            a=schedule[:,domain*4:(domain+1)*4]
            self.assertTrue((a[:,:,0]==domain).all());self.assertEqual(sorted(a[:,:,1].ravel().tolist()),list(range(24)))
    def test_schedule_changes_with_seed(self):self.assertFalse(np.array_equal(alg.make_schedule(0,24,4),alg.make_schedule(1,24,4)))
    def test_packing_uses_independent_caps(self):
        x,m=alg.pack([5]*300,[6]*300)
        self.assertEqual(x.tolist(),[5]*255+[2]+[6]*255+[1]);self.assertEqual(int(m.sum()),512)
    def test_packing_rejects_special_content(self):
        with self.assertRaises(ValueError):alg.pack([0,5],[6])
    def test_boolean_labels_rejected(self):
        with self.assertRaises(ValueError):alg.validate_label(True)
    def test_group_sampling_preserves_source_order_and_groups(self):
        values=[{'g':str(i//4),'n':i} for i in range(40)]
        selected,report=alg.select_groups(values,'g',13,20260925,'fixture')
        self.assertEqual(len(selected),12);self.assertEqual(selected,sorted(selected,key=lambda x:x['n']))
        self.assertTrue(all(sum(r['g']==g for r in selected)==4 for g in {r['g'] for r in selected}))
    def test_stream_final_eos_and_prefix(self):
        class Tok:
            def decode(self,ids,skip_special_tokens=False):return 'A'*len(ids)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);w=alg.StreamWriter(root,root/'streams','x',Tok(),budget=7)
            w.append('a',0,'AAAA',[4]*4);w.append('b',1,'AAAA',[5]*4);info=w.finish()
            self.assertEqual(np.fromfile(root/info['bin'],dtype='<u2').tolist(),[4,4,4,4,1,5,1])
            self.assertTrue(w.index_rows[-1]['truncated'])
    def test_m1_core_progress_and_output(self):
        from derived import confirmation
        from types import SimpleNamespace
        class Tok:
            def get_vocab_size(self):return 32000
            def token_to_id(self,s):return ['<|pad_v2|>','<|eos_v2|>','<|pair_v2|>','<|unk_v2|>'].index(s)
            def encode_batch(self,values,add_special_tokens=False):return [SimpleNamespace(ids=[ord(c)+4 for c in x]) for x in values]
            def decode(self,ids,skip_special_tokens=False):return ''.join(chr(i-4) for i in ids)
        candidate=[{'idx':1,'question1':'alpha','question2':'beta','label':1}]
        table=SimpleNamespace(to_pylist=lambda:candidate)
        refs={'files':{'tokenizers/mixed_bpe/tokenizer.json':{'path':'fixture-only'}}}
        docs=lambda refs: (('fixture',str(i),'unrelated text') for i in range(25000))
        with tempfile.TemporaryDirectory() as tmp, patch.object(confirmation.pq,'read_table',return_value=table), patch.object(confirmation.Tokenizer,'from_file',return_value=Tok()), patch('builtins.print'):
            report=confirmation.construct(Path(tmp),Path('unused'),{'selected_artifact':{'expected_rows':1}},refs,docs)
            self.assertEqual(report['retained_rows'],1)
            self.assertEqual(report['reference_documents'],{'fixture':25000})
    def test_target_alignment_rejects_reorder(self):
        with self.assertRaises(ValueError):alg.check_target_alignment({'id':'x','label':0,'component_id':'g'},{'id':'y'},[],[])


if __name__=='__main__':unittest.main()
