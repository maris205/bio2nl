import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE.parent))
from new_data_training_v1 import common, algorithms as alg, data_build


def spec(path):
    return {'path': str(path), 'sha256': common.sha(path), 'bytes': path.stat().st_size}


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    raw = root / 'release'; out = root / 'prepared'
    entries = []
    accepted = {}
    inputs = {}
    for member in common.RAW_MEMBERS:
        path = raw / member; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('new artifact: ' + member)
        value = spec(path)
        entries.append({'file': member, 'sha256': value['sha256'], 'bytes': value['bytes']})
        accepted[member] = {**entries[-1], 'equivalent_to_declared_reference': True}
        inputs['raw:' + member] = value
    for name, member in common.METADATA_NAMES.items():
        path = root / ('design' if name == 'design' else 'confirmation') / member
        dump(path, {'fixture': name}); inputs[name] = spec(path)
    gate = {'status': 'accepted_new_raw_release_training_disabled', 'release_id': '2026-10-01-portable-v2',
            'training_enabled': False, 'training_gate_pass': False, 'target_scoring_enabled': False,
            'files': accepted}
    acceptance = raw / 'portable_validation/acceptance.json'; dump(acceptance, gate)
    manifest = raw / 'manifest.json'
    dump(manifest, {**gate, 'files': entries, 'acceptance': {'sha256': common.sha(acceptance)}})
    protocol = {'schema_version': 1, 'training_enabled': False, 'target_scoring_enabled': False,
                'source_test_examples_allowed': False, 'raw_release_root': str(raw), 'output_root': str(out),
                'release_id': gate['release_id'], 'raw_acceptance': spec(acceptance), 'raw_manifest': spec(manifest),
                'inputs': inputs, 'code_pins': {str(PACKAGE / 'common.py'): {k: v for k, v in spec(PACKAGE / 'common.py').items() if k != 'path'}}}
    path = root / 'protocol.json'; dump(path, protocol)
    return protocol, path, out


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.protocol, self.path, self.out = fixture(self.root)
    def tearDown(self):
        self.tmp.cleanup()
    def store(self):
        return common.Store(self.path, common.sha(self.path), self.out)
    def rewrite(self):
        dump(self.path, self.protocol)
    def test_valid_gate_keeps_training_disabled(self):
        store = self.store()
        self.assertFalse(store.acceptance['training_gate_pass'])
        self.assertTrue(self.out.is_dir())
    def test_wrong_protocol_hash_rejected(self):
        with self.assertRaisesRegex(ValueError, 'protocol hash'):
            common.Store(self.path, '0' * 64, self.out)
    def test_existing_output_rejected(self):
        self.out.mkdir()
        with self.assertRaisesRegex(ValueError, 'Fresh output'):
            self.store()
    def test_old_gate_status_rejected(self):
        p = Path(self.protocol['raw_acceptance']['path']); v = common.read_json(p)
        v['status'] = 'accepted_for_derived_reconstruction'; dump(p, v)
        self.protocol['raw_acceptance'] = spec(p); self.rewrite()
        with self.assertRaisesRegex(ValueError, 'Raw release not accepted'):
            self.store()
    def test_training_permission_rejected(self):
        self.protocol['training_enabled'] = True; self.rewrite()
        with self.assertRaisesRegex(ValueError, 'disable training'):
            self.store()
    def test_extra_heldout_input_rejected(self):
        self.protocol['inputs']['raw:data/pairs/protein_sequence_similarity_test.tsv.gz'] = {}
        self.rewrite()
        with self.assertRaisesRegex(ValueError, 'allowlist'):
            self.store()
    def test_missing_required_input_rejected(self):
        self.protocol['inputs'].pop('confirmation_references'); self.rewrite()
        with self.assertRaisesRegex(ValueError, 'allowlist'):
            self.store()
    def test_raw_source_test_read_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not permitted'):
            self.store().raw('data/pairs/protein_sequence_similarity_test.tsv.gz')
    def test_raw_tamper_rejected(self):
        store = self.store(); member = common.RAW_MEMBERS[0]
        Path(self.protocol['inputs']['raw:' + member]['path']).write_text('mutated')
        with self.assertRaisesRegex(ValueError, 'hash/size changed'):
            store.raw(member)
    def test_raw_symlink_rejected(self):
        store = self.store(); member = common.RAW_MEMBERS[0]
        path = Path(self.protocol['inputs']['raw:' + member]['path'])
        relocated = self.root / 'moved'; path.rename(relocated); path.symlink_to(relocated)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            store.raw(member)
    def test_raw_spec_cannot_escape_release(self):
        member = common.RAW_MEMBERS[0]
        self.protocol['inputs']['raw:' + member]['path'] = str(self.root / 'wrong')
        self.rewrite()
        with self.assertRaisesRegex(ValueError, 'Raw path differs'):
            self.store().raw(member)
    def test_target_metadata_alias_rejected(self):
        self.protocol['inputs']['confirmation_references']['path'] = str(self.root / 'confirmation/data/references.json')
        self.rewrite()
        with self.assertRaisesRegex(ValueError, 'Example/model'):
            self.store().metadata('confirmation_references')
    def test_complete_readset_rehashes(self):
        store = self.store()
        for member in common.RAW_MEMBERS: store.raw(member)
        for name in common.METADATA_NAMES: store.metadata(name)
        store.finish_readset()
        Path(self.protocol['inputs']['design']['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash/size changed'):
            store.finish_readset()
    def test_incomplete_readset_rejected(self):
        with self.assertRaisesRegex(ValueError, 'not verified/read'):
            self.store().finish_readset()
    def test_raw_gate_manifest_link_required(self):
        path = Path(self.protocol['raw_manifest']['path']); value = common.read_json(path)
        value['acceptance']['sha256'] = '0' * 64; dump(path, value)
        self.protocol['raw_manifest'] = spec(path); self.rewrite()
        with self.assertRaisesRegex(ValueError, 'acceptance binding'):
            self.store()


class AlgorithmTests(unittest.TestCase):
    def test_whole_groups_nearest_prefix_not_row_truncation(self):
        values = [{'block_id': 'one', 'n': n} for n in range(6)]
        values += [{'block_id': 'two', 'n': n} for n in range(4)]
        chosen, report = alg.select_groups(values, 'block_id', 5, 20260925, 'protein_sequence_similarity')
        self.assertIn(len(chosen), (4, 6))
        self.assertEqual(len({r['block_id'] for r in chosen}), 1)
        self.assertTrue(report['whole_groups'])
    def test_selection_order_independent_of_group_insertion(self):
        values = [{'block_id': str(i), 'n': j} for i in range(10) for j in range(i + 2)]
        left, _ = alg.select_groups(values, 'block_id', 30, 20260925, 'protein_sequence_similarity')
        right, _ = alg.select_groups(list(reversed(values)), 'block_id', 30, 20260925, 'protein_sequence_similarity')
        self.assertEqual({r['block_id'] for r in left}, {r['block_id'] for r in right})
    def test_schedule_exact_single_pass_both_streams(self):
        value = alg.make_schedule(0)
        self.assertEqual(value.shape, (2048,16,2))
        for stream in (0,1):
            selected = value[value[:,:,0] == stream, 1]
            np.testing.assert_array_equal(np.sort(selected), np.arange(16384))
        np.testing.assert_array_equal(value, alg.make_schedule(0))
        self.assertFalse(np.array_equal(value, alg.make_schedule(1)))
    def test_shuffle_deterministic_composition(self):
        text = 'ACDEFGHIKLMNPQRSTVWY' * 3
        left, seed = alg.shuffle_record('test', text)
        right, other_seed = alg.shuffle_record('test', text)
        self.assertEqual((left, seed), (right, other_seed))
        self.assertEqual(sorted(left), sorted(text))
    def test_source_test_input_explicitly_forbidden(self):
        with self.assertRaisesRegex(ValueError, 'Source-test'):
            data_build.read_raw_source(Path('/no/such/file'), 'test')
        with self.assertRaisesRegex(ValueError, 'Source-test'):
            alg.make_source('test', None, None, None, None, None)
    def test_fixed_new_selection_not_old_count(self):
        design = {'pretraining': {'seeds':[0,1,2], 'input_tokens_per_run':2*alg.BUDGET, 'context_length':512, 'updates_per_run':1024}}
        data = {'seeds':[0,1,2], 'token_budget_per_stream':alg.BUDGET, 'source': {'target_rows':8000, 'selection_seed':20260925, 'expected_train_rows':8044, 'expected_train_groups':63, 'expected_validation_rows':20276, 'expected_raw_train_rows':99818}}
        data_build.check_design(design, data)
        data['source']['expected_train_rows'] = 8002
        with self.assertRaisesRegex(ValueError, 'new source selection'):
            data_build.check_design(design, data)
    def test_source_encoding_roundtrip_role_balance(self):
        class Encoded:
            def __init__(self, ids): self.ids=ids
        class Tokenizer:
            def encode(self, text, add_special_tokens=False): return Encoded([ord(c) for c in text])
            def decode(self, ids, skip_special_tokens=False): return ''.join(chr(i) for i in ids)
        texts_a = ['A'*40, 'C'*40, 'D'*40]; texts_b = ['E'*40, 'F'*40, 'G'*40]
        values=[]; canonical={}
        for i in range(3):
            for label, j in [(1,i),(0,(i+1)%3)]:
                a,b = texts_a[i],texts_b[j]
                metadata={'split':'train','block_id':'b','sequence_sha256_a':common.text_sha(a),'sequence_sha256_b':common.text_sha(b),'cluster_a':'a'+str(i),'cluster_b':'b'+str(j)}
                values.append({'row_id':f'{i}:{label}','label':label,'sentence1':a,'sentence2':b,'metadata':metadata})
                for side,text in [('a',a),('b',b)]: canonical[common.text_sha(text)]={'split':'train','cluster_id':metadata['cluster_'+side]}
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); common.write_rows(base/'raw.jsonl',values)
            report, ids=alg.make_source('train',base/'raw.jsonl',Tokenizer(),canonical,base/'source',base)
            self.assertEqual(report['count'],6); self.assertTrue(report['within_block_role_label_balance'])
            with np.load(base/report['npz']) as arr:
                self.assertEqual(arr['input_ids'].shape,(6,512))
                np.testing.assert_array_equal(arr['labels'],[1,0,1,0,1,0])
                self.assertEqual(arr['attention_mask'].sum(),6*82)
            self.assertEqual(len(ids['inputs']),6)


class InheritanceTests(unittest.TestCase):
    def setUp(self):
        class FakeStore:
            def metadata(self, key): return self.values[key]
        self.store = FakeStore()
        self.store.values = {
            'confirmation_acceptance': {'status':'accepted_for_local_confirmation','confirmation_data_ready':True,
                'qualification_scope':'historical lexical qualification',
                'files':{'manifest.json':'manifest','references.json':'references','verification/independent_verification.json':'audit'}},
            'confirmation_manifest': {'inputs':{}}, 'confirmation_references': {'files':{}},
            'confirmation_audit': {'status':'passed','confirmation_data_qualified':True,'file_sha256':{}}}
        self.store.inputs = {key:{'sha256':digest} for key,digest in [('confirmation_acceptance','acceptance'),('confirmation_manifest','manifest'),('confirmation_references','references'),('confirmation_audit','audit')]}
        self.store.acceptance = {'files':{}}
        for member in ('data/corpora/english/train.jsonl.gz','tokenizers/mixed_bpe/tokenizer.json'):
            path='/old/'+member
            self.store.values['confirmation_references']['files'][member]={'path':path,'sha256':'old'}
            self.store.values['confirmation_manifest']['inputs'][path]='old'
            self.store.values['confirmation_audit']['file_sha256'][path]='old'
            self.store.acceptance['files'][member]={'reference_kind':'accepted_v2','equivalent_to_declared_reference':True,'reference_sha256':'old','sha256':'new','comparison_mode':'scientific_equivalence'}
            self.store.inputs['raw:'+member]={'sha256':'new'}
    def test_metadata_only_equivalence_not_byte_identity(self):
        result=data_build.inherit_confirmation(self.store)
        self.assertTrue(result['historical_target_has_already_been_scored'])
        self.assertFalse(result['new_blind_confirmation_claimed'])
        self.assertFalse(result['target_raw_or_encoded_or_row_ledger_files_opened'])
    def test_stale_metadata_reference_rejected(self):
        self.store.values['confirmation_manifest']['inputs']['/old/data/corpora/english/train.jsonl.gz']='different'
        with self.assertRaisesRegex(ValueError,'Exposure reference identity'):
            data_build.inherit_confirmation(self.store)
    def test_unaccepted_current_reference_rejected(self):
        self.store.acceptance['files']['data/corpora/english/train.jsonl.gz']['equivalent_to_declared_reference']=False
        with self.assertRaisesRegex(ValueError,'semantic equivalence'):
            data_build.inherit_confirmation(self.store)
    def test_metadata_sha_not_accepted_rejected(self):
        self.store.inputs['confirmation_manifest']['sha256']='bad'
        with self.assertRaisesRegex(ValueError,'Qualification metadata binding'):
            data_build.inherit_confirmation(self.store)


class BoundaryTests(unittest.TestCase):
    def test_build_requires_cpu_environment_before_any_read(self):
        with mock.patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}):
            with self.assertRaisesRegex(ValueError, 'CUDA_VISIBLE_DEVICES empty'):
                data_build.build('/no/protocol', '0'*64, '/no/output')

    def test_duplicate_json_keys_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.json'; path.write_text('{"x":1,"x":2}')
            with self.assertRaisesRegex(ValueError,'Duplicate JSON'):
                common.read_json(path)
    def test_exclusive_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'out.json'; common.write_json(path,{})
            with self.assertRaises(FileExistsError): common.write_json(path,{})
    def test_copied_cli_imports_from_empty_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); package=root/'copied'; package.mkdir(); cwd=root/'empty'; cwd.mkdir()
            for name in ('common.py','algorithms.py','data_build.py'):
                (package/name).write_bytes((PACKAGE/name).read_bytes())
            result=subprocess.run([sys.executable,'-I','-B',str(package/'data_build.py'),'--help'],cwd=cwd,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--protocol-sha256',result.stdout)
    def test_copied_algorithm_bodies_match_pinned_source(self):
        import ast
        metadata=common.read_json(PACKAGE/'algorithm_sources.json')
        source=PACKAGE.parent.parent/metadata['source']
        # Resolve repository-relative evidence without importing its implementation.
        self.assertEqual(common.sha(source),metadata['source_sha256'])
        old={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        new={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((PACKAGE/'algorithms.py').read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        self.assertEqual(set(new),set(metadata['copied_symbols']))
        for name in new: self.assertEqual(new[name],old[name],name)


if __name__ == '__main__':
    unittest.main()
