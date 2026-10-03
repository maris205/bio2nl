"""CPU integration checks for offline orchestration and immutable phase barriers."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('corpus_phase_under_test', HERE / 'corpus_phase.py')
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


class CorpusPhaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.top = Path(self.temp.name)
        self.root = self.top / 'release'
        self.root.mkdir()
        self.logs = self.top / 'logs'
        self.smallweb = self.top / 'adapter.py'
        self.smallweb.write_text('print("adapter fixture")\n')

    def stages(self):
        return c.plan(self.root, 'initial', self.smallweb)

    def fake_initial(self, fail_stage=None):
        files, stages = {}, []
        names = [s['name'] for s in self.stages()]
        for name in names:
            relative = 'code/' + name + '.py'
            script = self.root / relative
            script.parent.mkdir(exist_ok=True)
            script.write_text('import json,os,pathlib,sys\n'
                              'root=pathlib.Path(sys.argv[1])\n'
                              + ('sys.exit(3)\n' if name == fail_stage else '') +
                              'path=root/"products"/' + repr(name + '.json') + '\n'
                              'path.parent.mkdir(exist_ok=True)\n'
                              'path.write_text(json.dumps({k:os.environ.get(k) for k in '
                              '["CUDA_VISIBLE_DEVICES","HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","PYTHONPATH","PYTHONSTARTUP"]}))\n')
            files[relative] = {'sha256': c.sha(script)}
            stages.append({'name': name, 'script': str(script),
                           'command': [sys.executable, '-B', str(script), str(self.root)],
                           'outputs': ['products/' + name + '.json'], 'output_dirs': []})
        directory = self.root / 'portable_validation'
        directory.mkdir()
        bootstrap = directory / 'bootstrap.json'
        bootstrap.write_text(json.dumps({'status': 'raw_assets_seeded_not_a_data_acceptance',
                                        'output_root': str(self.root), 'files': files}))
        self.bootstrap_sha = c.sha(bootstrap)
        return stages

    def run_fake(self, stages):
        argv = ['corpus_phase', '--release-root', str(self.root), '--phase', 'initial',
                '--log-root', str(self.logs), '--bootstrap-sha256', self.bootstrap_sha,
                '--smallweb-script', str(self.smallweb), '--smallweb-sha256', c.sha(self.smallweb)]
        with patch.object(sys, 'argv', argv), patch.dict(os.environ, {
            'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': '/untrusted', 'PYTHONSTARTUP': '/untrusted/startup'}), \
                patch.object(c, 'plan', return_value=stages), \
                patch.object(c.shutil, 'disk_usage', return_value=SimpleNamespace(free=12 << 30)):
            c.main()

    def test_verification_report_name_matches_frozen_builder(self):
        stage = next(s for s in self.stages() if s['name'] == 'nlp_independent_verify')
        frozen = HERE.parents[0] / 'data/rebuild_v2/nlp_synthetic/verify.py'
        self.assertEqual(stage['outputs'], ['validation/nlp_synthetic_independent_verification.json'])
        self.assertIn(stage['outputs'][0], frozen.read_text())
        self.assertIn('--offline', self.stages()[0]['command'])

    def test_existing_corpus_is_refused_even_without_metadata_report(self):
        path = self.root / 'data/corpora/english/train.jsonl.gz'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'preserve existing corpus')
        with self.assertRaisesRegex(ValueError, 'existing stage output directory'):
            c.require_new_outputs(self.root, self.stages())
        self.assertEqual(path.read_bytes(), b'preserve existing corpus')

    def test_existing_config_is_not_exception_to_overwrite_guard(self):
        path = self.root / 'configs/tokenization.json'
        path.parent.mkdir()
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'existing stage output'):
            c.require_new_outputs(self.root, c.plan(self.root, 'dependent', self.smallweb))

    def test_symlink_root_output_and_nested_member_are_refused(self):
        alias = self.top / 'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            c.checked(alias)
        (self.root / 'metadata').symlink_to(self.top, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            c.member(self.root, 'metadata/future.json')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            c.member(self.root, '../outside')

    def test_new_directory_files_cannot_hide_symlink(self):
        folder = self.root / 'products'
        folder.mkdir()
        (folder / 'linked').symlink_to(self.smallweb)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            c.tree_files(self.root, 'products')

    def test_real_subprocesses_offline_and_final_status_marker_matches(self):
        stages = self.fake_initial()
        self.run_fake(stages)
        marker = c.verify_initial(self.root, self.bootstrap_sha)
        self.assertEqual(marker['status_sha256'], c.sha(self.logs / 'status.json'))
        status = json.loads((self.logs / 'status.json').read_text())
        self.assertEqual(status['status'], 'completed')
        self.assertEqual(len(status['stages']), 7)
        env = json.loads((self.root / stages[0]['outputs'][0]).read_text())
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')
        self.assertEqual(env['TRANSFORMERS_OFFLINE'], '1')
        self.assertIsNone(env['PYTHONPATH'])
        self.assertIsNone(env['PYTHONSTARTUP'])

    def test_first_failed_worker_stops_queue_without_completed_marker(self):
        stages = self.fake_initial(fail_stage='nlp_independent_verify')
        with self.assertRaisesRegex(ValueError, 'Stage failed'):
            self.run_fake(stages)
        status = json.loads((self.logs / 'status.json').read_text())
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(len(status['stages']), 2)
        self.assertEqual(status['stages'][-1]['exit_code'], 3)
        self.assertFalse((self.root / '.corpus_initial_completed.json').exists())
        self.assertFalse((self.root / 'products/longrange.json').exists())

    def test_dependent_barrier_refuses_modified_final_status(self):
        self.run_fake(self.fake_initial())
        path = self.logs / 'status.json'
        path.write_text(path.read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'status changed'):
            c.verify_initial(self.root, self.bootstrap_sha)

    def test_dependent_barrier_rehashes_produced_rows_not_just_reports(self):
        self.run_fake(self.fake_initial())
        (self.root / 'products/english.json').write_text('altered')
        with self.assertRaisesRegex(ValueError, 'output changed'):
            c.verify_initial(self.root, self.bootstrap_sha)

    def test_dependent_barrier_refuses_wrong_bootstrap(self):
        self.run_fake(self.fake_initial())
        with self.assertRaisesRegex(ValueError, 'bound to this bootstrap'):
            c.verify_initial(self.root, '0' * 64)


if __name__ == '__main__':
    unittest.main()
