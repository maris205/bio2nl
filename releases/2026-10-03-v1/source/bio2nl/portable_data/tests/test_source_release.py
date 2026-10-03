import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('source_release', Path(__file__).parents[1] / 'source_release.py')
module = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(module)


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / 'workspace'; self.workspace.mkdir()
        records = {}
        for name in ('bio2nl', 'biopaws'):
            repo = self.workspace / name; repo.mkdir()
            self.git(repo, 'init', '-q'); self.git(repo, 'config', 'user.name', 'Test')
            self.git(repo, 'config', 'user.email', 'test@example.invalid')
            (repo / 'private_data.txt').write_text('must never be exported')
            (repo / 'paper.txt').write_text('historical paper')
            self.git(repo, 'add', 'private_data.txt', 'paper.txt'); self.git(repo, 'commit', '-qm', 'fixture')
            head = self.git(repo, 'rev-parse', 'HEAD').strip()
            (repo / 'paper.txt').write_text('user staged paper')
            self.git(repo, 'add', 'paper.txt')
            (repo / 'paper.txt').write_text('user newer unstaged paper')
            (repo / 'portable.py').write_text('print("portable")\n')
            data = (repo / 'portable.py').read_bytes()
            records[name] = {'observed_checkout_head': head, 'file_count': 1,
                'files': [{'path': 'portable.py', 'sha256': module.sha(data), 'bytes': len(data)}]}
        self.inventory = self.root / 'inventory.json'
        self.inventory.write_text(json.dumps({'status': 'uncommitted_candidates', 'missing_required_scopes': [],
            'raw_data_included': False, 'model_weights_included': False, 'repositories': records}))
        self.inventory_sha = module.sha(self.inventory.read_bytes())
        self.output = self.root / 'export'

    def git(self, repo, *args):
        return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE).decode()

    def run_capture(self):
        return module.capture(self.workspace, self.inventory, self.inventory_sha, self.output, 'release/test')

    def test_explicit_export_preserves_staged_and_unstaged_changes(self):
        before = {n: module.working_state(self.workspace / n) for n in ('bio2nl', 'biopaws')}
        result = self.run_capture()
        self.assertEqual(result['file_count'], 2)
        self.assertEqual(module.sha((self.output / 'candidate_inventory.json').read_bytes()), self.inventory_sha)
        self.assertFalse(result['remote_publication_performed'])
        for name, entry in result['repositories'].items():
            self.assertEqual(module.working_state(self.workspace / name), before[name])
            self.assertEqual(self.git(self.workspace / name, 'show', entry['git_commit'] + ':paper.txt'), 'historical paper')
            self.assertEqual(sorted(p.name for p in (self.output / 'code' / name).iterdir()), ['portable.py'])
            self.assertEqual(self.git(self.workspace / name, 'rev-parse', 'release/test').strip(), entry['git_commit'])

    def test_changed_candidate_fails_before_export_or_ref(self):
        (self.workspace / 'bio2nl/portable.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Candidate bytes changed'):
            self.run_capture()
        self.assertFalse(self.output.exists())

    def test_existing_branch_never_overwritten(self):
        self.git(self.workspace / 'bio2nl', 'branch', 'release/test')
        with self.assertRaisesRegex(ValueError, 'existing branch'):
            self.run_capture()
        self.assertFalse(self.output.exists())

    def test_symlink_candidate_rejected(self):
        path = self.workspace / 'bio2nl/portable.py'
        data = path.read_bytes(); path.unlink(); target = self.root / 'elsewhere.py'; target.write_bytes(data)
        path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, 'Symlinked'):
            self.run_capture()

    def test_inventory_digest_must_match(self):
        self.inventory.write_text(self.inventory.read_text() + '\n')
        with self.assertRaisesRegex(ValueError, 'inventory changed'):
            self.run_capture()

    def test_inherited_git_redirection_is_ignored(self):
        with patch.dict('os.environ', {'GIT_DIR': str(self.root / 'wrong.git'),
                                      'GIT_INDEX_FILE': str(self.root / 'wrong-index')}):
            result = self.run_capture()
        self.assertEqual(result['file_count'], 2)
        self.assertFalse((self.root / 'wrong-index').exists())


if __name__ == '__main__':
    unittest.main()
