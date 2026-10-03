"""Boundary tests: these exercise rejection and invariants, not copied outputs."""
import copy
import hashlib
import tempfile
from pathlib import Path
import unittest

import portable_prepare as pp


class PreparationBoundaries(unittest.TestCase):
    def setUp(self):
        self.lock = pp.read_json(pp.HERE / 'resource_lock.json')

    def test_original_lock_is_complete(self):
        self.assertEqual(len(pp.validate_lock(self.lock)['raw_members']), 18)

    def test_added_test_examples_are_rejected(self):
        lock = copy.deepcopy(self.lock)
        lock['raw_members']['data/pairs/protein_sequence_similarity_test.tsv.gz'] = next(iter(lock['raw_members'].values()))
        with self.assertRaisesRegex(ValueError, 'allowlist'):
            pp.validate_lock(lock)

    def test_changed_selection_seed_is_rejected(self):
        lock = copy.deepcopy(self.lock)
        lock['data']['source']['selection_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'design'):
            pp.validate_lock(lock)

    def test_changed_token_budget_is_rejected(self):
        lock = copy.deepcopy(self.lock)
        lock['data']['token_budget_per_stream'] += 512
        with self.assertRaisesRegex(ValueError, 'design'):
            pp.validate_lock(lock)

    def test_malformed_digest_is_rejected(self):
        lock = copy.deepcopy(self.lock)
        next(iter(lock['raw_members'].values()))['sha256'] = 'g' * 64
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            pp.validate_lock(lock)

    def test_build_cannot_grant_training(self):
        lock = copy.deepcopy(self.lock)
        lock['training_enabled'] = True
        with self.assertRaisesRegex(ValueError, 'scope'):
            pp.validate_lock(lock)

    def test_relative_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                pp.inside(Path(d), '../outside')

    def test_symlink_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / 'regular').mkdir(); (root / 'alias').symlink_to(root / 'regular', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'Symlink'):
                pp.inside(root, 'alias/data')

    def test_changed_raw_byte_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); value = b'original'; (root / 'raw').write_bytes(b'changed!')
            lock = {'raw_members': {'raw': {'sha256': hashlib.sha256(value).hexdigest(), 'bytes': len(value)}}}
            with self.assertRaisesRegex(ValueError, 'identity'):
                pp.verify_inputs(root, lock)

    def test_duplicate_json_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'metadata.json'; path.write_text('{"status": "passed", "status": "failed"}')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                pp.read_json(path)

    def test_nonfinite_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'metadata.json'; path.write_text('{"x": NaN}')
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                pp.read_json(path)

    def test_new_output_write_is_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'gate.json'; pp.write_json(path, {'status': 'old'})
            with self.assertRaises(FileExistsError):
                pp.write_json(path, {'status': 'new'})
            self.assertEqual(pp.read_json(path)['status'], 'old')


if __name__ == '__main__':
    unittest.main()
