import gzip
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location('verify_release', HERE / 'verify_release.py')
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)


class ContentComparisonTests(unittest.TestCase):
    def compare(self, a, b, name):
        return v.compare_content(a, b, name, v.sha(a), v.sha(b))

    def test_json_keys_may_reorder_but_arrays_and_numbers_may_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            a.write_text('{"vocab":{"a":0,"b":1},"merges":[["a","b"],["b","a"]]}')
            b.write_text('{"merges":[["a","b"],["b","a"]],"vocab":{"b":1,"a":0}}')
            self.assertTrue(self.compare(a, b, 'tokenizers/mixed_bpe/tokenizer.json')[0])
            b.write_text('{"vocab":{"a":0,"b":1},"merges":[["b","a"],["a","b"]]}')
            self.assertFalse(self.compare(a, b, 'tokenizers/mixed_bpe/tokenizer.json')[0])
            b.write_text('{"vocab":{"a":1,"b":0},"merges":[["a","b"],["b","a"]]}')
            self.assertFalse(self.compare(a, b, 'tokenizers/mixed_bpe/tokenizer.json')[0])
            a.write_text('{"x":1}'); b.write_text('{"x":1.0}')
            self.assertFalse(self.compare(a, b, 'configs/test.json')[0])

    def test_duplicate_json_keys_and_nonfinite_values_rejected(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError): v.loads(text)

    def test_gzip_headers_may_differ_but_ordered_rows_must_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            content = b'{"id":1,"label":0}\n{"id":2,"label":1}\n'
            a.write_bytes(gzip.compress(content, mtime=1)); b.write_bytes(gzip.compress(content, mtime=2))
            self.assertNotEqual(v.sha(a), v.sha(b))
            self.assertTrue(self.compare(a, b, 'data/corpora/protein/train.jsonl.gz')[0])
            b.write_bytes(gzip.compress(b'{"id":2,"label":1}\n{"id":1,"label":0}\n', mtime=2))
            equal, _, detail = self.compare(a, b, 'data/corpora/protein/train.jsonl.gz')
            self.assertFalse(equal); self.assertEqual(detail['first_different_row'], 1)
            b.write_bytes(gzip.compress(b'{"id":1,"label":0}\n', mtime=2))
            self.assertFalse(self.compare(a, b, 'data/corpora/protein/train.jsonl.gz')[0])

    def test_tsv_gzip_compares_all_decompressed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            a.write_bytes(gzip.compress(b'id\tlabel\na\t0\nb\t1\n', mtime=1))
            b.write_bytes(gzip.compress(b'id\tlabel\na\t0\nb\t1\n', mtime=2))
            self.assertTrue(self.compare(a, b, 'data/pairs/test.tsv.gz')[0])
            b.write_bytes(gzip.compress(b'id\tlabel\na\t1\nb\t0\n', mtime=2))
            self.assertFalse(self.compare(a, b, 'data/pairs/test.tsv.gz')[0])

    def test_parquet_requires_schema_dtype_and_ordered_rows(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            table = pa.table({'id': pa.array([1, 2], type=pa.int32()), 'seq': ['AAA', 'CCC']})
            pq.write_table(table, a)
            pq.write_table(table.replace_schema_metadata({'writer': 'new'}), b)
            self.assertTrue(self.compare(a, b, 'data/sequences/test.parquet')[0])
            pq.write_table(pa.table({'id': pa.array([1, 2], type=pa.int64()), 'seq': ['AAA', 'CCC']}), b)
            self.assertFalse(self.compare(a, b, 'data/sequences/test.parquet')[0])
            pq.write_table(table.take(pa.array([1, 0])), b)
            self.assertFalse(self.compare(a, b, 'data/sequences/test.parquet')[0])

    def test_binary_integer_exact_and_raw_assets_byte_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            a.write_bytes(b'\x01\x00\x04\x00'); b.write_bytes(a.read_bytes())
            self.assertTrue(self.compare(a, b, 'tokenized/a.bin')[0])
            b.write_bytes(b'\x01\x00\x05\x00')
            self.assertFalse(self.compare(a, b, 'tokenized/a.bin')[0])
            b.write_bytes(b'\x01')
            self.assertFalse(self.compare(a, b, 'tokenized/a.bin')[0])
            a.write_text('{"x":1}'); b.write_text('{ "x": 1 }')
            self.assertFalse(self.compare(a, b, 'tokenizers/gpt2_pinned/tokenizer.json')[0])


class MetadataNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path('/new/release')
        self.old_sha, self.new_sha = 'a' * 64, 'b' * 64
        self.bindings = {'data/a.gz': {'equivalent': True, 'reference_sha256': self.old_sha,
                         'sha256': self.new_sha, 'reference_bytes': 10, 'bytes': 12}}

    def normalize(self, value, name, side, bindings=None, entries=None):
        return v.normalize_metadata(value, name, side, self.root,
                    self.bindings if bindings is None else bindings, entries or {}, [])

    def test_verified_hash_and_descriptor_size_mapping_preserves_scientific_fields(self):
        old = {'file': 'data/a.gz', 'sha256': self.old_sha, 'bytes': 10, 'records': 3}
        new = {'file': 'data/a.gz', 'sha256': self.new_sha, 'bytes': 12, 'records': 3}
        self.assertEqual(self.normalize(old, 'metadata/test.json', 'reference'), self.normalize(new, 'metadata/test.json', 'current'))
        new['records'] = 4
        self.assertNotEqual(self.normalize(old, 'metadata/test.json', 'reference'), self.normalize(new, 'metadata/test.json', 'current'))
        new['bytes'] = 10
        with self.assertRaises(ValueError): self.normalize(new, 'metadata/test.json', 'current')

    def test_stale_or_unverified_hash_is_not_authorized(self):
        old = {'source_sha256': self.old_sha}
        stale = {'source_sha256': self.old_sha}
        self.assertNotEqual(self.normalize(old, 'metadata/test.json', 'reference'), self.normalize(stale, 'metadata/test.json', 'current'))
        failed = {'data/a.gz': {**self.bindings['data/a.gz'], 'equivalent': False}}
        current = {'source_sha256': self.new_sha}
        self.assertNotEqual(self.normalize(old, 'metadata/test.json', 'reference', failed), self.normalize(current, 'metadata/test.json', 'current', failed))
        # Digest-looking scientific values are not normalized merely by pattern.
        self.assertNotEqual(self.normalize({'row_id': self.old_sha}, 'metadata/test.json', 'reference'),
                            self.normalize({'row_id': self.new_sha}, 'metadata/test.json', 'current'))

    def test_only_explicit_timestamp_paths_are_relocatable(self):
        old = {'created_utc': '2026-09-25T00:00:00+00:00', 'source_records': 10}
        new = {'created_utc': '2026-10-01T00:00:00+00:00', 'source_records': 10}
        self.assertEqual(self.normalize(old, 'metadata/protein_sources.json', 'reference'), self.normalize(new, 'metadata/protein_sources.json', 'current'))
        self.assertNotEqual(self.normalize(old, 'metadata/other.json', 'reference'), self.normalize(new, 'metadata/other.json', 'current'))
        new['created_utc'] = '2026-10-01T00:00:00'
        with self.assertRaises(ValueError): self.normalize(new, 'metadata/protein_sources.json', 'current')

    def test_explicit_argv_root_mapping_retains_thresholds(self):
        old = {'argv': [v.ORIGINAL_RELEASE + 'a', '--min-seq-id', '.3'], 'seconds': 1, 'utc_completed': '2026-09-25T00:00:00+00:00'}
        new = {'argv': ['/new/release/a', '--min-seq-id', '.3'], 'seconds': 2, 'utc_completed': '2026-10-01T00:00:00+00:00'}
        name = 'metadata/protein_full_cluster_command.json'
        self.assertEqual(self.normalize(old, name, 'reference'), self.normalize(new, name, 'current'))
        new['argv'][-1] = '.4'
        self.assertNotEqual(self.normalize(old, name, 'reference'), self.normalize(new, name, 'current'))

    def test_current_metadata_cannot_keep_old_release_or_builder_paths(self):
        command = {'argv': [v.ORIGINAL_RELEASE + 'a'], 'seconds': 1, 'utc_completed': '2026-09-25T00:00:00+00:00'}
        name = 'metadata/protein_full_cluster_command.json'
        self.assertNotEqual(self.normalize(command, name, 'reference'), self.normalize(command, name, 'current'))
        command['argv'] = ['data_rebuild/2026-09-25-v2/a']
        self.assertNotEqual(self.normalize(command, name, 'reference'), self.normalize(command, name, 'current'))
        code = {'code': {'file': v.ORIGINAL_WORKSPACE + 'biopaws/data_v2_remote/build_remote.py'}}
        self.assertNotEqual(self.normalize(code, 'metadata/remote_manifest.json', 'reference'),
                            self.normalize(code, 'metadata/remote_manifest.json', 'current'))

    def test_descriptor_sha_binds_own_file_despite_an_unchanged_alias(self):
        bindings = {**self.bindings, 'data/b.gz': {**self.bindings['data/a.gz'], 'sha256': self.old_sha, 'bytes': 10}}
        wrong = {'file': 'data/a.gz', 'sha256': self.old_sha, 'bytes': 12}
        with self.assertRaisesRegex(ValueError, 'own file'):
            self.normalize(wrong, 'metadata/test.json', 'current', bindings)
        right = {'file': 'data/a.gz', 'sha256': self.new_sha, 'bytes': 12}
        old = {'file': 'data/a.gz', 'sha256': self.old_sha, 'bytes': 10}
        self.assertEqual(self.normalize(old, 'metadata/test.json', 'reference', bindings),
                         self.normalize(right, 'metadata/test.json', 'current', bindings))
        for side, digest in [('current', self.old_sha), ('current', self.new_sha), ('reference', self.old_sha)]:
            with self.assertRaisesRegex(ValueError, 'explicit file context'):
                self.normalize({'source_sha256': digest}, 'metadata/test.json', side, bindings)

    def test_pinned_nlp_inventory_transition_only(self):
        entries = {v.CODE_NLP + 'README.md': {'sha256': 'c' * 64}, v.CODE_NLP + 'build_longrange.py': {'sha256': 'd' * 64}}
        old = {'code_sha256': {'README.md': v.OLD_README_SHA, 'build.py': 'e' * 64}}
        new = {'code_sha256': {'README.md': 'c' * 64, 'build_longrange.py': 'd' * 64, 'build.py': 'e' * 64}}
        name = 'metadata/nlp_sources_manifest.json'
        self.assertEqual(self.normalize(old, name, 'reference', {}, entries), self.normalize(new, name, 'current', {}, entries))
        new['code_sha256']['build.py'] = 'f' * 64
        self.assertNotEqual(self.normalize(old, name, 'reference', {}, entries), self.normalize(new, name, 'current', {}, entries))
        new['code_sha256']['build_longrange.py'] = 'f' * 64
        with self.assertRaises(ValueError): self.normalize(new, name, 'current', {}, entries)


class TokenChecksTests(unittest.TestCase):
    def make_cell(self, root):
        folder = root / 'tokenized/pure_byte/protein'; folder.mkdir(parents=True)
        binary = folder / 'train.bin'; binary.write_bytes(bytes([4, 0, 1, 0, 5, 0, 1, 0]))
        index = folder / 'train.index.jsonl'
        rows = [{'record_id': 'a', 'offset': 0, 'length': 2, 'eos_offset': 1},
                {'record_id': 'b', 'offset': 2, 'length': 2, 'eos_offset': 3}]
        index.write_text(''.join(json.dumps(x) + '\n' for x in rows))
        metadata = {'tokens': 4, 'records': 2, 'sha256': v.sha(binary), 'index_sha256': v.sha(index), 'eos_token_id': 1}
        (folder / 'train.json').write_text(json.dumps(metadata))
        return folder, rows, metadata

    def test_exact_token_budget_index_and_eos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); self.make_cell(root)
            result = v.check_token_cell(root, 'tokenized/pure_byte/protein', 'train', 4, 1, 260)
            self.assertEqual(result['records'], 2)
            with self.assertRaises(ValueError): v.check_token_cell(root, 'tokenized/pure_byte/protein', 'train', 5, 1, 260)

    def test_bad_offset_and_wrong_token_rejected_even_with_updated_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder, rows, metadata = self.make_cell(root)
            rows[1]['offset'] = 1
            index = folder / 'train.index.jsonl'; index.write_text(''.join(json.dumps(x) + '\n' for x in rows))
            metadata['index_sha256'] = v.sha(index); (folder / 'train.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'record index'):
                v.check_token_cell(root, 'tokenized/pure_byte/protein', 'train', 4, 1, 260)
            rows[1]['offset'] = 2
            index.write_text(''.join(json.dumps(x) + '\n' for x in rows))
            binary = folder / 'train.bin'; binary.write_bytes(bytes([0, 0, 1, 0, 5, 0, 1, 0]))
            metadata.update(sha256=v.sha(binary), index_sha256=v.sha(index)); (folder / 'train.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'PAD/SEP/UNK'):
                v.check_token_cell(root, 'tokenized/pure_byte/protein', 'train', 4, 1, 260)

    def test_failed_current_gate_cannot_reuse_historical_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'validation').mkdir()
            (root / 'validation/integrated_release_acceptance.json').write_text('{"training_gate_pass":true}')
            gates, issues = v.current_gates(root)
            self.assertTrue(issues)
            self.assertFalse(any(gates.values()))


class AcceptanceBoundaryTests(unittest.TestCase):
    def fixture(self, temp):
        root, archive = Path(temp) / 'new', Path(temp) / 'archive'
        refroot = archive / v.PREFIX
        root.mkdir(); refroot.mkdir(parents=True)
        contents = {'data/task.jsonl': '{"row_id":"1","label":0}\n',
                    v.CODE_NLP + 'README.md': 'Accepted later documentation\n',
                    v.CODE_NLP + 'build_longrange.py': 'pass\n',
                    'metadata/nlp_synthetic_README.md': 'Historical shorter documentation\n'}
        entries = {}
        for name, text in contents.items():
            ref = refroot / name; ref.parent.mkdir(parents=True, exist_ok=True); ref.write_text(text)
            current = root / name; current.parent.mkdir(parents=True, exist_ok=True); current.write_text(text)
            entries[name] = {'file': name, 'sha256': v.sha(ref), 'bytes': ref.stat().st_size}
        (root / 'metadata/nlp_synthetic_README.md').write_text(contents[v.CODE_NLP + 'README.md'])
        (archive / 'archive_manifest.json').write_text('{"fixture":"archive"}\n')
        (refroot / 'manifest.json').write_text('{"fixture":"reference"}\n')
        archive_sha, release_sha = v.sha(archive / 'archive_manifest.json'), v.sha(refroot / 'manifest.json')
        (root / 'portable_validation').mkdir()
        copied = {name: {'sha256': item['sha256'], 'bytes': item['bytes'], 'kind': v.bootstrap_selected(name)}
                  for name, item in entries.items() if v.bootstrap_selected(name)}
        (root / 'portable_validation/bootstrap.json').write_text(json.dumps({
            'status': 'raw_assets_seeded_not_a_data_acceptance',
            'reference_release_manifest_sha256': release_sha, 'archive_manifest_sha256': archive_sha,
            'output_root': str(root), 'archive_root': str(archive),
            'prepared_data_copied': False, 'work_products_copied': False, 'new_shared_tokenizers_copied': False,
            'training_enabled': False, 'target_scoring_enabled': False,
            'raw_tool_assets': 0, 'raw_tool_bytes': 0, 'files': copied}))
        search = root / 'work/protein/remote_exclusion/remote_vs_swissprot.tsv'; search.parent.mkdir(parents=True)
        search.write_text('fresh search\n')
        for name in ('metadata/protein_remote_pretraining_exclusion.json', 'validation/cross_dataset_homology.json'):
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'search_output_sha256': v.sha(search)}))
        names = [n for n in sorted(entries) if n not in v.EXCLUDED_REFERENCE]
        protocol = Path(temp) / 'protocol.json'
        protocol.write_text(json.dumps({
            'accepted_reference_manifest_sha256': release_sha, 'accepted_archive_manifest_sha256': archive_sha,
            'files': names, 'selected_reference_file_count': len(names),
            'historical_files_excluded': v.EXCLUDED_REFERENCE, 'allowed_execution_fields': v.TIME_FIELDS,
            'allowed_command_root_fields': v.COMMAND_FIELDS, 'scientific_tolerance': 0,
            'required_token_cells': 33, 'success_status': 'accepted_for_derived_reconstruction',
            'training_enabled': False, 'target_scoring_enabled': False,
            'allowed_code_inventory_change': {'files': sorted(v.CODE_MANIFESTS), 'README_previous_sha256': v.OLD_README_SHA,
               'README_accepted_later_sha256': entries[v.CODE_NLP + 'README.md']['sha256'],
               'added_build_longrange_py_sha256': entries[v.CODE_NLP + 'build_longrange.py']['sha256'], 'other_entries': 'exact'}}))
        return root, archive, entries, protocol

    def fixture_context(self, root, archive, entries, protocol, gates=None):
        stack = ExitStack()
        stack.enter_context(patch.object(v, 'ARCHIVE_SHA', v.sha(archive / 'archive_manifest.json')))
        stack.enter_context(patch.object(v, 'V2_SHA', v.sha(archive / v.PREFIX / 'manifest.json')))
        stack.enter_context(patch.object(v, 'VERIFICATION_PROTOCOL_SHA', v.sha(protocol)))
        stack.enter_context(patch.object(v, 'authenticate', return_value=entries))
        stack.enter_context(patch.object(v, 'current_gates', side_effect=gates, return_value=({'fixture_gate': True}, [])))
        stack.enter_context(patch.object(v, 'current_token_checks', return_value=([{}] * 33, [])))
        stack.enter_context(patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}))
        return stack

    def test_full_gate_does_not_accept_changed_data_when_component_flags_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, archive, entries, protocol = self.fixture(tmp)
            (root / 'data/task.jsonl').write_text('{"row_id":"1","label":1}\n')
            with self.fixture_context(root, archive, entries, protocol):
                report = v.verify_release(archive, root, verification_protocol=protocol)
            self.assertEqual(report['status'], 'reconstruction_rejected')
            self.assertFalse(report['training_enabled']); self.assertFalse(report['target_scoring_enabled'])
            self.assertIn('data/task.jsonl', [x['file'] for x in report['mismatches']])
            self.assertFalse(report['files']['data/task.jsonl']['equivalent'])

    def test_success_is_scoped_and_existing_report_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, archive, entries, protocol = self.fixture(tmp)
            with self.fixture_context(root, archive, entries, protocol):
                report = v.verify_release(archive, root, verification_protocol=protocol)
                self.assertEqual(report['status'], 'accepted_for_derived_reconstruction')
                self.assertFalse(report['training_enabled']); self.assertFalse(report['target_scoring_enabled'])
                self.assertTrue(report['files']['data/task.jsonl']['equivalent'])
                with self.assertRaises(FileExistsError): v.verify_release(archive, root, verification_protocol=protocol)

    def test_final_recheck_rejects_data_changed_after_content_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, archive, entries, protocol = self.fixture(tmp)
            def mutate_after_comparison(_):
                (root / 'data/task.jsonl').write_text('{"row_id":"1","label":1}\n')
                return {'fixture_gate': True}, []
            with self.fixture_context(root, archive, entries, protocol, gates=mutate_after_comparison):
                report = v.verify_release(archive, root, verification_protocol=protocol)
            self.assertEqual(report['status'], 'reconstruction_rejected')
            self.assertFalse(report['final_integrity_recheck']['passed'])
            self.assertTrue(report['files']['data/task.jsonl']['changed_after_comparison'])

    def test_empty_bootstrap_inventory_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, archive, entries, protocol = self.fixture(tmp)
            path = root / 'portable_validation/bootstrap.json'
            boot = json.loads(path.read_text()); boot['files'] = {}; path.write_text(json.dumps(boot))
            with self.fixture_context(root, archive, entries, protocol):
                with self.assertRaisesRegex(ValueError, 'inventory'):
                    v.validate_bootstrap(root, archive, entries)

    def test_protocol_policy_change_is_rejected_even_if_hash_is_rebound_in_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, archive, entries, protocol = self.fixture(tmp)
            data = json.loads(protocol.read_text()); data['scientific_tolerance'] = 1e-5
            protocol.write_text(json.dumps(data))
            with self.fixture_context(root, archive, entries, protocol):
                with self.assertRaisesRegex(ValueError, 'protocol/code mismatch'):
                    v.validate_protocol(protocol, entries)


if __name__ == '__main__':
    unittest.main()
