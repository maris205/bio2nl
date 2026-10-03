"""Offline tokenizer guards and exact preservation of the historical EOS loop."""
import ast
import contextlib
import gzip
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('smallweb_local_under_test', HERE / 'smallweb_local.py')
s = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(s)
OLD_PATH = HERE.parent / 'data/rebuild_v2/build_gpt2_smallweb_data.py'
FILES = {name: ('fixture-' + name).encode() for name in s.TOKENIZER_FILES}
TARGETS = {'train': 7, 'validation': 5, 'test': 4}


class Tokenizer:
    vocab_size = 50257
    eos_token_id = 50256
    eos_token = bos_token = '<|endoftext|>'
    pad_token = None

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [self.eos_token_id] if text == 'EOS_FIXTURE' else [ord(c) for c in text]

    def save_pretrained(self, root):
        for name, content in FILES.items():
            (Path(root) / name).write_bytes(content)


class SmallWebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.top = Path(self.temp.name)
        self.root = self.top / 'local'
        self.seed(self.root)

    def seed(self, root, tokenizer=True):
        source = root / 'data/corpora/english'
        source.mkdir(parents=True)
        examples = {'train': ['abc', 'xyzuv'], 'validation': ['abcd'], 'test': ['z', '', 'qwer']}
        for split, texts in examples.items():
            text = ''.join(json.dumps({'record_id': split + str(i), 'text': value}) + '\n'
                           for i, value in enumerate(texts))
            (source / (split + '.jsonl.gz')).write_bytes(gzip.compress(text.encode(), mtime=0))
        if tokenizer:
            directory = root / 'tokenizers/gpt2_pinned'
            directory.mkdir(parents=True)
            Tokenizer().save_pretrained(directory)
            self.meta = directory / 'metadata.json'
            self.meta.write_text(json.dumps({'model_id': s.MODEL_ID, 'revision': s.REVISION,
                'tokenizer_files_sha256': {name: s.digest(directory / name) for name in FILES}}))

    def run_local(self, tokenizer=None):
        with patch.object(sys, 'argv', ['smallweb', '--root', str(self.root)]), \
                patch.object(s, 'TARGETS', TARGETS), \
                patch.object(s.AutoTokenizer, 'from_pretrained', return_value=tokenizer or Tokenizer()) as load, \
                contextlib.redirect_stdout(io.StringIO()):
            s.main()
        return load

    def test_old_new_token_loop_ast_is_identical(self):
        def loops(path):
            module = ast.parse(path.read_text())
            main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
            return [ast.dump(n, include_attributes=False) for n in main.body
                    if isinstance(n, ast.For) and ast.dump(n.iter) == ast.dump(ast.parse('TARGETS.items()', mode='eval').body)]
        self.assertEqual(len(loops(OLD_PATH)), 1)
        self.assertEqual(loops(OLD_PATH), loops(HERE / 'smallweb_local.py'))

    def test_token_bins_indices_metadata_equal_reference_builder(self):
        oldroot = self.top / 'old'
        self.seed(oldroot, tokenizer=False)
        oldspec = importlib.util.spec_from_file_location('old_smallweb_reference', OLD_PATH)
        old = importlib.util.module_from_spec(oldspec)
        with patch.dict(sys.modules, {'common': SimpleNamespace(digest=s.digest, write_json=s.write_json)}):
            oldspec.loader.exec_module(old)
        with patch.object(sys, 'argv', ['old', '--root', str(oldroot)]), \
                patch.object(old, 'TARGETS', TARGETS), \
                patch.object(old.AutoTokenizer, 'from_pretrained', return_value=Tokenizer()), \
                contextlib.redirect_stdout(io.StringIO()):
            old.main()
        load = self.run_local()
        load.assert_called_once_with(str(self.root / 'tokenizers/gpt2_pinned'), use_fast=True,
                                     local_files_only=True, trust_remote_code=False)
        files = list((oldroot / 'tokenized/gpt2_smallweb').iterdir())
        self.assertEqual(len(files), 9)
        for path in files:
            other = self.root / path.relative_to(oldroot)
            if path.suffix == '.json':
                self.assertEqual(json.loads(path.read_text()), json.loads(other.read_text()), path.name)
            else:
                self.assertEqual(path.read_bytes(), other.read_bytes(), path.name)
        self.assertEqual((oldroot / 'configs/gpt2_smallweb_data.json').read_bytes(),
                         (self.root / 'configs/gpt2_smallweb_data.json').read_bytes())

    def test_tokenizer_assets_not_rewritten(self):
        directory = self.root / 'tokenizers/gpt2_pinned'
        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        self.run_local()
        self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_missing_or_extra_tokenizer_bindings_fail_before_loading(self):
        original = json.loads(self.meta.read_text())
        for hashes in ({}, {**original['tokenizer_files_sha256'], 'extra.json': '0' * 64}):
            self.meta.write_text(json.dumps({**original, 'tokenizer_files_sha256': hashes}))
            with self.assertRaisesRegex(ValueError, 'five expected files'):
                self.run_local()
        self.assertFalse((self.root / 'tokenized').exists())

    def test_changed_tokenizer_bytes_are_rejected(self):
        (self.meta.parent / 'tokenizer.json').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            self.run_local()

    def test_unbound_extra_tokenizer_file_is_rejected(self):
        (self.meta.parent / 'added_tokens.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Unexpected or missing'):
            self.run_local()

    def test_symlink_tokenizer_is_rejected_even_with_matching_hash(self):
        member = self.meta.parent / 'merges.txt'
        outside = self.top / 'same-merges.txt'
        member.rename(outside)
        member.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.run_local()

    def test_symlink_corpus_is_rejected(self):
        member = self.root / 'data/corpora/english/train.jsonl.gz'
        outside = self.top / 'same-corpus.gz'
        member.rename(outside)
        member.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            self.run_local()

    def test_existing_config_is_preserved(self):
        path = self.root / 'configs/gpt2_smallweb_data.json'
        path.parent.mkdir()
        path.write_text('preserve')
        with self.assertRaises(FileExistsError):
            self.run_local()
        self.assertEqual(path.read_text(), 'preserve')

    def test_existing_encoding_is_preserved(self):
        path = self.root / 'tokenized/gpt2_smallweb/train.bin'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            self.run_local()
        self.assertEqual(path.read_bytes(), b'preserve')

    def test_insufficient_input_fails_without_complete_bin(self):
        with patch.object(sys, 'argv', ['smallweb', '--root', str(self.root)]), \
                patch.object(s, 'TARGETS', {'train': 100}), \
                patch.object(s.AutoTokenizer, 'from_pretrained', return_value=Tokenizer()), \
                self.assertRaisesRegex(RuntimeError, 'Insufficient'):
            s.main()
        self.assertFalse((self.root / 'tokenized/gpt2_smallweb/train.bin').exists())
        self.assertFalse((self.root / 'configs/gpt2_smallweb_data.json').exists())

    def test_corpus_that_encodes_to_special_eos_is_rejected(self):
        class BadTokenizer(Tokenizer):
            def encode(self, text, add_special_tokens):
                return [50256]
        with self.assertRaisesRegex(RuntimeError, 'unexpectedly encoded'):
            self.run_local(BadTokenizer())


if __name__ == '__main__':
    unittest.main()
