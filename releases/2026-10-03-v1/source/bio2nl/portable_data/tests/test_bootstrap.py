import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('portable_bootstrap', Path(__file__).resolve().parents[1] / 'bootstrap.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

@pytest.mark.parametrize('name', ['../outside', '/tmp/outside', 'data/../raw/x', 'raw\\x', 'raw/./x', ''])
def test_traversal_rejected(name):
    with pytest.raises(ValueError):
        MODULE.relative(name)

@pytest.mark.parametrize('name', ['data/sequences/protein_canonical_sequences.tsv.gz',
    'work/protein/full_cluster.tsv', 'tokenized/mixed_bpe/protein/train.bin',
    'tokenizers/mixed_bpe/tokenizer.json', 'validation/integrated_release_acceptance.json'])
def test_bootstrap_cannot_copy_prepared_outputs_or_old_acceptance(name):
    assert MODULE.selected(name) is None

def test_external_pinned_tokenizer_is_distinguished_from_new_shared_tokenizers():
    assert MODULE.selected('tokenizers/gpt2_pinned/tokenizer.json') == 'raw_or_external_tool_asset'
    assert MODULE.selected('metadata/remote_sources.json') == 'historical_source_provenance'

def test_symlink_archive_member_rejected(tmp_path):
    root = tmp_path / 'archive'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.write_text('do not read')
    (root / 'data').symlink_to(outside)
    with pytest.raises(ValueError, match='Symlink'):
        MODULE.member(root, 'data')

def test_cpu_requirement_precedes_input_access(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    with pytest.raises(ValueError, match='CPU-only'):
        MODULE.bootstrap(tmp_path / 'missing', tmp_path / 'output')

def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    output = tmp_path / 'output'
    output.mkdir()
    (output / 'original').write_bytes(b'preserved')
    with pytest.raises(ValueError, match='existing'):
        MODULE.bootstrap(tmp_path / 'missing', output)
    assert (output / 'original').read_bytes() == b'preserved'
