"""Pinned source-only data reads and exclusive writes for the new raw release."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

RAW_MEMBERS = (
    'data/pairs/protein_sequence_similarity_train.tsv.gz',
    'data/pairs/protein_sequence_similarity_validation.tsv.gz',
    'data/corpora/english/train.jsonl.gz',
    'data/corpora/protein/train.jsonl.gz',
    'data/sequences/protein_canonical_sequences.tsv.gz',
    'data/sequences/remote_sequences.tsv.gz',
    'tokenizers/mixed_bpe/tokenizer.json',
    'tokenizers/mixed_bpe/training_records.json',
    'tokenizers/mixed_bpe/metadata.json',
    'metadata/corpora_english.json', 'metadata/corpora_protein.json',
    'validation/corpora_english.json', 'validation/corpora_protein.json',
    'metadata/protein_split_policy.json',
    'metadata/protein_remote_pretraining_exclusion.json',
    'metadata/protein_pair_construction.json',
    'validation/protein_pair_acceptance.json',
    'validation/cross_dataset_homology.json',
)
METADATA_NAMES = {
    'confirmation_acceptance': 'confirmation_acceptance.json',
    'confirmation_manifest': 'manifest.json',
    'confirmation_references': 'references.json',
    'confirmation_audit': 'verification/independent_verification.json',
    'design': 'configs/joint_pretraining_design.json',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def require_regular(path):
    path = Path(path)
    require(path.is_absolute(), 'Absolute path required')
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink forbidden')
    require(path.is_file(), 'Missing regular file: ' + str(path))
    return path


def inside(root, member):
    root, member = Path(root), Path(member)
    require(root.is_absolute() and not member.is_absolute() and '..' not in member.parts, 'Unsafe relative member')
    path = root / member
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlink forbidden')
    require(path.resolve().is_relative_to(root.resolve()), 'Member escapes root')
    return path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'Duplicate JSON key')
        value[key] = item
    return value


def read_json(path):
    def invalid_constant(value):
        raise ValueError('Nonfinite JSON constant: ' + value)
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object, parse_constant=invalid_constant)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def rows(path):
    opener = gzip.open if Path(path).suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line, object_pairs_hook=unique_object)


def write_rows(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as raw:
        if path.suffix == '.gz':
            with gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as f:
                for value in values:
                    f.write((json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode())
        else:
            for value in values:
                raw.write((json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode())


class Store:
    """The exact read allowlist is frozen separately from raw release acceptance."""
    def __init__(self, protocol_path, protocol_sha256, output):
        self.inputs = {}
        self.protocol_path = require_regular(Path(protocol_path).absolute())
        require(sha(self.protocol_path) == protocol_sha256, 'Data protocol hash mismatch')
        self.protocol_sha256 = protocol_sha256
        self.protocol = read_json(self.protocol_path)
        p = self.protocol
        require(p.get('schema_version') == 1, 'Unknown data protocol schema')
        require(p.get('training_enabled') is False and p.get('target_scoring_enabled') is False, 'Build protocol must disable training/scoring')
        require(p.get('source_test_examples_allowed') is False, 'Source-test examples forbidden')
        self.root = Path(p['raw_release_root'])
        self.output = Path(output).absolute()
        require(self.output == Path(p['output_root']), 'Output differs from frozen protocol')
        require(not self.output.exists(), 'Fresh output directory required')
        require(not any(q.is_symlink() for q in (self.output, *self.output.parents)), 'Symlinked output forbidden')
        self.pin('data_protocol', {'path': str(self.protocol_path), 'sha256': protocol_sha256, 'bytes': self.protocol_path.stat().st_size})
        self.acceptance_path = self.pin('raw_acceptance', p['raw_acceptance'])
        self.manifest_path = self.pin('raw_manifest', p['raw_manifest'])
        require(self.acceptance_path == inside(self.root, 'portable_validation/acceptance.json'), 'Wrong raw acceptance path')
        require(self.manifest_path == inside(self.root, 'manifest.json'), 'Wrong raw manifest path')
        self.acceptance, self.manifest = read_json(self.acceptance_path), read_json(self.manifest_path)
        for v in (self.acceptance, self.manifest):
            require(v.get('status') == 'accepted_new_raw_release_training_disabled', 'Raw release not accepted')
            require(v.get('release_id') == p['release_id'], 'Wrong raw release identity')
            require(v.get('training_enabled') is False and v.get('training_gate_pass') is False and v.get('target_scoring_enabled') is False, 'Wrong raw gate scope')
        require(self.manifest['acceptance']['sha256'] == p['raw_acceptance']['sha256'], 'Raw acceptance binding mismatch')
        entries = self.manifest['files']
        self.raw_files = {v['file']: v for v in entries}
        require(len(entries) == len(self.raw_files), 'Duplicate raw manifest member')
        expected_ids = {'raw:' + rel for rel in RAW_MEMBERS} | set(METADATA_NAMES)
        require(set(p['inputs']) == expected_ids, 'Input allowlist must match source-only protocol')
        code = p['code_pins']
        require(isinstance(code, dict) and code, 'No frozen code pins')
        require(str(Path(__file__).resolve()) in code, 'Current common module not pinned')
        for path, spec in code.items():
            self.pin('code:' + path, {'path': path, **spec})
        self.output.mkdir(parents=True)

    def pin(self, key, spec):
        path = require_regular(Path(spec['path']))
        require(path.stat().st_size == spec['bytes'] and sha(path) == spec['sha256'], 'Input hash/size changed: ' + key)
        value = {'path': str(path), 'sha256': spec['sha256'], 'bytes': spec['bytes']}
        require(key not in self.inputs or self.inputs[key] == value, 'Conflicting input pin')
        self.inputs[key] = value
        return path

    def raw(self, member):
        require(member in RAW_MEMBERS, 'Raw member not permitted: ' + member)
        key = 'raw:' + member
        spec = self.protocol['inputs'][key]
        require(Path(spec['path']) == inside(self.root, member), 'Raw path differs from release')
        manifest = self.raw_files[member]
        acceptance = self.acceptance['files'][member]
        require(spec['sha256'] == manifest['sha256'] == acceptance['sha256'] and spec['bytes'] == manifest['bytes'] == acceptance['bytes'], 'Raw file gate binding differs')
        require(acceptance.get('equivalent_to_declared_reference') is True, 'Raw reference comparison not accepted')
        return self.pin(key, spec)

    def metadata(self, name):
        require(name in METADATA_NAMES, 'Metadata input not permitted')
        spec = self.protocol['inputs'][name]
        path = Path(spec['path'])
        require(path.as_posix().endswith('/' + METADATA_NAMES[name]), 'Unexpected metadata path')
        require(not any(x in path.parts for x in ('raw', 'data', 'predictions', 'checkpoints')), 'Example/model input forbidden')
        return read_json(self.pin(name, spec))

    def finish_readset(self):
        for key, spec in self.inputs.items():
            self.pin(key, spec)
        require(set(self.protocol['inputs']) <= set(self.inputs), 'A declared input was not verified/read')
