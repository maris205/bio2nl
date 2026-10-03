"""Pinned reads, fresh writes, and explicit semantic comparisons."""
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

ARCHIVE_SHA = '31db85eb9e88e0b12d5ed68cd0ea08e276bd509d2565fd11723ba192007aa44f'
V2 = 'data_rebuild/2026-09-25-v2/'
M1 = 'bio2nl/review/confirmation_data_v1_2026-09-29/'
M2 = 'bio2nl/review/joint_pretraining_v1_2026-09-29/'
M3 = 'bio2nl/review/joint_transfer_v1_2026-09-30/'
EVAL = 'bio2nl/review/eval_v2/prepared/'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def inside(root, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Nonrelative or escaping member')
    p = root / rel
    require(not any(q.is_symlink() for q in [p, *p.parents] if q != root.parent), 'Symlinked input/output forbidden')
    require(p.resolve().is_relative_to(root), 'Member escapes root')
    return p


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as raw:
        if path.suffix == '.gz':
            with gzip.GzipFile(fileobj=raw, filename='', mode='wb', mtime=0) as stream:
                for value in values:
                    stream.write((json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
        else:
            for value in values:
                raw.write((json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')) + '\n').encode())


def rows(path):
    opener = gzip.open if Path(path).suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def ordered_digest(values):
    h = hashlib.sha256(); count = 0
    for value in values:
        b = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        h.update(len(b).to_bytes(8, 'big')); h.update(b); count += 1
    return {'sha256': h.hexdigest(), 'rows': count}


def semantic(path):
    """No time/path stripping: data records are compared in full and in order."""
    name = Path(path).name
    if name.endswith(('.jsonl', '.jsonl.gz')):
        return {'mode': 'ordered_json_records', **ordered_digest(rows(path))}
    if name.endswith(('.tsv.gz', '.csv.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            return {'mode': 'ordered_tabular_records', **ordered_digest(csv.DictReader(f, delimiter='\t' if '.tsv.' in name else ','))}
    if name.endswith('.json'):
        value = json.loads(Path(path).read_text()); return {'mode': 'complete_json', **ordered_digest([value])}
    if name.endswith('.npz'):
        result = {}
        with np.load(path, allow_pickle=False) as f:
            for key in sorted(f.files):
                a = f[key]
                require(not a.dtype.hasobject, 'Object array forbidden')
                result[key] = {'dtype': a.dtype.str, 'shape': list(a.shape), 'sha256': hashlib.sha256(a.tobytes(order='C')).hexdigest()}
        return {'mode': 'exact_named_arrays', 'arrays': result}
    if name.endswith('.npy'):
        a = np.load(path, allow_pickle=False)
        return {'mode': 'exact_array', 'dtype': a.dtype.str, 'shape': list(a.shape), 'sha256': hashlib.sha256(a.tobytes(order='C')).hexdigest()}
    return {'mode': 'bytes', 'sha256': sha(path)}


class Inputs:
    def __init__(self, archive, release, output, expected_archive_sha=ARCHIVE_SHA):
        self.archive = Path(archive).resolve(); self.release = Path(release).resolve(); self.output = Path(output).resolve()
        require(self.archive.is_dir() and self.release.is_dir(), 'Missing archive/release root')
        manifest = inside(self.archive, 'archive_manifest.json')
        require(sha(manifest) == expected_archive_sha, 'Archive trust anchor changed')
        entries = json.loads(manifest.read_text())['files']
        self.entries = {x['path']: x for x in entries}
        require(len(entries) == len(self.entries), 'Duplicate archive paths')
        gate_path = inside(self.release, 'portable_validation/acceptance.json')
        self.gate_sha = sha(gate_path); self.gate = json.loads(gate_path.read_text())
        require(self.gate.get('status') == 'accepted_for_derived_reconstruction', 'New release data gate not accepted')
        require(self.gate.get('training_enabled') is False and self.gate.get('target_scoring_enabled') is False, 'Wrong gate scope')
        self.implementation = {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*')) if p.suffix in ('.py', '.json')}
        self.inputs = {'release_gate': {'path': str(gate_path), 'sha256': self.gate_sha, 'bytes': gate_path.stat().st_size}}; self.comparisons = []; self.archive_sha = expected_archive_sha

    def archived(self, member):
        require(member in self.entries, 'Unpinned archive input: ' + member)
        spec = self.entries[member]; path = inside(self.archive, member)
        digest = sha(path)
        require(path.stat().st_size == spec['bytes'] and digest == spec['sha256'], 'Changed archive input: ' + member)
        self.inputs['archive:' + member] = {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size}
        return path

    def release_file(self, member):
        require(member in self.gate['files'], 'Input not covered by new release gate: ' + member)
        spec = self.gate['files'][member]; path = inside(self.release, member)
        digest = sha(path); reference = self.entries.get(V2 + member)
        require(reference is not None, 'Input lacks accepted historical reference')
        require(spec.get('equivalent') is True and spec.get('comparison_mode'), 'No semantic acceptance for input')
        require(spec.get('reference_sha256') == reference['sha256'], 'Gate historical binding differs')
        require(digest == spec['sha256'] and path.stat().st_size == spec['bytes'], 'Changed new release input: ' + member)
        self.inputs['release:' + member] = {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size,
            'reference_sha256': reference['sha256'], 'comparison_mode': spec['comparison_mode'], 'equivalent': True}
        return path

    def previous(self, stage, member):
        folder = inside(self.output, stage); manifest_path = folder / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        self.inputs['stage_manifest:' + stage] = {'path': str(manifest_path), 'sha256': sha(manifest_path), 'bytes': manifest_path.stat().st_size}
        require(manifest.get('status') == 'passed_semantic_reconstruction', 'Prior stage not accepted')
        require(manifest['archive_manifest_sha256'] == self.archive_sha and manifest['release_gate_sha256'] == self.gate_sha, 'Prior stage belongs to another input gate')
        require(member in manifest['outputs'], 'Unbound prior output')
        path = inside(folder, member); digest = sha(path)
        require(digest == manifest['outputs'][member]['sha256'], 'Changed prior output')
        self.inputs[f'new:{stage}/{member}'] = {'path': str(path), 'sha256': digest, 'bytes': path.stat().st_size,
            'stage_manifest_sha256': sha(folder / 'manifest.json')}
        return path

    def compare(self, path, member):
        old = self.archived(member)
        left, right = semantic(path), semantic(old)
        require(left == right, 'Semantic reconstruction differs: ' + member)
        item = {'output': str(Path(path).relative_to(self.output)), 'reference': member,
            'sha256': sha(path), 'reference_sha256': sha(old), 'byte_identical': sha(path) == sha(old),
            'comparison': left, 'equivalent': True}
        self.comparisons.append(item)
        return item

    def finish(self, stage, report, started):
        folder = inside(self.output, stage)
        require(self.implementation == {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*')) if p.suffix in ('.py', '.json')}, 'Implementation changed during stage')
        require(sha(self.release / 'portable_validation/acceptance.json') == self.gate_sha, 'Release gate changed during reconstruction')
        for spec in self.inputs.values():
            require(sha(spec['path']) == spec['sha256'], 'Input changed during reconstruction')
        outputs = {str(p.relative_to(folder)): {'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(folder.rglob('*')) if p.is_file()}
        write_json(folder / 'manifest.json', {'schema_version': 1, 'status': 'passed_semantic_reconstruction',
            'stage': stage, 'started_at_utc': started, 'completed_at_utc': now(),
            'archive_manifest_sha256': self.archive_sha, 'release_gate_sha256': self.gate_sha,
            'inputs': self.inputs, 'outputs': outputs, 'semantic_comparisons': self.comparisons, 'report': report,
            'implementation_sha256': self.implementation,
            'model_loaded': False, 'model_training': False, 'target_scoring': False,
            'new_blind_qualification_claimed': False, 'text_redistribution_cleared': False})
        return {'stage': stage, 'status': 'passed_semantic_reconstruction', 'outputs': len(outputs), 'comparisons': len(self.comparisons)}
