"""Create the complete new executed code tree from authenticated bootstrap code."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

REVIEW = Path(__file__).resolve().parent
WORKSPACE = REVIEW.parents[2]
RELEASE = WORKSPACE / 'data_rebuild/2026-10-01-portable-v2'


def identity(path):
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Missing or symlinked preparation input: ' + str(path))
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'sha256': digest, 'bytes': path.stat().st_size}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    bootstrap_path = RELEASE / 'portable_validation/bootstrap.json'
    bootstrap_identity = identity(bootstrap_path)
    bootstrap = json.loads(bootstrap_path.read_text())
    if bootstrap['output_root'] != str(RELEASE) or bootstrap['prepared_data_copied'] is not False:
        raise ValueError('Incorrect fresh raw bootstrap')
    target = RELEASE / 'execution_code'
    manifest_path = RELEASE / 'portable_validation/builder_manifest.json'
    if target.exists() or target.is_symlink() or manifest_path.exists() or manifest_path.is_symlink():
        raise FileExistsError('Executed code preparation is exclusive')
    if any(p.is_symlink() for p in (*target.parents, *manifest_path.parents)):
        raise ValueError('Symlinked execution snapshot parent')
    helper_paths = {
        'corpus': WORKSPACE / 'bio2nl/portable_data/deterministic_v2/corpus_source.py',
        'protein': WORKSPACE / 'biopaws/portable_build/deterministic_v2/patches.py',
        'canonical': WORKSPACE / 'bio2nl/portable_data/deterministic_v1/patches.py',
    }
    helper_pins = {k: {'file': str(v), **identity(v)} for k, v in helper_paths.items()}
    helpers = {k: load('preparation_' + k, path) for k, path in helper_paths.items()}
    originals = {n: v for n, v in bootstrap['files'].items() if v['kind'] == 'historical_builder_source'}
    if len(originals) != 28:
        raise ValueError('Unexpected historical source inventory')
    prepared, descriptors, transformations = {}, {}, []
    for name, pin in sorted(originals.items()):
        source = RELEASE / name
        if identity(source) != {k: pin[k] for k in ('sha256', 'bytes')}:
            raise ValueError('Original bootstrap code changed: ' + name)
        original = source.read_bytes()
        transform, provenance = 'identity', None
        if name in helpers['corpus'].PATCHES:
            data, provenance = helpers['corpus'].patch_corpus_source(name, original)
            transform = provenance['transform']
        elif name == 'code/biopaws/data_v2/build_pairs.py':
            data, provenance = helpers['protein'].patch_pair_source(original)
            transform = 'protein_deterministic_raw_v2'
        elif name == 'code/biopaws/data_v2_remote/build_remote.py':
            data, provenance = helpers['protein'].patch_remote_source(original)
            transform = 'remote_release_identity_v2'
        elif name == 'code/biopaws/data_v2/build_protein.py':
            data, provenance = helpers['canonical'].patch_source(name, original)
            transform = 'canonical_columns_deterministic_v1'
        else:
            data = original
        if name.endswith('.py'):
            compile(data, name, 'exec')
        snapshot = target / name.removeprefix('code/')
        descriptors[name] = {
            'source': str(source), 'source_sha256': pin['sha256'], 'source_bytes': pin['bytes'],
            'snapshot': str(snapshot), 'sha256': hashlib.sha256(data).hexdigest(),
            'bytes': len(data), 'transform': transform,
        }
        prepared[snapshot] = data
        if provenance is not None:
            transformations.append({'file': name, 'transform': transform, 'provenance': provenance})
    if len(transformations) != 6:
        raise ValueError('Unexpected transformed source count')
    for k, path in helper_paths.items():
        if identity(path) != {n: helper_pins[k][n] for n in ('sha256', 'bytes')}:
            raise ValueError('Patch helper changed during preparation')
    if identity(bootstrap_path) != bootstrap_identity:
        raise ValueError('Bootstrap changed during source preparation')
    for path, data in prepared.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as f:
            f.write(data)
    for name, spec in descriptors.items():
        if identity(RELEASE / name) != {k: originals[name][k] for k in ('sha256', 'bytes')}:
            raise ValueError('Original source changed during preparation')
        if identity(Path(spec['snapshot'])) != {k: spec[k] for k in ('sha256', 'bytes')}:
            raise ValueError('Copied execution source differs')
    manifest = {
        'schema_version': 1, 'status': 'execution_code_overlay_not_data_acceptance',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'release_id': RELEASE.name,
        'release_root': str(RELEASE), 'builder_root': str(target),
        'bootstrap_sha256': bootstrap_identity['sha256'], 'files': descriptors,
        'patch_helper_files': helper_pins, 'transformations': transformations,
        'training_enabled': False, 'target_scoring_enabled': False,
    }
    with manifest_path.open('x') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')
    with (REVIEW / 'builder_preparation.json').open('x') as f:
        json.dump({'status': 'executed_tree_prepared_not_built', 'manifest': {'file': str(manifest_path), **identity(manifest_path)},
                   'preparation_script': {'file': str(Path(__file__)), **identity(Path(__file__))},
                   'source_files': 28, 'modified_copies': 6, 'originals_unchanged': True}, f, indent=2)
        f.write('\n')
    print(json.dumps({'status': 'executed_tree_prepared_not_built', 'manifest_sha256': identity(manifest_path)['sha256']}))


if __name__ == '__main__':
    main()
