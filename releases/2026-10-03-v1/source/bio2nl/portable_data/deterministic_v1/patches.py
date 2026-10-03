"""Pure, hash-pinned source edits. Original builders are never executed or edited."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

PATCH_VERSION = 'deterministic_serialization_v1'
CANONICAL_COLUMNS = (
    'accession', 'entry_name', 'taxon_id', 'sequence_version', 'sequence_sha256',
    'sequence', 'length', 'pretrain_eligible', 'pretrain_exclusion_reason',
    'cluster_id', 'split',
)
NLP_BEFORE = '        for key in bad:\n'
NLP_AFTER = '        for key in sorted(bad):\n'
CANONICAL_BEFORE = "    fields=list(records[0]); write_tsv(root/'data/sequences/protein_canonical_sequences.tsv.gz',records,fields)\n"
CANONICAL_AFTER = (
    '    # deterministic_serialization_v1: fixed named-column schema, preserving rows.\n'
    f'    fields = {list(CANONICAL_COLUMNS)!r}\n'
    '    expected_fields = set(fields)\n'
    '    for row in records:\n'
    '        if set(row) != expected_fields:\n'
    "            raise ValueError('Canonical named-field schema changed')\n"
    "    write_tsv(root/'data/sequences/protein_canonical_sequences.tsv.gz',records,fields)\n"
)
PATCHES = {
    'code/bio2nl/data/rebuild_v2/nlp_synthetic/build.py': {
        'original_sha256': '1b67ae2b9846f709c97cc2ddcd94ba90a9d320bb75e1c3da2de470884753d257',
        'before': NLP_BEFORE, 'after': NLP_AFTER,
        'reason': 'Sort only conflicting semantic-input group iteration in the exclusion ledger; retained examples and group-internal order are unchanged.',
    },
    'code/biopaws/data_v2/build_protein.py': {
        'original_sha256': '592c9a50b26bfcf1e6502363a6b1477c0f8a00a99bb5437e84c75c86a398af85',
        'before': CANONICAL_BEFORE, 'after': CANONICAL_AFTER,
        'reason': 'Fix canonical-sequence table columns to the accepted header and reject missing/extra named fields; preserve every row and value.',
    },
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_edit(source: bytes, expected_sha256: str, before: str, after: str) -> bytes:
    """Generic exact edit; production callers use the fixed pins in patch_source."""
    require(sha(source) == expected_sha256, 'Source SHA differs from pinned input')
    text = source.decode('utf-8')
    require(text.count(before) == 1, 'Expected exactly one source anchor')
    require(before != after, 'Empty source transformation')
    return text.replace(before, after, 1).encode('utf-8')


def patch_source(name: str, source: bytes):
    """Return (new bytes, provenance) for one of exactly two original builders.

    A second invocation on already patched bytes is rejected. To combine another
    adapter, pass these returned bytes explicitly to that separate transformer
    and record its subsequent before/after identities in the outer snapshot.
    """
    require(name in PATCHES, 'Source is outside this two-file patch scope')
    spec = PATCHES[name]
    result = exact_edit(source, spec['original_sha256'], spec['before'], spec['after'])
    return result, {
        'file': name, 'patch_version': PATCH_VERSION,
        'original_sha256': spec['original_sha256'], 'patched_sha256': sha(result),
        'original_bytes': len(source), 'patched_bytes': len(result),
        'reason': spec['reason'], 'replacement_count': 1,
        'original_file_modified': False,
    }


def regular(path: Path) -> Path:
    path = path.absolute()
    require(path.is_file(), 'Missing regular source: ' + str(path))
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'Symlinked source path')
    return path


def write_patched_snapshot(source_root: Path, output_root: Path):
    """Write only the two patched builders and provenance into a fresh directory.

    This is a patch overlay, not a complete executable builder tree. The caller
    supplies unchanged helper files in a separately recorded complete snapshot.
    No code/data is copied recursively, and no builder or original module loads.
    """
    source_root, output_root = Path(source_root).absolute(), Path(output_root).absolute()
    require(not output_root.exists() and not output_root.is_symlink(), 'Fresh snapshot destination required')
    require(not any(p.is_symlink() for p in output_root.parents), 'Symlinked destination ancestor')
    require(not output_root.is_relative_to(source_root) and not source_root.is_relative_to(output_root),
            'Source and snapshot roots must be disjoint')
    prepared = []
    for name in PATCHES:
        path = regular(source_root / name)
        original = path.read_bytes()
        patched, entry = patch_source(name, original)
        prepared.append((name, patched, entry, path, original))
    for _, _, _, path, original in prepared:
        require(path.read_bytes() == original, 'Source changed while preparing snapshot')
    output_root.mkdir(parents=True, exist_ok=False)
    for name, patched, _, path, _ in prepared:
        # Names are fixed constants, but still fail closed on accidental edits.
        rel = PurePosixPath(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe patch member')
        target = output_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(patched)
        target.chmod(path.stat().st_mode & 0o777)
    manifest = {
        'schema_version': 1, 'patch_version': PATCH_VERSION,
        'status': 'new_source_patch_overlay_not_data_acceptance',
        'source_root': str(source_root), 'snapshot_root': str(output_root),
        'files': [entry for _, _, entry, _, _ in prepared],
        'contains_complete_builder_tree': False, 'builders_executed': False,
        'training_enabled': False, 'target_scoring_enabled': False,
        'historical_exact_reproduction_claimed': False,
        'canonical_columns': list(CANONICAL_COLUMNS),
    }
    with (output_root / 'patch_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True); stream.write('\n')
    return manifest
