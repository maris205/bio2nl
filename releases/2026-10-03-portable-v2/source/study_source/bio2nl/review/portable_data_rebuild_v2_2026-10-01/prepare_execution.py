"""Freeze new raw construction commands, current code and acceptance policy."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys

REVIEW = Path(__file__).resolve().parent
WORKSPACE = REVIEW.parents[2]
RELEASE = WORKSPACE / 'data_rebuild/2026-10-01-portable-v2'
ARCHIVE = WORKSPACE / 'release_staging/m1_m3_2026-09-30_v1'
SNAPSHOT = REVIEW / 'code_snapshot'


def identity(path):
    if not path.is_file() or any(x.is_symlink() for x in (path, *path.parents)):
        raise ValueError('Missing or symlinked frozen input: ' + str(path))
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'sha256': digest, 'bytes': path.stat().st_size}


def pin(path):
    return {'file': str(path), **identity(path)}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def exclusive(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def main():
    if SNAPSHOT.exists() or (REVIEW / 'protocol.json').exists():
        raise FileExistsError('Execution freeze must use a fresh destination')
    prefix = 'bio2nl/portable_data/'
    source_map = {
        'run_queue.py': REVIEW / 'run_queue.py',
        prefix + 'bootstrap.py': REVIEW / 'bootstrap_code/bootstrap.py',
        prefix + 'verify_release.py': WORKSPACE / 'bio2nl/review/portable_data_rebuild_v1_2026-10-01/verification_code_snapshot/verify_release.py',
        prefix + 'smallweb_local.py': WORKSPACE / 'bio2nl/review/portable_data_rebuild_v1_2026-10-01/corpus_code_snapshot/smallweb_local.py',
        'support/pair_auditor.py': WORKSPACE / 'bio2nl/review/deterministic_rebuild_v1_2026-10-01/code_snapshot/audit/verify_pair_replays.py',
    }
    for rel in (
        prefix + 'deterministic_v2/corpus_phase.py', prefix + 'deterministic_v2/corpus_source.py',
        prefix + 'deterministic_v2/verify_release.py', prefix + 'deterministic_v2/verification_protocol.py',
        prefix + 'deterministic_v2/cross_dataset.py',
        prefix + 'deterministic_v2/ledger.py', prefix + 'deterministic_v2/metadata.py',
        'biopaws/portable_build/deterministic_v2/protein_phase.py',
        'biopaws/portable_build/deterministic_v2/patches.py',
        prefix + 'deterministic_v1/patches.py',
    ):
        source_map[rel] = WORKSPACE / rel
    source_pins = {rel: pin(source) for rel, source in source_map.items()}
    contents = {rel: source.read_bytes() for rel, source in source_map.items()}
    for rel, data in contents.items():
        compile(data, rel, 'exec')
        if hashlib.sha256(data).hexdigest() != source_pins[rel]['sha256']:
            raise ValueError('Source changed while freezing')
    bootstrap = RELEASE / 'portable_validation/bootstrap.json'
    builder_manifest = RELEASE / 'portable_validation/builder_manifest.json'
    bootstrap_identity = identity(bootstrap)
    builder_identity = identity(builder_manifest)
    archive_identity = identity(ARCHIVE / 'archive_manifest.json')
    manifest = json.loads(builder_manifest.read_text())
    if manifest['bootstrap_sha256'] != bootstrap_identity['sha256']:
        raise ValueError('Builder manifest bootstrap changed')
    execution_builders = {}
    for name, spec in manifest['files'].items():
        source, executed = Path(spec['source']), Path(spec['snapshot'])
        if identity(source) != {'sha256': spec['source_sha256'], 'bytes': spec['source_bytes']}:
            raise ValueError('Historical source changed')
        if identity(executed) != {k: spec[k] for k in ('sha256', 'bytes')}:
            raise ValueError('Executed builder changed')
        execution_builders[str(executed)] = identity(executed)
    for rel, data in contents.items():
        target = SNAPSHOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(data)
    copied = {str(SNAPSHOT / rel): identity(SNAPSHOT / rel) for rel in source_map}
    for rel, source in source_map.items():
        if identity(source) != {k: source_pins[rel][k] for k in ('sha256', 'bytes')}:
            raise ValueError('Source changed during code freeze')
    bootstrap_module = load('freeze_bootstrap_v2', SNAPSHOT / prefix / 'bootstrap.py')
    entries = bootstrap_module.authenticate(ARCHIVE)
    policy_module = load('freeze_policy_v2', SNAPSHOT / prefix / 'deterministic_v2/verification_protocol.py')
    replay_review = WORKSPACE / 'bio2nl/review/deterministic_rebuild_v1_2026-10-01'
    replay_root = replay_review / 'runs/order_a'
    replay_audit = replay_review / 'verification/pair_replay_audit.json'
    if identity(replay_audit)['sha256'] != policy_module.REPLAY_AUDIT_SHA:
        raise ValueError('Deterministic reference audit changed')
    policy = policy_module.make_policy(entries, ARCHIVE / bootstrap_module.PREFIX,
                                      replay_root, json.loads(replay_audit.read_text()))
    protein = SNAPSHOT / 'biopaws/portable_build/deterministic_v2/protein_phase.py'
    corpus = SNAPSHOT / prefix / 'deterministic_v2/corpus_phase.py'
    smallweb = SNAPSHOT / prefix / 'smallweb_local.py'
    auditor = SNAPSHOT / prefix / 'deterministic_v2/verify_release.py'
    manifest_sha, bootstrap_sha = builder_identity['sha256'], bootstrap_identity['sha256']
    py = sys.executable
    protein_command = [py, '-B', str(protein), '--release-root', str(RELEASE),
                       '--builder-root', str(RELEASE / 'execution_code/biopaws'),
                       '--log-root', str(REVIEW / 'protein_execution'),
                       '--builder-manifest', str(builder_manifest), '--builder-manifest-sha256', manifest_sha]
    def corpus_command(phase):
        command = [py, '-B', str(corpus), '--release-root', str(RELEASE), '--phase', phase,
                   '--log-root', str(REVIEW / ('corpus_' + phase + '_execution')),
                   '--bootstrap-sha256', bootstrap_sha, '--smallweb-script', str(smallweb),
                   '--smallweb-sha256', identity(smallweb)['sha256'],
                   '--builder-root', str(RELEASE / 'execution_code'),
                   '--builder-manifest', str(builder_manifest), '--builder-manifest-sha256', manifest_sha,
                   '--release-id', RELEASE.name]
        if phase == 'dependent':
            command += ['--protein-status', str(REVIEW / 'protein_execution/protein_phase_status.json')]
        return command
    phases = {
        'protein': {'command': protein_command, 'status_file': str(REVIEW / 'protein_execution/protein_phase_status.json'),
                    'stage_count': 9, 'stage_names': ['normalize', 'cluster', 'search', 'graph_split', 'remote_prepare',
                                                   'remote_exclusion', 'protein_pairs', 'remote_pairs', 'cross_dataset_audit']},
        'corpus_initial': {'command': corpus_command('initial'), 'status_file': str(REVIEW / 'corpus_initial_execution/status.json'),
                           'stage_count': 7, 'stage_names': ['nlp_and_dyck', 'nlp_independent_verify', 'longrange',
                                                          'english', 'dna', 'configs', 'smallweb_local']},
        'corpus_dependent': {'command': corpus_command('dependent'), 'status_file': str(REVIEW / 'corpus_dependent_execution/status.json'),
                            'stage_count': 3, 'stage_names': ['protein_and_controls', 'shared_tokenizers', 'encode_all_conditions']},
    }
    protocol = {
        'schema_version': 1, 'status': 'frozen_before_full_raw_reconstruction',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'review_root': str(REVIEW), 'release_root': str(RELEASE), 'release_id': RELEASE.name,
        'archive_root': str(ARCHIVE), 'archive_manifest': pin(ARCHIVE / 'archive_manifest.json'),
        'bootstrap': pin(bootstrap), 'execution_code_manifest': pin(builder_manifest),
        'execution_code': {**copied, **execution_builders}, 'phases': phases, 'auditor': str(auditor),
        'deterministic_reference': {'root': str(replay_root), 'audit': pin(replay_audit)},
        'pair_auditor': pin(SNAPSHOT / 'support/pair_auditor.py'),
        'cross_dataset_helper': pin(SNAPSHOT / prefix / 'deterministic_v2/cross_dataset.py'),
        'verification_policy': policy, 'success_status': policy_module.SUCCESS,
        'scope': 'complete_offline_raw_reconstruction_with_new_deterministic_pair_selection',
        'original_prepared_data_substitution_permitted': False,
        'training_enabled': False, 'target_scoring_enabled': False,
        'derived_reconstruction_enabled': False, 'remote_publication_performed': False,
        'exact_historical_pair_reproduction_claimed': False,
        'failure_policy': 'Stop first failed phase, stop only owned still-running sibling groups, retain every partial output; never lower acceptance rules or change samples after results.',
        'environment': {'python': platform.python_version(), 'cpu_only': True, 'python_hash_seed': 0,
                        'packages': {n: importlib.metadata.version(n) for n in
                                     ('numpy', 'scipy', 'biopython', 'pandas', 'scikit-learn', 'pyarrow',
                                      'tokenizers', 'transformers', 'datasets', 'huggingface-hub')}},
    }
    for control, original in ((bootstrap, bootstrap_identity), (builder_manifest, builder_identity),
                              (ARCHIVE / 'archive_manifest.json', archive_identity)):
        if identity(control) != original:
            raise ValueError('Pinned control changed during execution freeze: ' + str(control))
    if identity(replay_audit) != {k: protocol['deterministic_reference']['audit'][k] for k in ('sha256', 'bytes')}:
        raise ValueError('Deterministic reference audit changed during freeze')
    exclusive(REVIEW / 'protocol.json', protocol)
    exclusive(REVIEW / 'execution_freeze.json', {
        'protocol': pin(REVIEW / 'protocol.json'), 'copied_source_pins': source_pins,
        'preparation_script': pin(Path(__file__)), 'launcher': str(SNAPSHOT / 'run_queue.py'),
        'execution_source_files': len(copied), 'executed_builder_files': len(execution_builders),
    })
    print(json.dumps({'status': 'frozen_not_launched', 'protocol_sha256': identity(REVIEW / 'protocol.json')['sha256']}))


if __name__ == '__main__':
    main()
