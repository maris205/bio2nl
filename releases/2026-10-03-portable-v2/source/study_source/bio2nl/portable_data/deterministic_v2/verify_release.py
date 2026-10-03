"""Independent acceptance of a new raw rebuild; never authorizes training.

The archived release and audited deterministic pair replay are separate,
explicitly pinned scientific references. Historical gates are not modified.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

try:
    from . import verification_protocol as rules
except ImportError:
    import verification_protocol as rules


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def identity(path):
    path = Path(path)
    require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), 'Missing/symlinked file: ' + str(path))
    return {'sha256': sha(path), 'bytes': path.stat().st_size}


def check_pin(path, pin, watch):
    path = Path(path).absolute()
    actual = identity(path)
    require(actual == {k: pin[k] for k in ('sha256', 'bytes')}, 'Pinned identity mismatch: ' + str(path))
    previous = watch.get(str(path))
    require(previous is None or previous == actual, 'File changed between uses: ' + str(path))
    watch[str(path)] = actual
    return path


def load_module(name, path, expected_sha):
    path = Path(path).absolute()
    require(sha(path) == expected_sha, 'Helper code identity mismatch: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def load_old_helper():
    parent = Path(__file__).resolve().parent.parent
    require(sha(parent / 'bootstrap.py') == rules.BOOTSTRAP_SHA, 'Bootstrap helper identity mismatch')
    return load_module('_new_raw_reference_helpers', parent / 'verify_release.py', rules.OLD_HELPER_SHA)


def validate_builder_manifest(protocol, root, entries, old, watch):
    pin = protocol['execution_code_manifest']
    path = check_pin(pin['file'], pin, watch)
    manifest = old.loads(path.read_text())
    require(manifest['schema_version'] == 1 and manifest['status'] == 'execution_code_overlay_not_data_acceptance', 'Builder manifest scope mismatch')
    require(manifest['release_id'] == protocol['release_id'] and Path(manifest['release_root']) == root and Path(manifest['builder_root']) == root / 'execution_code', 'Builder manifest root/identity mismatch')
    require(manifest['training_enabled'] is False and manifest['target_scoring_enabled'] is False, 'Builder enables prohibited work')
    require(manifest['bootstrap_sha256'] == protocol['bootstrap']['sha256'], 'Builder bootstrap binding mismatch')
    expected = {n for n in entries if n.startswith('code/')}
    require(set(manifest['files']) == expected and len(expected) == 28, 'Builder inventory must cover exactly 28 source files')
    for name, item in manifest['files'].items():
        source, target = root / name, root / 'execution_code' / name.removeprefix('code/')
        require(item['source'] == str(source) and item['snapshot'] == str(target), 'Builder manifest path mismatch')
        require(item['source_sha256'] == entries[name]['sha256'] and item['source_bytes'] == entries[name]['bytes'], 'Builder original lineage mismatch')
        check_pin(source, entries[name], watch)
        check_pin(target, item, watch)
        require(protocol['execution_code'][str(target)] == {k: item[k] for k in ('sha256', 'bytes')}, 'Executed builder absent from frozen code pins')
    return manifest


def validate_build_status(protocol, protocol_sha, root, path, old, watch):
    check_pin(path, identity(path), watch)
    status = old.loads(Path(path).read_text())
    require(status['schema_version'] == 1 and status['status'] == 'completed', 'Raw build did not complete')
    require(status['protocol_sha256'] == protocol_sha and status['release_root'] == str(root) and status['release_id'] == protocol['release_id'], 'Raw build identity mismatch')
    names = {'protein', 'corpus_initial', 'corpus_dependent'}
    require(set(status['phases']) == set(protocol['phases']) == names, 'Raw phase coverage mismatch')
    final_outputs = {}
    producer_chain = {}
    for name in ('protein', 'corpus_initial', 'corpus_dependent'):
        bound, declared = status['phases'][name], protocol['phases'][name]
        require(bound['status'] == 'completed' and bound['exit_code'] == 0, 'Raw phase failed: ' + name)
        for field in ('command', 'status_file', 'stage_count'):
            require(bound[field] == declared[field], 'Phase protocol mismatch: ' + field)
        phase_path = Path(bound['status_file'])
        phase_pin = identity(phase_path)
        require(phase_pin['sha256'] == bound['status_sha256'], 'Phase status SHA mismatch')
        require('status_bytes' not in bound or phase_pin['bytes'] == bound['status_bytes'], 'Phase status byte count mismatch')
        check_pin(phase_path, phase_pin, watch)
        phase = old.loads(phase_path.read_text())
        require(phase['status'] == 'completed' and phase['release_root'] == str(root) and phase['release_id'] == protocol['release_id'], 'Phase completion/root mismatch')
        require(phase['bootstrap_sha256'] == protocol['bootstrap']['sha256'] and phase['builder_manifest_sha256'] == protocol['execution_code_manifest']['sha256'], 'Phase source lineage mismatch')
        require(len(phase['stages']) == declared['stage_count'] and [s['name'] for s in phase['stages']] == declared['stage_names'], 'Phase stage coverage/order mismatch')
        for stage in phase['stages']:
            require(stage['status'] == 'completed' and stage.get('returncode', stage.get('exit_code')) == 0, 'Incomplete scientific stage')
            code_sha = stage.get('builder_sha256', stage.get('code_sha256'))
            code_paths = [Path(x) for x in stage['command'] if isinstance(x, str) and x in protocol['execution_code']]
            require(any(protocol['execution_code'][str(p)]['sha256'] == code_sha for p in code_paths), 'Stage code not bound to executed snapshot')
            for rel, pin in stage.get('inputs', {}).items() if isinstance(stage.get('inputs'), dict) else ():
                if rel in producer_chain:
                    require(pin == producer_chain[rel], 'Intermediate input does not bind its preceding producer')
                else:
                    check_pin(old.member(root, rel), pin, watch)
            outputs = stage.get('output_sha256')
            if outputs is not None:
                outputs = {rel: {'sha256': digest} for rel, digest in outputs.items()}
            else:
                outputs = stage['outputs']
            require(isinstance(outputs, dict) and outputs, 'Stage has no output identities')
            for rel, pin in outputs.items():
                old.member(root, rel)
                producer_chain[rel] = pin
                final_outputs[rel] = pin
    for rel, pin in final_outputs.items():
        current = old.member(root, rel)
        actual = identity(current)
        require(actual['sha256'] == pin['sha256'] and ('bytes' not in pin or actual['bytes'] == pin['bytes']), 'Last producer output changed: ' + rel)
        check_pin(current, actual, watch)
    return {'phase_count': 3, 'stage_count': sum(x['stage_count'] for x in status['phases'].values()), 'last_writer_outputs': final_outputs}


class PairInputView:
    """Read-only file routing for the unchanged frozen pair-auditor function."""
    def __init__(self, root):
        self.root = root

    def __truediv__(self, name):
        if name == 'work/protein/positive_candidates.jsonl':
            name = rules.SORTED_CANDIDATES
        return self.root / name


def verify_current_pair_inputs(root, protocol, builders, replay_root, replay_audit, auditor, old, watch):
    order_name = 'metadata/protein_candidate_order.json'
    order_path = old.member(root, order_name)
    check_pin(order_path, identity(order_path), watch)
    order = old.loads(order_path.read_text())
    expected = {'schema_version': 1, 'release_id': protocol['release_id'],
                'ordering': 'ascending ASCII (a,b) before unchanged seeded shuffle',
                'canonical_serialization': {'sort_keys': True, 'separators': [',', ':'], 'ensure_ascii': True, 'newline': 'LF'},
                'seed': 20260925, 'score_values_changed': False, 'duplicates_policy': 'fail',
                'training_enabled': False, 'data_release_gate_passed': False,
                'sorted_candidate_file': rules.SORTED_CANDIDATES,
                'builder_sha256': builders['files']['code/biopaws/data_v2/build_pairs.py']['sha256'],
                'python_hash_seed': str(protocol['environment']['python_hash_seed']),
                'python_version': protocol['environment']['python'],
                'python_implementation': platform.python_implementation(),
                'random_engine': 'stdlib random.Random; original seed and calls preserved'}
    require(all(type(order.get(k)) is type(v) and order.get(k) == v for k, v in expected.items()), 'Fresh candidate ordering policy/runtime mismatch')
    probe_env = {k: v for k, v in os.environ.items() if not k.startswith('PYTHON')}
    probe_env['PYTHONHASHSEED'] = expected['python_hash_seed']
    probe = old.loads(subprocess.check_output([sys.executable, '-P', '-s', '-B', '-c',
        "import json,sys; print(json.dumps([hash('protein-deterministic-raw-v2'),sys.flags.hash_randomization]))"], env=probe_env, text=True))
    require([order['hash_probe'], order['hash_randomization']] == probe, 'Actual builder hash-seed probe differs')
    raw = old.member(root, 'work/protein/positive_candidates.jsonl')
    sorted_file = old.member(root, rules.SORTED_CANDIDATES)
    raw_pin, sorted_pin = identity(raw), identity(sorted_file)
    check_pin(raw, raw_pin, watch)
    check_pin(sorted_file, sorted_pin, watch)
    require(order['input_candidate_sha256'] == raw_pin['sha256'], 'Ordering metadata does not bind current raw candidates')
    require(order['canonical_sorted_jsonl_sha256'] == sorted_pin['sha256'] and order['sorted_candidate_bytes'] == sorted_pin['bytes'], 'Ordering metadata does not bind sorted candidates')
    reference_sorted = replay_root / 'work/protein/positive_candidates.jsonl'
    reference_pin = replay_audit['verified_files'][str(reference_sorted)]
    check_pin(reference_sorted, reference_pin, watch)
    require(sorted_pin == reference_pin, 'Fresh complete sorted candidate content differs from audited replay')
    header, rows = auditor.table(root / auditor.PAIR_FILES[0])
    require(header == auditor.HEADERS and rows, 'New pair schema differs')
    for split in auditor.SPLITS:
        fields, projected = auditor.table(root / f'data/pairs/protein_sequence_similarity_{split}.tsv.gz')
        require(fields == header and projected == [r for r in rows if r['split'] == split], 'New split rows are not ordered projections')
    result = auditor.check_row_semantics(PairInputView(root), rows)
    require(result == replay_audit['scientific_audit'], 'Independent new pair invariants differ from deterministic reference')
    require(order['records'] == result['candidate_records'], 'Candidate ordering count differs from independent scan')
    return result


def finish_verification(root, output, protocol, protocol_path, protocol_sha, old, files,
                        entries, builders, bootstrap, build_info, pair_result, issues, watch):
    document = old.member(root, 'metadata/nlp_synthetic_README.md')
    doc_pin = entries['code/bio2nl/data/rebuild_v2/nlp_synthetic/README.md']
    check_pin(document, doc_pin, watch)
    files['metadata/nlp_synthetic_README.md'] = {**identity(document), 'comparison_mode': 'accepted_later_document_exact', 'equivalent_to_declared_reference': True}
    gates, gate_errors = old.current_gates(root)
    cells, token_errors = old.current_token_checks(root)
    issues.extend({'file': 'current_gates', 'error': e} for e in gate_errors)
    issues.extend({'file': 'token_cells', 'error': e} for e in token_errors)
    require(len(cells) == 33 and not token_errors, 'Current 33 token cells failed independent budget/index/EOS checks')
    require(len(files) == 333, 'Incomplete accepted-reference/new-metadata/document coverage')
    require(not issues and all(x['equivalent_to_declared_reference'] for x in files.values()), 'New release comparison rejected: ' + old.canonical(issues))
    for name, entry in builders['files'].items():
        current_name = 'execution_code/' + name.removeprefix('code/')
        files[current_name] = {'sha256': entry['sha256'], 'bytes': entry['bytes'],
                               'comparison_mode': 'frozen_new_execution_source', 'equivalent_to_declared_reference': True,
                               'source_file': name, 'source_sha256': entry['source_sha256'], 'transform': entry['transform']}
    for filename, pin in watch.items():
        require(identity(filename) == pin, 'Source/control/output changed during verification: ' + filename)
    expected_scientific = {n for n in entries if n.startswith(('data/', 'tokenized/', 'tokenizers/'))}
    actual_scientific = {p.relative_to(root).as_posix() for prefix in ('data', 'tokenized', 'tokenizers') for p in (root / prefix).rglob('*') if p.is_file()}
    require(actual_scientific == expected_scientific, 'Scientific artifact set changed during verification')
    report = {'schema_version': 1, 'status': rules.SUCCESS, 'release_id': protocol['release_id'],
              'training_enabled': False, 'target_scoring_enabled': False, 'training_gate_pass': False,
              'derived_reconstruction_authorized': False, 'historical_exact_reproduction_claimed': False,
              'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'protocol': {'file': str(protocol_path), 'sha256': protocol_sha},
              'reference_policy': protocol['verification_policy'], 'files': files,
              'bootstrap': bootstrap, 'build_lineage': build_info, 'independent_pair_audit': pair_result,
              'current_checks': gates, 'token_cells': cells, 'all_read_files_final_rehashed': len(watch),
              'verified_read_files': watch, 'mismatches': [],
              'limitations': ['This is a new deterministic raw release, not exact recreation of historical pair samples.',
                              'Pair baseline CSV is compared completely with the independently audited replay; logistic probes are not refitted by this verifier.',
                              'Training, target scoring, derived reconstruction and publication are not authorized by this acceptance.']}
    report_bytes = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    manifest = {'schema_version': 1, 'release_id': protocol['release_id'], 'status': rules.SUCCESS,
                'training_gate_pass': False, 'training_enabled': False, 'target_scoring_enabled': False,
                'derived_reconstruction_authorized': False, 'historical_exact_reproduction_claimed': False,
                'file_count': len(files), 'files': [{'file': n, 'sha256': x['sha256'], 'bytes': x['bytes']} for n, x in sorted(files.items())],
                'acceptance': {'file': 'portable_validation/acceptance.json', 'sha256': hashlib.sha256(report_bytes).hexdigest(), 'bytes': len(report_bytes)},
                'protocol': {'file': str(protocol_path), **identity(protocol_path)},
                'builder_manifest': protocol['execution_code_manifest'], 'bootstrap': protocol['bootstrap'],
                'reference_archive': {'root': protocol['archive_root'], 'archive_manifest_sha256': old.ARCHIVE_SHA, 'accepted_release_manifest_sha256': old.V2_SHA},
                'deterministic_pair_reference': protocol['deterministic_reference']}
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    created = []
    try:
        for path, payload in ((root / 'manifest.json', manifest_bytes), (output, report_bytes)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                created.append(path)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        require(identity(output) == {k: manifest['acceptance'][k] for k in ('sha256', 'bytes')}, 'Written report integrity failure')
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return report


def verify_release(protocol_path, protocol_sha, build_status_path, output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Verification must be CPU-only')
    old = load_old_helper()
    protocol_path, output = Path(protocol_path).absolute(), Path(output).absolute()
    watch = {}
    require(sha(protocol_path) == protocol_sha, 'Protocol SHA mismatch')
    check_pin(protocol_path, identity(protocol_path), watch)
    protocol = old.loads(protocol_path.read_text())
    root, archive = Path(protocol['release_root']), Path(protocol['archive_root'])
    require(root.is_absolute() and archive.is_absolute() and root != archive and not root.is_relative_to(archive), 'Unsafe release/reference roots')
    require(protocol['schema_version'] == 1 and protocol['release_id'] not in ('', '2026-09-25-v2'), 'Invalid new release identity')
    for flag in ('training_enabled', 'target_scoring_enabled', 'remote_publication_performed'):
        require(protocol[flag] is False, 'Forbidden protocol authorization: ' + flag)
    require(output == root / 'portable_validation/acceptance.json' and not output.exists() and not (root / 'manifest.json').exists(), 'Fresh scoped acceptance/manifest required')
    require(not (root / 'validation/integrated_release_acceptance.json').exists(), 'Historical training gate must not exist')
    require(str(Path(__file__).resolve()) in protocol['execution_code'], 'Verifier source not frozen in protocol')
    for filename, pin in protocol['execution_code'].items():
        check_pin(filename, pin, watch)
    base = Path(__file__).resolve().parent
    for filename in (base / 'verification_protocol.py', base.parent / 'bootstrap.py', base.parent / 'verify_release.py'):
        require(str(filename) in protocol['execution_code'], 'Missing helper code pin')
    entries = old.authenticate(archive)
    reference_root = archive / old.PREFIX
    check_pin(archive / 'archive_manifest.json', identity(archive / 'archive_manifest.json'), watch)
    check_pin(reference_root / 'manifest.json', identity(reference_root / 'manifest.json'), watch)
    bootstrap_pin = protocol['bootstrap']
    require(Path(bootstrap_pin['file']) == root / 'portable_validation/bootstrap.json', 'Bootstrap path mismatch')
    check_pin(bootstrap_pin['file'], bootstrap_pin, watch)
    bootstrap = old.validate_bootstrap(root, archive, entries)
    builders = validate_builder_manifest(protocol, root, entries, old, watch)
    build_info = validate_build_status(protocol, protocol_sha, root, build_status_path, old, watch)
    replay = protocol['deterministic_reference']
    replay_root = Path(replay['root'])
    require(replay['audit']['sha256'] == rules.REPLAY_AUDIT_SHA, 'Wrong completed deterministic reference audit')
    replay_audit_path = check_pin(replay['audit']['file'], replay['audit'], watch)
    replay_audit = old.loads(replay_audit_path.read_text())
    require(replay_audit['status'] == 'pair_replay_determinism_passed' and replay_audit['training_enabled'] is False, 'Unaccepted deterministic reference')
    policy = rules.make_policy(entries, reference_root, replay_root, replay_audit)
    require(old.canonical(protocol['verification_policy']) == old.canonical(policy), 'Frozen complete comparison policy differs')
    ledger = load_module('_new_raw_ledger', base / 'ledger.py', protocol['execution_code'][str(base / 'ledger.py')]['sha256'])
    metadata = load_module('_new_raw_metadata', base / 'metadata.py', protocol['execution_code'][str(base / 'metadata.py')]['sha256'])
    pair_spec = protocol['pair_auditor']
    require(pair_spec['sha256'] == rules.PAIR_AUDITOR_SHA, 'Frozen pair auditor differs')
    check_pin(pair_spec['file'], pair_spec, watch)
    pair_auditor = load_module('_new_raw_pair_auditor', pair_spec['file'], pair_spec['sha256'])
    cross_spec = protocol['cross_dataset_helper']
    check_pin(cross_spec['file'], cross_spec, watch)
    require(protocol['execution_code'][cross_spec['file']] == {k: cross_spec[k] for k in ('sha256', 'bytes')}, 'Cross helper not execution pinned')
    cross_helper = load_module('_new_raw_cross_dataset', cross_spec['file'], cross_spec['sha256'])
    pair_reference_path = replay_root / 'metadata/original_builder_manifest.json'
    check_pin(pair_reference_path, replay_audit['verified_files'][str(pair_reference_path)], watch)
    pair_reference = old.loads(pair_reference_path.read_text())
    actual_scientific = {p.relative_to(root).as_posix() for prefix in ('data', 'tokenized', 'tokenizers') for p in (root / prefix).rglob('*') if p.is_file()}
    expected_scientific = {n for n in entries if n.startswith(('data/', 'tokenized/', 'tokenizers/'))}
    require(actual_scientific == expected_scientific, 'Scientific payload file coverage differs')
    files, bindings, pending, issues = {}, {}, [], []
    for name, spec in policy['files'].items():
        reference, current = Path(spec['reference_file']), old.member(root, name)
        try:
            check_pin(reference, {'sha256': spec['reference_sha256'], 'bytes': spec['reference_bytes']}, watch)
            actual = identity(current)
            check_pin(current, actual, watch)
            files[name] = {**actual, **spec, 'equivalent_to_declared_reference': False}
            bindings[name] = {**actual, 'reference_sha256': spec['reference_sha256'], 'reference_bytes': spec['reference_bytes'], 'equivalent': False}
            special_json = name in rules.IDENTITY_FIELDS or old.is_metadata_json(name)
            if special_json:
                pending.append(name)
                continue
            if name == 'validation/nlp_clean_exclusions.jsonl':
                details = ledger.compare_ledger(reference, reference_root / 'validation/nlp_clean_label_conflicts.json', current)
                equal, mode = True, spec['mode']
            else:
                equal, mode, details = old.compare_content(reference, current, name, spec['reference_sha256'], actual['sha256'])
            require(equal, 'Scientific reference comparison failed: ' + name)
            files[name].update(equivalent_to_declared_reference=True, comparison_mode=mode, details=details)
            bindings[name]['equivalent'] = True
        except Exception as error:
            issues.append({'file': name, 'error': f'{type(error).__name__}: {error}'})
    with gzip.open(root / 'data/sequences/protein_canonical_sequences.tsv.gz', 'rt') as f:
        require(next(csv.reader(f, delimiter='\t')) == rules.CANONICAL_COLUMNS, 'Canonical column schema differs')
    for name in policy['files']:
        if not old.bootstrap_selected(name):
            require(name in build_info['last_writer_outputs'], 'No fresh stage producer for required output: ' + name)
    search_path = old.member(root, 'work/protein/remote_exclusion/remote_vs_swissprot.tsv')
    search_pin = identity(search_path)
    check_pin(search_path, search_pin, watch)
    cross_old = old.loads((reference_root / 'validation/cross_dataset_homology.json').read_text())
    cross_expected = cross_helper.recompute_cross_dataset(root, cross_old)
    require(old.loads((root / 'validation/cross_dataset_homology.json').read_text()) == cross_expected, 'Independent cross-dataset audit differs')
    pair_result = verify_current_pair_inputs(root, protocol, builders, replay_root, replay_audit, pair_auditor, old, watch)
    order_name = 'metadata/protein_candidate_order.json'
    require(order_name in build_info['last_writer_outputs'] and rules.SORTED_CANDIDATES in build_info['last_writer_outputs'], 'Fresh producer binding missing for candidate canonicalization')
    order_pin = identity(root / order_name)
    files[order_name] = {**order_pin, 'comparison_mode': 'fresh_current_candidate_order_bound_to_audited_sorted_reference', 'equivalent_to_declared_reference': True}
    bindings[order_name] = {**order_pin, 'reference_sha256': order_pin['sha256'], 'reference_bytes': order_pin['bytes'], 'equivalent': True}
    # Only scientific referents enter generic metadata relocation. The retained
    # original code tree never authorizes a metadata reference to execution code.
    metadata_bindings = {n: x for n, x in bindings.items() if not n.startswith('code/')}
    while pending:
        later, errors, progress = [], {}, False
        for name in pending:
            try:
                a = old.loads(Path(policy['files'][name]['reference_file']).read_text())
                b = old.loads((root / name).read_text())
                a, b, effective_name, changes = metadata.prepare_metadata_pair(
                    a, b, name, root=root, release_id=protocol['release_id'], entries=entries,
                    bindings=metadata_bindings, builder_manifest=builders, search_sha=search_pin['sha256'],
                    cross_expected=cross_expected, pair_reference=pair_reference)
                normalizations = {'reference': [], 'current': []}
                an = old.normalize_metadata(a, effective_name, 'reference', root, metadata_bindings, entries, normalizations['reference'])
                bn = old.normalize_metadata(b, effective_name, 'current', root, metadata_bindings, entries, normalizations['current'])
                require(old.canonical(an) == old.canonical(bn), 'Unpermitted metadata scientific difference')
                files[name].update(equivalent_to_declared_reference=True, comparison_mode='strict_declared_new_version_metadata', transitions=changes, relocations=normalizations)
                bindings[name]['equivalent'] = True
                metadata_bindings[name]['equivalent'] = True
                progress = True
            except Exception as error:
                later.append(name)
                errors[name] = f'{type(error).__name__}: {error}'
        if not progress:
            issues.extend({'file': n, 'error': errors[n]} for n in later)
            break
        pending = later
    return finish_verification(root, output, protocol, protocol_path, protocol_sha, old, files,
                               entries, builders, bootstrap, build_info, pair_result, issues, watch)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--build-status', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite validation output')
    try:
        result = verify_release(args.protocol, args.protocol_sha256, args.build_status, args.output)
    except Exception as error:
        result = {'status': 'new_raw_release_rejected', 'training_enabled': False, 'target_scoring_enabled': False,
                  'error': f'{type(error).__name__}: {error}'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps(result), flush=True)
        raise SystemExit(1)
    print(json.dumps({'status': result['status'], 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
