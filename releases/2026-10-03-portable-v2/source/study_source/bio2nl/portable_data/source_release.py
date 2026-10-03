"""Capture explicit source candidates as local Git commits and export only those blobs.

An isolated temporary index preserves both repositories' working trees and indexes.
No checkout, network, remote publication, data or model export is performed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tempfile


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args, data=None, env=None, check=True):
    effective = os.environ.copy()
    redirected = {'GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE', 'GIT_COMMON_DIR',
                  'GIT_OBJECT_DIRECTORY', 'GIT_ALTERNATE_OBJECT_DIRECTORIES',
                  'GIT_NAMESPACE', 'GIT_REPLACE_REF_BASE', 'GIT_CONFIG'}
    for key in list(effective):
        if key in redirected or key.startswith('GIT_CONFIG_'):
            effective.pop(key)
    effective['GIT_NO_REPLACE_OBJECTS'] = '1'
    if env is not None:
        # Only this explicit per-call value may redirect the temporary index.
        effective['GIT_INDEX_FILE'] = env['GIT_INDEX_FILE']
    return subprocess.run(['git', '-C', str(repo), *args], input=data, env=effective,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write('\n')


def safe_file(root, name):
    relative = PurePosixPath(name)
    require(name and not relative.is_absolute() and '..' not in relative.parts and
            relative.as_posix() == name and '\\' not in name and '\x00' not in name and
            ':' not in name and not any(x.startswith('.') for x in relative.parts), 'Unsafe source path')
    path = root
    for component in relative.parts:
        path = path / component
        require(not path.is_symlink(), 'Symlinked source path')
    require(path.is_file(), 'Missing candidate: ' + name)
    return path


def working_state(repo):
    # Includes tracked edits and index state, without exposing unrelated content.
    return {kind: sha(git(repo, *args).stdout) for kind, args in {
        'head': ('rev-parse', 'HEAD'), 'worktree_diff': ('diff', '--binary'),
        'index_diff': ('diff', '--cached', '--binary'),
        'status': ('status', '--porcelain=v1', '-uno'),
    }.items()}


def capture(workspace, inventory_path, expected_sha, output, branch):
    workspace, inventory_path, output = map(Path, (workspace, inventory_path, output))
    raw = inventory_path.read_bytes()
    require(sha(raw) == expected_sha, 'Candidate inventory changed')
    inventory = json.loads(raw)
    require(inventory['status'] == 'uncommitted_candidates' and not inventory['missing_required_scopes'], 'Need complete uncommitted inventory')
    require(inventory['raw_data_included'] is False and inventory['model_weights_included'] is False,
            'Only the source candidate inventory is supported')
    require(set(inventory['repositories']) == {'bio2nl', 'biopaws'}, 'Unexpected repositories')
    require(not output.exists() and not output.is_symlink(), 'Fresh export directory required')
    for parent in output.parents:
        require(not parent.is_symlink(), 'Symlinked export parent')
    content, before = {}, {}
    for name, record in inventory['repositories'].items():
        repo = workspace / name
        require(not repo.is_symlink(), 'Symlinked repository')
        git(repo, 'check-ref-format', '--branch', branch)
        require(git(repo, 'show-ref', '--verify', '--quiet', 'refs/heads/' + branch, check=False).returncode == 1,
                'Refuse existing branch: ' + name + '/' + branch)
        head = git(repo, 'rev-parse', 'HEAD').stdout.decode().strip()
        require(head == record['observed_checkout_head'], 'Checkout HEAD changed since inventory')
        before[name] = working_state(repo)
        files = {}
        for item in record['files']:
            require(item['path'] not in files, 'Duplicate candidate')
            path = safe_file(repo, item['path']); data = path.read_bytes()
            require(len(data) == item['bytes'] and sha(data) == item['sha256'], 'Candidate bytes changed: ' + item['path'])
            files[item['path']] = (data, '100755' if path.stat().st_mode & 0o111 else '100644')
        require(len(files) == record['file_count'], 'Candidate count differs')
        require(not output.resolve().is_relative_to(repo.resolve()), 'Export must be outside source repositories')
        content[name] = files
    output.mkdir(parents=True)
    with (output / 'candidate_inventory.json').open('xb') as stream:
        stream.write(raw)
    result = {'status': 'in_progress', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'inventory_sha256': expected_sha, 'repositories': {}, 'exported_files': [],
              'remote_publication_performed': False, 'hf_dataset_revision': None,
              'hf_model_revision': None, 'public_release_ready': False,
              'whole_git_tree_exported': False, 'historical_experiment_identity_changed': False}
    for name, record in inventory['repositories'].items():
        repo = workspace / name
        with tempfile.TemporaryDirectory(prefix='source-release-index-') as temporary:
            env = os.environ.copy(); env['GIT_INDEX_FILE'] = str(Path(temporary) / 'index')
            git(repo, 'read-tree', record['observed_checkout_head'], env=env)
            for relative, (data, mode) in content[name].items():
                oid = git(repo, 'hash-object', '-w', '--stdin', data=data).stdout.decode().strip()
                git(repo, 'update-index', '--add', '--cacheinfo', mode, oid, relative, env=env)
            tree = git(repo, 'write-tree', env=env).stdout.decode().strip()
            message = ('Capture portable reconstruction and source runtime entrypoints\n\n'
                       'Local source capture only; historical experiment identities remain per-file hashes.\n'
                       'Candidate inventory SHA-256: ' + expected_sha + '\n').encode()
            commit = git(repo, 'commit-tree', tree, '-p', record['observed_checkout_head'], data=message).stdout.decode().strip()
        changed = set(filter(None, git(repo, 'diff-tree', '--no-commit-id', '--name-only', '-r', '-z',
                                      record['observed_checkout_head'], commit).stdout.decode().split('\0')))
        require(changed <= set(content[name]), 'Commit changed unlisted paths')
        for relative, (data, mode) in content[name].items():
            blob = git(repo, 'cat-file', 'blob', commit + ':' + relative).stdout
            require(blob == data, 'Commit blob mismatch')
            destination = output / 'code' / name / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(blob)
            destination.chmod(0o755 if mode == '100755' else 0o644)
            require(sha(destination.read_bytes()) == sha(data), 'Export readback mismatch')
            result['exported_files'].append({'path': destination.relative_to(output).as_posix(),
                'repository': name, 'git_commit': commit, 'sha256': sha(blob), 'bytes': len(blob)})
        git(repo, 'update-ref', 'refs/heads/' + branch, commit, '0' * 40)
        after = working_state(repo)
        require(before[name] == after, 'Original checkout/index changed during isolated capture')
        entry = {'git_commit': commit, 'base_head': record['observed_checkout_head'], 'branch': branch,
                 'file_count': len(content[name]), 'changed_paths': sorted(changed),
                 'original_checkout_unchanged': True, 'working_state_hashes': before[name]}
        result['repositories'][name] = entry
        write(output / (name + '_local_commit.json'), entry)
    result.update(status='local_commit_bound_source_export',
                  completed_at_utc=datetime.now(timezone.utc).isoformat(),
                  file_count=len(result['exported_files']))
    write(output / 'source_export_manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--inventory-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--branch', required=True)
    args = parser.parse_args()
    result = capture(args.workspace, args.inventory, args.inventory_sha256, args.output, args.branch)
    print(json.dumps({k: result[k] for k in ('status', 'file_count', 'remote_publication_performed')}))


if __name__ == '__main__':
    main()
