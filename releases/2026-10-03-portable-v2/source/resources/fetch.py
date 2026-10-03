#!/usr/bin/env python3
"""Acquire fixed public inputs and reconstruct the current-study English pool locally."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def verify(path, spec):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size != spec['bytes'] or sha(path) != spec['sha256']:
        raise ValueError('File differs from locked size/SHA: ' + str(path))


def safe_path(root, name):
    part = PurePosixPath(name)
    if not name or part.is_absolute() or '..' in part.parts or str(part) != name or '\\' in name:
        raise ValueError('Unsafe relative member')
    root = Path(root).absolute()
    result = root.joinpath(*part.parts)
    if any(p.is_symlink() for p in [result, *result.parents]):
        raise ValueError('Symlink destinations are forbidden')
    return result


def https(url):
    p = urllib.parse.urlsplit(url)
    if p.scheme != 'https' or not p.hostname or p.username or p.password:
        raise ValueError('Expected credential-free HTTPS URL')
    return url


class VersionMismatch(ValueError):
    """Wrong byte ranges, total size or final content are never accepted/retried."""


class IncompleteTransport(OSError):
    """An early EOF is a transport failure; retained bytes can be resumed."""


class HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        https(newurl)
        return super().redirect_request(request, fp, code, msg, headers, newurl)


def _open(request):
    return urllib.request.build_opener(HTTPSRedirectHandler()).open(request, timeout=60)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def _range_url(url, start, end):
    # Distinct query keys prevent a proxy from confusing cached Range responses.
    # The locked path/revision and required final content hash are unchanged.
    parsed = urllib.parse.urlsplit(https(url))
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query.append(('bio2nl_range', f'{start}-{end}'))
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(query)))


def _transport_retryable(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 429) or 500 <= error.code <= 599
    return isinstance(error, (urllib.error.URLError, TimeoutError, OSError, http.client.IncompleteRead))


def _download_chunk(spec, folder, start, end, attempts=3):
    name = f'{start:012d}-{end:012d}'
    complete, partial = folder / (name + '.chunk'), folder / (name + '.partial')
    record = folder / (name + '.json')
    expected_size = end - start + 1
    events = []
    if complete.exists():
        prior = json.loads(record.read_text())
        if complete.stat().st_size != expected_size or sha(complete) != prior['sha256']:
            raise VersionMismatch('Resumed complete chunk differs from its recorded hash')
        return {**prior, 'resumed_complete_chunk': True}
    if partial.exists() and partial.stat().st_size > expected_size:
        raise VersionMismatch('Retained partial exceeds its fixed range')
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected_size:
            break
        actual_start = start + offset
        expected_range = f'bytes {actual_start}-{end}/{spec["bytes"]}'
        request = urllib.request.Request(_range_url(spec['url'], actual_start, end), headers={
            'User-Agent': 'bio2nl-reproduction/2', 'Range': f'bytes={actual_start}-{end}',
            'Accept-Encoding': 'identity', 'Cache-Control': 'no-cache'})
        try:
            with _open(request) as response:
                https(response.geturl())
                if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                    raise VersionMismatch('Range status/Content-Range differs: expected ' + expected_range +
                                          '; received status ' + str(response.status) + ' and ' + str(response.headers.get('Content-Range')))
                length = response.headers.get('Content-Length')
                if length is not None and int(length) != expected_size - offset:
                    raise VersionMismatch('Range Content-Length differs')
                if response.headers.get('Content-Encoding', 'identity') not in ('identity', ''):
                    raise VersionMismatch('Encoded HTTP transfer would change fixed byte offsets')
                with partial.open('ab') as stream:
                    while True:
                        try:
                            chunk = response.read(min(1 << 20, expected_size - offset + 1))
                        except http.client.IncompleteRead as error:
                            chunk = error.partial
                            if offset + len(chunk) > expected_size:
                                raise VersionMismatch('Partial response exceeds fixed range')
                            stream.write(chunk); stream.flush(); offset += len(chunk)
                            raise IncompleteTransport('HTTP stream ended before fixed range completed') from None
                        if not chunk:
                            break
                        if offset + len(chunk) > expected_size:
                            raise VersionMismatch('Response exceeds fixed range')
                        stream.write(chunk); stream.flush(); offset += len(chunk)
            if offset != expected_size:
                raise IncompleteTransport('HTTP stream ended before fixed range completed')
            events.append({'attempt': attempt, 'range_start': actual_start, 'status': 'complete'})
            break
        except BaseException as error:
            size = partial.stat().st_size if partial.exists() else 0
            event = {'attempt': attempt, 'range_start': actual_start, 'status': 'failed',
                     'error_type': type(error).__name__, 'retained_bytes': size,
                     'retained_sha256': sha(partial) if partial.exists() else None,
                     'retryable_transport': _transport_retryable(error)}
            events.append(event)
            _write_json(record, {'start': start, 'end': end, 'events': events, 'status': 'partial_failed'})
            if not _transport_retryable(error) or attempt == attempts:
                raise
            time.sleep(min(attempt, 2))
    if not partial.exists() or partial.stat().st_size != expected_size:
        raise IncompleteTransport('Missing or short completed chunk')
    result = {'start': start, 'end': end, 'bytes': expected_size, 'sha256': sha(partial),
              'status': 'chunk_complete', 'events': events, 'resumed_complete_chunk': False}
    _write_json(record, result)
    os.replace(partial, complete)
    return result


def _acquire_ranges(spec, destination, *, chunk_size=8 << 20, workers=6):
    folder = safe_path(destination.parent, '.' + destination.name + '.download')
    folder.mkdir(exist_ok=True)
    identity = {'bytes': spec['bytes'], 'sha256': spec['sha256'],
                'source_url_sha256': hashlib.sha256(https(spec['url']).encode()).hexdigest(), 'chunk_size': chunk_size}
    identity_path = folder / 'identity.json'
    lock = os.open(folder / 'lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise VersionMismatch('Retained download belongs to another immutable resource')
        else:
            _write_json(identity_path, identity)
        parts = [(start, min(start + chunk_size, spec['bytes']) - 1) for start in range(0, spec['bytes'], chunk_size)]
        records = []
        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(_download_chunk, spec, folder, start, end): (start, end) for start, end in parts}
                for future in as_completed(futures):
                    records.append(future.result())
            records.sort(key=lambda x: x['start'])
            assembled = folder / 'assembled.partial'
            with assembled.open('wb') as out:
                for start, end in parts:
                    with (folder / f'{start:012d}-{end:012d}.chunk').open('rb') as stream:
                        shutil.copyfileobj(stream, out, 4 << 20)
            actual_sha, actual_bytes = sha(assembled), assembled.stat().st_size
            if actual_bytes != spec['bytes'] or actual_sha != spec['sha256']:
                raise VersionMismatch('Assembled bytes do not match locked version')
            os.link(assembled, destination)
            report = {'status': 'downloaded_verified', 'strategy': 'strict_parallel_ranges', 'workers': workers,
                      'chunk_size': chunk_size, 'chunks': records, 'bytes': actual_bytes, 'sha256': actual_sha,
                      'completed_at_utc': datetime.now(timezone.utc).isoformat()}
            _write_json(folder / 'receipt.json', report)
            return 'downloaded_verified'
        except BaseException as error:
            files = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)}
                     for p in sorted(folder.iterdir()) if p.is_file() and p.suffix in ('.chunk', '.partial')]
            _write_json(folder / 'failure.json', {'status': 'failed_partials_retained', 'error_type': type(error).__name__,
                        'locked_bytes': spec['bytes'], 'locked_sha256': spec['sha256'], 'files': files,
                        'failed_at_utc': datetime.now(timezone.utc).isoformat()})
            raise
    finally:
        os.close(lock)


def acquire(spec, destination):
    destination = Path(destination).absolute()
    safe_path(destination.parent, destination.name)
    if destination.exists():
        verify(destination, spec)
        return 'existing_verified'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if spec['bytes'] > 32 << 20:
        return _acquire_ranges(spec, destination)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.download-', delete=False) as f:
            temporary = Path(f.name)
            h, size = hashlib.sha256(), 0
            request = urllib.request.Request(https(spec['url']), headers={'User-Agent': 'bio2nl-reproduction/1'})
            with _open(request) as response:
                https(response.geturl())
                while chunk := response.read(1 << 20):
                    size += len(chunk)
                    if size > spec['bytes']:
                        raise ValueError('Upstream bytes exceed fixed version')
                    f.write(chunk)
                    h.update(chunk)
            if size != spec['bytes'] or h.hexdigest() != spec['sha256']:
                raise ValueError('Upstream bytes do not match fixed version')
        os.link(temporary, destination)
        return 'downloaded_verified'
    except BaseException as error:
        if temporary is not None and temporary.exists():
            retained = safe_path(destination.parent, '.' + destination.name + '.download')
            retained.mkdir(exist_ok=True)
            failed = retained / ('full-' + str(time.time_ns()) + '.partial')
            os.replace(temporary, failed)
            _write_json(failed.with_suffix('.json'), {'status': 'failed_partial_retained',
                'error_type': type(error).__name__, 'bytes': failed.stat().st_size, 'sha256': sha(failed),
                'locked_bytes': spec['bytes'], 'locked_sha256': spec['sha256'],
                'failed_at_utc': datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def unpack(archive, root, manifest):
    entries = manifest['files']
    expected = {x['path']: x for x in entries}
    if len(expected) != len(entries):
        raise ValueError('Duplicate lock members')
    root = Path(root).absolute()
    safe_path(root.parent, root.name)
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError('Archive member set differs from explicit lock')
        for m in members:
            dest = safe_path(root, m.name)
            if not m.isfile() or m.size != expected[m.name]['bytes']:
                raise ValueError('Unsafe type or member size')
            if dest.exists():
                verify(dest, expected[m.name])
        # Manual extraction never follows archive symlinks/hardlinks.
        for m in members:
            dest = safe_path(root, m.name)
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as f:
                tmp = Path(f.name)
                try:
                    with tar.extractfile(m) as stream:
                        shutil.copyfileobj(stream, f, 1 << 20)
                    f.flush()
                    verify(tmp, expected[m.name])
                    os.link(tmp, dest)
                finally:
                    tmp.unlink(missing_ok=True)
    return len(expected)


def build_english(root, raw_file, recipe):
    """Run the original English algorithm with a frozen hash-only exclusion index."""
    root = Path(root).absolute()
    verify(raw_file, recipe['raw'])
    index = safe_path(root, recipe['exclusion']['file'])
    verify(index, recipe['exclusion'])
    data = index.read_bytes()
    n = recipe['exclusion']['entries']
    if len(data) != 16 * n:
        raise ValueError('Exclusion index width differs')
    digests = [data[i:i+16] for i in range(0, len(data), 16)]
    if digests != sorted(set(digests)):
        raise ValueError('Exclusion index must be sorted and unique')
    for name, spec in recipe['vendor'].items():
        verify(safe_path(HERE / 'vendor', name), spec)
    sys.path.insert(0, str(HERE / 'vendor'))
    # Vendor imports resolve only from this verified adapter directory.
    for name in ('common', 'fetch_corpora', 'build_corpora'):
        loaded = sys.modules.get(name)
        if loaded and Path(loaded.__file__).resolve().parent != HERE / 'vendor':
            raise ValueError('Conflicting module already imported: ' + name)
    import build_corpora
    build_corpora.benchmark_shingles = lambda unused_root: (set(digests), recipe['exclusion']['source_files'])
    with tempfile.TemporaryDirectory(prefix='english-build-', dir=root.parent) as temporary:
        staging = Path(temporary)
        rawdest = staging / 'raw/corpora/openwebtext-00000.parquet'
        rawdest.parent.mkdir(parents=True)
        # Input link is internal to this temporary build, not a published artifact.
        rawdest.symlink_to(Path(raw_file).absolute())
        build_corpora.english(staging)
        actual_report = json.loads((staging / 'validation/corpora_english.json').read_text())
        if actual_report != recipe['expected_report']:
            raise ValueError('Reconstructed English report differs')
        for name, spec in recipe['outputs'].items():
            verify(staging / name, spec)
            dest = safe_path(root, name)
            if dest.exists():
                verify(dest, spec)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staging / name, dest)
                verify(dest, spec)
    return {'status': 'exact_five_files_rebuilt', 'source_rows': actual_report['source_rows'],
            'train_rows': actual_report['splits']['train']['records'],
            'historical_exclusion_hash_projection_used': True,
            'original_nlp_preparation_rerun': False, 'outputs': recipe['outputs']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('fetch')
    a.add_argument('--lock', type=Path, default=HERE / 'resources.json')
    a.add_argument('--id', required=True)
    a.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('unpack')
    b.add_argument('--archive', type=Path, required=True)
    b.add_argument('--root', type=Path, required=True)
    b.add_argument('--manifest', type=Path, default=HERE / 'input_bundle_manifest.json')
    c = sub.add_parser('english')
    c.add_argument('--raw', type=Path, required=True)
    c.add_argument('--root', type=Path, required=True)
    c.add_argument('--recipe', type=Path, default=HERE / 'english_recipe.json')
    args = p.parse_args()
    if args.command == 'fetch':
        lock = json.loads(args.lock.read_text())
        spec = lock['resources'][args.id]
        report = {'status': acquire(spec, args.output), 'id': args.id, 'sha256': spec['sha256']}
    elif args.command == 'unpack':
        report = {'status': 'unpacked_and_verified', 'members': unpack(args.archive, args.root, json.loads(args.manifest.read_text()))}
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('English construction requires CUDA_VISIBLE_DEVICES empty')
        report = build_english(args.root, args.raw, json.loads(args.recipe.read_text()))
    report['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
