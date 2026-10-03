import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import fetch


class RangeResponse(io.BytesIO):
    def __init__(self, value, start, end, total, status=206, content_range=None):
        super().__init__(value)
        self.status = status
        self.headers = {'Content-Range': content_range or f'bytes {start}-{end}/{total}',
                        'Content-Length': str(end - start + 1)}

    def geturl(self):
        return 'https://cdn.example.test/locked-object'


class ResourceTests(unittest.TestCase):
    def spec(self, content):
        return {'url': 'https://example.test/fixed-revision/file', 'bytes': len(content),
                'sha256': hashlib.sha256(content).hexdigest()}

    def test_paths(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ('../escape', '/absolute', 'x/../../escape', 'a\\b', './x'):
                with self.assertRaises(ValueError): fetch.safe_path(d, name)
            (Path(d) / 'link').symlink_to('/tmp')
            with self.assertRaises(ValueError): fetch.safe_path(d, 'link/file')

    def test_urls(self):
        for value in ('http://example.com/x', 'https://secret@example.com/x', 'file:///tmp/x'):
            with self.assertRaises(ValueError): fetch.https(value)

    def test_existing_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x'; p.write_bytes(b'ABC')
            s = {'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}
            with self.assertRaises(ValueError): fetch.acquire(s, p)

    def test_archive_exact_and_tamper(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); archive = root / 'x.tar.gz'
            s = {'path': 'nested/file', 'bytes': 3, 'sha256': hashlib.sha256(b'abc').hexdigest()}
            with tarfile.open(archive, 'w:gz') as t:
                m = tarfile.TarInfo('nested/file'); m.size = 3; t.addfile(m, io.BytesIO(b'abc'))
            self.assertEqual(fetch.unpack(archive, root/'out', {'files': [s]}), 1)
            (root/'out/nested/file').write_bytes(b'ABC')
            with self.assertRaises(ValueError): fetch.unpack(archive, root/'out', {'files': [s]})

    def test_archive_links(self):
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d)/'x.tar.gz'
            with tarfile.open(archive, 'w:gz') as t:
                m = tarfile.TarInfo('nested/file'); m.type = tarfile.SYMTYPE; m.linkname = '/tmp'; t.addfile(m)
            with self.assertRaises(ValueError):
                fetch.unpack(archive, Path(d)/'out', {'files':[{'path':'nested/file','bytes':0,'sha256':hashlib.sha256(b'').hexdigest()}]})

    def test_parallel_ranges_exact_offsets_and_final_hash(self):
        content = b'0123456789abcdefghijklmnopqrstuv'
        requests = []
        def opening(request):
            start, end = map(int, request.get_header('Range').removeprefix('bytes=').split('-'))
            requests.append((start, end))
            return RangeResponse(content[start:end+1], start, end, len(content))
        with tempfile.TemporaryDirectory() as d, patch('fetch._open', side_effect=opening):
            dest = Path(d)/'asset'
            self.assertEqual(fetch._acquire_ranges(self.spec(content), dest, chunk_size=8, workers=3), 'downloaded_verified')
            self.assertEqual(dest.read_bytes(), content)
            self.assertEqual(sorted(requests), [(0,7),(8,15),(16,23),(24,31)])
            receipt = json.loads((Path(d)/'.asset.download/receipt.json').read_text())
            self.assertEqual(receipt['sha256'], self.spec(content)['sha256'])

    def test_truncated_range_resumes_from_retained_offset(self):
        content = b'01234567'
        requested = []
        def opening(request):
            value = request.get_header('Range'); requested.append(value)
            if len(requested) == 1:
                return RangeResponse(content[:3], 0, 7, 8)
            self.assertEqual(value, 'bytes=3-7')
            return RangeResponse(content[3:], 3, 7, 8)
        with tempfile.TemporaryDirectory() as d, patch('fetch._open', side_effect=opening), patch('fetch.time.sleep'):
            item = fetch._download_chunk(self.spec(content), Path(d), 0, 7)
            self.assertEqual(item['sha256'], self.spec(content)['sha256'])
            self.assertEqual(requested, ['bytes=0-7','bytes=3-7'])
            self.assertEqual(item['events'][0]['retained_bytes'], 3)

    def test_ignored_range_rejected_without_retry(self):
        content = b'01234567'
        with tempfile.TemporaryDirectory() as d, patch('fetch._open', return_value=RangeResponse(content,0,7,8,status=200)) as opening:
            with self.assertRaises(fetch.VersionMismatch):
                fetch._acquire_ranges(self.spec(content), Path(d)/'asset', chunk_size=8)
            self.assertEqual(opening.call_count, 1)
            self.assertFalse((Path(d)/'asset').exists())
            self.assertEqual(json.loads((Path(d)/'.asset.download/failure.json').read_text())['status'], 'failed_partials_retained')

    def test_wrong_cached_offset_rejected(self):
        content = b'01234567'
        with tempfile.TemporaryDirectory() as d, patch('fetch._open', return_value=RangeResponse(content[:4],0,3,8)) as opening:
            with self.assertRaises(fetch.VersionMismatch):
                fetch._download_chunk(self.spec(content), Path(d), 4, 7)
            self.assertEqual(opening.call_count, 1)

    def test_transport_attempts_bounded_and_partial_preserved(self):
        content = b'01234567'
        calls = []
        def opening(request):
            start = int(request.get_header('Range').split('=')[1].split('-')[0]); calls.append(start)
            return RangeResponse(content[start:start+1], start, 7, 8)
        with tempfile.TemporaryDirectory() as d, patch('fetch._open', side_effect=opening), patch('fetch.time.sleep'):
            with self.assertRaises(fetch.IncompleteTransport):
                fetch._download_chunk(self.spec(content), Path(d), 0, 7)
            self.assertEqual(calls,[0,1,2])
            self.assertEqual((Path(d)/'000000000000-000000000007.partial').read_bytes(),b'012')
            metadata=json.loads((Path(d)/'000000000000-000000000007.json').read_text())
            self.assertEqual(len(metadata['events']),3)
            self.assertEqual(metadata['events'][-1]['retained_sha256'], hashlib.sha256(b'012').hexdigest())

    def test_final_content_mismatch_keeps_evidence_without_retry(self):
        content = b'01234567'
        with tempfile.TemporaryDirectory() as d, patch('fetch._open', return_value=RangeResponse(b'ABCDEFGH',0,7,8)) as opening:
            with self.assertRaises(fetch.VersionMismatch):
                fetch._acquire_ranges(self.spec(content),Path(d)/'asset',chunk_size=8)
            self.assertEqual(opening.call_count,1)
            evidence=json.loads((Path(d)/'.asset.download/failure.json').read_text())
            assembled=next(x for x in evidence['files'] if x['name']=='assembled.partial')
            self.assertEqual(assembled['bytes'],8)
            self.assertEqual(assembled['sha256'],hashlib.sha256(b'ABCDEFGH').hexdigest())
            self.assertFalse((Path(d)/'asset').exists())

    def test_completed_chunk_reuse_checks_recorded_hash(self):
        content=b'01234567'
        with tempfile.TemporaryDirectory() as d, patch('fetch._open',return_value=RangeResponse(content,0,7,8)) as opening:
            fetch._download_chunk(self.spec(content),Path(d),0,7)
            again=fetch._download_chunk(self.spec(content),Path(d),0,7)
            self.assertTrue(again['resumed_complete_chunk'])
            self.assertEqual(opening.call_count,1)
            (Path(d)/'000000000000-000000000007.chunk').write_bytes(b'ABCDEFGH')
            with self.assertRaises(fetch.VersionMismatch): fetch._download_chunk(self.spec(content),Path(d),0,7)

    def test_https_redirect_rejects_downgrade(self):
        handler=fetch.HTTPSRedirectHandler()
        with self.assertRaises(ValueError):
            handler.redirect_request(None,None,302,'',{},'http://example.test/asset')

    def test_small_failure_also_preserves_actual_bytes_and_hash(self):
        content=b'01234567'
        with tempfile.TemporaryDirectory() as d, patch('fetch._open',return_value=RangeResponse(b'012',0,7,8,status=200)):
            with self.assertRaises(ValueError): fetch.acquire(self.spec(content),Path(d)/'asset')
            folder=Path(d)/'.asset.download';parts=list(folder.glob('full-*.partial'))
            self.assertEqual(len(parts),1)
            self.assertEqual(parts[0].read_bytes(),b'012')
            record=json.loads(parts[0].with_suffix('.json').read_text())
            self.assertEqual(record['bytes'],3)
            self.assertEqual(record['sha256'],hashlib.sha256(b'012').hexdigest())


if __name__ == '__main__': unittest.main()
