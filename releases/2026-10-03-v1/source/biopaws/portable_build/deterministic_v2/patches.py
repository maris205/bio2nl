"""Exact, hash-pinned edits for a new builder snapshot; never edit old sources."""
from __future__ import annotations

import hashlib

RELEASE_ID = "2026-10-01-portable-v2"
PATCH_VERSION = "protein_deterministic_raw_v2"
PAIR_SHA = "783a46065459e641d8d27209970e085654ece8c756e78b73f53b3c72f491767b"
REMOTE_SHA = "20c00994ca244eb3843345e479b7dcfc8fbe7c0dc0ecbba1e3a722bb15f369e7"
ORDER_ANCHOR = "    rng=random.Random(a.seed); rng.shuffle(candidates); used=set(); buckets=defaultdict(list)\n"
ORDER_INSERT = '''    # protein_deterministic_raw_v2: canonical ordering before the unchanged RNG.
    import math, platform, sys
    if len(byid) != len(rows):
        raise ValueError('Duplicate canonical accession; refuse silent overwrite')
    if not candidates:
        raise ValueError('Empty positive candidates')
    expected_candidate_fields = {'a','b','identity','query_coverage','target_coverage','evalue','alignment_length','split','cluster_id'}
    for p in candidates:
        if set(p) != expected_candidate_fields:
            raise ValueError('Unexpected candidate fields')
        x, y = p['a'], p['b']
        if not all(isinstance(v,str) and v and v.isascii() and not any(c.isspace() or ord(c)<32 for c in v) for v in (x,y)):
            raise ValueError('Invalid candidate identifier')
        if not (x < y and x in byid and y in byid):
            raise ValueError('Candidate endpoints must be canonical a < b')
        if not ((byid[x]['split'],byid[x]['cluster_id']) == (byid[y]['split'],byid[y]['cluster_id']) == (p['split'],p['cluster_id'])):
            raise ValueError('Candidate/table assignment differs')
        if not all(type(p[k]) in (int,float) and math.isfinite(p[k]) for k in ('identity','query_coverage','target_coverage','evalue')):
            raise ValueError('Invalid candidate numeric value')
        if not (.4 <= p['identity'] <= .9 and .8 <= p['query_coverage'] <= 1 and .8 <= p['target_coverage'] <= 1 and 0 <= p['evalue'] <= .001):
            raise ValueError('Candidate violates frozen positive definition')
        if not (type(p['alignment_length']) is int and p['alignment_length'] > 0):
            raise ValueError('Invalid alignment length')
    candidates.sort(key=lambda p:(p['a'],p['b']))
    previous = None
    for p in candidates:
        key = (p['a'],p['b'])
        if key == previous:
            raise ValueError('Duplicate candidate edge; refuse silent removal')
        previous = key
    sorted_file = root/'work/protein/positive_candidates_sorted.jsonl'
    canonical_digest = hashlib.sha256()
    with sorted_file.open('xb') as stream:
        for p in candidates:
            encoded = (json.dumps(p,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)+'\\n').encode('utf-8')
            stream.write(encoded); canonical_digest.update(encoded)
    dump(root/'metadata/protein_candidate_order.json',{
        'schema_version':1,'release_id':'2026-10-01-portable-v2',
        'ordering':'ascending ASCII (a,b) before unchanged seeded shuffle',
        'records':len(candidates),'input_candidate_sha256':sha(root/'work/protein/positive_candidates.jsonl'),
        'canonical_sorted_jsonl_sha256':canonical_digest.hexdigest(),
        'sorted_candidate_file':'work/protein/positive_candidates_sorted.jsonl',
        'sorted_candidate_bytes':sorted_file.stat().st_size,
        'canonical_serialization':{'sort_keys':True,'separators':[',',':'],'ensure_ascii':True,'newline':'LF'},
        'builder_sha256':sha(Path(__file__)),'seed':a.seed,'score_values_changed':False,
        'duplicates_policy':'fail','training_enabled':False,'data_release_gate_passed':False,
        'python_version':platform.python_version(),'python_implementation':platform.python_implementation(),
        'python_hash_seed':os.environ.get('PYTHONHASHSEED'),'hash_randomization':sys.flags.hash_randomization,
        'hash_probe':hash('protein-deterministic-raw-v2'),'random_engine':'stdlib random.Random; original seed and calls preserved'})
'''
PAIR_ID_BEFORE = "{'release_id':'2026-09-25-v2','acceptance':acceptance,"
PAIR_ID_AFTER = "{'release_id':'2026-10-01-portable-v2','original_algorithm_release_id':'2026-09-25-v2','candidate_order_record':{'file':'metadata/protein_candidate_order.json','sha256':sha(root/'metadata/protein_candidate_order.json')},'acceptance':acceptance,"
REMOTE_ID_BEFORE = '{"release_id": "2026-09-25-v2", "task":'
REMOTE_ID_AFTER = '{"release_id": "2026-10-01-portable-v2", "original_algorithm_release_id": "2026-09-25-v2", "task":'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _patch(source, expected, changes, name):
    if digest(source) != expected:
        raise ValueError("Source SHA differs from pinned original: " + name)
    text = source.decode("utf-8")
    for before, after in changes:
        if text.count(before) != 1 or before == after:
            raise ValueError("Expected one exact nonempty patch anchor: " + name)
        text = text.replace(before, after, 1)
    result = text.encode("utf-8")
    return result, {"schema_version": 1, "patch_version": PATCH_VERSION, "file": name,
        "original_sha256": expected, "original_bytes": len(source),
        "patched_sha256": digest(result), "patched_bytes": len(result),
        "replacement_count": len(changes), "original_file_modified": False,
        "release_id": RELEASE_ID, "original_algorithm_release_id": "2026-09-25-v2",
        "scientific_thresholds_changed": False, "seed_changed": False,
        "data_release_gate_passed": False, "training_enabled": False}


def patch_pair_source(source: bytes):
    """Return bytes/provenance: validate/order complete candidates, label new ID."""
    return _patch(source, PAIR_SHA,
                  [(ORDER_ANCHOR, ORDER_INSERT + ORDER_ANCHOR), (PAIR_ID_BEFORE, PAIR_ID_AFTER)],
                  "code/biopaws/data_v2/build_pairs.py")


def patch_remote_source(source: bytes):
    """Change only manifest identity; current descriptors are emitted by builder."""
    return _patch(source, REMOTE_SHA, [(REMOTE_ID_BEFORE, REMOTE_ID_AFTER)],
                  "code/biopaws/data_v2_remote/build_remote.py")
