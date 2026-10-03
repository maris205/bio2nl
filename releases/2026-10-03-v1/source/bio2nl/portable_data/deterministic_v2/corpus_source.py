"""Pure exact-source transformations for the second portable raw rebuild.

The NLP edit is the existing deterministic_v1 transformation, reproduced here
without a runtime dependency on a mutable workspace module. Its implementation
source is patches.py SHA 5b7f9067f7353a4f9d1bc61dd7793ef6bdb575312c129c49afadf976c330df80.
No file is changed or builder executed by this module.
"""
import hashlib

RELEASE_ID = '2026-10-01-portable-v2'
PREFIX = 'code/bio2nl/data/rebuild_v2/'
PATCHES = {
    PREFIX + 'nlp_synthetic/build.py': {
        'source_sha256': '1b67ae2b9846f709c97cc2ddcd94ba90a9d320bb75e1c3da2de470884753d257',
        'before': '        for key in bad:\n',
        'after': '        for key in sorted(bad):\n',
        'count': 1, 'transform': 'nlp_conflict_order_deterministic_v1',
    },
    PREFIX + 'write_experiment_configs.py': {
        'source_sha256': 'ad0b01a7a3f2916d147dceca29abc5165546ee6d2dfafdb3f979f0ac5cf9bbb4',
        'before': '"data_release_id":"2026-09-25-v2"',
        'after': '"data_release_id":"' + RELEASE_ID + '"',
        'count': 2, 'transform': 'corpus_config_release_identity_v2',
    },
    PREFIX + 'nlp_synthetic/build_longrange.py': {
        'source_sha256': 'bc19fd80bbd079d7f28fecab93df3eee30579fcc313f337561cab3ff054da315',
        'before': "'release_id':'2026-09-25-v2'",
        'after': "'release_id':'" + RELEASE_ID + "'",
        'count': 1, 'transform': 'longrange_release_identity_v2',
    },
}


def patch_corpus_source(name, source, release_id=RELEASE_ID):
    """Return (executed bytes, provenance) for exactly three pinned sources."""
    if release_id != RELEASE_ID or name not in PATCHES:
        raise ValueError('Unexpected corpus patch scope or release identity')
    spec = PATCHES[name]
    if hashlib.sha256(source).hexdigest() != spec['source_sha256']:
        raise ValueError('Corpus source SHA differs from fixed original')
    text = source.decode('utf-8')
    if text.count(spec['before']) != spec['count']:
        raise ValueError('Corpus source patch anchor count differs')
    output = text.replace(spec['before'], spec['after']).encode('utf-8')
    return output, {
        'transform': spec['transform'], 'source_sha256': spec['source_sha256'],
        'source_bytes': len(source), 'sha256': hashlib.sha256(output).hexdigest(),
        'bytes': len(output), 'replacement_count': spec['count'],
        'release_id': release_id, 'source_modified': False,
    }
