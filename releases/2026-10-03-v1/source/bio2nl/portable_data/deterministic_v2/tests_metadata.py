"""CPU-only metadata lineage fixtures; no builders, models or acceptance writes."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    spec = importlib.util.spec_from_file_location('metadata_v2_test', HERE / 'metadata.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
finally:
    sys.path.pop(0)


def normalize_pair(ctx, a, b, name):
    # Exercise the exact frozen generic normalizer used by the main verifier.
    sys.path.insert(0, str(HERE.parent))
    try:
        spec = importlib.util.spec_from_file_location('old_metadata_normalizer_test', HERE.parent / 'verify_release.py')
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
    finally:
        sys.path.pop(0)
    bindings = {k:v for k,v in ctx['bindings'].items() if not k.startswith('code/')}
    x = old.normalize_metadata(a, name, 'reference', ctx['root'], bindings, ctx['entries'], [])
    y = old.normalize_metadata(b, name, 'current', ctx['root'], bindings, ctx['entries'], [])
    return old.canonical(x), old.canonical(y)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return {'sha256': digest(value), 'bytes': len(value)}


@pytest.fixture
def ctx(tmp_path):
    root = tmp_path / 'release'
    root.mkdir()
    entries, files, bindings = {}, {}, {}
    for key in [m.NLP_PREFIX + v for v in ('README.md', 'build.py', 'build_longrange.py', 'verify.py')]+[m.PAIR_CODE, m.REMOTE_CODE]:
        source = root / key
        target = root / 'execution_code' / key[5:]
        original = write(source, ('original ' + key).encode())
        executed = write(target, ('executed ' + key).encode())
        entries[key] = original
        files[key] = {'source': str(source), 'source_sha256': original['sha256'], 'source_bytes': original['bytes'],
                      'snapshot': str(target), **executed, 'transform': 'fixture'}
    return {'root': root, 'release_id': m.NEW_ID, 'entries': entries, 'bindings': bindings,
            'builder_manifest': {'files': files}, 'search_sha': 'a' * 64}


def artifact(ctx, name, value, reference=None):
    current = write(ctx['root'] / name, value)
    old = reference or current
    ctx['bindings'][name] = {**current, 'reference_sha256': old['sha256'], 'reference_bytes': old['bytes'], 'equivalent': True}
    return current


def prepare(ctx, a, b, name, **kwargs):
    return m.prepare_metadata_pair(a, b, name, **ctx, **kwargs)


def nlp(ctx):
    files = ctx['builder_manifest']['files']
    a = {'code_sha256': {Path(k).name: v['source_sha256'] for k,v in files.items() if k.startswith(m.NLP_PREFIX)}}
    a['code_sha256']['README.md'] = m.OLD_README_SHA
    del a['code_sha256']['build_longrange.py']
    b = {'code_sha256': {Path(k).name: v['sha256'] for k,v in files.items() if k.startswith(m.NLP_PREFIX)}}
    return a,b


def test_nlp_inventory_exact_current_code_and_immutable_inputs(ctx):
    a,b = nlp(ctx)
    originals = copy.deepcopy((a,b))
    x,y,name,changes = prepare(ctx,a,b,'metadata/nlp_clean_manifest.json')
    assert x == y and name == 'metadata/versioned_inventory.json' and changes
    assert (a,b) == originals
    assert len(set(normalize_pair(ctx,x,y,name))) == 1


@pytest.mark.parametrize('fault', ['source_hash', 'missing', 'extra', 'wrong_readme', 'executed_mutated', 'path_alias'])
def test_nlp_inventory_rejects_code_lineage_faults(ctx, fault):
    a,b = nlp(ctx)
    code = ctx['builder_manifest']['files'][m.NLP_PREFIX+'build.py']
    if fault == 'source_hash': b['code_sha256']['build.py'] = code['source_sha256']
    if fault == 'missing': del b['code_sha256']['verify.py']
    if fault == 'extra': b['code_sha256']['extra.py'] = '0'*64
    if fault == 'wrong_readme': a['code_sha256']['README.md'] = '0'*64
    if fault == 'executed_mutated': Path(code['snapshot']).write_bytes(b'changed')
    if fault == 'path_alias': code['snapshot'] = code['source']
    with pytest.raises(ValueError): prepare(ctx,a,b,'metadata/nlp_sources_manifest.json')


@pytest.mark.parametrize('name,field', list(m.rules.IDENTITY_FIELDS.items())[:2])
def test_identity_transition_is_explicit_and_other_fields_remain(ctx,name,field):
    a,b={field:m.OLD_ID,'seed':1},{field:m.NEW_ID,'seed':2}
    x,y,_,_=prepare(ctx,a,b,name)
    assert x[field] == y[field] == m.OLD_ID and x['seed'] != y['seed']
    b[field] = m.OLD_ID
    with pytest.raises(ValueError): prepare(ctx,a,b,name)


def test_unknown_identity_is_not_normalized(ctx):
    x,y,_,changes = prepare(ctx,{'release_id':m.OLD_ID},{'release_id':m.NEW_ID},'metadata/unlisted.json')
    assert x != y and not changes


def test_longrange_exact_builder(ctx):
    pin=ctx['builder_manifest']['files'][m.NLP_PREFIX+'build_longrange.py']
    a={'release_id':m.OLD_ID,'code_sha256':pin['source_sha256']}
    b={'release_id':m.NEW_ID,'code_sha256':pin['sha256']}
    assert prepare(ctx,a,b,'metadata/synthetic_longrange_manifest.json')[:2] == (a,a)
    b['code_sha256']=pin['source_sha256']
    with pytest.raises(ValueError): prepare(ctx,a,b,'metadata/synthetic_longrange_manifest.json')


def test_descriptor_hash_of_other_valid_file_rejected(ctx):
    one=artifact(ctx,'data/one',b'one');two=artifact(ctx,'data/two',b'two')
    a={'files':[{'file':'data/one',**one}]}
    b={'files':[{'file':'data/one',**two}]}
    with pytest.raises(ValueError,match='own current file'):prepare(ctx,a,b,'metadata/plain.json')
    b['files'][0]=dict(a['files'][0],bytes=True)
    with pytest.raises(ValueError,match='bytes'):prepare(ctx,a,b,'metadata/plain.json')


def test_binding_does_not_replace_current_physical_check(ctx):
    one=artifact(ctx,'data/one',b'one')
    a={'file':'data/one',**one}
    (ctx['root']/'data/one').write_bytes(b'changed')
    with pytest.raises(ValueError,match='binding'):prepare(ctx,a,a,'metadata/plain.json')


def pair(ctx):
    code=ctx['builder_manifest']['files'][m.PAIR_CODE]
    accepted={'release_id':m.OLD_ID,'acceptance':{'rows':4},'source_release':'fixture',
              'builder_sha256':code['source_sha256'],'data_files':{},'duration_seconds':1,
              'remote_pretraining_exclusion_required':True}
    replay=copy.deepcopy(accepted);replay['acceptance']={'rows':8}
    current=copy.deepcopy(replay);current['release_id']=m.NEW_ID;current['builder_sha256']=code['sha256']
    current['original_algorithm_release_id']=m.OLD_ID
    for name in m.rules.PAIR_FILES:
        old={'sha256':digest(('replay '+name).encode()),'bytes':5}
        pin=artifact(ctx,name,('current '+name).encode(),old)
        replay['data_files'][Path(name).name]=old['sha256']
        accepted['data_files'][Path(name).name]='f'*64
        current['data_files'][Path(name).name]=pin['sha256']
    artifact(ctx,'validation/protein_pair_acceptance.json',json.dumps(current['acceptance']).encode())
    pin=artifact(ctx,'metadata/protein_candidate_order.json',b'{"fixture":true}')
    current['candidate_order_record']={'file':'metadata/protein_candidate_order.json','sha256':pin['sha256']}
    return accepted,replay,current


def test_pair_uses_replay_and_current_own_files(ctx):
    a,p,b=pair(ctx)
    x,y,_,changes=prepare(ctx,a,b,'metadata/protein_pair_manifest.json',pair_reference=p)
    assert x['acceptance'] == y['acceptance'] == {'rows':8}
    assert x['builder_sha256']==y['builder_sha256']
    assert 'candidate_order_record' not in y and len(changes)>=4
    assert len(set(normalize_pair(ctx,x,y,'metadata/protein_pair_manifest.json'))) == 1


@pytest.mark.parametrize('fault',['missing_reference','source_code','acceptance','pair_sha','order_sha','algorithm'])
def test_pair_faults(ctx,fault):
    a,p,b=pair(ctx)
    if fault=='missing_reference':p=None
    if fault=='source_code':b['builder_sha256']=a['builder_sha256']
    if fault=='acceptance':b['acceptance']['rows']=99
    if fault=='pair_sha':b['data_files'][next(iter(b['data_files']))]='0'*64
    if fault=='order_sha':b['candidate_order_record']['sha256']='0'*64
    if fault=='algorithm':p['source_release']='changed'
    with pytest.raises(ValueError):prepare(ctx,a,b,'metadata/protein_pair_manifest.json',pair_reference=p)


def test_cross_changes_only_recomputed_pair_counts_and_search(ctx):
    a={'search_output_sha256':m.OLD_SEARCH_SHA,'pair_task_overlap':{'sequence_similarity':{'rows':4},'remote':7},'threshold':.3}
    b=copy.deepcopy(a);b['search_output_sha256']=ctx['search_sha'];b['pair_task_overlap']['sequence_similarity']={'rows':8}
    x,y,_,_=prepare(ctx,a,b,'validation/cross_dataset_homology.json',cross_expected=b)
    assert x==y
    bad=copy.deepcopy(b);bad['threshold']=.4
    with pytest.raises(ValueError,match='recomputation'):prepare(ctx,a,bad,'validation/cross_dataset_homology.json',cross_expected=b)
    x,y,_,_=prepare(ctx,a,bad,'validation/cross_dataset_homology.json',cross_expected=bad)
    assert x!=y  # independently computed changes outside the whitelist still fail final comparison


def test_search_reference_and_current_both_bound(ctx):
    a={'search_output_sha256':m.OLD_SEARCH_SHA};b={'search_output_sha256':ctx['search_sha']}
    assert prepare(ctx,a,b,'metadata/protein_remote_pretraining_exclusion.json')[:2]==(a,a)
    b['search_output_sha256']='b'*64
    with pytest.raises(ValueError):prepare(ctx,a,b,'metadata/protein_remote_pretraining_exclusion.json')


def test_remote_pinned_erratum_current_descriptors_and_code(ctx):
    archive=HERE.parents[2]/'release_staging/m1_m3_2026-09-30_v1/data_rebuild/2026-09-25-v2'
    oldpath=archive/'metadata/remote_manifest.json'
    if not oldpath.exists():pytest.skip('Archived metadata integration fixture unavailable')
    a=json.loads(oldpath.read_text());b=copy.deepcopy(a)
    ctx['entries']['metadata/remote_manifest.json']={'sha256':m.rules.REMOTE_MANIFEST_SHA}
    ctx['entries'][m.REMOTE_SOURCE]={'sha256':m.rules.REMOTE_SOURCE_SHA,'bytes':1663}
    pin=ctx['builder_manifest']['files'][m.REMOTE_CODE]
    # Synthetic source tree uses the authenticated old code bytes for this integration.
    original=(archive/m.REMOTE_CODE).read_bytes()
    source_pin=write(Path(pin['source']),original)
    pin.update(source_sha256=source_pin['sha256'],source_bytes=source_pin['bytes'])
    ctx['entries'][m.REMOTE_CODE]=source_pin
    b['release_id']=m.NEW_ID;b['original_algorithm_release_id']=m.OLD_ID
    b['code'].update(file=pin['snapshot'],sha256=pin['sha256'])
    for item in b['files']:
        old=next(v for v in a['files'] if v['file']==item['file'])
        content=(archive/m.REMOTE_SOURCE).read_bytes() if item['file']==m.REMOTE_SOURCE else ('fixture '+item['file']).encode()
        ref={'sha256':m.rules.REMOTE_SOURCE_SHA,'bytes':1663} if item['file']==m.REMOTE_SOURCE else old
        item.update(artifact(ctx,item['file'],content,ref))
    x,y,_,changes=prepare(ctx,a,b,'metadata/remote_manifest.json')
    assert x['files'][11]['sha256']==m.rules.REMOTE_SOURCE_SHA and a['files'][11]['sha256']==m.rules.REMOTE_STALE_SHA
    assert any(c['rule']=='pinned_reference_only_descriptor_erratum' for c in changes)
    assert len(set(normalize_pair(ctx,x,y,'metadata/remote_manifest.json'))) == 1
    bad=copy.deepcopy(b);bad['files'][11]['bytes']=1825
    with pytest.raises(ValueError):prepare(ctx,a,bad,'metadata/remote_manifest.json')
    bad=copy.deepcopy(b);bad['code']['file']=pin['source']
    with pytest.raises(ValueError):prepare(ctx,a,bad,'metadata/remote_manifest.json')
    badref=copy.deepcopy(a);badref['files'][11]['sha256']='0'*64
    with pytest.raises(ValueError):prepare(ctx,badref,b,'metadata/remote_manifest.json')
