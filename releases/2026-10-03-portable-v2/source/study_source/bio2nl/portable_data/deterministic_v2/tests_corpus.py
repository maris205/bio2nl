"""CPU fixtures for separate source trees, exact patches and phase barriers."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = load('corpus_phase_deterministic_v2_test', HERE / 'corpus_phase.py')
p = load('corpus_source_deterministic_v2_test', HERE / 'corpus_source.py')


def original(name):
    path = WORKSPACE / 'data_rebuild/2026-09-25-v2' / name
    if not path.exists():
        pytest.skip('Exact archived source integration fixture is unavailable')
    return path.read_bytes()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj) + '\n')


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / 'release'
    root.mkdir()
    execution = root / 'execution_code'
    entries, code = {}, {}
    names = list(p.PATCHES) + ['code/bio2nl/data/rebuild_v2/common.py']
    names += [f'code/biopaws/fixture_{i}.py' for i in range(24)]
    for name in names:
        source = root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        content = original(name) if name in p.PATCHES or name.endswith('/common.py') else b'# unchanged fixture\n'
        source.write_bytes(content)
        entries[name] = {**c.identity(source), 'kind': 'historical_builder_source'}
        output, provenance = p.patch_corpus_source(name, content) if name in p.PATCHES else (content, {'transform': 'identity'})
        snapshot = execution / name[5:]
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(output)
        code[name] = {'source': str(source), 'source_sha256': c.sha(source), 'source_bytes': len(content),
                      'snapshot': str(snapshot), **c.identity(snapshot), 'transform': provenance['transform']}
    for kind, count, prefix in [('raw_or_external_tool_asset', 77, 'raw'),
                                ('historical_source_provenance', 5, 'metadata')]:
        for i in range(count):
            name = f'{prefix}/fixture_{i}.txt'
            path = root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text('tiny immutable fixture ' + name)
            entries[name] = {**c.identity(path), 'kind': kind}
    bootstrap = root / 'portable_validation/bootstrap.json'
    write(bootstrap, {'status': 'raw_assets_seeded_not_a_data_acceptance', 'output_root': str(root), 'files': entries})
    manifest = root / 'portable_validation/builder_manifest.json'
    write(manifest, {'schema_version': 1, 'status': 'execution_code_overlay_not_data_acceptance',
                     'release_id': p.RELEASE_ID, 'release_root': str(root), 'builder_root': str(execution),
                     'bootstrap_sha256': c.sha(bootstrap), 'training_enabled': False,
                     'target_scoring_enabled': False, 'files': code})
    worker = tmp_path / 'tiny_worker.py'
    worker.write_text("import json,os,sys\nfrom pathlib import Path\n"
                      "root=Path(sys.argv[1]); name=sys.argv[2]\n"
                      "if name==os.environ.get('FIXTURE_FAIL'):sys.exit(7)\n"
                      "out=root/'products'/name;out.parent.mkdir(exist_ok=True)\n"
                      "out.write_text(json.dumps({k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','HF_HUB_OFFLINE','PYTHONPATH','PYTHONHASHSEED']}))\n"
                      "if name==os.environ.get('FIXTURE_MUTATE'):(root/'execution_code/biopaws/fixture_0.py').write_text('# changed')\n")
    names = [s['name'] for s in c.plan(root, 'initial', worker, execution)]
    stages = [{'name': name, 'script': str(worker),
               'command': [sys.executable, '-B', str(worker), str(root), name],
               'outputs': ['products/' + name], 'output_dirs': []} for name in names]
    return SimpleNamespace(root=root, execution=execution, bootstrap=bootstrap, manifest=manifest,
                           worker=worker, stages=stages, top=tmp_path)


def invoke(f, extra_env=None):
    args = ['phase', '--release-root', str(f.root), '--phase', 'initial',
            '--log-root', str(f.top / 'logs'), '--bootstrap-sha256', c.sha(f.bootstrap),
            '--smallweb-script', str(f.worker), '--smallweb-sha256', c.sha(f.worker),
            '--builder-root', str(f.execution), '--builder-manifest', str(f.manifest),
            '--builder-manifest-sha256', c.sha(f.manifest), '--release-id', p.RELEASE_ID]
    with patch.object(sys, 'argv', args), patch.object(c, 'plan', return_value=f.stages), \
            patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '', 'PYTHONPATH': '/untrusted', **(extra_env or {})}), \
            patch.object(c.shutil, 'disk_usage', return_value=SimpleNamespace(free=20 << 30)):
        c.main()


def check_overlay(f):
    bootstrap = c.verify_bootstrap(f.root, f.bootstrap, c.sha(f.bootstrap))
    return c.verify_builder_manifest(f.root, f.execution, f.manifest, c.sha(f.manifest),
                                     bootstrap, c.sha(f.bootstrap), p.RELEASE_ID)


def test_nlp_patch_exactly_reuses_previous_source_transformation():
    old = load('existing_deterministic_v1_patches', HERE.parent / 'deterministic_v1/patches.py')
    name = p.PREFIX + 'nlp_synthetic/build.py'
    source = original(name)
    assert p.patch_corpus_source(name, source)[0] == old.patch_source(name, source)[0]


@pytest.mark.parametrize('name', list(p.PATCHES))
def test_source_changes_and_repeated_patches_rejected(name):
    source = original(name)
    actual, record = p.patch_corpus_source(name, source)
    assert record['source_sha256'] == hashlib.sha256(source).hexdigest()
    with pytest.raises(ValueError, match='SHA'):
        p.patch_corpus_source(name, source + b'\n')
    with pytest.raises(ValueError, match='SHA'):
        p.patch_corpus_source(name, actual)
    with pytest.raises(ValueError, match='identity'):
        p.patch_corpus_source(name, source, 'different-release')


def test_real_config_builder_only_changes_release_identity(tmp_path):
    name = p.PREFIX + 'write_experiment_configs.py'
    generated = []
    for label, content in [('old', original(name)), ('new', p.patch_corpus_source(name, original(name))[0])]:
        folder = tmp_path / label
        folder.mkdir()
        (folder / 'common.py').write_bytes(original(p.PREFIX + 'common.py'))
        script = folder / 'write_experiment_configs.py'
        script.write_bytes(content)
        result = subprocess.run([sys.executable, '-B', str(script), '--root', str(folder)], capture_output=True, text=True,
                                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
        assert result.returncode == 0, result.stderr
        generated.append([json.loads((folder / 'configs' / fn).read_text())
                          for fn in ('training_matrix_v2.json', 'smallweb_gpt2_v2.json')])
    for old, new in zip(*generated):
        assert old.pop('data_release_id') == '2026-09-25-v2'
        assert new.pop('data_release_id') == p.RELEASE_ID
        assert old == new and new['training_not_started'] is True


def test_longrange_identity_patch_changes_only_declared_literal():
    name = p.PREFIX + 'nlp_synthetic/build_longrange.py'
    source = original(name)
    new, record = p.patch_corpus_source(name, source)
    assert new.replace(p.RELEASE_ID.encode(), b'2026-09-25-v2') == source
    assert record['replacement_count'] == 1


def test_overlay_is_separate_complete_and_preserves_bootstrap(fixture):
    f = fixture
    before = {name: c.identity(f.root / name) for name in json.loads(f.bootstrap.read_text())['files']}
    m = check_overlay(f)
    assert len(m['files']) == 28 and len(before) == 110
    assert {name: c.identity(f.root / name) for name in before} == before
    assert c.sha(f.root / next(iter(p.PATCHES))) != c.sha(f.execution / next(iter(p.PATCHES))[5:])


@pytest.mark.parametrize('tamper', ['missing', 'extra', 'executed', 'source', 'scientific_patch', 'source_descriptor', 'wrong_id'])
def test_overlay_tampering_refused(fixture, tamper):
    f = fixture
    m = json.loads(f.manifest.read_text()); name = p.PREFIX + 'write_experiment_configs.py'
    if tamper == 'missing':del m['files'][name]
    if tamper == 'extra':(f.execution / 'extra.py').write_text('')
    if tamper == 'executed':(f.execution / name[5:]).write_text('changed')
    if tamper == 'source':(f.root / name).write_text('changed')
    if tamper == 'scientific_patch':
        dest = f.execution / name[5:]
        dest.write_bytes(dest.read_bytes().replace(b'16_777_216', b'16_777_217'))
        m['files'][name].update(c.identity(dest))
    if tamper == 'source_descriptor':m['files'][name]['source'] = str(f.worker)
    if tamper == 'wrong_id':m['release_id'] = '2026-09-25-v2'
    write(f.manifest, m)
    with pytest.raises(ValueError):check_overlay(f)


def test_symlinked_tree_or_path_escape_refused(fixture):
    f = fixture
    (f.execution / 'linked.py').symlink_to(f.worker)
    with pytest.raises(ValueError, match='Symlink'):check_overlay(f)
    with pytest.raises(ValueError, match='Unsafe'):c.member(f.root, '../outside')


def test_real_tiny_workers_offline_marker_bound(fixture):
    f = fixture
    invoke(f)
    marker = c.verify_initial(f.root, c.sha(f.bootstrap), c.sha(f.manifest), p.RELEASE_ID, f.execution)
    state = json.loads((f.top / 'logs/status.json').read_text())
    assert marker['status_sha256'] == c.sha(f.top / 'logs/status.json')
    assert state['status'] == 'completed' and len(state['stages']) == 7
    assert state['release_id'] == p.RELEASE_ID and not state['training_enabled']
    env = json.loads((f.root / f.stages[0]['outputs'][0]).read_text())
    assert env == {'CUDA_VISIBLE_DEVICES': '', 'HF_HUB_OFFLINE': '1', 'PYTHONPATH': None, 'PYTHONHASHSEED': '0'}


@pytest.mark.parametrize('error', ['failure', 'post_worker_code_mutation'])
def test_first_worker_error_stops_and_no_completion(fixture, error):
    f = fixture
    env = {'FIXTURE_FAIL' if error == 'failure' else 'FIXTURE_MUTATE': f.stages[1]['name']}
    with pytest.raises(ValueError):invoke(f, env)
    state = json.loads((f.top / 'logs/status.json').read_text())
    assert state['status'] == 'failed' and len(state['stages']) == 2
    assert state['stages'][-1]['status'] == 'failed'
    assert not (f.root / '.corpus_initial_completed.json').exists()
    assert not (f.root / f.stages[2]['outputs'][0]).exists()


def test_existing_output_fails_before_launch(fixture):
    f = fixture
    (f.root / 'products').mkdir()
    path = f.root / f.stages[0]['outputs'][0]
    path.write_text('preserve')
    with pytest.raises(ValueError, match='existing stage output'):invoke(f)
    assert path.read_text() == 'preserve' and not (f.top / 'logs').exists()


@pytest.mark.parametrize('tamper', ['status', 'output', 'overlay_binding'])
def test_initial_barrier_rejects_stale_artifacts(fixture, tamper):
    f = fixture
    invoke(f)
    expected = c.sha(f.manifest)
    if tamper == 'status':
        path = f.top / 'logs/status.json'; path.write_text(path.read_text() + ' ')
    elif tamper == 'output':(f.root / f.stages[0]['outputs'][0]).write_text('changed')
    else:expected = '0' * 64
    with pytest.raises(ValueError):c.verify_initial(f.root, c.sha(f.bootstrap), expected, p.RELEASE_ID, f.execution)


def test_protein_dependency_uses_last_writer_and_binds_new_release(fixture):
    f = fixture
    stages = []
    for i in range(9):
        path = f.root / f'protein_output_{i}.txt'; path.write_text(str(i))
        stages.append({'name': c.PROTEIN_STAGE_NAMES[i], 'status': 'completed', 'returncode': 0,
                       'outputs': {path.name: c.identity(path)}})
    changed = f.root / 'protein_output_0.txt'; changed.write_text('final content')
    stages[-1]['outputs'][changed.name] = c.identity(changed)
    status = f.top / 'protein_status.json'
    state = {'status': 'completed', 'stages': stages, 'release_root': str(f.root), 'release_id': p.RELEASE_ID,
             'bootstrap_sha256': c.sha(f.bootstrap), 'builder_manifest_sha256': c.sha(f.manifest),
             'builder_root': str(f.execution / 'biopaws'), 'builder_manifest_file': str(f.manifest),
             'training_enabled': False, 'target_scoring_enabled': False, 'complete_release_gate_passed': False}
    write(status, state)
    actual = c.verify_protein_status(status, f.root, c.sha(f.bootstrap), c.sha(f.manifest), p.RELEASE_ID)
    assert actual['last_writer_outputs'][changed.name] == c.identity(changed)
    stages[0]['name'] = 'different_stage'; write(status, state)
    with pytest.raises(ValueError, match='incomplete'):
        c.verify_protein_status(status, f.root, c.sha(f.bootstrap), c.sha(f.manifest), p.RELEASE_ID)
    stages[0]['name'] = c.PROTEIN_STAGE_NAMES[0]
    state['release_id'] = '2026-09-25-v2'; write(status, state)
    with pytest.raises(ValueError, match='provenance'):
        c.verify_protein_status(status, f.root, c.sha(f.bootstrap), c.sha(f.manifest), p.RELEASE_ID)
