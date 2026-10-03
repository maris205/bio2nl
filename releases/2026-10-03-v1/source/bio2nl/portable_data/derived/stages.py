"""Explicit stage boundaries for data construction; no model libraries imported."""
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from .common import V2, M1, M2, M3, EVAL, inside, now, require, rows, semantic, sha, write_json, write_rows
from . import algorithms as alg


def fresh(store, name):
    folder = inside(store.output, name)
    require(not folder.exists(), 'Refusing existing stage: ' + name)
    folder.mkdir(parents=True)
    return folder


def tokenizer(store):
    from tokenizers import Tokenizer
    path = store.release_file('tokenizers/mixed_bpe/tokenizer.json')
    # Configuration and token identity, not merely vocab size, are exact.
    require(semantic(path) == semantic(store.archived(V2 + 'tokenizers/mixed_bpe/tokenizer.json')), 'Tokenizer semantics changed')
    value = Tokenizer.from_file(str(path)); value.no_padding(); value.no_truncation()
    require(value.get_vocab_size() == 32000, 'Unexpected vocabulary size')
    require([value.token_to_id(s) for s in ('<|pad_v2|>','<|eos_v2|>','<|pair_v2|>','<|unk_v2|>')] == [0,1,2,3], 'Special IDs changed')
    return value


def raw_source(store, split):
    require(split in ('train', 'validation', 'test'), 'Unknown source role')
    path = store.release_file(f'data/pairs/protein_sequence_similarity_{split}.tsv.gz')
    with gzip.open(path, 'rt') as stream:
        values = list(csv.DictReader(stream, delimiter='\t'))
    for row in values:
        row['label'] = int(row['label']); row.setdefault('row_id', row.get('pair_id'))
    alg.audit_split(values, 'protein_sequence_similarity', split)
    return values


def identities(values):
    return {'row_ids': {r['row_id'] for r in values}, 'groups': {r['metadata']['block_id'] for r in values},
        'sequences': {r['metadata']['sequence_sha256_' + x] for r in values for x in ('a','b')},
        'clusters': {r['metadata']['cluster_' + x] for r in values for x in ('a','b')}}


def check_separation(left, right):
    a,b = identities(left), identities(right)
    result = {key: len(a[key] & b[key]) for key in a}
    require(not any(result.values()), 'Cross-split source identities overlap')
    return result


def source(store):
    started = now(); out = fresh(store, 'source'); report = {'selection_seed': 20260925, 'train_target': 8000, 'roles': {}}
    prepared = {}
    for split, expected in [('train',8002), ('validation',20338)]:
        values = raw_source(store, split); source_count = len(values)
        selection = None
        if split == 'train':
            values, selection = alg.select_groups(values, 'block_id', 8000, 20260925, 'protein_sequence_similarity')
            require(selection['selected_groups'] == 90 and source_count == 99894, 'Source selection population changed')
        prepared[split] = [alg.prepared_row(row, 'protein_sequence_similarity') for row in values]
        require(len(values) == expected, 'Source row count changed')
        path = out / f'{split}.jsonl'; write_rows(path, prepared[split])
        store.compare(path, EVAL + f'protein_sequence_similarity/{split}.jsonl')
        report['roles'][split] = {'count': len(values), 'source_count': source_count, 'selection': selection}
    report['cross_split_overlap'] = check_separation(prepared['train'], prepared['validation'])
    report['source_test_examples_read'] = False; report['target_examples_read'] = False
    write_json(out / 'report.json', report)
    return store.finish('source', report, started)


def reference_documents(refs):
    import pyarrow.parquet as pq
    for entry in refs['old_nlp_files']:
        path = refs['files'][entry['file']]['path']
        domain = f"old_nlp:{entry['task']}:{entry['stage']}:{entry['split']}"
        records = pq.read_table(path, columns=entry['text_fields']).to_pylist() if entry['format'] == 'parquet' else rows(path)
        count = 0
        for i, row in enumerate(records):
            count += 1
            for field in entry['text_fields']:
                value = row[field]
                if value is None:
                    continue
                require(isinstance(value, str), 'Nontext reference')
                yield domain, f"{entry['file']}:{i}:{field}", value
        require(count == entry['rows'], 'Reference count changed')
    for entry in refs['english_corpora']:
        count = 0
        for row in rows(refs['files'][entry['file']]['path']):
            count += 1
            yield f"english:{entry['split']}", row[entry['id_field']], row[entry['text_field']]
        require(count == entry['rows'], 'English reference count changed')
    prefix = refs['tokenizer_prefix']; used = []; nbytes = 0
    expected_ids = json.loads(Path(refs['files'][prefix['record_ids_file']]['path']).read_text())[prefix['record_ids_key']]
    for row in rows(refs['files'][prefix['source_corpus_file']]['path']):
        if nbytes >= prefix['utf8_bytes']:
            break
        text = row['text'].encode()[:prefix['utf8_bytes']-nbytes].decode('utf-8', errors='ignore')
        if text:
            used.append(row['record_id']); nbytes += len(text.encode())
            yield 'tokenizer:english_prefix', row['record_id'], text
    require(used == expected_ids and nbytes == prefix['utf8_bytes'] and len(used) == prefix['records'], 'Tokenizer prefix changed')


def compare_report(actual, expected, ignore=()):
    # JSON object keys are strings after serialization (e.g. label counters).
    actual = json.loads(json.dumps(actual, allow_nan=False))
    expected = json.loads(json.dumps(expected, allow_nan=False))
    a = {k:v for k,v in actual.items() if k not in ignore}
    b = {k:v for k,v in expected.items() if k not in ignore}
    require(a == b, 'Reconstructed fixed-rule report differs')
    return len(a)


def m1(store):
    from .confirmation import construct
    started = now(); out = fresh(store, 'm1')
    refs = json.loads(store.archived(M1 + 'references.json').read_text())
    lock = json.loads(store.archived(M1 + 'sources/source_lock.json').read_text())
    # Metadata declares references. Actual reads resolve only through the new gate.
    required = {e['file'] for e in refs['old_nlp_files']} | {e['file'] for e in refs['english_corpora']}
    required |= {refs['tokenizer_prefix']['record_ids_file'], refs['tokenizer_prefix']['source_corpus_file'], 'tokenizers/mixed_bpe/tokenizer.json'}
    refs['files'] = {name: {'path': str(store.release_file(name))} for name in sorted(required)}
    tokenizer(store)
    raw = store.archived(M1 + 'raw/qqp_validation.parquet')
    require(sha(raw) == lock['selected_artifact']['expected_sha256'] and raw.stat().st_size == lock['selected_artifact']['expected_bytes'], 'Candidate bytes changed')
    report = construct(out, raw, lock, refs, reference_documents)
    require(report['qualification_passed'] and report['retained_rows'] == 39893, 'Fixed qualification differs')
    for name in ['data/confirmation.jsonl','data/encoded.jsonl','audit/exclusions.jsonl','audit/overlap_hits.jsonl']:
        store.compare(out / name, M1 + name)
    expected = json.loads(store.archived(M1 + 'results/build_report.json').read_text())
    fields = compare_report(report, expected, ('completed_at_utc','formal_wall_seconds'))
    summary = {'rows': 39893, 'components': 33128, 'all_fixed_rule_report_fields_reproduced': fields,
        'reference_documents': report['reference_documents'], 'source_test_examples_read': False,
        'target_examples_read_for_fixed_data_reconstruction': True, 'offline': True, 'new_first_access_claimed': False}
    return store.finish('m1', summary, started)


def m2(store):
    started = now(); out = fresh(store, 'm2'); tok = tokenizer(store)
    design = json.loads(store.archived('bio2nl/review/next_round_preparation_v1_2026-09-29/configs/joint_pretraining_design.json').read_text())
    require(design['pretraining']['seeds'] == [0,1,2] and design['pretraining']['input_tokens_per_run'] == 2*alg.BUDGET, 'Frozen budget changed')
    english_path = store.release_file('data/corpora/english/train.jsonl.gz')
    protein_path = store.release_file('data/corpora/protein/train.jsonl.gz')
    canonical_path = store.release_file('data/sequences/protein_canonical_sequences.tsv.gz')
    remote_path = store.release_file('data/sequences/remote_sequences.tsv.gz')
    canonical, by_sequence, clusters = alg.read_canonical_metadata(canonical_path)
    forbidden = {key for key,value in by_sequence.items() if value['split'] != 'train'}
    remote, remote_count = alg.remote_hashes(remote_path); forbidden |= remote
    tokenizer_records = json.loads(store.release_file('tokenizers/mixed_bpe/training_records.json').read_text())
    require(all(p in canonical and canonical[p]['split']=='train' and canonical[p]['eligible'] for p in tokenizer_records['protein']), 'Tokenizer protein training membership changed')
    # M2 reads M1 aggregate acceptance only, never its example files or row ledgers.
    m1_report = json.loads(store.previous('m1','results/build_report.json').read_text())
    require(m1_report['qualification_passed'] and m1_report['retained_rows']==39893, 'Reconstructed M1 not accepted')
    encoded = {}; source_ids = {}
    for split, count in [('train',8002), ('validation',20338)]:
        raw = store.previous('source', f'{split}.jsonl')
        encoded[split],source_ids[split] = alg.make_source(split,raw,tok,by_sequence,out/'source',out)
        require(encoded[split]['count']==count,'Source count changed')
        for suffix in ('.npz','.rows.jsonl.gz'):
            store.compare(out/'source'/f'{split}{suffix}',M2+'prepared/source/'+split+suffix)
    separation = {k:len(source_ids['train'][k]&source_ids['validation'][k]) for k in source_ids['train']}
    require(not any(separation.values()),'Source encoded overlap')
    writers = {name:alg.StreamWriter(out,out/'streams',name,tok) for name in ('english_common','english_extra')}
    seen_ids=set(); seen_text=set()
    for position,row in enumerate(rows(english_path)):
        parent,text=row['record_id'],row['text']
        require(row['split']=='train' and parent not in seen_ids and row['text_sha256'] not in seen_text,'English repeats or wrong split')
        require(alg.text_sha(text)==row['text_sha256'],'English text identity differs')
        seen_ids.add(parent);seen_text.add(row['text_sha256'])
        name='english_extra' if writers['english_common'].complete else 'english_common'
        writers[name].append(parent,position,text,alg.encode_content(tok,text))
        if writers['english_extra'].complete: break
    streams={name:w.finish() for name,w in writers.items()}
    require(not {x['parent_id'] for x in writers['english_common'].index_rows}&{x['parent_id'] for x in writers['english_extra'].index_rows},'English streams share parents')
    protein_entries,parent_file,parent_report=alg.build_protein(list(rows(protein_path)),canonical,forbidden,tok,out,out)
    streams.update(protein_entries)
    write_json(out/'protein_control_audit.json',parent_report)
    store.compare(out/'protein_control_audit.json',M2+'prepared/protein_control_audit.json')
    store.compare(out/parent_file,M2+'prepared/protein_parent_controls.jsonl.gz')
    for name in streams:
        for suffix in ('.bin','.index.jsonl'):
            store.compare(out/'streams'/(name+suffix),M2+'prepared/streams/'+name+suffix)
    schedules={}
    (out/'schedules').mkdir()
    for seed in range(3):
        path=out/'schedules'/f'seed{seed}.npy';np.save(path,alg.make_schedule(seed),allow_pickle=False)
        store.compare(path,M2+f'prepared/schedules/seed{seed}.npy')
        schedules[str(seed)]={'file':str(path.relative_to(out)),'sha256':sha(path),'shape':[2048,16,2],'dtype':'int64'}
    report={'source':encoded,'streams':streams,'schedules':schedules,'protein_parent_controls':parent_report,
        'condition_streams':{'EP':['english_common','protein'],'ES':['english_common','shuffled'],'EE':['english_common','english_extra']},
        'canonical_clusters':len(clusters),'remote_rows':remote_count,'source_overlap':separation,
        'target_examples_read':False,'source_test_examples_read':False,'near_homology_alignment_rerun_in_this_derived_stage':False}
    write_json(out/'report.json',report)
    return store.finish('m2',report,started)


def historical_selection_gate(store):
    barrier=json.loads(store.archived(M3+'source_barrier.json').read_text())
    protocol=json.loads(store.archived(M3+'protocol.json').read_text())
    require(barrier['status']=='frozen_sources_for_heldout_evaluation' and protocol['source_barrier_sha256']==sha(store.archived(M3+'source_barrier.json')),'Historical source barrier changed')
    selection_path=store.archived(M2+'results/source_selection.json')
    require(sha(selection_path)==protocol['source_selection']['sha256'],'Historical source selection changed')
    selection=json.loads(selection_path.read_text())
    require(selection['status']=='frozen' and selection['conditions_filtered_by_score'] is False,'Historical selection not complete')
    grid={(x['job']['condition'],x['job']['pt_seed'],x['job']['ft_seed']) for x in selection['selected']}
    require(len(selection['selected'])==27 and grid=={(c,p,f) for c in ('EP','ES','EE') for p in range(3) for f in range(3)},'Historical selection grid differs')
    return {'barrier_sha256':sha(store.archived(M3+'source_barrier.json')),'source_selection_sha256':sha(selection_path),
        'purpose':'historical provenance gate for data reconstruction only; no new model selection or inference'}


def m3(store):
    started=now();out=fresh(store,'m3');gate=historical_selection_gate(store);tok=tokenizer(store)
    # The source test is opened only in this held-out data stage, after the gate.
    values=raw_source(store,'test');test=[alg.prepared_row(x,'protein_sequence_similarity') for x in values]
    require(len(test)==20808,'Source-test size differs')
    write_rows(out/'source_test.raw.jsonl',test)
    store.compare(out/'source_test.raw.jsonl',EVAL+'protein_sequence_similarity/test.jsonl')
    for split in ('train','validation'):
        previous=list(rows(store.previous('source',split+'.jsonl')))
        check_separation(previous,test)
    roles={'source_test':alg.encode_role('source_test',test,tok,out)}
    target=list(rows(store.previous('m1','data/confirmation.jsonl')))
    prior=list(rows(store.previous('m1','data/encoded.jsonl')))
    roles['target']=alg.encode_role('target',target,tok,out,prior)
    for role in roles:
        for suffix in ('.npz','.rows.jsonl.gz'):
            store.compare(out/(role+suffix),M3+'prepared/'+role+suffix)
    report={'roles':roles,'historical_source_selection_gate':gate,'target_examples_read_for_fixed_data_reconstruction':True,
        'source_test_examples_read':True,'row_selection_changed':False,'model_inference_performed':False}
    write_json(out/'report.json',report)
    return store.finish('m3',report,started)


STAGES={'source':source,'m1':m1,'m2':m2,'m3':m3}
