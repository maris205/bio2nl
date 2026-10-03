import csv
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

# Works with unittest discover and with the exported source in an empty cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deterministic_v1 import patches as p
from deterministic_v1 import remote_provenance as r


def serialize_canonical(records):
    """Execute the actual inserted block against a CSV writer fixture."""
    result = io.StringIO(newline='')
    def write_tsv(path, rows, fields):
        writer = csv.DictWriter(result, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    exec(textwrap.dedent(p.CANONICAL_AFTER), {'records': records, 'root': Path('/fixture'), 'write_tsv': write_tsv})
    return result.getvalue()


class DeterministicTests(unittest.TestCase):
    def test_exact_source_change_requires_pin_and_unique_anchor(self):
        source=b'prefix\n        for key in bad:\nbody\n'
        changed=p.exact_edit(source,p.sha(source),p.NLP_BEFORE,p.NLP_AFTER)
        self.assertEqual(changed,source.replace(p.NLP_BEFORE.encode(),p.NLP_AFTER.encode()))
        with self.assertRaisesRegex(ValueError,'Source SHA'):
            p.exact_edit(source+b' ',p.sha(source),p.NLP_BEFORE,p.NLP_AFTER)
        twice=source+source
        with self.assertRaisesRegex(ValueError,'exactly one'):
            p.exact_edit(twice,p.sha(twice),p.NLP_BEFORE,p.NLP_AFTER)

    def test_production_scope_and_altered_source_rejected(self):
        with self.assertRaisesRegex(ValueError,'two-file'):
            p.patch_source('unknown.py',b'x')
        for name in p.PATCHES:
            with self.assertRaisesRegex(ValueError,'Source SHA'):
                p.patch_source(name,b'changed builder')

    def test_bad_source_creates_no_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'source';output=Path(tmp)/'new'
            for name in p.PATCHES:
                file=root/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Source SHA'):
                p.write_patched_snapshot(root,output)
            self.assertFalse(output.exists())

    def test_existing_snapshot_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'new';output.mkdir();sentinel=output/'keep';sentinel.write_text('unchanged')
            with self.assertRaisesRegex(ValueError,'Fresh'):
                p.write_patched_snapshot(Path(tmp)/'source',output)
            self.assertEqual(sentinel.read_text(),'unchanged')

    def test_conflict_group_permutations_preserve_complete_rows(self):
        groups={'group-z':[{'row':'z1'},{'row':'z2'}], 'group-a':[{'row':'a1'}], 'group-m':[{'row':'m1'}]}
        expected=[{'row':'a1'},{'row':'m1'},{'row':'z1'},{'row':'z2'}]
        code='def collect():\n    if True:\n'+p.NLP_AFTER+'            out.extend(groups[key])\ncollect()\n'
        for keys in itertools.permutations(groups):
            scope={'bad':set(keys),'groups':groups,'out':[]};exec(code,scope)
            self.assertEqual(scope['out'],expected)
        groups['group-a'][0]['row']='changed'
        scope={'bad':set(groups),'groups':groups,'out':[]};exec(code,scope)
        self.assertNotEqual(scope['out'],expected)

    def test_real_hash_seeds_produce_identical_exclusion_order(self):
        code=("import json\ngroups={str(i): [{'row': i, 'slot': j} for j in range(2)] for i in range(30)}\n"
              "bad=set(groups)\nout=[]\ndef collect():\n    if True:\n"+p.NLP_AFTER+
              "            out.extend(groups[key])\ncollect()\nprint(json.dumps(out,sort_keys=True))\n")
        results=[]
        with tempfile.TemporaryDirectory() as tmp:
            for seed in ('0','1','17','random'):
                env=os.environ.copy();env.update(PYTHONHASHSEED=seed,PYTHONDONTWRITEBYTECODE='1')
                env.pop('PYTHONPATH',None);env.pop('PYTHONSTARTUP',None)
                results.append(subprocess.run([sys.executable,'-B','-c',code],cwd=tmp,env=env,check=True,
                                              stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout)
        self.assertEqual(len(set(results)),1)
        self.assertEqual(len(json.loads(results[0])),60)

    def test_canonical_named_values_and_row_order_preserved(self):
        original={key:'value-'+key for key in p.CANONICAL_COLUMNS}
        second={key:'second-'+key for key in reversed(p.CANONICAL_COLUMNS)}
        baseline=serialize_canonical([original,second])
        for offset in range(len(p.CANONICAL_COLUMNS)):
            keys=p.CANONICAL_COLUMNS[offset:]+p.CANONICAL_COLUMNS[:offset]
            changed={key:original[key] for key in keys}
            self.assertEqual(serialize_canonical([changed,second]),baseline)
        parsed=list(csv.DictReader(io.StringIO(baseline),delimiter='\t'))
        self.assertEqual(parsed,[original,second])
        self.assertEqual(baseline.splitlines()[0].split('\t'),list(p.CANONICAL_COLUMNS))
        self.assertNotEqual(serialize_canonical([second,original]),baseline)

    def test_canonical_missing_or_extra_field_fails(self):
        original={key:'x' for key in p.CANONICAL_COLUMNS}
        for record in ({k:v for k,v in original.items() if k!='split'}, {**original,'new_field':'must not vanish'}):
            with self.assertRaisesRegex(ValueError,'schema changed'):
                serialize_canonical([original,record])

    def test_canonical_value_change_remains_identifiable(self):
        original={key:'x' for key in p.CANONICAL_COLUMNS}
        before=serialize_canonical([original]);after=serialize_canonical([{**original,'split':'changed'}])
        self.assertNotEqual(hashlib.sha256(before.encode()).hexdigest(),hashlib.sha256(after.encode()).hexdigest())

    def test_real_hash_seeds_produce_identical_column_serialization(self):
        code=("import csv,io\nfrom pathlib import Path\n"
              f"columns={p.CANONICAL_COLUMNS!r}\nrecords=[{{key: 'value-'+key for key in set(columns)}}]\n"
              "root=Path('/fixture')\nout=io.StringIO(newline='')\n"
              "def write_tsv(path, rows, fields):\n"
              "    writer=csv.DictWriter(out,fieldnames=fields,delimiter='\\t',extrasaction='ignore')\n"
              "    writer.writeheader(); writer.writerows(rows)\n"+
              textwrap.dedent(p.CANONICAL_AFTER)+"print(out.getvalue(),end='')\n")
        results=[]
        with tempfile.TemporaryDirectory() as tmp:
            for seed in ('0','1','17','random'):
                env=os.environ.copy();env.update(PYTHONHASHSEED=seed,PYTHONDONTWRITEBYTECODE='1')
                env.pop('PYTHONPATH',None);env.pop('PYTHONSTARTUP',None)
                results.append(subprocess.run([sys.executable,'-B','-c',code],cwd=tmp,env=env,check=True,
                                              stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout)
        self.assertEqual(len(set(results)),1)

    def test_wrong_historical_erratum_bytes_rejected(self):
        with self.assertRaisesRegex(ValueError,'historical remote manifest'):
            r.historical_remote_erratum(b'{}',b'{}')


class RemoteManifestTests(unittest.TestCase):
    def fixture(self,root):
        paths={'metadata/remote_sources.json':b'{"source":"fixed"}\n','data/pairs/remote_fixture.tsv':b'id\tvalue\na\t1\n'}
        entries=[]
        for name,data in paths.items():
            file=root/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(data)
            entries.append({'file':name,'bytes':len(data),'sha256':p.sha(data)})
        code=root/'code/build.py';code.parent.mkdir();code.write_text('# source fixture\n')
        value={'source_record':r.SOURCE_NAME,'files':entries,'code':{'file':str(code),'sha256':p.sha(code.read_bytes())}}
        (root/'metadata/remote_manifest.json').write_text(json.dumps(value));return value

    def update(self,root,value):
        (root/'metadata/remote_manifest.json').write_text(json.dumps(value))

    def test_current_references_verified_without_historical_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);result=r.verify_current_remote_manifest(root)
            self.assertEqual(result['file_count'],2)
            self.assertFalse(result['historical_exact_reproduction_claimed'])
            self.assertFalse(result['training_enabled']);self.assertFalse(result['target_scoring_enabled'])

    def test_changed_file_same_size_different_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);file=root/r.SOURCE_NAME
            file.write_bytes(file.read_bytes().replace(b'fixed',b'other'))
            with self.assertRaisesRegex(ValueError,'own file'):
                r.verify_current_remote_manifest(root)

    def test_stale_hash_or_size_and_boolean_size_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);value=self.fixture(root)
            for key,new in [('sha256','0'*64),('bytes',1825),('bytes',True)]:
                before=value['files'][0][key];value['files'][0][key]=new;self.update(root,value)
                with self.assertRaises(ValueError):r.verify_current_remote_manifest(root)
                value['files'][0][key]=before

    def test_descriptor_for_another_files_valid_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);value=self.fixture(root)
            value['files'][0].update({key:value['files'][1][key] for key in ('sha256','bytes')});self.update(root,value)
            with self.assertRaisesRegex(ValueError,'own file'):r.verify_current_remote_manifest(root)

    def test_duplicate_and_escaping_descriptors_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);value=self.fixture(root);value['files'].append(value['files'][0]);self.update(root,value)
            with self.assertRaisesRegex(ValueError,'Duplicate'):r.verify_current_remote_manifest(root)
            value['files'].pop();value['files'][0]['file']='../outside';self.update(root,value)
            with self.assertRaisesRegex(ValueError,'Escaping'):r.verify_current_remote_manifest(root)

    def test_symlinked_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);file=root/r.SOURCE_NAME;target=root/'kept';file.rename(target);file.symlink_to(target)
            with self.assertRaisesRegex(ValueError,'Symlinked'):r.verify_current_remote_manifest(root)

    def test_external_code_requires_explicit_snapshot_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'release';root.mkdir();value=self.fixture(root)
            outside=Path(tmp)/'snapshot';outside.mkdir();file=outside/'build.py';file.write_text('# external snapshot\n')
            value['code']={'file':str(file),'sha256':p.sha(file.read_bytes())};self.update(root,value)
            with self.assertRaisesRegex(ValueError,'declared roots'):r.verify_current_remote_manifest(root)
            self.assertEqual(r.verify_current_remote_manifest(root,allowed_code_roots=[outside])['code']['file'],str(file))


if __name__=='__main__':
    unittest.main()
