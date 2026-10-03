"""New hash-bound portable prepared-input reader; old gates remain untouched."""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import stat
import sys

import numpy as np

CONDITION_STREAMS = {"EP": ("english_common", "protein"),
                     "ES": ("english_common", "shuffled"),
                     "EE": ("english_common", "english_extra")}
RUNTIME_FILES = {"runtime.py", "runtime_data.py", "model.py"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def regular_file(path):
    path = Path(path).expanduser().absolute()
    for part in [*reversed(path.parents), path]:
        require(not part.is_symlink(), "Symlinked file or ancestor forbidden: " + str(part))
    require(stat.S_ISREG(path.stat().st_mode), "Expected regular file: " + str(path))
    return path


def file_sha256(path):
    path = regular_file(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    after = regular_file(path).stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "File changed while hashing")
    return digest.hexdigest()


def relative_path(value):
    require(isinstance(value, str) and value and not value.startswith("/") and "\\" not in value and
            "\0" not in value and all(x not in ("", ".", "..") for x in value.split("/")), "Noncanonical relative path")
    return value


def validate_schedule(value):
    require(value.dtype == np.int64 and value.shape == (2048, 16, 2), "Schedule shape/dtype differs")
    require(np.all(value[:, :8, 0] == 0) and np.all(value[:, 8:, 0] == 1), "Schedule stream assignment differs")
    for stream in (0, 1):
        used = value[:, :, 1][value[:, :, 0] == stream]
        require(np.array_equal(np.sort(used), np.arange(16384)), "Schedule omitted or repeated stream blocks")


def validate_source_arrays(arrays, rows, split, count):
    require(split in ("train", "validation"), "Only source train/validation is authorized")
    require(set(arrays) == {"input_ids", "attention_mask", "labels"}, "Unexpected source array fields")
    require(type(count) is int and count >= 64 and len(rows) == count, "Source row count differs")
    require(len({r["row_id"] for r in rows}) == count, "Duplicate source row identity")
    ids, masks, labels = (arrays[k] for k in ("input_ids", "attention_mask", "labels"))
    require(all(a.dtype == np.int64 for a in arrays.values()), "Source arrays must be int64")
    require(ids.shape == masks.shape == (count, 512) and labels.shape == (count,), "Source array dimensions differ")
    require(np.isin(labels, (0, 1)).all() and np.isin(masks, (0, 1)).all(), "Invalid source label/mask")
    for index, row in enumerate(rows):
        require(row["row_index"] == index and type(row["row_index"]) is int, "Source row order differs")
        require(isinstance(row["row_id"], str) and row["row_id"] and row["metadata"]["split"] == split and
                row["metadata"].get("block_id"), "Source row/split/group identity differs")
        require(type(row["label"]) is int and row["label"] == int(labels[index]), "Source label differs")
        a, b = row["ids_a"], row["ids_b"]
        require(isinstance(a, list) and isinstance(b, list) and 1 <= len(a) <= 255 and 1 <= len(b) <= 255,
                "Source endpoint cap differs")
        require(all(type(x) is int and 4 <= x < 32000 for x in a + b), "Invalid source content token")
        sequence = a + [2] + b + [1]
        n = len(sequence)
        require(np.array_equal(ids[index, :n], sequence) and np.all(ids[index, n:] == 0), "Source pair encoding differs")
        require(np.all(masks[index, :n] == 1) and np.all(masks[index, n:] == 0), "Source attention mask differs")



class ReleaseData:
    """New portable source-only gate. It never loads or rewrites old acceptance."""
    def __init__(self, protocol_path, expected_protocol_sha256, mode="prepare"):
        require(mode in ('prepare','smoke','full'), 'Unknown access mode')
        self.mode=mode;self.full_training_accepted=False;self.accessed={}
        self.protocol_path=regular_file(protocol_path)
        require(re.fullmatch('[0-9a-f]{64}',expected_protocol_sha256 or '') is not None,'Explicit protocol SHA required')
        self.protocol_sha256=expected_protocol_sha256
        self._pin(self.protocol_path,{'sha256':expected_protocol_sha256,'bytes':self.protocol_path.stat().st_size})
        p=json.loads(self.protocol_path.read_text());self.execution_protocol=p
        require(p.get('schema_version')==1 and p.get('status')=='frozen_portable_source_training_v1' and p.get('scope')=='source_only_portable_reproduction','Portable protocol identity differs')
        for flag in ('target_scoring_enabled','source_test_scoring_enabled','old_weights_allowed'):
            require(p.get(flag) is False,'Forbidden scope: '+flag)
        require(type(p.get('cpu_validation_only')) is bool,'Explicit compiler execution mode required')
        require(p.get('bounded_smoke_training_enabled') is (not p['cpu_validation_only']) and p.get('full_budget_training_enabled') is (not p['cpu_validation_only']),'Compiler execution scope differs')
        require(mode=='prepare' or not p['cpu_validation_only'],'CPU-validation protocol cannot authorize GPU smoke/full training')
        here=Path(__file__).absolute().parent
        required={'runtime.py','runtime_data.py','model.py','audit_runtime.py','fit_surface.py','surface_features.py','audit_surface.py','audit_source.py','support.py','run_queue.py','accept_training.py','cpu_check.py','design.json','scientific_payload_lock.json','algorithm_lock.json'}
        code=p['runtime_code'];require(set(code)=={str(here/n) for n in required},'Runtime code inventory differs')
        for path,desc in code.items():self._pin(path,desc)
        env=p['environment'];require(env['python']==sys.version,'Runtime Python identity changed')
        require(set(env['packages'])=={'numpy','torch','transformers','tokenizers','scipy','scikit-learn','pyarrow'},'Incomplete runtime environment pins')
        import torch
        require(torch.__version__==env['torch_build_version'] and torch.version.cuda==env['torch_cuda_build'],'PyTorch binary/CUDA build changed')
        if mode in ('smoke','full'):
            from support import require_gpu_environment
            require_gpu_environment(env)
        for package,version in env['packages'].items():require(importlib.metadata.version(package)==version,'Runtime package changed: '+package)
        self.design=p['design'];require(canonical_hash(self.design)==canonical_hash(json.loads((here/'design.json').read_text())),'Locked October 2 design differs')
        self.output_root=Path(p['output_root']).absolute();self.source_code_root=Path(p['source_code_root']).absolute()
        require(self.output_root==self.protocol_path.parent/'results','Output location differs')
        prepared=p['prepared'];self.root=Path(prepared['root']).absolute()
        require(not self.output_root.is_relative_to(self.root) and not self.root.is_relative_to(self.output_root),'Prepared/output roots overlap')
        require(Path(prepared['manifest']['file'])==self.root/'manifest.json' and Path(prepared['acceptance']['file'])==self.root/'acceptance.json','Prepared controls must stay in declared root')
        self.manifest=self._json_descriptor(prepared['manifest']);self.manifest_sha256=prepared['manifest']['sha256']
        accepted=self._json_descriptor(prepared['acceptance'])
        m=self.manifest
        require(m.get('artifact_type')=='bio2nl_portable_prepared_inputs_v1' and m.get('status')=='prepared_inputs_built','Not portable prepared data')
        require(accepted.get('artifact_type')=='bio2nl_portable_preparation_acceptance_v1' and accepted.get('status')=='passed','Preparation independent audit missing')
        require(accepted.get('prepared_manifest_sha256')==self.manifest_sha256 and accepted.get('protocol_sha256')==m.get('protocol_sha256') and accepted.get('resource_lock_sha256')==m.get('input_resource_lock_sha256'),'Preparation audit bindings differ')
        require(accepted.get('scientific_payloads_exact')==22 and accepted.get('independent_invariant_checks'),'Incomplete independent data checks')
        checks=accepted['independent_invariant_checks']
        require(checks.get('scientific_payloads_exact')==22 and checks.get('output_payloads_rehashed')==len(m['outputs']),'Incomplete preparation inventory audit')
        require(checks.get('source_overlap')==dict(blocks=0,clusters=0,rows=0,sequences=0),'Preparation overlap audit differs')
        for role,count in (('train',8044),('validation',20276)):
            require(checks['source_roles'][role]==dict(labels_per_class=count//2,rows=count,tensor_row_replay='exact'),'Independent source replay differs')
        require(set(checks['schedules'])=={'0','1','2'} and all(v.get('complete_per_stream_permutation') is True for v in checks['schedules'].values()),'Independent schedule audit incomplete')
        require(set(checks['streams'])=={'english_common','english_extra','protein','shuffled'} and all(v.get('complete_index') is True and v.get('tokens')==8388608 for v in checks['streams'].values()),'Independent stream audit incomplete')
        require(accepted.get('historical_gate_reused') is False and accepted.get('new_preparation_only') is True,'Historical acceptance cannot authorize the new inputs')
        for value in (m,accepted):
            require(value.get('training_enabled') is False and value.get('target_scoring_enabled') is False,'Data acceptance may not grant training/target authority')
        require(m.get('source_test_examples_allowed') is False,'Held-out data access forbidden')
        require(re.fullmatch('[0-9a-f]{64}',m.get('input_resource_lock_sha256','')) is not None,'Resource-lock provenance missing')
        self.source_counts=m['source_counts'];require(self.source_counts=={'train':8044,'validation':20276},'Source counts differ')
        require(m['condition_streams']=={k:list(v) for k,v in CONDITION_STREAMS.items()},'Condition streams differ')
        require(set(m['source'])=={'train','validation'} and set(m['streams'])=={'english_common','english_extra','protein','shuffled'},'Prepared roles differ')
        expected=json.loads((here/'scientific_payload_lock.json').read_text())
        require(len(expected)==22 and all(m['outputs'].get(name)==desc for name,desc in expected.items()),'Scientific payload identities differ from October 2')
        for name,desc in m['outputs'].items():
            relative_path(name)
            require(not any(x in name.lower() for x in ('source/test','target/','heldout/')),'Held-out output in source manifest')
        token=m['tokenizer'];require(token['file']=='tokenizer/tokenizer.json' and token['sha256']==self.design['tokenizer']['sha256'],'Shared tokenizer differs')
        self._pin(self.root/token['file'],m['outputs'][token['file']])
        require(m['outputs'][token['file']]['sha256']==token['sha256'],'Tokenizer output pin differs')
        if mode=='full':self._accept_full_training()
        self.verify_current_inputs()

    def _accept_full_training(self):
        # Evidence is produced by the new runtime, never copied from October 2.
        from accept_training import validate_evidence
        evidence=validate_evidence(self)
        path=self.protocol_path.parent/'training_acceptance.json'
        require(str(path)==self.execution_protocol['training_acceptance_file'],'Acceptance path differs')
        value=self._json_descriptor({'file':str(path),'sha256':file_sha256(path),'bytes':path.stat().st_size})
        require(value.get('status')=='accepted_for_portable_source_training' and value.get('protocol_sha256')==self.protocol_sha256 and value.get('prepared_manifest_sha256')==self.manifest_sha256,'New full training acceptance missing/different')
        require(value.get('target_scoring_enabled') is False and value.get('source_test_scoring_enabled') is False and value.get('full_budget_training_enabled') is True,'Acceptance scope differs')
        require(value.get('verified_files')==evidence,'Acceptance evidence closure differs')
        for path,pin in evidence.items():self._pin(path,pin)
        self.full_training_accepted=True

    def _pin(self, path, expected):
        path = regular_file(path)
        require(type(expected.get("bytes")) is int and expected["bytes"] >= 0 and path.stat().st_size == expected["bytes"],
                "Input bytes differ: " + str(path))
        require(file_sha256(path) == expected["sha256"], "Input SHA-256 differs: " + str(path))
        value = {"sha256": expected["sha256"], "bytes": expected["bytes"]}
        previous = self.accessed.setdefault(str(path), value)
        require(previous == value, "Conflicting input pin")
        return path

    def _json_descriptor(self, item):
        path = self._pin(item["file"], item)
        return json.loads(path.read_text())

    def output_path(self, relative):
        relative = relative_path(relative)
        require(relative in self.manifest["outputs"], "Unbound prepared member")
        require(relative.startswith(("source/", "streams/", "schedules/")) and
                not any(word in relative.lower() for word in ("test", "target", "confirmation")), "Forbidden prepared role")
        return self._pin(self.root / relative, self.manifest["outputs"][relative])

    def source_context(self):
        return {"protocol": self.design, "protocol_sha256": self.protocol_sha256,
                "prepared_manifest": self.manifest, "prepared_manifest_sha256": self.manifest_sha256}

    def verify_current_inputs(self):
        for path, pin in tuple(self.accessed.items()):
            self._pin(path, pin)
        return dict(self.accessed)

    def load_source(self, split, *, smoke=False):
        require(split in ("train", "validation"), "Only source train or validation is authorized")
        entry = self.manifest["source"][split]
        count = self.source_counts[split]
        require(entry["count"] == count and entry["npz"] == f"source/{split}.npz" and
                entry["rows"] == f"source/{split}.rows.jsonl.gz", "Source descriptor differs")
        for key in ("npz", "rows"):
            require(self.manifest["outputs"][entry[key]]["sha256"] == entry[key + "_sha256"], "Source descriptor hash differs")
        with np.load(self.output_path(entry["npz"]), allow_pickle=False) as loaded:
            arrays = {k: loaded[k] for k in loaded.files}
        with gzip.open(self.output_path(entry["rows"]), "rt", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        validate_source_arrays(arrays, rows, split, count)
        prefix = self.manifest["source_prefix64"][split]
        require(prefix["rows"] == 64 and prefix["row_ids"] == [r["row_id"] for r in rows[:64]] and
                prefix["row_ids_sha256"] == canonical_hash(prefix["row_ids"]), "Frozen source prefix differs")
        smoke_spec = self.manifest["smoke_source"][split]
        require(smoke_spec == prefix and smoke_spec["indices"] == list(range(64)), "Source smoke selection differs")
        if smoke:
            arrays = {k: v[:64].copy() for k, v in arrays.items()}
            rows = rows[:64]
        for value in arrays.values():
            value.flags.writeable = False
        return arrays, rows

    def load_pretrain(self, condition, seed):
        require(condition in CONDITION_STREAMS and type(seed) is int and seed in (0, 1, 2), "Unknown pretraining condition/seed")
        require(self.mode != "smoke" or seed == 0, "Only pt0 smoke streams are authorized")
        streams = []
        for name in CONDITION_STREAMS[condition]:
            spec = self.manifest["streams"][name]
            require(spec["bin"] == f"streams/{name}.bin" and spec["dtype"] == "<u2" and
                    spec["tokens"] == 8388608 and spec["blocks"] == 16384, "Pretraining stream descriptor differs")
            require(self.manifest["outputs"][spec["bin"]]["sha256"] == spec["bin_sha256"], "Pretraining stream hash differs")
            values = np.memmap(self.output_path(spec["bin"]), mode="r", dtype="<u2").reshape(16384, 512)
            require(np.all((values == 1) | ((values >= 4) & (values < 32000))), "Invalid pretraining token")
            streams.append(values)
        spec = self.manifest["schedules"][str(seed)]
        require(spec["file"] == f"schedules/seed{seed}.npy" and self.manifest["outputs"][spec["file"]]["sha256"] == spec["sha256"],
                "Schedule binding differs")
        schedule = np.load(self.output_path(spec["file"]), allow_pickle=False)
        validate_schedule(schedule)
        schedule.flags.writeable = False
        return {"streams": tuple(streams), "schedule": schedule, "stream_names": CONDITION_STREAMS[condition]}
