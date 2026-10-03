"""Hash-bound, source-only input reader for the newly rebuilt data smoke.

There is no historical archive resolver, model-weight loader or held-out role.
The raw release remains training-disabled. Only the separately frozen execution
protocol and independently audited derived inputs authorize this bounded smoke.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import stat
import sys

import numpy as np

CONDITION_STREAMS = {"EP": ("english_common", "protein"),
                     "ES": ("english_common", "shuffled"),
                     "EE": ("english_common", "english_extra")}
JOBS = ("pretrain_EP", "pretrain_ES", "pretrain_EE", "sft_EP", "audit_runtime")
RUNTIME_FILES = {"runtime.py", "runtime_data.py", "model.py", "audit_runtime.py"}


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
    def __init__(self, protocol_path, expected_protocol_sha256):
        require(isinstance(expected_protocol_sha256, str) and
                re.fullmatch(r"[0-9a-f]{64}", expected_protocol_sha256), "Explicit protocol SHA-256 required")
        self.accessed = {}
        self.protocol_path = regular_file(protocol_path)
        require(file_sha256(self.protocol_path) == expected_protocol_sha256, "Execution protocol SHA-256 differs")
        self.protocol_sha256 = expected_protocol_sha256
        self._pin(self.protocol_path, {"sha256": expected_protocol_sha256, "bytes": self.protocol_path.stat().st_size})
        protocol = json.loads(self.protocol_path.read_text())
        require(protocol.get("schema_version") == 1 and protocol.get("status") == "frozen_for_new_data_smoke" and
                protocol.get("scope") == "source_only_new_data_smoke", "Execution protocol is not frozen for this smoke")
        require(protocol.get("bounded_smoke_training_enabled") is True and
                protocol.get("full_budget_training_enabled") is False and
                protocol.get("target_scoring_enabled") is False and
                protocol.get("source_test_scoring_enabled") is False, "Execution scope differs")
        require(protocol["environment"]["python"] == sys.version, "Python environment changed")
        require({"numpy", "torch", "transformers", "tokenizers", "scikit-learn"} <=
                set(protocol["environment"]["packages"]), "Incomplete environment identity")
        for name, version in protocol["environment"]["packages"].items():
            require(importlib.metadata.version(name) == version, "Package version changed: " + name)
        code = protocol["runtime_code"]
        require(isinstance(code, dict) and code, "Missing runtime code identities")
        here = Path(__file__).absolute().parent
        require(all(str(here / name) in code for name in RUNTIME_FILES), "Incomplete runtime code inventory")
        for path, pin in code.items():
            self._pin(path, pin)
        self.execution_protocol = protocol
        self.design = protocol["design"]
        raw = protocol["raw_release"]
        self.raw_root = Path(raw["root"]).absolute()
        raw_manifest = self._json_descriptor(raw["manifest"])
        raw_acceptance = self._json_descriptor(raw["acceptance"])
        require(Path(raw["manifest"]["file"]).absolute() == self.raw_root / "manifest.json" and
                Path(raw["acceptance"]["file"]).absolute() == self.raw_root / "portable_validation/acceptance.json",
                "Raw control path differs")
        for value in (raw_manifest, raw_acceptance):
            require(value.get("status") == "accepted_new_raw_release_training_disabled" and
                    value.get("release_id") == "2026-10-01-portable-v2" and
                    value.get("training_enabled") is False and value.get("training_gate_pass") is False and
                    value.get("target_scoring_enabled") is False, "Raw release identity or preserved gate differs")
        require(raw_manifest["acceptance"]["sha256"] == raw["acceptance"]["sha256"], "Raw manifest acceptance binding differs")
        prepared = protocol["prepared"]
        self.root = Path(prepared["root"]).absolute()
        require(not self.root.is_relative_to(self.raw_root) and self.root != self.raw_root, "Prepared root must be separate")
        require(Path(prepared["manifest"]["file"]).absolute() == self.root / "manifest.json", "Prepared manifest path differs")
        self.manifest = self._json_descriptor(prepared["manifest"])
        self.manifest_sha256 = prepared["manifest"]["sha256"]
        accepted = self._json_descriptor(prepared["acceptance"])
        require(self.manifest.get("status") == "built_pending_independent_audit", "Unexpected prepared build status")
        require(self.manifest.get("raw_release_id") == "2026-10-01-portable-v2" and
                self.manifest.get("raw_acceptance_sha256") == raw["acceptance"]["sha256"] and
                self.manifest.get("raw_manifest_sha256") == raw["manifest"]["sha256"], "Prepared raw lineage differs")
        require(accepted.get("status") == "source_only_smoke_inputs_accepted" and
                accepted.get("prepared_manifest_sha256") == self.manifest_sha256 and
                accepted.get("data_protocol_sha256") == self.manifest["build_protocol_sha256"] and
                accepted.get("raw_release_manifest_sha256") == raw["manifest"]["sha256"] and
                accepted.get("raw_release_acceptance_sha256") == raw["acceptance"]["sha256"], "Independent source data gate differs")
        require(accepted.get("full_training_enabled") is False and accepted.get("target_scoring_enabled") is False and
                accepted.get("source_test_scoring_enabled") is False and accepted.get("all_checks_passed") is True,
                "Derived acceptance scope/checks differ")
        self.source_counts = self.manifest["source_counts"]
        require(set(self.source_counts) == {"train", "validation"} and
                accepted.get("source_counts") == self.source_counts and
                all(type(n) is int and n >= 64 for n in self.source_counts.values()), "Accepted source counts differ")
        require(self.manifest["condition_streams"] == {k: list(v) for k, v in CONDITION_STREAMS.items()}, "Condition stream assignment differs")
        require(set(self.manifest["source"]) == {"train", "validation"} and
                set(self.manifest["streams"]) == {"english_common", "english_extra", "protein", "shuffled"}, "Prepared roles differ")
        token = self.manifest["tokenizer"]
        require(Path(token["file"]).absolute() == self.raw_root / "tokenizers/mixed_bpe/tokenizer.json", "Tokenizer must come from new raw release")
        require(token["sha256"] == self.design["tokenizer"]["sha256"] and
                self.design["tokenizer"]["vocab_size"] == 32000 and
                [self.design["tokenizer"][k] for k in ("pad_token_id", "eos_token_id", "pair_separator_id", "unk_token_id")] == [0, 1, 2, 3],
                "Tokenizer configuration differs")
        self._pin(token["file"], {"sha256": token["sha256"], "bytes": Path(token["file"]).stat().st_size})
        require(isinstance(self.manifest["outputs"], dict), "Prepared output inventory missing")
        self.verify_current_inputs()

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
        require(condition in CONDITION_STREAMS and type(seed) is int and seed == 0, "Only the three pt0 smokes are authorized")
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
        require(spec["file"] == "schedules/seed0.npy" and self.manifest["outputs"][spec["file"]]["sha256"] == spec["sha256"],
                "Schedule binding differs")
        schedule = np.load(self.output_path(spec["file"]), allow_pickle=False)
        validate_schedule(schedule)
        schedule.flags.writeable = False
        return {"streams": tuple(streams), "schedule": schedule, "stream_names": CONDITION_STREAMS[condition]}
