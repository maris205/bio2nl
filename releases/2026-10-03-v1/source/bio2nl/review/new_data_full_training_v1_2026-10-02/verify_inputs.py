"""Fresh CPU verification of accepted new inputs; no model or held-out loader."""
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np

R = Path(__file__).absolute().parent
W = R.parents[2]
PRIOR = W / "bio2nl/review/new_data_training_smoke_v1_2026-10-01"
D = W / "data_rebuild/2026-10-01-portable-v2"


def identity(path):
    path = Path(path).absolute()
    assert path.is_file() and not any(p.is_symlink() for p in (path, *path.parents))
    before = path.stat()
    with path.open("rb") as handle:
        sha = hashlib.file_digest(handle, "sha256").hexdigest()
    after = path.stat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
    return {"sha256": sha, "bytes": after.st_size}


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    destination = R / "verification/input_revalidation.json"
    assert not destination.exists()
    pins = {}

    def bind(path, expected=None):
        path = Path(path).absolute()
        value = identity(path)
        if isinstance(expected, str):
            assert value["sha256"] == expected, str(path)
        elif expected is not None:
            assert value == {k: expected[k] for k in ("sha256", "bytes")}, str(path)
        assert pins.setdefault(str(path), value) == value
        return path

    def read(path, expected=None):
        return json.loads(bind(path, expected).read_text())

    raw = read(D / "manifest.json", "c65ad951dced3285856f8edf49618e3d6b5329fc4ad006520dab748d35b407bf")
    raw_gate = read(D / "portable_validation/acceptance.json", "dc35bd6dd36566fef25120434419ab2b358f204678b5d83f7daa89903f5be3f2")
    prepared = PRIOR / "prepared"
    manifest = read(prepared / "manifest.json", "2ea23ffd2793ed2142156f2942cc4bfb77767a0c3e87ba668a77c34a3a16c010")
    gate = read(PRIOR / "data_acceptance.json", "395752ab8a836d1d82c829d60c37b189071e5b7b6a023bbad3bc29b3d5a97043")
    assert gate["status"] == "source_only_smoke_inputs_accepted" and gate["all_checks_passed"]
    assert gate["prepared_manifest_sha256"] == pins[str(prepared / "manifest.json")]["sha256"]
    assert gate["raw_release_manifest_sha256"] == pins[str(D / "manifest.json")]["sha256"]
    assert gate["raw_release_acceptance_sha256"] == pins[str(D / "portable_validation/acceptance.json")]["sha256"]
    for value in (raw, raw_gate):
        assert value["status"] == "accepted_new_raw_release_training_disabled"
        assert value["training_enabled"] is False and value["training_gate_pass"] is False
    assert gate["full_training_enabled"] is False and gate["target_scoring_enabled"] is False
    for path, expected in gate["verified_read_files"].items():
        bind(path, expected)
    for relative, expected in manifest["outputs"].items():
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        bind(prepared / relative, expected)
    assert len(manifest["outputs"]) == 24
    source = {}
    for split, expected_rows, expected_blocks in (("train", 8044, 63), ("validation", 20276, 695)):
        entry = manifest["source"][split]
        with np.load(prepared / entry["npz"], allow_pickle=False) as loaded:
            labels = loaded["labels"]
            assert loaded["input_ids"].shape == loaded["attention_mask"].shape == (expected_rows, 512)
            assert labels.dtype == np.int64 and labels.shape == (expected_rows,)
            assert np.bincount(labels, minlength=2).tolist() == [expected_rows // 2] * 2
        with gzip.open(prepared / entry["rows"], "rt") as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == len({row["row_id"] for row in rows}) == expected_rows
        assert len({row["metadata"]["block_id"] for row in rows}) == expected_blocks
        assert [row["label"] for row in rows] == labels.tolist()
        assert all(row["metadata"]["split"] == split for row in rows)
        source[split] = {"rows": expected_rows, "blocks": expected_blocks,
                         "labels": [expected_rows // 2] * 2}
    for name in ("english_common", "english_extra", "protein", "shuffled"):
        tokens = np.memmap(prepared / "streams" / (name + ".bin"), dtype=np.uint16, mode="r")
        assert len(tokens) == 8388608 and int(tokens.max()) < 32000
        del tokens
    for seed in range(3):
        schedule = np.load(prepared / "schedules" / f"seed{seed}.npy", allow_pickle=False)
        assert schedule.dtype == np.int64 and schedule.shape == (2048, 16, 2)
        assert (schedule[:, :8, 0] == 0).all() and (schedule[:, 8:, 0] == 1).all()
        for sid in (0, 1):
            assert np.array_equal(np.sort(schedule[:, :, 1][schedule[:, :, 0] == sid]), np.arange(16384))
    failed = read(PRIOR / "execution_status.json")
    repaired = read(PRIOR / "reload_review_v2/review_result.json")
    replay = read(PRIOR / "reload_review_v2/results/metadata.json")
    read(PRIOR / "reload_review_v2/completion_review.json")
    assert failed["status"] == "failed"
    assert repaired["status"] == "completed" and repaired["checkpoints_replayed"] == 4 and repaired["final_pins_unchanged"]
    assert replay["all_new_checkpoint_replays_passed"] and replay["fixed_absolute_tolerance"] == 1e-5
    assert replay["optimizer_updates_performed"] == 0
    bind(Path(__file__).absolute())
    for path, value in pins.items():
        assert identity(path) == value
    report = {"schema_version": 1, "status": "passed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_counts": {"train": 8044, "validation": 20276}, "source": source,
              "prepared_manifest_sha256": pins[str(prepared / "manifest.json")]["sha256"],
              "prepared_outputs_rehashed": 24, "streams_checked": 4, "schedules_checked": 3,
              "full_source_updates_per_fit": 1260, "full_source_presentations_per_fit": 40220,
              "original_raw_and_derived_gates_unchanged": True, "original_failed_queue_preserved": True,
              "prior_separate_reload_review_passed": True, "models_loaded": False,
              "source_test_pair_or_target_examples_opened": False,
              "identity_tables_physically_contain_sequence_bytes": True,
              "verified_files": pins, "verified_file_count": len(pins), "all_final_hashes_unchanged": True}
    with destination.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("verified_files", "source")}))


if __name__ == "__main__":
    main()
