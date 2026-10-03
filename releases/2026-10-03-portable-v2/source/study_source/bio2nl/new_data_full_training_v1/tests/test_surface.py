"""Numerical and integrity tests; fixtures never read research data or weights."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_surface as audit
import fit_surface as fitting
from runtime_data import file_sha256
from surface_features import vector_features


def surface_config():
    return dict(lambdas=[1e-4, 1e-3, .01, .1, 1.],
                optimizer=dict(accept_gradient_inf_le=1e-5, ftol=1e-12, gtol=1e-8,
                               maxiter=2000, maxls=50, method="L-BFGS-B"),
                dtype="float64", expected_cpu_fits=5, family="length_unigram_bigram",
                feature_dimension=9, initialization="all_zero", scaler_std_floor=1e-6,
                scaler_floor_replacement=1., scaler_fit="source_train_only_float64_population_mean_std",
                objective="mean_binary_CE_plus_half_lambda_squared_weight_norm_intercept_unpenalized",
                selection="minimum_source_validation_CE_ties_smaller_lambda",
                target_calibration=False, target_scaler_refit=False)


def fixture_release(tmp_path):
    rng = np.random.default_rng(19)
    sources = {}
    for split in ("train", "validation"):
        rows = []
        for index in range(80):
            # Equal endpoint lengths deliberately leave two constant scaler coordinates.
            first = rng.integers(4, 20, 6).tolist()
            second = rng.integers(4, 20, 6).tolist()
            if index % 2:
                second[:3] = first[:3]
            rows.append(dict(row_id=f"{split}_{index}", ids_a=first, ids_b=second,
                             label=index % 2))
        sources[split] = ({"labels": np.arange(80, dtype=np.int64) % 2}, rows)
    output_root = tmp_path / "results"

    class FakeRelease:
        def __init__(self, protocol, digest, mode):
            assert mode == "prepare"
            assert Path(protocol) == tmp_path / "protocol.json"
            assert digest == "a" * 64
            self.execution_protocol = {"output_root": str(output_root)}
            self.design = {"surface_reference": surface_config()}
            self.protocol_sha256 = digest
            self.manifest_sha256 = "b" * 64
            self.source_counts = {"train": 80, "validation": 80}
            self.accessed = {}

        def load_source(self, split, *, smoke):
            assert smoke is False and split in ("train", "validation")
            return sources[split]

        def verify_current_inputs(self):
            return dict(self.accessed)

    return FakeRelease, sources


@pytest.fixture
def integrated(tmp_path, monkeypatch):
    fake, sources = fixture_release(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(fitting, "ReleaseData", fake)
    monkeypatch.setattr(audit, "ReleaseData", fake)
    fitting.fit(tmp_path / "protocol.json", "a" * 64)
    return tmp_path, sources


def dump(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def rebind_selection_and_status(root, selection):
    """Allow an intentional data corruption through hash guards to exercise math checks."""
    surface = root / "results/surface"
    for candidate in selection["candidates"]:
        dump(surface / (candidate["candidate_id"] + ".result.json"), candidate)
    dump(surface / "selection.json", selection)
    status = json.loads((surface / "status.json").read_text())
    status["selection_sha256"] = file_sha256(surface / "selection.json")
    status["artifacts"] = {str(p.relative_to(root / "results")): file_sha256(p)
                           for p in sorted(surface.iterdir()) if p.name != "status.json"}
    dump(surface / "status.json", status)


@pytest.mark.parametrize("a,b", [([4], [5]), ([4], [4]), ([4, 5, 4], [5, 4, 5, 6]),
                                  ([9, 9, 9], [9, 9]), ([7, 8], [8]), ([4, 6], [5, 7])])
def test_independent_features_and_symmetry(a, b):
    assert np.allclose(vector_features(a, b), audit.independent_vector(a, b), atol=1e-14, rtol=0)
    assert np.array_equal(audit.independent_vector(a, b), audit.independent_vector(b, a))


@pytest.mark.parametrize("ids", [[], [3], [32000], [True], [4.0]])
def test_invalid_content_ids(ids):
    with pytest.raises(ValueError):
        vector_features(ids, [5])
    with pytest.raises(ValueError):
        audit.independent_vector(ids, [5])


def test_surface_gradient_finite_difference():
    rng = np.random.default_rng(1)
    x, y, theta = rng.normal(size=(17, 9)), np.arange(17) % 2, rng.normal(size=10)
    value, gradient = fitting.objective(theta, x, y, .01)
    approximate = []
    for i in range(10):
        delta = np.zeros(10)
        delta[i] = 1e-5
        approximate.append((fitting.objective(theta + delta, x, y, .01)[0]
                            - fitting.objective(theta - delta, x, y, .01)[0]) / 2e-5)
    assert np.isfinite(value) and np.allclose(gradient, approximate, atol=1e-8, rtol=0)


def test_intercept_unpenalized():
    x = np.zeros((4, 9))
    y = np.array([0, 0, 1, 1])
    theta = np.r_[np.zeros(9), 2.]
    a, ag = fitting.objective(theta, x, y, .0001)
    b, bg = fitting.objective(theta, x, y, 1.)
    assert a == b and np.array_equal(ag, bg)


def test_independent_metrics_fixed_direction_ties():
    y = np.array([0, 1, 0, 1], dtype=np.int64)
    margin = np.array([2., -2., 1., -1.])
    logp = np.column_stack((-np.logaddexp(0, margin), -np.logaddexp(0, -margin)))
    assert audit.independent_metrics(y, logp)["auroc"] == 0
    for scores in (logp, np.full((4, 2), -np.log(2))):
        for key, expected in fitting.metrics(y, scores).items():
            assert np.allclose(audit.independent_metrics(y, scores)[key], expected, atol=1e-14, rtol=0)
    tied = audit.independent_metrics(y, np.full((4, 2), -np.log(2)))
    assert tied["predicted_positive_fraction"] == 0 and tied["balanced_accuracy"] == .5


def test_real_five_fit_and_independent_replay(integrated):
    root, _ = integrated
    report = audit.audit(root / "protocol.json", "a" * 64)
    assert report["status"] == "passed" and report["candidates"] == 5
    assert report["prediction_artifacts"] == 10 and len(report["files"]) == 22
    first = json.loads((root / "results/surface/lambda0.json").read_text())
    assert first["floored_coordinate_count"] == 3
    assert first["scale"][:3] == [1., 1., 1.]
    with pytest.raises(ValueError, match="existing"):
        audit.audit(root / "protocol.json", "a" * 64)
    with pytest.raises(ValueError, match="existing"):
        fitting.fit(root / "protocol.json", "a" * 64)


def test_changed_artifact_hash_refused(integrated):
    root, _ = integrated
    path = root / "results/surface/lambda0.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="SHA-256"):
        audit.audit(root / "protocol.json", "a" * 64)
    assert json.loads((root / "verification/surface_audit.status.json").read_text())["status"] == "failed"
    assert not (root / "verification/surface_audit.json").exists()


def test_rebound_scaler_corruption_detected_independently(integrated):
    root, _ = integrated
    surface = root / "results/surface"
    selection = json.loads((surface / "selection.json").read_text())
    state = json.loads((surface / "lambda0.json").read_text())
    state["mean"][0] += .01
    dump(surface / "lambda0.json", state)
    selection["candidates"][0]["checkpoint_sha256"] = file_sha256(surface / "lambda0.json")
    rebind_selection_and_status(root, selection)
    with pytest.raises(ValueError, match="Train mean"):
        audit.audit(root / "protocol.json", "a" * 64)


def test_rebound_metric_corruption_detected_independently(integrated):
    root, _ = integrated
    selection = json.loads((root / "results/surface/selection.json").read_text())
    selection["candidates"][0]["metrics"]["train"]["accuracy"] += .01
    rebind_selection_and_status(root, selection)
    with pytest.raises(ValueError, match="Surface metric differs: accuracy"):
        audit.audit(root / "protocol.json", "a" * 64)


def test_rebound_label_corruption_detected_independently(integrated):
    root, _ = integrated
    path = root / "results/surface/lambda0.train.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = dict(data)
    arrays["labels"] = 1 - arrays["labels"]
    with path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    selection = json.loads((root / "results/surface/selection.json").read_text())
    descriptor = selection["candidates"][0]["predictions"]["train"]
    descriptor.update(sha256=file_sha256(path), bytes=path.stat().st_size)
    rebind_selection_and_status(root, selection)
    with pytest.raises(ValueError, match="Surface labels differ"):
        audit.audit(root / "protocol.json", "a" * 64)


def test_changed_selection_refused(integrated):
    root, _ = integrated
    selection = json.loads((root / "results/surface/selection.json").read_text())
    selection["selected_candidate"] = "not_selected"
    rebind_selection_and_status(root, selection)
    with pytest.raises(ValueError, match="Selected surface identity"):
        audit.audit(root / "protocol.json", "a" * 64)


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "surface/../outside", "target/input", "surface/nested/x"])
def test_artifact_path_escape_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        audit.Artifacts(tmp_path).pin(name)


def test_artifact_symlink_rejected(tmp_path):
    (tmp_path / "surface").mkdir()
    (tmp_path / "outside").write_text("x")
    (tmp_path / "surface/a").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="Symlink"):
        audit.Artifacts(tmp_path).pin("surface/a")


def test_no_gpu_scope_before_input_access(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ValueError, match="CPU-only"):
        fitting.fit(tmp_path / "absent", "a" * 64)
    with pytest.raises(ValueError, match="CPU-only"):
        audit.audit(tmp_path / "absent", "a" * 64)


def test_failed_optimizer_stops_without_selection(tmp_path, monkeypatch):
    fake, _ = fixture_release(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(fitting, "ReleaseData", fake)
    monkeypatch.setattr(fitting, "minimize", lambda *args, **kwargs:
                        SimpleNamespace(x=np.zeros(10), success=False, nit=2000))
    with pytest.raises(ValueError, match="optimizer failed"):
        fitting.fit(tmp_path / "protocol.json", "a" * 64)
    assert json.loads((tmp_path / "results/surface/status.json").read_text())["status"] == "failed"
    assert not (tmp_path / "results/surface/selection.json").exists()


def test_prediction_no_overwrite(tmp_path):
    path = tmp_path / "pred.npz"
    y, scores = np.array([0, 1]), np.full((2, 2), -np.log(2))
    fitting.save_predictions(path, y, scores, ["a", "b"])
    with pytest.raises(ValueError, match="existing"):
        fitting.save_predictions(path, y, scores, ["a", "b"])


@pytest.mark.parametrize("key,value", [("lambdas", [1., 2.]), ("initialization", "random"),
                                         ("target_calibration", True), ("scaler_std_floor", 1e-5)])
def test_protocol_extension_refused(key, value):
    config = surface_config()
    config[key] = value
    with pytest.raises(ValueError):
        fitting.validate_design(config)
