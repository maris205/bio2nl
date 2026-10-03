"""Independent source surface feature, optimizer, prediction and selection audit.

Does not import the fitting program or its feature implementation. Source input
identity and role guards are shared through the reviewed ReleaseData reader.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

try:
    from .runtime_data import ReleaseData, file_sha256, regular_file, relative_path, require
except ImportError:
    from runtime_data import ReleaseData, file_sha256, regular_file, relative_path, require

FEATURE_COLUMNS = ["length_log1p_min", "length_log1p_max", "length_min_max_ratio",
                   "unigram_set_jaccard", "unigram_multiset_dice", "unigram_count_cosine",
                   "bigram_set_jaccard", "bigram_multiset_dice", "bigram_count_cosine"]


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value, *, replace=False):
    path = Path(path)
    require(replace or not path.exists(), "Refuse existing audit artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".partial")
    with pending.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    pending.replace(path)


def independent_vector(a, b):
    require(a and b and all(type(x) is int and 4 <= x < 32000 for x in a + b),
            "Invalid independent content IDs")
    values = [math.log1p(min(len(a), len(b))), math.log1p(max(len(a), len(b))),
              min(len(a), len(b)) / max(len(a), len(b))]
    for width in (1, 2):
        ac = Counter(tuple(a[j:j + width]) for j in range(len(a) - width + 1))
        bc = Counter(tuple(b[j:j + width]) for j in range(len(b) - width + 1))
        sa, sb = set(ac), set(bc)
        if not ac and not bc:
            values.extend([1., 1., 1.])
            continue
        if not ac or not bc:
            values.extend([0., 0., 0.])
            continue
        intersection = sa & sb
        values.extend([len(intersection) / len(sa | sb),
                       2 * sum(min(ac[k], bc[k]) for k in intersection) / (sum(ac.values()) + sum(bc.values())),
                       sum(ac[k] * bc[k] for k in intersection)
                       / math.sqrt(sum(v * v for v in ac.values()) * sum(v * v for v in bc.values()))])
    return values


def near(actual, expected, message, tolerance=1e-10):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a, b, atol=tolerance, rtol=0), message)


def independent_metrics(labels, logp):
    labels, logp = np.asarray(labels), np.asarray(logp)
    require(labels.dtype == np.int64 and labels.ndim == 1 and set(np.unique(labels)) == {0, 1},
            "Invalid independent metric labels")
    require(logp.dtype == np.float64 and logp.shape == (len(labels), 2) and np.isfinite(logp).all(),
            "Invalid independent score matrix")
    near(np.exp(logp).sum(1), np.ones(len(labels)), "Unnormalized independent scores", tolerance=2e-6)
    pred = np.argmax(logp, axis=1)
    confusion = np.bincount(2 * labels + pred, minlength=4).reshape(2, 2)
    tn, fp, fn, tp = (int(v) for v in confusion.ravel())
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    n0, n1 = tn + fp, tp + fn
    positive_ranks = rankdata(logp[:, 1] - logp[:, 0], method="average")[labels == 1]
    return dict(rows=len(labels), accuracy=(tn + tp) / len(labels),
                balanced_accuracy=.5 * (tn / n0 + tp / n1),
                mcc=(tp * tn - fp * fn) / denominator if denominator else 0.,
                auroc=float((positive_ranks.sum() - n1 * (n1 + 1) / 2) / (n0 * n1)),
                cross_entropy=float(-logp[np.arange(len(labels)), labels].mean()),
                predicted_positive_fraction=float(pred.mean()), confusion_matrix=confusion.tolist())


class Artifacts:
    def __init__(self, root):
        self.root, self.files = Path(root).absolute(), {}

    def pin(self, name, expected=None):
        name = relative_path(name)
        require(name.startswith("surface/") and len(Path(name).parts) == 2, "Unexpected surface artifact path")
        path = regular_file(self.root / name)
        digest = file_sha256(path)
        require(expected is None or digest == expected, "Surface artifact SHA-256 differs: " + name)
        previous = self.files.setdefault(str(path), digest)
        require(previous == digest, "Surface artifact changed during audit")
        return path

    def read(self, name, expected=None):
        return json.loads(self.pin(name, expected).read_text())

    def verify(self):
        for path, digest in self.files.items():
            require(file_sha256(path) == digest, "Surface artifact changed after audit: " + path)


def audit(protocol_path, protocol_sha256):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Surface audit must be CPU-only")
    release = ReleaseData(protocol_path, protocol_sha256, mode="prepare")
    output_root = Path(release.execution_protocol["output_root"]).absolute()
    destination = Path(protocol_path).absolute().parent / "verification/surface_audit.json"
    status_path = destination.with_name("surface_audit.status.json")
    require(not destination.exists() and not status_path.exists(), "Refuse existing independent surface audit")
    status = dict(status="running", started_at_utc=now(), protocol_sha256=protocol_sha256)
    atomic_json(status_path, status)
    try:
        files = Artifacts(output_root)
        selection = files.read("surface/selection.json")
        fit_status = files.read("surface/status.json")
        require(selection["status"] == "selected_pending_independent_audit"
                and fit_status["status"] == "completed" and fit_status["completed_candidates"] == 5,
                "Surface fitting is incomplete")
        require(selection["protocol_sha256"] == release.protocol_sha256
                and selection["prepared_manifest_sha256"] == release.manifest_sha256
                and selection["source_counts"] == release.source_counts,
                "Surface protocol/data binding differs")
        require(selection["source_only"] is True and selection["target_examples_read"] is False
                and selection["all_read_files_final_rehashed"] is True
                and selection["selection_metric"] == "source_validation_cross_entropy", "Surface role differs")
        require(fit_status["protocol_sha256"] == release.protocol_sha256
                and fit_status["prepared_manifest_sha256"] == release.manifest_sha256
                and fit_status["selection_sha256"] == files.files[str(output_root / "surface/selection.json")],
                "Surface completion bindings differ")
        expected_names = {"surface/selection.json"}
        for i in range(5):
            expected_names.update({f"surface/lambda{i}.json", f"surface/lambda{i}.result.json",
                                   f"surface/lambda{i}.train.npz", f"surface/lambda{i}.validation.npz"})
        require(set(fit_status["artifacts"]) == expected_names, "Surface artifact inventory differs")
        require({str(p.relative_to(output_root)) for p in (output_root / "surface").iterdir()}
                == expected_names | {"surface/status.json"}, "Unexpected or missing surface file")
        for name, digest in fit_status["artifacts"].items():
            files.pin(name, digest)
        arrays, rows, features = {}, {}, {}
        for split in ("train", "validation"):
            arrays[split], rows[split] = release.load_source(split, smoke=False)
            features[split] = np.asarray([independent_vector(r["ids_a"], r["ids_b"]) for r in rows[split]],
                                         dtype=np.float64)
        mean = features["train"].mean(0)
        std = features["train"].std(0, ddof=0)
        scale = np.where(std < 1e-6, 1., std)
        candidates = selection["candidates"]
        require(len(candidates) == 5 and release.design["surface_reference"]["lambdas"]
                == [1e-4, 1e-3, .01, .1, 1.], "Five fixed surface penalties required")
        for i, candidate in enumerate(candidates):
            require(candidate["candidate_id"] == f"lambda{i}"
                    and candidate["penalty"] == [1e-4, 1e-3, .01, .1, 1.][i], "Surface grid changed")
            require(candidate["optimizer"]["success"] is True
                    and type(candidate["optimizer"]["iterations"]) is int
                    and 0 <= candidate["optimizer"]["iterations"] <= 2000, "Optimizer convergence record differs")
            require(files.read(f"surface/lambda{i}.result.json") == candidate, "Candidate record differs")
            require(candidate["checkpoint"] == f"surface/lambda{i}.json", "Candidate checkpoint path differs")
            state = files.read(candidate["checkpoint"], candidate["checkpoint_sha256"])
            weights, bias = np.asarray(state["coefficients"], dtype=np.float64), state["bias"]
            require(weights.shape == (9,) and np.isfinite(weights).all() and math.isfinite(bias), "Invalid surface head")
            require(state["feature_columns"] == FEATURE_COLUMNS and state["source_train_only"] is True
                    and state["dtype"] == "float64", "Surface feature/training role differs")
            require(state["penalty"] == candidate["penalty"] and state["candidate_id"] == candidate["candidate_id"]
                    and state["protocol_sha256"] == release.protocol_sha256
                    and state["prepared_manifest_sha256"] == release.manifest_sha256
                    and state["source_counts"] == release.source_counts, "Surface state identity differs")
            near(state["mean"], mean, "Train mean differs")
            near(state["scale"], scale, "Train scale differs")
            near(state["train_population_std"], std, "Unfloored train standard deviation differs")
            require(state["floored_coordinate_count"] == int((std < 1e-6).sum()), "SD floor count differs")
            require(set(candidate["predictions"]) == set(candidate["metrics"]) == {"train", "validation"},
                    "Surface scored roles differ")
            for split in ("train", "validation"):
                z = (features[split] - mean) / scale
                margin = z @ weights + bias
                logp = np.stack((-np.logaddexp(0, margin), -np.logaddexp(0, -margin)), axis=1)
                descriptor = candidate["predictions"][split]
                require(descriptor["file"] == f"surface/lambda{i}.{split}.npz", "Surface prediction role/path differs")
                path = files.pin(descriptor["file"], descriptor["sha256"])
                require(path.stat().st_size == descriptor["bytes"], "Surface prediction bytes differ")
                with np.load(path, allow_pickle=False) as stored:
                    require(set(stored.files) == {"labels", "row_ids", "log_probabilities", "predictions"},
                            "Surface prediction fields differ")
                    require(stored["labels"].dtype == np.int64
                            and np.array_equal(stored["labels"], arrays[split]["labels"]), "Surface labels differ")
                    require(stored["row_ids"].tolist() == [r["row_id"] for r in rows[split]], "Surface row order differs")
                    require(stored["log_probabilities"].dtype == np.float64, "Surface score precision differs")
                    near(stored["log_probabilities"], logp, "Independent surface score replay differs")
                    require(stored["predictions"].dtype == np.int64
                            and np.array_equal(stored["predictions"], logp.argmax(1)), "Surface decisions differ")
                expected = independent_metrics(arrays[split]["labels"], logp)
                require(set(candidate["metrics"][split]) == set(expected), "Surface metric inventory differs")
                for key, value in expected.items():
                    near(candidate["metrics"][split][key], value, "Surface metric differs: " + key)
                if split == "train":
                    residual = expit(margin) - arrays[split]["labels"]
                    gradient = np.r_[z.T @ residual / len(z) + candidate["penalty"] * weights, residual.mean()]
                    norm = float(np.max(np.abs(gradient)))
                    require(norm <= 1e-5, "Surface solution is not stationary")
                    near(candidate["optimizer"]["gradient_inf_norm"], norm, "Surface gradient differs")
                    loss = float(np.mean(np.logaddexp(0, margin) - arrays[split]["labels"] * margin)
                                 + .5 * candidate["penalty"] * np.dot(weights, weights))
                    near(candidate["optimizer"]["objective"], loss, "Surface objective differs")
        selected = min(candidates, key=lambda c: (c["metrics"]["validation"]["cross_entropy"], c["penalty"]))
        require(selection["selected_candidate"] == selected["candidate_id"]
                and selection["checkpoint"] == selected["checkpoint"]
                and selection["checkpoint_sha256"] == selected["checkpoint_sha256"], "Selected surface identity differs")
        for name, pin in selection["verified_read_files"].items():
            require(name in release.accessed and release.accessed[name] == pin,
                    "Unrecognized fitting input in read-set")
            require(regular_file(name).stat().st_size == pin["bytes"] and file_sha256(name) == pin["sha256"],
                    "Surface fitting input changed")
        require(selection["verified_read_files"] == release.accessed, "Surface fitting/auditing input read-sets differ")
        read_files = release.verify_current_inputs()
        files.verify()
        report = dict(status="passed", completed_at_utc=now(), source_only=True, candidates=5,
                      prediction_artifacts=10, checkpoint_artifacts=5, selected_candidate=selected["candidate_id"],
                      checkpoint=selected["checkpoint"], checkpoint_sha256=selected["checkpoint_sha256"],
                      selection_file=str(output_root / "surface/selection.json"),
                      selection_sha256=files.files[str(output_root / "surface/selection.json")],
                      protocol_sha256=release.protocol_sha256, prepared_manifest_sha256=release.manifest_sha256,
                      source_counts=release.source_counts, files=files.files, verified_read_files=read_files,
                      all_read_files_final_rehashed=True, independent_feature_implementation="counted_local_ngrams",
                      independent_metrics_implementation="confusion_counts_and_average_score_ranks",
                      training_role="source_train", selection_role="source_validation", target_examples_read=False)
        atomic_json(destination, report)
        status.update(status="passed", completed_at_utc=now(), report_sha256=file_sha256(destination))
        atomic_json(status_path, status, replace=True)
        print("Independent new-source surface audit passed: 5 candidates / 10 predictions", flush=True)
        return report
    except BaseException as error:
        status.update(status="failed", failed_at_utc=now(), error=f"{type(error).__name__}: {error}")
        atomic_json(status_path, status, replace=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        audit(args.protocol, args.protocol_sha256)


if __name__ == "__main__":
    main()
