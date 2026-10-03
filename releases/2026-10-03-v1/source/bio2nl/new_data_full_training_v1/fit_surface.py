"""Five fixed CPU surface fits on the separately authorized new source data.

The mathematical design is inherited from the frozen 2026-09-29 protocol.
No historical examples, fitted scaler, weights or target inputs are loaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from threadpoolctl import threadpool_limits

try:
    from .runtime_data import ReleaseData, file_sha256, require
    from .surface_features import vector_features, FEATURE_COLUMNS
except ImportError:
    from runtime_data import ReleaseData, file_sha256, require
    from surface_features import vector_features, FEATURE_COLUMNS


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value, *, replace=False):
    path = Path(path)
    require(replace or not path.exists(), "Refuse existing artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".partial")
    with pending.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    pending.replace(path)


def metrics(labels, logp):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, matthews_corrcoef, roc_auc_score)
    labels, logp = np.asarray(labels, dtype=np.int64), np.asarray(logp, dtype=np.float64)
    require(labels.ndim == 1 and len(labels) > 0 and logp.shape == (len(labels), 2),
            "Invalid prediction dimensions")
    require(set(np.unique(labels)) == {0, 1} and np.isfinite(logp).all(), "Invalid prediction values")
    require(np.allclose(np.exp(logp).sum(1), 1, atol=2e-6, rtol=0), "Unnormalized probabilities")
    pred = logp.argmax(1)
    return dict(rows=len(labels), accuracy=float(accuracy_score(labels, pred)),
                balanced_accuracy=float(balanced_accuracy_score(labels, pred)),
                mcc=float(matthews_corrcoef(labels, pred)),
                auroc=float(roc_auc_score(labels, logp[:, 1] - logp[:, 0])),
                cross_entropy=float(-logp[np.arange(len(labels)), labels].mean()),
                predicted_positive_fraction=float(pred.mean()),
                confusion_matrix=confusion_matrix(labels, pred, labels=[0, 1]).tolist())


def save_predictions(path, labels, logp, row_ids):
    path = Path(path)
    require(not path.exists(), "Refuse existing predictions")
    require(len(row_ids) == len(labels) and len(set(row_ids)) == len(row_ids), "Invalid prediction row identities")
    value = metrics(labels, logp)
    pending = path.with_name(path.name + ".partial")
    with pending.open("xb") as handle:
        np.savez_compressed(handle, labels=np.asarray(labels, dtype=np.int64),
                            log_probabilities=np.asarray(logp, dtype=np.float64),
                            predictions=np.asarray(logp).argmax(1), row_ids=np.asarray(row_ids, dtype=str))
    pending.replace(path)
    return value


def objective(theta, x, y, penalty):
    margin = x @ theta[:-1] + theta[-1]
    residual = expit(margin) - y
    value = float(np.mean(np.logaddexp(0, margin) - y * margin)
                  + .5 * penalty * np.dot(theta[:-1], theta[:-1]))
    gradient = np.r_[x.T @ residual / len(y) + penalty * theta[:-1], residual.mean()]
    return value, gradient


def validate_design(config):
    require(config["lambdas"] == [1e-4, 1e-3, .01, .1, 1.], "Surface penalty grid changed")
    require(config["optimizer"] == dict(accept_gradient_inf_le=1e-5, ftol=1e-12, gtol=1e-8,
                                        maxiter=2000, maxls=50, method="L-BFGS-B"), "Surface optimizer changed")
    expected = dict(dtype="float64", expected_cpu_fits=5, family="length_unigram_bigram",
                    feature_dimension=9, initialization="all_zero", scaler_std_floor=1e-6,
                    scaler_floor_replacement=1., scaler_fit="source_train_only_float64_population_mean_std",
                    objective="mean_binary_CE_plus_half_lambda_squared_weight_norm_intercept_unpenalized",
                    selection="minimum_source_validation_CE_ties_smaller_lambda",
                    target_calibration=False, target_scaler_refit=False)
    require(all(config.get(k) == v for k, v in expected.items()), "Surface mathematical protocol changed")


def fit(protocol_path, protocol_sha256):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Surface fitting must be CPU-only")
    started = time.monotonic()
    release = ReleaseData(protocol_path, protocol_sha256, mode="prepare")
    config = release.design["surface_reference"]
    validate_design(config)
    output_root = Path(release.execution_protocol["output_root"]).absolute()
    out = output_root / "surface"
    require(not out.exists(), "Refuse existing surface outputs")
    out.mkdir(parents=True)
    status = dict(status="running", started_at_utc=now(), protocol_sha256=release.protocol_sha256,
                  prepared_manifest_sha256=release.manifest_sha256, completed_candidates=0,
                  source_only=True, target_examples_read=False)
    atomic_json(out / "status.json", status)
    try:
        arrays, rows, features = {}, {}, {}
        for split in ("train", "validation"):
            arrays[split], rows[split] = release.load_source(split, smoke=False)
            features[split] = np.stack([vector_features(r["ids_a"], r["ids_b"]) for r in rows[split]])
        mean = features["train"].mean(0)
        std = features["train"].std(0, ddof=0)
        scale = np.where(std < 1e-6, 1., std)
        normalized = {key: (value - mean) / scale for key, value in features.items()}
        results = []
        for i, penalty in enumerate(config["lambdas"]):
            solution = minimize(objective, np.zeros(10, dtype=np.float64),
                                args=(normalized["train"], arrays["train"]["labels"], penalty),
                                method="L-BFGS-B", jac=True,
                                options={k: config["optimizer"][k] for k in ("maxiter", "ftol", "gtol", "maxls")})
            value, gradient = objective(solution.x, normalized["train"], arrays["train"]["labels"], penalty)
            norm = float(np.max(np.abs(gradient)))
            require(solution.success and np.isfinite(solution.x).all() and np.isfinite(value)
                    and norm <= config["optimizer"]["accept_gradient_inf_le"],
                    "Fixed surface optimizer failed; no grid or budget extension")
            candidate = f"lambda{i}"
            checkpoint = dict(candidate_id=candidate, penalty=penalty, feature_columns=FEATURE_COLUMNS,
                              coefficients=solution.x[:-1].tolist(), bias=float(solution.x[-1]),
                              mean=mean.tolist(), scale=scale.tolist(), train_population_std=std.tolist(),
                              floored_coordinate_count=int((std < 1e-6).sum()),
                              protocol_sha256=release.protocol_sha256,
                              prepared_manifest_sha256=release.manifest_sha256,
                              source_counts=release.source_counts, dtype="float64", source_train_only=True)
            ckpt = out / (candidate + ".json")
            atomic_json(ckpt, checkpoint)
            record = dict(candidate_id=candidate, penalty=penalty,
                          checkpoint=str(ckpt.relative_to(output_root)), checkpoint_sha256=file_sha256(ckpt),
                          optimizer=dict(success=bool(solution.success), iterations=int(solution.nit),
                                         objective=value, gradient_inf_norm=norm), metrics={}, predictions={})
            for split in ("train", "validation"):
                margin = normalized[split] @ solution.x[:-1] + solution.x[-1]
                logp = np.column_stack((-np.logaddexp(0., margin), -np.logaddexp(0., -margin)))
                pred = out / f"{candidate}.{split}.npz"
                record["metrics"][split] = save_predictions(pred, arrays[split]["labels"], logp,
                                                            [r["row_id"] for r in rows[split]])
                record["predictions"][split] = dict(file=str(pred.relative_to(output_root)),
                                                   sha256=file_sha256(pred), bytes=pred.stat().st_size)
            atomic_json(out / (candidate + ".result.json"), record)
            results.append(record)
            status["completed_candidates"] = len(results)
            atomic_json(out / "status.json", status, replace=True)
        selected = min(results, key=lambda r: (r["metrics"]["validation"]["cross_entropy"], r["penalty"]))
        read_files = release.verify_current_inputs()
        selection = dict(status="selected_pending_independent_audit", selected_candidate=selected["candidate_id"],
                         checkpoint=selected["checkpoint"], checkpoint_sha256=selected["checkpoint_sha256"],
                         selection_metric="source_validation_cross_entropy", candidates=results,
                         source_only=True, target_examples_read=False,
                         protocol_sha256=release.protocol_sha256, prepared_manifest_sha256=release.manifest_sha256,
                         source_counts=release.source_counts, verified_read_files=read_files,
                         all_read_files_final_rehashed=True, completed_at_utc=now(),
                         elapsed_seconds=time.monotonic() - started)
        atomic_json(out / "selection.json", selection)
        status.update(status="completed", completed_at_utc=now(), elapsed_seconds=time.monotonic() - started,
                      selection_sha256=file_sha256(out / "selection.json"),
                      artifacts={str(p.relative_to(output_root)): file_sha256(p)
                                 for p in sorted(out.iterdir()) if p.name != "status.json"})
        atomic_json(out / "status.json", status, replace=True)
        print("Completed 5 new-source surface candidates; selected", selected["candidate_id"], flush=True)
        return selection
    except BaseException as error:
        status.update(status="failed", failed_at_utc=now(), error=f"{type(error).__name__}: {error}")
        atomic_json(out / "status.json", status, replace=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        fit(args.protocol, args.protocol_sha256)


if __name__ == "__main__":
    main()
