#!/usr/bin/env python3
"""Standalone, CPU-only release inspection. Does not invoke historical workers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import statistics
import subprocess
import sys

CONDITIONS = ("EP", "ES", "EE")
ROLES = ("source_test", "target")
METRICS = ("accuracy", "balanced_accuracy", "auroc", "cross_entropy", "mcc", "predicted_positive_fraction")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def member(root, name):
    root = Path(root).absolute()
    part = PurePosixPath(name)
    if part.is_absolute() or not part.parts or any(x in ("..", ".") for x in part.parts):
        raise ValueError(f"Unsafe relative member: {name}")
    path = root.joinpath(*part.parts)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Symlinked member/root: {name}")
    return path


def verify_files(root, records, *, hash_content=True):
    seen = set()
    for item in records:
        name = item["path"]
        if name in seen:
            raise ValueError(f"Duplicate member: {name}")
        seen.add(name)
        path = member(root, name)
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Missing file or byte-size mismatch: {name}")
        if hash_content and digest(path) != item["sha256"]:
            raise ValueError(f"SHA256 mismatch: {name}")
    return len(seen)


def verify(root):
    manifest = read(member(root, "MANIFEST.json"))
    n = verify_files(root, manifest["files"])
    expected={r["path"] for r in manifest["files"]}|{"MANIFEST.json"}
    actual=set()
    for base, directories, files in os.walk(Path(root).absolute(),followlinks=False):
        base=Path(base)
        for name in directories+files:
            if (base/name).is_symlink():raise ValueError("Symlink in source tree")
        if base==Path(root).absolute() and ".git" in directories:
            directories.remove(".git")  # Isolated local Git metadata is not payload.
        actual.update(str((base/name).relative_to(Path(root).absolute())) for name in files)
    if actual!=expected:
        raise ValueError("Source members differ from manifest closed set")
    return {"status": "passed", "files": n, "manifest_sha256": digest(member(root, "MANIFEST.json")), "hash_content": True}


def safe_output(root, output):
    root=Path(root).absolute()
    output=Path(output).absolute()
    for path in (root,output):
        if any(p.is_symlink() for p in (path,*path.parents)):
            raise ValueError("Output/source ancestors must not be symlinks")
    resolved=output.resolve(strict=False)
    if resolved.is_relative_to(root.resolve(strict=True)):
        raise ValueError("Keep generated outputs outside immutable source root")
    return resolved


def csv_rows(root, name):
    with member(root, "results/oct2/fixed_transfer/" + name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b, field):
    if not math.isfinite(float(a)) or not math.isfinite(float(b)) or abs(float(a) - float(b)) > 1e-12:
        raise ValueError(f"Aggregate mismatch: {field}")


def report(root):
    jobs = csv_rows(root, "per_job_metrics.csv")
    expected = {(c, str(p), str(f), r) for c in CONDITIONS for p in range(3) for f in range(3) for r in ROLES}
    keyed = {(r["condition"], r["pt_seed"], r["ft_seed"], r["role"]): r for r in jobs}
    if len(jobs) != 54 or set(keyed) != expected:
        raise ValueError("Expected exactly 54 unique neural cells, all 3 PT x 3 FT seeds")
    for row in jobs:
        if int(row["rows"]) != (20854 if row["role"] == "source_test" else 39893):
            raise ValueError("Role count changed")
    means, nested = {}, []
    for c in CONDITIONS:
        for role in ROLES:
            for metric in METRICS:
                groups = [[float(keyed[(c, str(p), str(f), role)][metric]) for f in range(3)] for p in range(3)]
                pts = [statistics.mean(g) for g in groups]
                means[c, role, metric] = pts
                nested.append({"condition": c, "role": role, "metric": metric, "pt_means": pts,
                               "mean": statistics.mean(pts), "pretraining_sample_sd": statistics.stdev(pts),
                               "within_pt_ft_sample_sd": [statistics.stdev(g) for g in groups]})
    originals = {(r["condition"], r["role"], r["metric"]): r for r in csv_rows(root, "nested_metrics.csv")}
    if len(originals) != 36:
        raise ValueError("Expected 36 nested metrics")
    for row in nested:
        old = originals[row["condition"], row["role"], row["metric"]]
        for field in ["mean", "pretraining_sample_sd"]:
            close(row[field], old[field], field)
        for p in range(3):
            close(row["pt_means"][p], old[f"pt{p}_mean"], "PT mean")
            close(row["within_pt_ft_sample_sd"][p], old[f"pt{p}_ft_sample_sd"], "FT sample SD")
    contrasts = []
    for left, right in [("EP", "ES"), ("EP", "EE"), ("ES", "EE")]:
        for role in ROLES:
            for metric in METRICS:
                differences = [a - b for a, b in zip(means[left, role, metric], means[right, role, metric])]
                contrasts.append({"contrast": f"{left}-{right}", "role": role, "metric": metric,
                                  "pt_differences": differences, "mean_difference": statistics.mean(differences),
                                  "pretraining_sample_sd": statistics.stdev(differences),
                                  "all_three_pt_differences_positive": all(x > 0 for x in differences)})
    originals = {(r["contrast"], r["role"], r["metric"]): r for r in csv_rows(root, "paired_contrasts.csv")}
    if len(originals) != 36:
        raise ValueError("Expected 36 paired contrasts")
    for row in contrasts:
        old = originals[row["contrast"], row["role"], row["metric"]]
        for field in ["mean_difference", "pretraining_sample_sd"]:
            close(row[field], old[field], field)
        for p in range(3):
            close(row["pt_differences"][p], old[f"pt{p}_difference"], "paired PT difference")
        if row["all_three_pt_differences_positive"] != (old["all_three_pt_differences_positive"].lower() == "true"):
            raise ValueError("Contrast direction flag changed")
    refs = csv_rows(root, "reference_metrics.csv")
    if len(refs) != 6 or {(r["reference"], r["role"]) for r in refs} != {(c, r) for c in ("surface", "constant0", "constant1") for r in ROLES}:
        raise ValueError("Expected 6 fixed-reference cells")
    primary = next(r for r in contrasts if (r["contrast"], r["role"], r["metric"]) == ("EP-ES", "target", "auroc"))
    summary = read(member(root, "results/oct2/fixed_transfer/summary.json"))
    if summary["primary_direction_consistency_flag"] != primary["all_three_pt_differences_positive"]:
        raise ValueError("Summary primary direction flag changed")
    return {"status": "passed", "neural_cells": 54, "reference_cells": 6, "nested_metrics": nested,
            "paired_contrasts": contrasts, "primary_target_auc_contrast": primary,
            "primary_direction_consistency_flag": primary["all_three_pt_differences_positive"],
            "hypothesis_tests": False, "new_blind_confirmation_claimed": False,
            "recomputed_from_aggregate_cells_not_row_predictions": True}


def figures(root, output):
    output=safe_output(root,output)
    results = report(root)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, metric, title in zip(axes, ("auroc", "balanced_accuracy"), ("QQP fixed-direction AUC", "QQP fixed-decision balanced accuracy")):
        for index, condition in enumerate(CONDITIONS):
            row = next(r for r in results["nested_metrics"] if (r["condition"], r["role"], r["metric"]) == (condition, "target", metric))
            ax.errorbar(index, row["mean"], yerr=row["pretraining_sample_sd"], fmt="o", capsize=4)
            ax.scatter([index] * 3, row["pt_means"], s=16, alpha=0.5)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xticks(range(3), CONDITIONS)
        ax.set_title(title)
        ax.set_ylabel("Mean / sample SD across 3 PT seeds")
    fig.tight_layout()
    paths=[]
    for suffix in ("pdf", "svg", "png"):
        p=output/f"fixed_transfer.{suffix}";fig.savefig(p, dpi=180);paths.append({"path":p.name,"sha256":digest(p),"bytes":p.stat().st_size})
    plt.close(fig)
    (output/"figure_data.json").write_text(json.dumps(results, indent=2)+"\n")
    return {"status":"passed", "figures":paths, "new_inference_or_training":False,
            "note":"Release diagnostic figure, not the separately prepared manuscript figure set."}


def manuscript_figures(root, output):
    output=safe_output(root,output)
    verify(root)
    report(root)
    source = member(root, "paper_assets")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Figure output must be fresh")
    shutil.copytree(source, output)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}
    for name in ("gen_fig1_target.py", "gen_fig2_paired.py", "gen_fig3_source.py"):
        script = output / "figures" / name
        wrapper = "import runpy,sys; from pathlib import Path; p=Path(sys.argv[1]); sys.path.insert(0,str(p.parent)); runpy.run_path(str(p),run_name='__main__')"
        subprocess.run([sys.executable, "-I", "-B", "-c", wrapper, str(script.absolute())], cwd=output, env=env, check=True)
    files = sorted((output / "figures").glob("*.pdf")) + sorted((output / "figures").glob("*.png"))
    if len(files) != 6:
        raise ValueError("Expected 3 PDF and 3 PNG figures")
    return {"status": "passed", "figures": [{"path":str(p.relative_to(output)),"bytes":p.stat().st_size,"sha256":digest(p)} for p in files], "new_inference_or_training":False}


def source_input(root, asset_root, kind, size_only=False):
    manifest = read(member(root, f"metadata/{kind}_assets.json"))
    n=verify_files(asset_root, manifest["files"], hash_content=not size_only)
    return {"status":"passed", "kind":kind,"files":n,"hash_content":not size_only,
            "training_authorization":False,"historical_gates_changed":False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    sub=parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify")
    p=sub.add_parser("report");p.add_argument("--output", type=Path)
    p=sub.add_parser("rebuild-figures");p.add_argument("--output", type=Path, required=True);p.add_argument("--manuscript",action="store_true")
    p=sub.add_parser("source-input");p.add_argument("--asset-root", type=Path, required=True);p.add_argument("--kind",choices=("protein","raw_inputs","models"),required=True);p.add_argument("--size-only",action="store_true")
    args=parser.parse_args(argv)
    verification=verify(args.source_root)
    if args.command=="verify": result=verification
    elif args.command=="report":
        result=report(args.source_root)
        if args.output:
            output=safe_output(args.source_root,args.output)
            with output.open("x") as stream:json.dump(result,stream,indent=2);stream.write("\n")
    elif args.command=="rebuild-figures":
        result=(manuscript_figures if args.manuscript else figures)(args.source_root,args.output)
    else:result=source_input(args.source_root,args.asset_root,args.kind,args.size_only)
    print(json.dumps(result if args.command!="report" else {k:v for k,v in result.items() if k not in ("nested_metrics","paired_contrasts")},indent=2))


if __name__=="__main__":
    try:main()
    except (ValueError, OSError, KeyError) as error:
        print(f"Release check failed: {error}",file=sys.stderr)
        sys.exit(2)
