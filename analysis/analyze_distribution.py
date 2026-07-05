"""
Analyze the Tier-1 reverse-transfer seed distribution.
Reads results/benchmarks/tier1_distribution.jsonl (one record per seed) and reports:
  - distribution stats (mean/std/min/max/quantiles) for transfer vs baseline
  - paired comparison: per-seed (transfer - baseline)
  - one-sided test: does the transfer distribution beat baseline?
  - tail: how many seeds clear baseline by a margin (the "some runs work" claim)
Run anytime while the batch is still writing (reads whatever is present).
"""
import json, numpy as np
from pathlib import Path

JSONL = "/root/autodl-tmp/bio-trans/bio2nl/results/benchmarks/tier1_distribution.jsonl"


def col(rows, key, sub):
    return np.array([r[key][sub] for r in rows])


def stats(x):
    return dict(n=len(x), mean=round(float(x.mean()), 4), std=round(float(x.std()), 4),
                min=round(float(x.min()), 4), p25=round(float(np.percentile(x, 25)), 4),
                median=round(float(np.median(x)), 4), p75=round(float(np.percentile(x, 75)), 4),
                max=round(float(x.max()), 4))


def main():
    rows = [json.loads(l) for l in open(JSONL)]
    print(f"seeds analyzed: {len(rows)}\n")

    for metric in ["acc_raw", "acc_flipbest"]:
        tr = col(rows, "transfer_pawsx_test", metric)
        bl = col(rows, "baseline_pawsx_test", metric)
        diff = tr - bl
        print(f"=== metric: {metric} ===")
        print(f"  transfer : {stats(tr)}")
        print(f"  baseline : {stats(bl)}")
        print(f"  paired Δ (transfer-baseline): mean={diff.mean():.4f} std={diff.std():.4f} "
              f"min={diff.min():.4f} max={diff.max():.4f}")
        print(f"  seeds with Δ>0    : {(diff>0).sum()}/{len(diff)}")
        print(f"  seeds with Δ>0.02 : {(diff>0.02).sum()}/{len(diff)}")
        print(f"  seeds with Δ>0.05 : {(diff>0.05).sum()}/{len(diff)}")
        # bootstrap CI on mean Δ
        if len(diff) >= 5:
            boot = np.array([np.random.choice(diff, len(diff), replace=True).mean()
                             for _ in range(5000)])
            print(f"  mean Δ 95% CI     : [{np.percentile(boot,2.5):.4f}, {np.percentile(boot,97.5):.4f}]")
        print()

    ind = col(rows, "protein_indomain", "acc_raw")
    print(f"protein in-domain acc: {stats(ind)}  (sanity: did it learn the bio task)")

    # best transfer seeds (the tail story)
    tr_flip = col(rows, "transfer_pawsx_test", "acc_flipbest")
    order = np.argsort(-tr_flip)[:5]
    print("\ntop-5 transfer seeds (flipbest):")
    for i in order:
        r = rows[i]
        print(f"  seed {r['seed']:>3}  transfer_flip={tr_flip[i]:.4f}  "
              f"baseline_flip={r['baseline_pawsx_test']['acc_flipbest']:.4f}  "
              f"indomain={r['protein_indomain']['acc_raw']:.4f}")


if __name__ == "__main__":
    main()
