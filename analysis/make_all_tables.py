"""
Master analysis: regenerate EVERY paper table/number from raw result JSONL files.
Single source of truth. Run after any new results land.

    python bio2nl/analysis/make_all_tables.py

Reads results/benchmarks/*.jsonl, prints all tables, and writes results/SUMMARY.md.
"""
import json, numpy as np
from collections import defaultdict
from pathlib import Path

B = Path("/root/autodl-tmp/bio-trans/bio2nl/results/benchmarks")
OUT = Path("/root/autodl-tmp/bio-trans/bio2nl/results/SUMMARY.md")
lines = []
def p(s=""):
    print(s); lines.append(s)


def load(name):
    fp = B / name
    return [json.loads(l) for l in open(fp)] if fp.exists() else []


def ms(xs):
    a = np.array(xs, float); return (a.mean(), a.std(), len(a))


# ---------- Table 1: bidirectional symmetry (GPT-2, one protocol) ----------
def table_bidirectional():
    rows = load("bidirectional_symmetry.jsonl")
    if not rows: return
    p("## Table 1 — Bidirectional symmetry (GPT-2, identical protocol)")
    d = defaultdict(list)
    for r in rows:
        d[r["direction"]].append(r["transfer"]["acc_flipbest"])
    p(f"{'direction':10s} {'transfer(flip)':18s} n")
    for dr in ["NL->Bio", "Bio->NL"]:
        if d[dr]:
            mu, sd, n = ms(d[dr]); p(f"{dr:10s} {mu:.3f} ± {sd:.3f}      {n}")
    p("(NL->Bio majority≈0.50 protein; Bio->NL majority=0.545 PAWS-X)\n")


# ---------- Table 2: reverse FT distribution (100 seeds) ----------
def table_reverse_dist():
    rows = load("tier1_distribution.jsonl")
    if not rows: return
    p("## Table 2 — Reverse (Bio->NL) FT, 100-seed distribution on PAWS-X")
    tr = [r["transfer_pawsx_test"]["acc_flipbest"] for r in rows]
    bl = [r["baseline_pawsx_test"]["acc_flipbest"] for r in rows]
    ind = [r["protein_indomain"]["acc_raw"] for r in rows]
    mt, st, n = ms(tr); mb, sb, _ = ms(bl); mi, si, _ = ms(ind)
    p(f"protein in-domain : {mi:.3f} ± {si:.3f}")
    p(f"transfer -> PAWS  : {mt:.3f} ± {st:.3f}  (majority 0.5465)")
    p(f"untrained baseline: {mb:.3f} ± {sb:.3f}")
    p(f"seeds beating baseline by >0.05: {sum(1 for a,b in zip(tr,bl) if a-b>0.05)}/{n}  (n={n})\n")


# ---------- Table 3: CPT iso-token ablation (124M vs 355M, PAWS-X sample-eff) ----------
def table_cpt():
    r124 = load("exp2_structural_nl_gpt2_124M.jsonl")
    r355 = [r for r in load("exp2_structural_nl.jsonl") if r.get("base") == "gpt2-medium"]
    if not r124: return
    p("## Table 3 — Iso-token CPT ablation, PAWS-X sample-efficiency (3 seeds)")
    for label, rows in [("124M", r124), ("355M", r355)]:
        if not rows: continue
        agg = defaultdict(list)
        for r in rows:
            if r["task"] == "paws-x":
                agg[(r["frac"], r["group"])].append(r["score"])
        p(f"  [{label}]  (protein−shuffled isolates STRUCTURE; protein−text isolates BIOLOGY-vs-tokens)")
        for fr in [0.05, 0.1, 1.0]:
            def g(x):
                v = agg.get((fr, x)); return np.mean(v) if v else float("nan")
            pr, sh, tx, bs = g("protein"), g("shuffled"), g("text"), g("base")
            p(f"    frac{fr}: base={bs:.3f} text={tx:.3f} protein={pr:.3f} shuffled={sh:.3f}  Δ(p−s)={pr-sh:+.3f} Δ(p−t)={pr-tx:+.3f}")
    p("")


# ---------- Table 4: 2x2 cross-modal matrix ----------
def table_2x2():
    nl = defaultdict(list); bio = defaultdict(list)
    for r in load("esm_on_nl.jsonl"):
        if r["task"] == "paws-x": nl[r["model"]].append(r["score"])
    for r in load("crossmodal_2x2.jsonl"):
        bio[r["model"]].append(r["acc"])
    if not nl or not bio: return
    p("## Table 4 — 2x2 cross-modal transfer matrix (acc, 3-seed mean)")
    p(f"{'backbone':12s}{'pretrain':10s}{'->protein':12s}{'->PAWS-X'}")
    for mk, pre in [("esm2_8M", "protein"), ("bert_small", "English"), ("bert_tiny", "English")]:
        b = np.mean(bio[mk]) if bio.get(mk) else float("nan")
        n = np.mean(nl[mk]) if nl.get(mk) else float("nan")
        p(f"{mk:12s}{pre:10s}{b:<12.3f}{n:.3f}")
    if bio.get("esm2_8M") and nl.get("bert_small"):
        p(f"  off-domain drop: language-model(bio) {np.mean(nl['bert_small'])-np.mean(bio['bert_small']):+.3f} | bio-model(NL) {np.mean(bio['esm2_8M'])-np.mean(nl['esm2_8M']):+.3f}")
    p("(majority: protein 0.50, PAWS-X 0.545)\n")


# ---------- Table 5: scaling divergence + cross-family ----------
def table_scaling():
    d = defaultdict(list)
    for r in load("esm_on_nl.jsonl"):
        if r["task"] == "paws-x": d[r["model"]].append(r["score"])
    if not d: return
    p("## Table 5 — Scaling on PAWS-X: language scales UP, bio->language DECAYS (3 seeds)")
    params = {"esm2_8M":8,"esm2_35M":35,"esm2_150M":150,"esm2_650M":650,"protbert":420,
              "bert_tiny":4,"bert_small":11,"bert_base":110}
    p("  ESM-2 ladder (protein):")
    for m in ["esm2_8M","esm2_35M","esm2_150M","esm2_650M"]:
        if d.get(m): mu,sd,_=ms(d[m]); p(f"    {m:11s} {params[m]:4d}M  {mu:.3f} ± {sd:.3f}")
    p("  Cross-family large protein LMs (universality):")
    for m in ["esm2_650M","protbert"]:
        if d.get(m): mu,sd,_=ms(d[m]); p(f"    {m:11s} {params[m]:4d}M  {mu:.3f} ± {sd:.3f}")
    p("  BERT ladder (English):")
    for m in ["bert_tiny","bert_small","bert_base"]:
        if d.get(m): mu,sd,_=ms(d[m]); p(f"    {m:11s} {params[m]:4d}M  {mu:.3f} ± {sd:.3f}")
    p("")


# ---------- Table 6: ESM structural-only (CoLA/RTE collapse) ----------
def table_esm_tasks():
    d = defaultdict(dict);
    agg = defaultdict(list)
    for r in load("esm_on_nl.jsonl"):
        agg[(r["task"], r["model"])].append(r["score"])
    if not agg: return
    p("## Table 6 — ESM-2 is STRUCTURAL-only (above chance on PAWS, collapses on CoLA/RTE)")
    for t in ["paws-x","cola","rte"]:
        cells=[]
        for m in ["esm2_8M","bert_small"]:
            v=agg.get((t,m)); cells.append(f"{m}={np.mean(v):.3f}" if v else f"{m}=--")
        p(f"  {t:8s} "+"  ".join(cells))
    p("")


# ---------- Table 7: Dyck hard task ----------
def table_dyck():
    rows=[r for r in load("exp2_structural_nl.jsonl") if 'dyck' in r.get('task','')]
    if not rows: return
    p("## Table 7 — Hard structural task (Dyck): protein>shuffled does NOT hold (3 seeds)")
    agg=defaultdict(list)
    for r in rows: agg[(r.get('base','gpt2'),r['task'],r['frac'],r['group'])].append(r['score'])
    for base in ["gpt2","gpt2-medium"]:
        for t in ["dyck_L40_t3","dyck_L60_t3"]:
            fr=1.0
            def g(x):
                v=agg.get((base,t,fr,x)); return np.mean(v) if v else float('nan')
            pr,sh,tx=g('protein'),g('shuffled'),g('text')
            p(f"  {base:11s} {t} f{fr}: protein={pr:.3f} shuffled={sh:.3f} text={tx:.3f}  Δ(p−s)={pr-sh:+.3f}")
    p("")


ROUTER = B.parent / "router"


def table_mechanism():
    def load_r(name):
        fp = ROUTER / name
        return [json.loads(l) for l in open(fp)] if fp.exists() else []
    m2 = load_r("M2_learning_dynamics.jsonl")
    m3c = load_r("M3c_rigorous.jsonl")
    if not m2 and not m3c:
        return
    p("## Table 8 — Mechanism (M1 CKA / M2 dynamics / M3 difference-heads)")
    if m2:
        agg = defaultdict(lambda: {"start": [], "end": []})
        for r in m2:
            t = r["trajectory"]; agg[r["direction"]]["start"].append(t[0]["transfer_flip"])
            agg[r["direction"]]["end"].append(t[-1]["transfer_flip"])
        p("  M2 learning dynamics (transfer start->end during training):")
        for dr in ["NL->Bio", "Bio->NL"]:
            if agg[dr]["start"]:
                p(f"    {dr}: {np.mean(agg[dr]['start']):.3f} -> {np.mean(agg[dr]['end']):.3f}")
    if m3c:
        p("  M3c rigorous difference-head z-scores (diff-head is pretraining artifact, not induced):")
        for r in m3c:
            p(f"    {r['model']:12s} {r['modality']:7s} max={r['max']} z={r['z_of_max']} head L{r['max_head'][0]}H{r['max_head'][1]}")
    p("")


if __name__ == "__main__":
    p("# bio2nl — All Result Tables (auto-generated)\n")
    table_bidirectional()
    table_reverse_dist()
    table_cpt()
    table_2x2()
    table_scaling()
    table_esm_tasks()
    table_dyck()
    table_mechanism()
    OUT.write_text("\n".join(lines))
    print(f"\n[written] {OUT}")
