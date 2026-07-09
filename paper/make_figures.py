"""Generate publication figures for the paper: Fig1 (directionality schematic) + Fig2 (M2 dynamics)."""
import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict

FIGDIR = "/root/autodl-tmp/bio-trans/bio2nl/paper/figures"
plt.rcParams.update({"font.size": 11, "font.family": "sans-serif", "axes.linewidth": 0.8,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})

# ---------------- Fig 1: directionality schematic ----------------
def fig1():
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5)
    # two domain boxes
    lang = plt.Rectangle((0.6, 1.7), 2.6, 1.6, fc="#cfe3f7", ec="#2b6cb0", lw=1.5, zorder=2)
    bio  = plt.Rectangle((6.8, 1.7), 2.6, 1.6, fc="#d8f0d8", ec="#2f855a", lw=1.5, zorder=2)
    ax.add_patch(lang); ax.add_patch(bio)
    ax.text(1.9, 2.5, "Natural\nLanguage", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(8.1, 2.5, "Biological\nSequences", ha="center", va="center", fontsize=12, fontweight="bold")
    # strong forward arrow (thick)
    a1 = FancyArrowPatch((3.3, 3.0), (6.7, 3.0), arrowstyle="-|>", mutation_scale=28,
                         lw=6, color="#2b6cb0", zorder=3)
    ax.add_patch(a1)
    ax.text(5.0, 3.5, "strong  (drop 0.08)", ha="center", color="#2b6cb0", fontsize=11, fontweight="bold")
    # weak reverse arrow (thin)
    a2 = FancyArrowPatch((6.7, 2.0), (3.3, 2.0), arrowstyle="-|>", mutation_scale=14,
                         lw=1.2, color="#a0aec0", zorder=3, linestyle=(0,(4,2)))
    ax.add_patch(a2)
    ax.text(5.0, 1.5, "weak  (drop 0.36)", ha="center", color="#718096", fontsize=10)
    ax.text(5.0, 0.5, "Structural transfer is bidirectional but strongly asymmetric",
            ha="center", fontsize=11, style="italic")
    fig.savefig(f"{FIGDIR}/fig1_directionality.pdf")
    print("wrote fig1_directionality.pdf")

# ---------------- Fig 2: M2 learning dynamics ----------------
def fig2():
    traj = defaultdict(list)  # direction -> list of (steps, accs)
    for l in open("/root/autodl-tmp/bio-trans/bio2nl/results/router/M2_learning_dynamics.jsonl"):
        r = json.loads(l)
        steps = [x["step"] for x in r["trajectory"]]
        accs = [x["transfer_flip"] for x in r["trajectory"]]
        traj[r["direction"]].append((steps, accs))

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    colors = {"NL->Bio": "#2b6cb0", "Bio->NL": "#c05621"}
    labels = {"NL->Bio": "Language$\\rightarrow$Biology", "Bio->NL": "Biology$\\rightarrow$Language"}
    for dr in ["NL->Bio", "Bio->NL"]:
        runs = traj[dr]
        # align on common step grid (min length)
        m = min(len(s) for s, _ in runs)
        steps = runs[0][0][:m]
        M = np.array([a[:m] for _, a in runs])
        mean = M.mean(0); sd = M.std(0)
        ax.plot(steps, mean, color=colors[dr], lw=2.2, label=labels[dr])
        ax.fill_between(steps, mean - sd, mean + sd, color=colors[dr], alpha=0.18)
    ax.axhline(0.545, ls="--", color="#c05621", lw=0.9, alpha=0.7)
    ax.axhline(0.50, ls="--", color="#2b6cb0", lw=0.9, alpha=0.7)
    ax.text(ax.get_xlim()[1]*0.98, 0.552, "PAWS-X chance", ha="right", fontsize=8, color="#c05621")
    ax.text(ax.get_xlim()[1]*0.98, 0.508, "protein chance", ha="right", fontsize=8, color="#2b6cb0")
    ax.set_xlabel("Training step (source domain)")
    ax.set_ylabel("Zero-shot transfer accuracy (target domain)")
    ax.set_title("Transfer emerges during forward training only", fontsize=11)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=9)
    ax.set_ylim(0.45, 0.90)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(f"{FIGDIR}/fig2_dynamics.pdf")
    print("wrote fig2_dynamics.pdf")

# ---------------- Fig 3: scaling divergence ----------------
def fig3():
    d = defaultdict(list)
    for l in open("/root/autodl-tmp/bio-trans/bio2nl/results/benchmarks/esm_on_nl.jsonl"):
        r = json.loads(l)
        if r["task"] == "paws-x":
            d[r["model"]].append(r["score"])
    params = {"esm2_8M":8,"esm2_35M":35,"esm2_150M":150,"esm2_650M":650,
              "protbert":420,"bert_tiny":4,"bert_small":11,"bert_base":110}
    def pts(models):
        xs, ys, es = [], [], []
        for m in models:
            if d.get(m):
                xs.append(params[m]); ys.append(np.mean(d[m])); es.append(np.std(d[m]))
        order = np.argsort(xs)
        return np.array(xs)[order], np.array(ys)[order], np.array(es)[order]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    # language ladder (up)
    x, y, e = pts(["bert_tiny", "bert_small", "bert_base"])
    ax.errorbar(x, y, yerr=e, color="#2b6cb0", lw=2.2, marker="o", ms=7, capsize=3,
                label="English models (BERT)")
    # ESM ladder (down)
    x, y, e = pts(["esm2_8M", "esm2_35M", "esm2_150M", "esm2_650M"])
    ax.errorbar(x, y, yerr=e, color="#c05621", lw=2.2, marker="s", ms=7, capsize=3,
                label="Protein models (ESM-2)")
    # ProtBERT cross-family point
    if d.get("protbert"):
        ax.scatter([params["protbert"]], [np.mean(d["protbert"])], color="#9c4221",
                   marker="D", s=70, zorder=5, label="ProtBERT (cross-family)")
    ax.axhline(0.545, ls="--", color="#718096", lw=0.9)
    ax.text(4.3, 0.556, "PAWS-X chance", fontsize=8, color="#718096")
    ax.set_xscale("log")
    ax.set_xlabel("Model size (M parameters, log scale)")
    ax.set_ylabel("PAWS-X accuracy (transfer to language)")
    ax.set_title("Language transfer scales up; biology$\\rightarrow$language transfer decays", fontsize=10.5)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=8.5)
    ax.set_ylim(0.50, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(f"{FIGDIR}/fig3_scaling.pdf")
    print("wrote fig3_scaling.pdf")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
