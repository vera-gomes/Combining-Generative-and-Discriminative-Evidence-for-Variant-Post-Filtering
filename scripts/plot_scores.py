#!/usr/bin/env python3
"""
plot_scores.py — Figure 2: apply-mode score distributions of the
nine distinct models on NA06985, from the corrected run. Reads the large scores
file in chunks (low memory). By default only calls on chr1-22 and chrX are used
(4,963,081 calls, the main analysis); --chroms all uses all 5,069,016 calls.

  python3 plot_scores.py \
      --scores output/apply/scored_variants_with_external_validation.csv \
      --out fig_score_distributions
"""
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["GM", "BGM", "LogReg", "RandForest", "LGB", "LGB_Bayes", "LGB_MultiObj",
         "Hybrid_LGB_alpha08", "Hybrid_LGB_alpha07"]
DISPLAY = {"GM": "GM", "BGM": "BGM", "LogReg": "LogReg", "RandForest": "RF", "LGB": "LGB",
           "LGB_Bayes": "LGB_Bayes", "LGB_MultiObj": "LGB_MultiObj",
           "Hybrid_LGB_alpha08": "Hybrid_0.8", "Hybrid_LGB_alpha07": "Hybrid_0.7"}
# one colour + line style per model, so every curve in panel (a) is distinguishable
STYLE = {"GM": ("#eb6834", "-"), "BGM": ("#eb6834", "--"),
         "LogReg": ("#8fb8ea", "-"), "RandForest": ("#5b95dd", "-."), "LGB": ("#2a78d6", ":"),
         "LGB_Bayes": ("#1a4f94", "-"), "LGB_MultiObj": ("#1a4f94", "--"),
         "Hybrid_LGB_alpha08": ("#1baf7a", "-"), "Hybrid_LGB_alpha07": ("#1baf7a", "--")}
FAMILY = {"GM": "#eb6834", "BGM": "#eb6834", "LogReg": "#2a78d6", "RandForest": "#2a78d6",
          "LGB": "#2a78d6", "LGB_Bayes": "#2a78d6", "LGB_MultiObj": "#2a78d6",
          "Hybrid_LGB_alpha08": "#1baf7a", "Hybrid_LGB_alpha07": "#1baf7a"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#d9d8d2"
BINS = np.linspace(0, 1, 201)

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
ap.add_argument("--popdb-col", default="IsExternalVariant")
ap.add_argument("--out", default="fig_score_distributions")
ap.add_argument("--chroms", choices=["all", "primary"], default="primary",
                help="primary: chr1-22 and chrX only (main analysis of the revised paper)")
a = ap.parse_args()

header = pd.read_csv(a.scores, nrows=0).columns
cols = [f"{m}_score" for m in ORDER if f"{m}_score" in header]
has_pop = a.popdb_col in header
use = cols + ([a.popdb_col] if has_pop else []) + ["CHROM"]
counts = {c: {"all": np.zeros(len(BINS) - 1), "pos": np.zeros(len(BINS) - 1), "neg": np.zeros(len(BINS) - 1)}
          for c in cols}
n = 0
PRIMARY = {str(i) for i in range(1, 23)} | {"X"}
for chunk in pd.read_csv(a.scores, usecols=use, chunksize=500_000, dtype={"CHROM": str}):
    if a.chroms == "primary":
        chunk = chunk[chunk["CHROM"].str.replace("chr", "", regex=False).isin(PRIMARY)]
    n += len(chunk)
    y = chunk[a.popdb_col].to_numpy() == 1 if has_pop else None
    for c in cols:
        v = np.clip(chunk[c].to_numpy(dtype=float), 0, 1)
        counts[c]["all"] += np.histogram(v, BINS)[0]
        if has_pop:
            counts[c]["pos"] += np.histogram(v[y], BINS)[0]
            counts[c]["neg"] += np.histogram(v[~y], BINS)[0]
print(f"read {n:,} calls")

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig = plt.figure(figsize=(10, 9.5))
gs = fig.add_gridspec(4, 3, height_ratios=[1.3, 1, 1, 1], hspace=0.75, wspace=0.25)

# (a) CDFs of all calls, all models in one panel
ax = fig.add_subplot(gs[0, :2])
for c in cols:
    m = c[:-6]
    col, ls = STYLE[m]
    cdf = np.concatenate([[0], np.cumsum(counts[c]["all"])]) / counts[c]["all"].sum()
    ax.plot(BINS, cdf, linestyle=ls, color=col, lw=1.6, label=DISPLAY[m])
ax.set_xlim(-0.02, 1.05); ax.set_ylim(0, 1.02)
ax.set_xlabel("Score"); ax.set_ylabel("Cumulative fraction of calls")
ax.set_title(f"(a) Cumulative distribution of the scores of all {n:,} calls", loc="left", fontsize=9.5, color=INK)
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.legend(ncol=1, frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1.0), title="Model", title_fontsize=8)

# (b) densities per model, catalogue-supported vs background
for i, c in enumerate(cols):
    m = c[:-6]
    ax = fig.add_subplot(gs[1 + i // 3, i % 3])
    w = np.diff(BINS)
    if has_pop:
        for key, col, lab in [("pos", INK, "catalogue-supported"), ("neg", "#8f8d84", "background")]:
            dens = counts[c][key] / (counts[c][key].sum() * w)
            ax.stairs(dens, BINS, color=col, lw=1.2, label=lab, linestyle="-" if key == "pos" else (0, (3, 1.5)))
    else:
        ax.stairs(counts[c]["all"] / (counts[c]["all"].sum() * w), BINS, color=FAMILY[m], lw=1.3)
    ax.set_xlim(-0.02, 1.05)
    ax.set_title(DISPLAY[m], loc="left", fontsize=9, family="DejaVu Sans Mono", color=INK)
    if i % 3 == 0:
        ax.set_ylabel("Density")
    if i >= 6:
        ax.set_xlabel("Score")
    if i == 0 and has_pop:
        ax.legend(frameon=False, fontsize=7)
fig.text(0.07, 0.672, "(b) Score density by class (population-catalogue label)", fontsize=9.5, color=INK)
for ext in ("pdf", "png"):
    fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight")
print(f"saved {a.out}.pdf / .png")
