#!/usr/bin/env python3
"""
fig_alpha_three.py — Supplementary Figure S2: AUC of the fixed-weight hybrid as a
function of alpha in three settings: (a) illustrative simulation
(scripts/hybrid_alpha_simulation.py), (b) NA12878 benchmark block CV with 10 seeds
(scripts/alpha_convergence.py), (c) NA06985 after transfer, per autosome, with the
pooled primary-chromosome AUC of the evaluated models.

  python3 scripts/fig_alpha_three.py --results results --out figures/fig_alpha_three
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures/fig_alpha_three")
    a = ap.parse_args()
    sim = pd.read_csv(os.path.join(a.results, "alpha_simulation", "alpha_simulation_summary.csv"))
    conv = pd.read_csv(os.path.join(a.results, "alpha_convergence", "alpha_convergence_summary.csv"))
    pc = pd.read_csv(os.path.join(a.results, "alpha_external", "alpha_external_per_chrom.csv"))
    pc["chrom"] = pc.chrom.astype(str).str.replace("chr", "")
    pc = pc[pc.chrom.isin([str(i) for i in range(1, 23)])]
    s = pd.read_csv(os.path.join(a.results, "external_validation", "popdb_summary.csv")).set_index("Model").AUC
    pooled = {0.7: s["Hybrid_LGB_alpha07"], 0.8: s["Hybrid_LGB_alpha08"], 1.0: s["LGB"]}

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))

    def mark(axis, x, y, txt):
        axis.plot(x, y, "o", color="#c0392b", zorder=5)
        axis.annotate(txt, (x, y), textcoords="offset points", xytext=(-8, -22), ha="right", fontsize=8.5, color="#c0392b")

    x0 = ax[0]
    x0.fill_between(sim.alpha, sim.mean_AUC - sim.CI_halfwidth, sim.mean_AUC + sim.CI_halfwidth, color="#9ecae1", alpha=.6, lw=0)
    x0.plot(sim.alpha, sim.mean_AUC, color="#08519c", lw=2)
    i = sim.mean_AUC.idxmax(); mark(x0, sim.alpha[i], sim.mean_AUC[i], f"max at α={sim.alpha[i]:.2f}")
    x0.set_title("(a) Simulated scores (illustrative)", fontsize=10, loc="left"); x0.set_ylabel("AUC")

    x1 = ax[1]
    x1.fill_between(conv.alpha, conv.CI_low, conv.CI_high, color="#bdbdbd", alpha=.7, lw=0)
    x1.plot(conv.alpha, conv.mean_AUC, color="black", lw=2)
    i = conv.mean_AUC.idxmax(); mark(x1, conv.alpha[i], conv.mean_AUC[i], f"max at α={conv.alpha[i]:.2f}")
    x1.set_title("(b) NA12878 benchmark, 5-fold block CV", fontsize=10, loc="left")

    x2 = ax[2]
    for _, g in pc.groupby("chrom"):
        x2.plot(g.alpha, g.AUC, color="#d9d9d9", lw=.8)
    med = pc.groupby("alpha").AUC.median()
    x2.plot(med.index, med.values, color="#7b3294", lw=2, label="median over chromosomes")
    x2.plot(list(pooled), list(pooled.values()), "s", color="#e66101", ms=6, label="all primary chromosomes")
    i = med.idxmax(); mark(x2, i, med[i], f"median max at α={i:.2f}")
    x2.set_ylim(0.65, 1.0)
    x2.set_title("(c) NA06985, PopDB-supported vs. background", fontsize=10, loc="left")
    x2.legend(fontsize=8, loc="lower right", frameon=False)
    for axis in ax:
        axis.set_xlabel("α (weight of the LightGBM probability)")
        axis.axvline(0.8, ls=":", color="grey", lw=1); axis.set_xlim(-0.02, 1.02)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"saved {a.out}.pdf / .png")


if __name__ == "__main__":
    main()
