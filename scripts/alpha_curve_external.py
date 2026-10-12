#!/usr/bin/env python3
"""
alpha_curve_external.py — AUC of the hybrid score alpha*p_LGB + (1-alpha)*p_GM
across the alpha grid AFTER TRANSFER, on the NA06985 apply-mode scores, with
the PopDB label. No model is refitted: the saved LGB and GM scores are blended.

Variability is shown across chromosomes (one curve per autosome), the analogue
of the per-fold curves of the benchmark analysis (alpha_convergence.py).

  python scripts/alpha_curve_external.py \
      --scores output/apply/scored_variants_with_external_validation.csv \
      --out results/alpha_external

Outputs: alpha_external_summary.csv, alpha_external_per_chrom.csv,
fig_alpha_external.pdf/.png
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
ap.add_argument("--label-col", default="IsExternalVariant")
ap.add_argument("--base", default="LGB", help="LightGBM base model: LGB, LGB_Bayes or LGB_MultiObj")
ap.add_argument("--out", default="alpha_external")
ap.add_argument("--alpha-step", type=float, default=0.05)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

d = pd.read_csv(a.scores, usecols=["CHROM", a.label_col, f"{a.base}_score", "GM_score"])
d = d.dropna()
d["CHROM"] = d["CHROM"].astype(str).str.replace("^chr", "", regex=True)
y = d[a.label_col].to_numpy(int)
pl, pg = d[f"{a.base}_score"].to_numpy(float), d["GM_score"].to_numpy(float)
alphas = np.round(np.arange(0, 1 + 1e-9, a.alpha_step), 4)

overall = [roc_auc_score(y, al * pl + (1 - al) * pg) for al in alphas]
rows = []
for c in [str(i) for i in range(1, 23)]:
    m = (d["CHROM"] == c).to_numpy()
    if m.sum() == 0 or y[m].min() == y[m].max():
        continue
    for al in alphas:
        rows.append((c, al, roc_auc_score(y[m], al * pl[m] + (1 - al) * pg[m])))
pc = pd.DataFrame(rows, columns=["chrom", "alpha", "AUC"])
pc.to_csv(os.path.join(a.out, "alpha_external_per_chrom.csv"), index=False)
sd = pc.groupby("alpha").AUC.std(ddof=1)
best_per_chrom = pc.loc[pc.groupby("chrom").AUC.idxmax()].alpha
summ = pd.DataFrame({"alpha": alphas, "AUC_all_calls": overall,
                     "mean_AUC_over_chrom": pc.groupby("alpha").AUC.mean().values,
                     "SD_across_chrom": sd.values})
summ.to_csv(os.path.join(a.out, "alpha_external_summary.csv"), index=False)
ib = int(np.argmax(overall))
print(f"{len(y):,} calls ({y.mean():.3f} PopDB-supported), base {a.base}")
print(f"AUC: alpha=0 {overall[0]:.4f} | 0.7 {overall[14]:.4f} | 0.8 {overall[16]:.4f} | 1 {overall[-1]:.4f}")
print(f"best alpha on all calls = {alphas[ib]:g} (AUC {overall[ib]:.4f})")
print(f"best alpha per chromosome: median {best_per_chrom.median():g}, "
      f"range {best_per_chrom.min():g}-{best_per_chrom.max():g}")
print(f"SD across chromosomes lowest at alpha = {sd.idxmin():g}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
INK, MUTED, GRID, BLUE, PURPLE = "#1f1f1e", "#6b6a63", "#d9d8d2", "#2a78d6", "#7a4fc4"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
for i, (c, g) in enumerate(pc.groupby("chrom")):
    ax1.plot(g.alpha, g.AUC, color=BLUE, alpha=0.25, lw=0.8, label="chromosome" if i == 0 else None)
ax1.plot(alphas, overall, color=INK, lw=2, label="all calls")
ax1.scatter([alphas[ib]], [overall[ib]], color=INK, zorder=5, s=25)
ax2.plot(alphas, sd.values, color=PURPLE, lw=2, marker="o", ms=3)
for ax in (ax1, ax2):
    for x in (0.7, 0.8):
        ax.axvline(x, color=MUTED, ls=":", lw=1)
        ax.annotate(f"{x:g}", (x, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                    fontsize=7.5, color=MUTED)
    ax.set_xlabel(r"$\alpha$ (weight of the LightGBM probability)")
    ax.grid(color=GRID, lw=0.5)
ax1.set_ylabel("AUC (PopDB-supported vs background)")
ax1.set_title("(a) AUC across the $\\alpha$ grid after transfer (NA06985)", loc="left", fontsize=9.5, color=INK)
ax1.legend(frameon=False, fontsize=8, loc="lower right")
ax2.set_ylabel("SD of AUC across autosomes")
ax2.set_title("(b) Variability across chromosomes", loc="left", fontsize=9.5, color=INK)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(a.out, f"fig_alpha_external.{ext}"), dpi=300, bbox_inches="tight")
print(f"saved {a.out}/fig_alpha_external.pdf / .png")
