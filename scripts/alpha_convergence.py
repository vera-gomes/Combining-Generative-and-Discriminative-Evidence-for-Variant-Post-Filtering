#!/usr/bin/env python3
"""
alpha_convergence.py — convergence analysis of the hybrid weight alpha on the
real benchmark data (NA12878), with the corrected pipeline settings.

For each random seed, the labelled benchmark calls are split into 5 folds of
1 Mb genomic blocks (StratifiedGroupKFold, as in the pipeline). In each fold
the GM pair (5 components, standardized features) and default LightGBM are
fitted on the training folds, and the AUC of

    p_H(x; alpha) = alpha * p_LGB(x) + (1 - alpha) * p_GM(x)

is computed on the held-out fold for every alpha on a grid. Imputation and
scaling are fitted on the training folds only. Repeating over seeds gives

  (a) the per-fold AUC curves and their grand mean with a 95% CI, and
  (b) the cross-fold standard deviation of AUC for each alpha (mean over seeds),

i.e. the analysis of the original Figures 3-5, on real data.

Usage (from the repository root, with the package installed):

  python scripts/alpha_convergence.py \
      --annotations <train_run_dir>/annotations_labeled.csv \
      --out results/alpha_convergence --seeds 10

  Quick check first:  --seeds 1 --max-blocks 300

Outputs: alpha_convergence_per_fold.csv (seed, fold, alpha, AUC),
alpha_convergence_summary.csv (per alpha: mean AUC, CI, mean cross-fold SD),
fig_alpha_convergence.pdf/.png.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture

from postfilter_pipeline.data_prep import build_genomic_groups, make_block_splitter
from postfilter_pipeline.models import (LeakageAwareImputer, LeakageAwareScaler,
                                        compute_vqslod_prob)

FEATURES = ["MQ", "QD", "FS", "SOR", "MQRankSum", "ReadPosRankSum", "BaseQRankSum", "DP"]
# default LightGBM of the pipeline (models.lightgbm in configs/train.example.yaml)
LGB_PARAMS = dict(objective="binary", boosting_type="gbdt", n_estimators=100, learning_rate=0.05,
                  max_depth=7, num_leaves=31, min_child_samples=20, subsample=0.8,
                  colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0, verbose=-1,
                  force_row_wise=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True, help="annotations_labeled.csv from a train run")
    ap.add_argument("--label-col", default="TruthLabel")
    ap.add_argument("--out", default="results/alpha_convergence")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--alpha-step", type=float, default=0.05)
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--max-blocks", type=int, default=0,
                    help="use a random subset of this many 1 Mb blocks (0 = all data)")
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.annotations, usecols=lambda c: c in FEATURES + ["CHROM", "POS", a.label_col])
    groups = build_genomic_groups(df, block_size_bp=1_000_000)
    if a.max_blocks:
        rng = np.random.default_rng(0)
        keep = rng.choice(np.unique(groups), size=min(a.max_blocks, len(np.unique(groups))), replace=False)
        m = np.isin(groups, keep)
        df, groups = df[m].reset_index(drop=True), groups[m]
    X = df[FEATURES].to_numpy(dtype=float)
    y = df[a.label_col].to_numpy(dtype=int)
    alphas = np.round(np.arange(0, 1 + 1e-9, a.alpha_step), 4)
    print(f"{len(y):,} calls, {y.mean():.3f} true, {len(np.unique(groups)):,} blocks, "
          f"{a.seeds} seeds x {a.folds} folds, {len(alphas)} alpha values", flush=True)

    rows = []
    for seed in range(a.seeds):
        splitter, _ = make_block_splitter(a.folds, random_state=seed)
        for fold, (tr, va) in enumerate(splitter.split(X, y, groups)):
            t0 = time.time()
            imp = LeakageAwareImputer().fit(X[tr])
            Xtr, Xva = imp.transform(X[tr]), imp.transform(X[va])
            sc = LeakageAwareScaler().fit(Xtr)
            Str, Sva = sc.transform(Xtr), sc.transform(Xva)
            gm_kw = dict(n_components=a.n_components, max_iter=500, random_state=seed * 100 + fold)
            g = GaussianMixture(**gm_kw).fit(Str[y[tr] == 1])
            b = GaussianMixture(**gm_kw).fit(Str[y[tr] == 0])
            p_gm = compute_vqslod_prob(g, b, Sva)
            lgb = LGBMClassifier(**LGB_PARAMS, random_state=seed * 100 + fold, n_jobs=a.n_jobs).fit(Xtr, y[tr])
            p_lgb = lgb.predict_proba(Xva)[:, 1]
            for al in alphas:
                rows.append((seed, fold, al, roc_auc_score(y[va], al * p_lgb + (1 - al) * p_gm)))
            print(f"seed {seed} fold {fold}: AUC(GM)={rows[-len(alphas)][3]:.4f} "
                  f"AUC(LGB)={rows[-1][3]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows, columns=["seed", "fold", "alpha", "AUC"]).to_csv(
            os.path.join(a.out, "alpha_convergence_per_fold.csv"), index=False)

    r = pd.DataFrame(rows, columns=["seed", "fold", "alpha", "AUC"])
    # grand mean over all seed x fold estimates; CI from the seed-level means
    seed_mean = r.groupby(["alpha", "seed"]).AUC.mean().unstack()
    cv_sd = r.groupby(["alpha", "seed"]).AUC.std(ddof=1).unstack()   # cross-fold SD per seed
    n_s = seed_mean.shape[1]
    half = 1.96 * seed_mean.std(axis=1, ddof=1) / np.sqrt(n_s) if n_s > 1 else 0 * seed_mean.mean(axis=1)
    summ = pd.DataFrame({"alpha": seed_mean.index, "mean_AUC": seed_mean.mean(axis=1).values,
                         "CI_low": (seed_mean.mean(axis=1) - half).values,
                         "CI_high": (seed_mean.mean(axis=1) + half).values,
                         "mean_crossfold_SD": cv_sd.mean(axis=1).values,
                         "min_crossfold_SD": cv_sd.min(axis=1).values,
                         "max_crossfold_SD": cv_sd.max(axis=1).values})
    summ.to_csv(os.path.join(a.out, "alpha_convergence_summary.csv"), index=False)
    best = summ.loc[summ.mean_AUC.idxmax()]
    print(f"grand-mean AUC is highest at alpha = {best.alpha:g} ({best.mean_AUC:.4f}); "
          f"cross-fold SD is lowest at alpha = {summ.loc[summ.mean_crossfold_SD.idxmin(), 'alpha']:g}")
    plot(r, summ, a.out)


def plot(r, summ, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID, BLUE, PURPLE = "#1f1f1e", "#6b6a63", "#d9d8d2", "#2a78d6", "#7a4fc4"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    fold_curves = r.groupby(["fold", "alpha"]).AUC.mean().unstack(0)   # each fold averaged over seeds
    for f in fold_curves:
        ax1.plot(fold_curves.index, fold_curves[f], color=BLUE, alpha=0.35, lw=1,
                 label="fold (mean over seeds)" if f == fold_curves.columns[0] else None)
    ax1.fill_between(summ.alpha, summ.CI_low, summ.CI_high, color=INK, alpha=0.15, lw=0)
    ax1.plot(summ.alpha, summ.mean_AUC, color=INK, lw=2, label="grand mean (95% CI)")
    ax1.axvline(0.8, color=MUTED, ls=":", lw=1); ax1.axvline(0.7, color=MUTED, ls=":", lw=1)
    ax1.set_xlabel(r"$\alpha$ (weight of the LightGBM probability)"); ax1.set_ylabel("AUC (held-out fold)")
    ax1.set_title("(a) Held-out AUC across the $\\alpha$ grid", loc="left", fontsize=9.5, color=INK)
    ax1.grid(color=GRID, lw=0.5); ax1.legend(frameon=False, fontsize=8, loc="lower right")
    ax2.plot(summ.alpha, summ.mean_crossfold_SD, color=PURPLE, lw=2, marker="o", ms=3,
             label="mean over seeds")
    ax2.plot(summ.alpha, summ.min_crossfold_SD, color=PURPLE, lw=0.8, ls="--", label="min / max over seeds")
    ax2.plot(summ.alpha, summ.max_crossfold_SD, color=PURPLE, lw=0.8, ls="--")
    ax2.axvline(0.8, color=MUTED, ls=":", lw=1); ax2.axvline(0.7, color=MUTED, ls=":", lw=1)
    for ax in (ax1, ax2):
        for x, lab in ((0.7, "0.7"), (0.8, "0.8")):
            ax.annotate(lab, (x, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                        fontsize=7.5, color=MUTED)
    ax2.set_xlabel(r"$\alpha$ (weight of the LightGBM probability)"); ax2.set_ylabel("Cross-fold SD of AUC")
    ax2.set_title("(b) Variability of AUC across folds", loc="left", fontsize=9.5, color=INK)
    ax2.grid(color=GRID, lw=0.5); ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_alpha_convergence.{ext}"), dpi=300, bbox_inches="tight")
    print(f"saved {out}/fig_alpha_convergence.pdf / .png")


if __name__ == "__main__":
    main()
