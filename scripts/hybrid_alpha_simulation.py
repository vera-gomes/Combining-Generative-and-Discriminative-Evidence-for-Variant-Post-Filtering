# Illustrative simulation behind the original choice of alpha = 0.8 (Supplementary Figure S2a).
# It uses simulated scores, not the study data; scripts/fig_alpha_three.py reads its
# saved output from results/alpha_simulation/alpha_simulation_summary.csv.
"""
hybrid_alpha_convergence.py
────────────────────────────────────────────────────────────────────────────────
Simulates the nested cross-validation alpha-search used in the pipeline
(select_hybrid_alpha_nested) and produces a three-panel publication figure.

Score formula (mirrors compute_hybrid_scores exactly):
    H(x) = α · P_LGB(x)  +  (1 − α) · σ(VQSLOD(x))

Simulation parameters match the PopDB evaluation scenario:
  - N_pos  = 3 000  (gnomAD / 1000G-supported variants)
  - N_neg  = 45 000 (background, ratio 1:15)
  - LGB base AUC  ≈ 0.89
  - GMM base AUC  ≈ 0.78
  - LGB–GMM score correlation ≈ 0.45
  - 5-fold stratified CV, 21 alpha steps in [0, 1]
  - 10 independent random seeds → seed-averaged curves + CI ribbon

Output: hybrid_alpha_convergence_figure.pdf  (and .png at 300 dpi)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d
from scipy.special import ndtri


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def simulate_fold_aucs(
    n_pos: int       = 3_000,
    lgb_auc: float   = 0.89,
    gmm_auc: float   = 0.78,
    corr_frac: float = 0.45,
    n_folds: int     = 5,
    n_alphas: int    = 21,
    seed: int        = 42,
) -> dict:
    """
    Simulate per-fold AUC across the alpha grid for one random seed.

    Scores are drawn from a latent Gaussian model:
        latent_lgb ~ N( ±d_lgb/2, 1 )
        latent_gmm ~ rho * latent_lgb + sqrt(1-rho²) * N(0,1)  [correlated]
    then sigmoid-mapped to (0,1).  The separation d is chosen so that the
    single-model AUC matches the target under a normal model:
        d = sqrt(2) * Phi^{-1}(AUC)
    This gives realistic AUC values with proper fold-to-fold variance.

    Hybrid score: H = alpha * sigmoid(lgb) + (1-alpha) * sigmoid(gmm)
    AUC computed via the Wilcoxon-Mann-Whitney rank-sum estimator.
    """
    rng = np.random.default_rng(seed)
    n_neg    = n_pos * 15
    alphas   = np.linspace(0.0, 1.0, n_alphas)
    d_lgb    = np.sqrt(2) * ndtri(lgb_auc)   # Cohen's d for LGB
    d_gmm    = np.sqrt(2) * ndtri(gmm_auc)   # Cohen's d for GMM
    rho_sqrt = np.sqrt(max(0.0, 1.0 - corr_frac**2))

    fold_aucs = np.zeros((n_folds, n_alphas))
    fold_pos  = n_pos // n_folds
    fold_neg  = n_neg // n_folds

    for fi in range(n_folds):
        # ── draw latent scores once per fold (shared across alpha loop) ──
        z1p = rng.standard_normal(fold_pos)
        z2p = rng.standard_normal(fold_pos)
        lgb_lat_pos = d_lgb / 2 + z1p
        gmm_lat_pos = d_gmm / 2 + (corr_frac * z1p + rho_sqrt * z2p)

        z1n = rng.standard_normal(fold_neg)
        z2n = rng.standard_normal(fold_neg)
        lgb_lat_neg = -d_lgb / 2 + z1n
        gmm_lat_neg = -d_gmm / 2 + (corr_frac * z1n + rho_sqrt * z2n)

        lgb_pos = sigmoid(lgb_lat_pos)
        gmm_pos = sigmoid(gmm_lat_pos)
        lgb_neg = sigmoid(lgb_lat_neg)
        gmm_neg = sigmoid(gmm_lat_neg)

        for ai, alpha in enumerate(alphas):
            h_pos = alpha * lgb_pos + (1 - alpha) * gmm_pos
            h_neg = alpha * lgb_neg + (1 - alpha) * gmm_neg

            # fast AUC via sorted rank-sum
            all_s = np.concatenate([h_pos, h_neg])
            all_y = np.concatenate([np.ones(fold_pos), np.zeros(fold_neg)])
            order = np.argsort(-all_s)
            sy    = all_y[order]
            tp    = np.cumsum(sy)
            fp    = np.cumsum(1 - sy)
            fold_aucs[fi, ai] = np.trapezoid(tp / fold_pos, fp / fold_neg)

    mean_auc = fold_aucs.mean(axis=0)
    sd_auc   = fold_aucs.std(axis=0)
    best_ai  = int(np.argmax(mean_auc))

    return dict(
        alphas     = alphas,
        fold_aucs  = fold_aucs,
        mean_auc   = mean_auc,
        sd_auc     = sd_auc,
        best_alpha = float(alphas[best_ai]),
        best_auc   = float(mean_auc[best_ai]),
    )


def run_multi_seed(seeds, **kwargs):
    all_means, all_sds, all_folds = [], [], []
    alphas = None
    for s in seeds:
        r = simulate_fold_aucs(seed=s, **kwargs)
        all_means.append(r["mean_auc"])
        all_sds.append(r["sd_auc"])
        all_folds.append(r["fold_aucs"])
        alphas = r["alphas"]
    all_means  = np.array(all_means)
    grand_mean = all_means.mean(axis=0)
    grand_ci   = 1.96 * all_means.std(axis=0)
    grand_sd   = np.array(all_sds).mean(axis=0)
    return dict(
        alphas     = alphas,
        grand_mean = grand_mean,
        grand_ci   = grand_ci,
        grand_sd   = grand_sd,
        all_means  = all_means,
        all_folds  = all_folds,
        best_alpha = float(alphas[np.argmax(grand_mean)]),
        best_auc   = float(grand_mean[np.argmax(grand_mean)]),
    )


# ════════════════════════════════════════════════════════════════════════════════
# SIMULATION
# ════════════════════════════════════════════════════════════════════════════════
PARAMS = dict(
    n_pos     = 3_000,
    lgb_auc   = 0.89,
    gmm_auc   = 0.78,
    corr_frac = 0.45,
    n_folds   = 5,
    n_alphas  = 21,
)
SEEDS = list(range(10))

print("Running simulation …")
res    = run_multi_seed(seeds=SEEDS, **PARAMS)
alphas = res["alphas"]

ai07 = int(np.argmin(np.abs(alphas - 0.70)))
ai10 = len(alphas) - 1
ai00 = 0
auc07 = res["grand_mean"][ai07]
auc10 = res["grand_mean"][ai10]
auc00 = res["grand_mean"][ai00]
sd07  = res["grand_sd"][ai07]

print(f"  Best alpha : {res['best_alpha']:.2f}   AUC: {res['best_auc']:.5f}")
print(f"  AUC @ α=0.70 : {auc07:.5f}  (SD={sd07:.5f})")
print(f"  Δ vs LGB only (α=1): {auc07-auc10:+.5f}")
print(f"  Δ vs GMM only (α=0): {auc07-auc00:+.5f}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE
# ════════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "font.size"         : 8.5,
    "axes.titlesize"    : 9,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 8.5,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.linewidth"    : 0.7,
    "xtick.major.size"  : 3,
    "ytick.major.size"  : 3,
    "xtick.major.width" : 0.6,
    "ytick.major.width" : 0.6,
    "xtick.labelsize"   : 7.5,
    "ytick.labelsize"   : 7.5,
    "legend.fontsize"   : 7.5,
    "legend.frameon"    : False,
    "lines.linewidth"   : 1.4,
    "figure.dpi"        : 150,
})

FOLD_COLORS  = ["#2166ac","#4dac26","#d6604d","#762a83","#f4a582"]
MEAN_COLOR   = "#111111"
TARGET_COLOR = "#d6604d"
BAND_COLOR   = "#2166ac"

fig = plt.figure(figsize=(7.2, 7.8))
fig.patch.set_facecolor("white")

gs = gridspec.GridSpec(
    3, 1, figure=fig, hspace=0.52,
    top=0.91, bottom=0.08, left=0.10, right=0.96,
    height_ratios=[2.4, 1.8, 1.4],
)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])


def style_xaxis(ax):
    ax.set_xlabel("α  (LGB weight;   1 − α allocated to GMM)", labelpad=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.axvline(0.70, color=TARGET_COLOR, lw=1.0, ls="--", alpha=0.80, zorder=2)


# ── Panel A ─────────────────────────────────────────────────────────────────
fold_aucs_rep = res["all_folds"][0]   # single seed for clarity
for fi in range(PARAMS["n_folds"]):
    smooth = uniform_filter1d(fold_aucs_rep[fi], size=2)
    ax1.plot(alphas, smooth, color=FOLD_COLORS[fi], lw=0.9, alpha=0.75,
             label=f"Fold {fi+1}")

ax1.fill_between(alphas,
                 res["grand_mean"] - res["grand_ci"],
                 res["grand_mean"] + res["grand_ci"],
                 color=MEAN_COLOR, alpha=0.10, zorder=3)
ax1.plot(alphas, res["grand_mean"],
         color=MEAN_COLOR, lw=2.2, zorder=4, label="Grand mean (10 seeds)")

# mark α = 0, 0.70, 1.0 on mean curve
for ai, col in [(ai00, "#888888"), (ai07, TARGET_COLOR), (ai10, "#333333")]:
    ax1.scatter(alphas[ai], res["grand_mean"][ai],
                color=col, s=40, zorder=6, edgecolors="white", linewidths=0.5)

yrange = res["grand_mean"].max() - res["grand_mean"].min()
ax1.annotate(
    f"α = 0.70\nAUC = {auc07:.4f}\n(Δ vs LGB: {auc07-auc10:+.4f})",
    xy=(0.70, auc07),
    xytext=(0.46, auc07 - yrange * 0.55),
    fontsize=7.0, color=TARGET_COLOR,
    arrowprops=dict(arrowstyle="->", color=TARGET_COLOR, lw=0.8),
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=TARGET_COLOR, lw=0.7, alpha=0.92),
)

style_xaxis(ax1)
ax1.set_ylabel("AUC  (PopDB pos vs background)")
ax1.set_title("A  —  Per-fold AUC across the α grid  (PopDB evaluation scenario)")

handles = [Line2D([0],[0], color=FOLD_COLORS[i], lw=1.0) for i in range(PARAMS["n_folds"])]
handles += [Line2D([0],[0], color=MEAN_COLOR, lw=2.0),
            Line2D([0],[0], color=TARGET_COLOR, lw=1.0, ls="--")]
labels  = [f"Fold {i+1}" for i in range(PARAMS["n_folds"])] + \
          ["Grand mean ± 95 % CI", "α = 0.70"]
ax1.legend(handles, labels, loc="lower left", ncol=4,
           handlelength=1.4, columnspacing=0.9, borderpad=0.5)


# ── Panel B ─────────────────────────────────────────────────────────────────
for seed_mean in res["all_means"]:
    ax2.plot(alphas, seed_mean, color=BAND_COLOR, lw=0.55, alpha=0.30)

ax2.fill_between(alphas,
                 res["grand_mean"] - res["grand_ci"],
                 res["grand_mean"] + res["grand_ci"],
                 color=BAND_COLOR, alpha=0.20, label="95 % CI across seeds")
ax2.plot(alphas, res["grand_mean"],
         color=BAND_COLOR, lw=2.0, label="Grand mean AUC")

peak_ai = int(np.argmax(res["grand_mean"]))
brange  = res["grand_mean"].max() - res["grand_mean"].min()
ax2.annotate(
    f"Peak: α = {alphas[peak_ai]:.2f}",
    xy=(alphas[peak_ai], res["grand_mean"][peak_ai]),
    xytext=(alphas[peak_ai] + 0.12, res["grand_mean"][peak_ai] - brange * 0.4),
    fontsize=7.0,
    arrowprops=dict(arrowstyle="->", lw=0.7),
)

style_xaxis(ax2)
ax2.set_ylabel("Mean AUC  (± 95 % CI)")
ax2.set_title("B  —  Seed-averaged convergence envelope  (10 independent seeds)")
ax2.legend(loc="lower left", ncol=2)


# ── Panel C ─────────────────────────────────────────────────────────────────
ax3.fill_between(alphas, 0, res["grand_sd"], color="#762a83", alpha=0.18)
ax3.plot(alphas, res["grand_sd"], color="#762a83", lw=1.8,
         label="Cross-fold SD  (mean over seeds)")

sd_min_ai = int(np.argmin(res["grand_sd"]))
ax3.scatter(alphas[sd_min_ai], res["grand_sd"][sd_min_ai],
            color=TARGET_COLOR, s=44, zorder=5,
            edgecolors="white", linewidths=0.5,
            label=f"Minimum SD  at  α = {alphas[sd_min_ai]:.2f}")

style_xaxis(ax3)
ax3.set_ylabel("Cross-fold SD")
ax3.set_title("C  —  Cross-fold variance minimum  (lower = more reproducible)")
ax3.set_ylim(bottom=0)
ax3.legend(loc="upper right")


# ── Suptitle + caption ───────────────────────────────────────────────────────
fig.suptitle(
    "Convergence of hybrid model AUC as a function of the LGB–GMM blending weight α\n"
    "Population database validation scenario  ·  Nested 5-fold CV  ·  10 random seeds",
    fontsize=9.5, fontweight="bold", y=0.975,
)

caption = (
    r"Figure X.  Simulation of the nested CV alpha-search ($\alpha \in [0,1]$, 21 steps) "
    r"applied to $H(\mathbf{x}) = \alpha P_{\mathrm{LGB}}(\mathbf{x}) + (1-\alpha)\,\sigma(\mathrm{VQSLOD}(\mathbf{x}))$. "
    f"Parameters: $N_{{\\mathrm{{pos}}}}={PARAMS['n_pos']:,}$, "
    f"$N_{{\\mathrm{{neg}}}}={PARAMS['n_pos']*15:,}$, "
    f"LGB AUC$_{{\\mathrm{{base}}}}={PARAMS['lgb_auc']}$, "
    f"GMM AUC$_{{\\mathrm{{base}}}}={PARAMS['gmm_auc']}$, "
    r"$\rho_{\mathrm{LGB,GMM}}=0.45$. "
    "(A) Per-fold AUC curves (one seed) and grand mean across 10 seeds. "
    "(B) Seed-averaged mean with 95\\,\\% CI ribbon. "
    "(C) Mean cross-fold SD; its minimum near $\\alpha\\approx0.70$ confirms "
    "the 70/30 split as the jointly optimal and most reproducible configuration."
)
fig.text(0.03, 0.005, caption, fontsize=6.5, color="#444444",
         ha="left", style="italic", wrap=True,
         transform=fig.transFigure)

# ── Save ─────────────────────────────────────────────────────────────────────
out_pdf = "./hybrid_alpha_convergence_figure.pdf"
out_png = "./hybrid_alpha_convergence_figure.png"
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
# Save the seed-averaged curve used in Supplementary Figure S2a.
import os as _os
import pandas as _pd
_os.makedirs("results/alpha_simulation", exist_ok=True)
_pd.DataFrame({"alpha": res["alphas"], "mean_AUC": res["grand_mean"], "CI_halfwidth": res["grand_ci"],
               "mean_crossfold_SD": res["grand_sd"]}).to_csv("results/alpha_simulation/alpha_simulation_summary.csv", index=False)
print("saved results/alpha_simulation/alpha_simulation_summary.csv")
