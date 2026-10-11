#!/usr/bin/env python3
"""
make_results.py — builds every table and figure of the revised manuscript and
supplement from the raw result files, so each number in the paper is traceable.

  python3 scripts/make_results.py --reeval results/external_validation --train results/train \
      --clinvar-records results/external_validation/clinvar_records_in_callset.csv.gz

Outputs: tables/*.tex (included with \\input) and figures/*.pdf|png.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- naming
ORDER = ["GM", "BGM", "LogReg", "RandForest", "LGB", "LGB_Bayes", "LGB_MultiObj",
         "Hybrid_LGB_alpha08", "Hybrid_LGB_alpha07"]
DISPLAY = {"GM": "GM", "BGM": "BGM", "LogReg": "LogReg", "RandForest": "RF",
           "LGB": "LGB", "LGB_Bayes": "LGB_Bayes", "LGB_MultiObj": "LGB_MultiObj",
           "Hybrid_LGB_alpha08": "Hybrid_0.8", "Hybrid_LGB_alpha07": "Hybrid_0.7"}
FAMILY = {"GM": "gen", "BGM": "gen", "LogReg": "disc", "RandForest": "disc", "LGB": "disc",
          "LGB_Bayes": "disc", "LGB_MultiObj": "disc",
          "Hybrid_LGB_alpha08": "hyb", "Hybrid_LGB_alpha07": "hyb"}
# validated categorical slots (light mode) from the reference palette
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FAM_COLOR = {"gen": C_ORANGE, "disc": C_BLUE, "hyb": C_AQUA}
FAM_MARK = {"gen": "s", "disc": "o", "hyb": "D"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#d9d8d2"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
})


def tm(name):
    """Model name as LaTeX (monospace macro defined in the manuscript)."""
    return r"\mdl{" + DISPLAY.get(name, name).replace("_", r"\_") + "}"


def fmt_p(p):
    if not np.isfinite(p):
        return "--"
    if p >= 0.001:
        return f"{p:.3f}"
    if p == 0:
        return r"$<10^{-300}$"
    e = int(np.floor(np.log10(p)))
    if e < -300:
        return r"$<10^{-300}$"
    m = p / 10 ** e
    return rf"${m:.1f}\times10^{{{e}}}$"


def fnum(x, d=4):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def write(path, text):
    with open(path, "w") as f:
        f.write(text)
    print("wrote", path)


def load_regime(reeval, name):
    p = lambda k: os.path.join(reeval, f"{name}_{k}.csv")
    if not os.path.exists(p("summary")):
        return None
    s = pd.read_csv(p("summary")).set_index("Model").reindex(ORDER).dropna(how="all")
    pw = pd.read_csv(p("pairwise"))
    tr = pd.read_csv(p("tranches"))
    return s, pw, tr


# ---------------------------------------------------------------- tables
def table_cv(train, out):
    pf = pd.read_csv(os.path.join(train, "cv_metrics_per_fold.csv"))
    pf = pf[pf.Model.isin(ORDER)]
    g = pf.groupby("Model")[["AUC", "Precision", "Recall", "F1"]].agg(["mean", "std"]).reindex(ORDER)
    rows = []
    for m in ORDER:
        cells = [f"{g.loc[m, (k, 'mean')]:.3f} ({g.loc[m, (k, 'std')]:.3f})"
                 for k in ["AUC", "Precision", "Recall", "F1"]]
        rows.append(" & ".join([tm(m)] + cells) + r" \\")
    tex = (r"\begin{tabular}{@{}lcccc@{}}" "\n" r"\toprule" "\n"
           r"\textbf{Model} & \textbf{AUC} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} \\" "\n"
           r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write(os.path.join(out, "tab_cv.tex"), tex)


def table_main(reg, out, fname, fp_label):
    s, pw, tr = reg
    t90 = tr[np.isclose(tr.Target_sensitivity, 0.90)].set_index("Model")
    t99 = tr[np.isclose(tr.Target_sensitivity, 0.99)].set_index("Model")
    best_auc = s["AUC"].max()
    rows = []
    for m in s.index:
        r = s.loc[m]
        auc = f"{r.AUC:.4f}"
        if np.isclose(r.AUC, best_auc):
            auc = r"\textbf{" + auc + "}"
        ci = f"[{r.AUC_DeLong_CI_low:.4f}, {r.AUC_DeLong_CI_high:.4f}]"
        rows.append(" & ".join([
            tm(m), auc, ci, f"{r.AP:.4f}",
            f"{int(t90.loc[m, 'FP']):,}", f"{100 * t90.loc[m, 'Retained_fraction']:.2f}",
            f"{int(t99.loc[m, 'FP']):,}",
        ]) + r" \\")
    tex = (r"\begin{tabular}{@{}lcccrrr@{}}" "\n" r"\toprule" "\n"
           r" & & & & \multicolumn{2}{c}{\textbf{Sensitivity 0.90}} & \textbf{Sens.\ 0.99} \\"
           "\n" r"\cmidrule(lr){5-6}\cmidrule(l){7-7}" "\n"
           r"\textbf{Model} & \textbf{AUC} & \textbf{95\% CI (DeLong)} & \textbf{AP} & "
           rf"\textbf{{{fp_label}}} & \textbf{{Retained (\%)}} & \textbf{{{fp_label}}} \\" "\n"
           r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write(os.path.join(out, fname), tex)


def table_pairwise(reg, out, fname):
    s, pw, tr = reg
    rows = []
    for _, r in pw.iterrows():
        if r.Model_A not in ORDER or r.Model_B not in ORDER:
            continue
        rows.append(" & ".join([
            tm(r.Model_A) + " -- " + tm(r.Model_B),
            f"{r.Delta_AUC:+.4f}",
            f"[{r.Delta_AUC_bootstrap_CI_low:+.4f}, {r.Delta_AUC_bootstrap_CI_high:+.4f}]",
            fmt_p(r.DeLong_p_Holm),
            f"{r.Delta_AP:+.5f}",
            f"[{r.Delta_AP_bootstrap_CI_low:+.5f}, {r.Delta_AP_bootstrap_CI_high:+.5f}]",
        ]) + r" \\")
    tex = (r"\begin{tabular}{@{}lccccc@{}}" "\n" r"\toprule" "\n"
           r"\textbf{Comparison (A -- B)} & $\boldsymbol{\Delta}$\textbf{AUC} & \textbf{95\% CI (bootstrap)} & "
           r"\textbf{Holm $p$ (DeLong)} & $\boldsymbol{\Delta}$\textbf{AP} & \textbf{95\% CI (bootstrap)} \\" "\n"
           r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write(os.path.join(out, fname), tex)


def table_tranches(reg, out, fname):
    s, pw, tr = reg
    tr = tr[tr.Model.isin(ORDER)].copy()
    tr["o"] = tr.Model.map({m: i for i, m in enumerate(ORDER)})
    tr = tr.sort_values(["o", "Target_sensitivity"], ascending=[True, False])
    rows, last = [], None
    for _, r in tr.iterrows():
        if last is not None and r.Model != last:
            rows.append(r"\addlinespace")
        last = r.Model
        thr = f"{r.Threshold:.4f}" if r.Threshold >= 1e-3 else fmt_p(r.Threshold).replace(r"$<10^{-300}$", "0")
        rows.append(" & ".join([
            tm(r.Model), f"{r.Target_sensitivity:.3f}", thr, f"{int(r.TP):,}", f"{int(r.FP):,}",
            f"{r.Precision:.6f}" if r.Precision > 0.01 else fmt_p(r.Precision),
            f"{r.Recall:.4f}", f"{100 * r.Retained_fraction:.2f}",
        ]) + r" \\")
    tex = (r"\begin{tabular}{@{}lccrrccc@{}}" "\n" r"\toprule" "\n"
           r"\textbf{Model} & \textbf{Target sens.} & \textbf{Threshold} & \textbf{TP} & \textbf{FP} & "
           r"\textbf{Precision} & \textbf{Recall} & \textbf{Retained (\%)} \\" "\n"
           r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write(os.path.join(out, fname), tex)


def table_pbest(reg, out, fname):
    s, pw, tr = reg
    cols = [("AUC", "AUC"), ("AP", "AP"), ("F1_at_0.5", "F1@0.5"), ("LogLoss", "LogLoss")]
    rows = []
    for m in s.index:
        cells = []
        for k, _ in cols:
            a, b = s.loc[m, f"Pbest_{k}_bootstrap"], s.loc[m, f"Pbest_{k}_bayesboot"]
            cells.append(f"{100 * a:.1f} / {100 * b:.1f}")
        rows.append(" & ".join([tm(m)] + cells) + r" \\")
    head = " & ".join([r"\textbf{Model}"] + [rf"\textbf{{{lab}}}" for _, lab in cols])
    tex = (r"\begin{tabular}{@{}lcccc@{}}" "\n" r"\toprule" "\n" + head + r" \\" "\n"
           r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write(os.path.join(out, fname), tex)


def table_alpha(train, out):
    h = json.load(open(os.path.join(train, "hybrid_tuning_summary.json")))
    bases = [("Hybrid_LGB", "LGB"), ("Hybrid_LGB_Bayes", "LGB_Bayes"), ("Hybrid_LGB_MultiObj", "LGB_MultiObj")]
    alphas = [round(a, 1) for a, _ in h["Hybrid_LGB"]["alpha_cv_scores"]]
    rows = []
    for i, a in enumerate(alphas):
        cells = [f"{h[k]['alpha_cv_scores'][i][1]:.4f}" for k, _ in bases]
        rows.append(" & ".join([f"{a:.1f}"] + cells) + r" \\")
    head = " & ".join([r"$\boldsymbol{\alpha}$"] + [r"\textbf{Base: }" + tm(b) for _, b in bases])
    tex = (r"\begin{tabular}{@{}cccc@{}}" "\n" r"\toprule" "\n" + head + r" \\" "\n" r"\midrule" "\n"
           + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write(os.path.join(out, "tab_alpha_cv.tex"), tex)


def table_plp(records, tranches_popdb, out):
    d = pd.read_csv(records)
    p = d[d.Class == "P"].copy()
    if p.empty:
        return
    t90 = tranches_popdb[np.isclose(tranches_popdb.Target_sensitivity, 0.90)].set_index("Model").Threshold
    p["Gene"] = p.GENEINFO.str.split(":").str[0]
    rows = []
    p["_c"] = pd.to_numeric(p.CHROM, errors="coerce").fillna(99)
    for _, r in p.sort_values(["_c", "POS"]).iterrows():
        cells = []
        for m in ORDER:
            sc = r[f"{m}_score"]
            mark = r"\checkmark" if sc >= t90[m] else r"$\times$"
            cells.append(mark)
        var = rf"\textit{{{r.Gene}}} chr{r.CHROM}:{r.POS:,} {r.REF}$>${r.ALT}"
        rows.append(" & ".join([var] + cells) + r" \\")
    kept = [str(int(sum(p[f"{m}_score"] >= t90[m]))) for m in ORDER]
    rows.append(r"\midrule")
    rows.append(" & ".join([r"\textbf{Retained (of " + str(len(p)) + ")}"] + kept) + r" \\")
    head = " & ".join([r"\textbf{Variant}"] + [r"\rotatebox{90}{" + tm(m) + "}" for m in ORDER])
    tex = (r"\begin{tabular}{@{}l" + "c" * len(ORDER) + r"@{}}" "\n" r"\toprule" "\n" + head + r" \\" "\n"
           r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write(os.path.join(out, "tab_plp_retention.tex"), tex)


def table_threshold(reg, out, fname):
    s, pw, tr = reg
    rows = []
    for m in s.index:
        r = s.loc[m]
        rows.append(" & ".join([tm(m), f"{r['F1_at_0.5']:.4f}",
            f"[{r['F1_at_0.5_bootstrap_CI_low']:.4f}, {r['F1_at_0.5_bootstrap_CI_high']:.4f}]",
            f"{r.LogLoss:.4f}", f"[{r.LogLoss_bootstrap_CI_low:.4f}, {r.LogLoss_bootstrap_CI_high:.4f}]",
            f"{100*r.Pbest_AUC_bootstrap:.1f} / {100*r.Pbest_AUC_bayesboot:.1f}",
            f"{100*r.Pbest_AP_bootstrap:.1f} / {100*r.Pbest_AP_bayesboot:.1f}",
            f"{100*r['Pbest_F1_at_0.5_bootstrap']:.1f} / {100*r['Pbest_F1_at_0.5_bayesboot']:.1f}",
            f"{100*r.Pbest_LogLoss_bootstrap:.1f} / {100*r.Pbest_LogLoss_bayesboot:.1f}"]) + r" \\")
    tex = (r"\begin{tabular}{@{}lcccccccc@{}}" "\n" r"\toprule" "\n"
           r" & \multicolumn{4}{c}{\textbf{Threshold 0.5}} & \multicolumn{4}{c}{\textbf{Probability of being best (\%), standard / Bayesian bootstrap}} \\" "\n"
           r"\cmidrule(lr){2-5}\cmidrule(l){6-9}" "\n"
           r"\textbf{Model} & \textbf{F1} & \textbf{95\% CI} & \textbf{LogLoss} & \textbf{95\% CI} & \textbf{AUC} & \textbf{AP} & \textbf{F1} & \textbf{LogLoss} \\" "\n"
           r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write(os.path.join(out, fname), tex)


def table_hyper(train, out):
    j = json.load(open(os.path.join(train, "lightgbm_tuning_summary.json")))
    keys = ["n_estimators", "learning_rate", "max_depth", "num_leaves", "min_child_samples",
            "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "min_split_gain"]
    names = [("LGB", "LGB"), ("LGB_Bayes", "LGB_Bayes"), ("LGB_MultiObj", "LGB_MultiObj")]
    def f(v):
        if v is None: return "--"
        if isinstance(v, int): return str(v)
        if abs(v) < 1e-3 and v != 0: return fmt_p(v)
        return f"{v:.4g}"
    rows = []
    for k in keys:
        rows.append(" & ".join([r"\texttt{" + k.replace("_", r"\_") + "}"] +
                               [f(j[n]["best_params"].get(k)) for n, _ in names]) + r" \\")
    rows.append(r"\midrule")
    rows.append(" & ".join(["Tuning criterion", "none (defaults)", "AUC", r"$0.85\,$AP $-\,0.15\min(t/5\,\mathrm{s},1)$"]) + r" \\")
    rows.append(" & ".join(["Trials", "--", str(j["LGB_Bayes"].get("n_trials", "--")), str(j["LGB_MultiObj"].get("n_trials", "--"))]) + r" \\")
    rows.append(" & ".join(["Validation score of selected trial", f"{j['LGB']['primary_metric_value']:.4f} (AUC)",
                            f"{j['LGB_Bayes']['primary_metric_value']:.4f} (AUC)",
                            f"{j['LGB_MultiObj']['primary_metric_value']:.4f} (AP); $J=${j['LGB_MultiObj']['best_value']:.4f}"]) + r" \\")
    rows.append(" & ".join(["Fit time of selected trial (s)", "--", f"{j['LGB_Bayes'].get('runtime_sec', float('nan')):.1f}",
                            f"{j['LGB_MultiObj'].get('runtime_sec', float('nan')):.1f}"]) + r" \\")
    head = " & ".join([r"\textbf{Hyperparameter}"] + [r"\textbf{" + tm(n) + "}" for n, _ in names])
    tex = (r"\begin{tabular}{@{}lccc@{}}" "\n" r"\toprule" "\n" + head + r" \\" "\n" r"\midrule" "\n"
           + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write(os.path.join(out, "tab_hyper.tex"), tex)


# ---------------------------------------------------------------- original-format tables
def table_auc_simple(reg, out, fname):
    """Original-style ranking table (sorted by AUC), now with CI and AP."""
    s, pw, tr = reg
    s = s.sort_values("AUC", ascending=False)
    best = s["AUC"].max()
    rows = []
    for m in s.index:
        r = s.loc[m]
        auc = f"{r.AUC:.4f}"
        if np.isclose(r.AUC, best):
            auc = r"\textbf{" + auc + "}"
        ap = f"{r.AP:.4f}" if r.AP >= 0.01 else rf"${r.AP*1e4:.2f}\times10^{{-4}}$"
        rows.append(" & ".join([tm(m), auc, f"[{r.AUC_DeLong_CI_low:.4f}, {r.AUC_DeLong_CI_high:.4f}]", ap]) + r" \\")
    tex = (r"\begin{tabular}{@{}lccc@{}}" "\n" r"\toprule" "\n"
           r"\textbf{Model} & \textbf{AUC} & \textbf{95\% CI (DeLong)} & \textbf{AP} \\" "\n"
           r"\midrule" "\n" + "\n".join(rows) + "\n" r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write(os.path.join(out, fname), tex)


def _sci(x):
    if x >= 0.01:
        return f"{x:.6f}"
    e = int(np.floor(np.log10(x))); m = x / 10 ** e
    return rf"\({m:.2f}\times10^{{{e}}}\)"


def table_tranche_selected(reg, out, fname):
    """Original-style tranche table: 0.90 and 0.99, sorted by precision, plus retained fraction."""
    s, pw, tr = reg
    blocks = []
    for sens in (0.90, 0.99):
        t = tr[np.isclose(tr.Target_sensitivity, sens) & tr.Model.isin(ORDER)].sort_values("Precision", ascending=False)
        rows = []
        for _, r in t.iterrows():
            thr = f"{r.Threshold:.4f}" if r.Threshold >= 1e-3 else _sci(r.Threshold)
            rows.append(" & ".join([tm(r.Model), f"{sens:.2f}", thr, _sci(r.Precision),
                                    f"{r.Recall:.4f}", _sci(r.F1), f"{int(r.FP):,}",
                                    f"{100*r.Retained_fraction:.1f}\\%"]) + r" \\")
        blocks.append("\n".join(rows))
    tex = (r"\begin{tabular}{@{}lccccccr@{}}" "\n" r"\toprule" "\n"
           r"\textbf{Model} & \textbf{Target sens.} & \textbf{Threshold} & \textbf{Precision} & "
           r"\textbf{Recall} & \textbf{F1} & \textbf{Background kept} & \textbf{Retained} \\" "\n"
           r"\midrule" "\n" + blocks[0] + "\n" r"\midrule" "\n" + blocks[1] + "\n"
           r"\bottomrule" "\n" r"\end{tabular}" "\n")
    write(os.path.join(out, fname), tex)


def fig_heatmap_single(reg, title, out, stem):
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#c2410c", "#eb6834", "#f3f2ee", "#2a78d6", "#184f95"])
    s, pw, _ = reg
    ms = [m for m in ORDER if m in s.index]; k = len(ms); idx = {m: i for i, m in enumerate(ms)}
    M = np.full((k, k), np.nan); sig = np.zeros((k, k), bool)
    for _, r in pw.iterrows():
        if r.Model_A in idx and r.Model_B in idx:
            i, j = idx[r.Model_A], idx[r.Model_B]
            a, b = (i, j) if i > j else (j, i)
            M[a, b] = 100 * (r.Delta_AUC if i > j else -r.Delta_AUC)
            sig[a, b] = r.DeLong_p_Holm < 0.05
    v = np.nanmax(np.abs(M))
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -v, v))
    for a in range(k):
        for b in range(k):
            if np.isfinite(M[a, b]):
                ax.text(b, a, f"{M[a, b]:+.2f}" + ("" if sig[a, b] else "\nn.s."), ha="center", va="center",
                        fontsize=7, color="white" if abs(M[a, b]) > 0.6 * v else INK)
    lab = [DISPLAY[m] for m in ms]
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(lab, rotation=45, ha="right", family="DejaVu Sans Mono", fontsize=7.5)
    ax.set_yticklabels(lab, family="DejaVu Sans Mono", fontsize=7.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("Column model"); ax.set_ylabel("Row model")
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.outline.set_visible(False)
    cb.set_label("ΔAUC × 100 (row − column)", color=INK)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{stem}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig); print("wrote", stem)


# ---------------------------------------------------------------- figures
def fig_auc_dots(regs, out):
    panels = [(k, v) for k, v in regs if v is not None]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (title, (s, _, _)) in zip(axes, panels):
        ys = np.arange(len(s))[::-1]
        for y, m in zip(ys, s.index):
            f = FAMILY[m]
            r = s.loc[m]
            ax.plot([r.AUC_DeLong_CI_low, r.AUC_DeLong_CI_high], [y, y], color=FAM_COLOR[f], lw=2,
                    solid_capstyle="round")
            ax.plot(r.AUC, y, FAM_MARK[f], ms=6, color=FAM_COLOR[f], mec="white", mew=1.2)
        ax.set_yticks(ys)
        ax.set_yticklabels([DISPLAY[m] for m in s.index], family="DejaVu Sans Mono", color=INK)
        ax.set_xlabel("AUC (point estimate, 95% DeLong CI)")
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        lo = (s.AUC_DeLong_CI_low.min()); hi = (s.AUC_DeLong_CI_high.max())
        pad = (hi - lo) * 0.08
        ax.set_xlim(lo - pad, hi + pad)
    handles = [plt.Line2D([], [], marker=FAM_MARK[f], color=FAM_COLOR[f], lw=2, mec="white", label=l)
               for f, l in [("gen", "Generative (class-conditional mixtures)"),
                            ("disc", "Discriminative"), ("hyb", "Fixed-weight hybrid")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_auc_external.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_auc_external")


def fig_delta_heatmaps(regs, out):
    panels = [(k, v) for k, v in regs if v is not None]
    fig, axes = plt.subplots(len(panels), 1, figsize=(7.0, 5.6 * len(panels)))
    axes = np.atleast_1d(axes)
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    cmap = LinearSegmentedColormap.from_list("div", ["#c2410c", "#eb6834", "#f3f2ee", "#2a78d6", "#184f95"])
    for ax, (title, (s, pw, _)) in zip(axes, panels):
        ms = [m for m in ORDER if m in s.index]
        k = len(ms)
        M = np.full((k, k), np.nan)
        sig = np.zeros((k, k), dtype=bool)
        idx = {m: i for i, m in enumerate(ms)}
        for _, r in pw.iterrows():
            if r.Model_A in idx and r.Model_B in idx:
                i, j = idx[r.Model_A], idx[r.Model_B]
                # row model minus column model, lower triangle
                a, b = (i, j) if i > j else (j, i)
                d = r.Delta_AUC if i > j else -r.Delta_AUC
                M[a, b] = d
                sig[a, b] = r.DeLong_p_Holm < 0.05
        M = 100 * M
        v = np.nanmax(np.abs(M))
        im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -v, v))
        for a in range(k):
            for b in range(k):
                if np.isfinite(M[a, b]):
                    txt = f"{M[a, b]:+.2f}" + ("" if sig[a, b] else "\nn.s.")
                    ax.text(b, a, txt, ha="center", va="center", fontsize=7,
                            color="white" if abs(M[a, b]) > 0.6 * v else INK)
        ax.set_xticks(range(k)); ax.set_yticks(range(k))
        ax.set_xticklabels([DISPLAY[m] for m in ms], rotation=45, ha="right", family="DejaVu Sans Mono", fontsize=7.5)
        ax.set_yticklabels([DISPLAY[m] for m in ms], family="DejaVu Sans Mono", fontsize=7.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("ΔAUC × 100 (row − column)", color=INK)
        cb.outline.set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_delta_auc_heatmaps.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_delta_auc_heatmaps")


def fig_alpha(train, out):
    h = json.load(open(os.path.join(train, "hybrid_tuning_summary.json")))
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    spec = [("Hybrid_LGB", "LGB", C_BLUE, "o", "-"),
            ("Hybrid_LGB_Bayes", "LGB-Bayes", C_ORANGE, "s", "--"),
            ("Hybrid_LGB_MultiObj", "LGB-MultiObj", C_AQUA, "D", ":")]
    for key, lab, col, mk, ls in spec:
        a = np.array(h[key]["alpha_cv_scores"], dtype=float)
        ax.plot(a[:, 0], a[:, 1], ls, color=col, lw=2, marker=mk, ms=5, mec="white", mew=1, label=f"base: {lab}")
    ax.axvline(0.8, color=MUTED, lw=0.8, ls=(0, (2, 2)))
    ax.text(0.785, 0.806, "α = 0.8\n(fixed-weight\ncomparator)", ha="right", va="bottom",
            fontsize=7.5, color=MUTED)
    ax.set_xlabel("α (weight on the LightGBM probability; 1 − α on the GM score)")
    ax.set_ylabel("Mean inner-CV AUC")
    ax.set_xlim(-0.03, 1.05)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"fig_alpha_cv.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_alpha_cv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reeval", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--clinvar-records")
    ap.add_argument("--tables", default="tables")
    ap.add_argument("--figures", default="figures")
    a = ap.parse_args()
    os.makedirs(a.tables, exist_ok=True); os.makedirs(a.figures, exist_ok=True)

    regimes = {
        "popdb": load_regime(a.reeval, "popdb"),
        "cv2": load_regime(a.reeval, "clinvar_2star_B_vs_bg"),
        "cvexp": load_regime(a.reeval, "clinvar_expert_B_vs_bg"),
        "cv2pop": load_regime(a.reeval, "clinvar_2star_B_vs_popdbbg"),
        "cvexppop": load_regime(a.reeval, "clinvar_expert_B_vs_popdbbg"),
    }
    table_cv(a.train, a.tables)
    table_alpha(a.train, a.tables)
    table_hyper(a.train, a.tables)
    fig_alpha(a.train, a.figures)
    for key, reg in regimes.items():
        if reg is None:
            print("missing regime:", key)
            continue
        table_pairwise(reg, a.tables, f"tab_pairwise_{key}.tex")
        table_tranches(reg, a.tables, f"tab_tranches_{key}.tex")
        table_pbest(reg, a.tables, f"tab_pbest_{key}.tex")
        table_threshold(reg, a.tables, f"tab_threshold_{key}.tex")
    if regimes["popdb"] is not None:
        table_main(regimes["popdb"], a.tables, "tab_main_popdb.tex", "Background kept")
    if regimes["cv2"] is not None:
        table_main(regimes["cv2"], a.tables, "tab_main_cv2.tex", "Background kept")
    if regimes["cvexp"] is not None:
        table_main(regimes["cvexp"], a.tables, "tab_main_cvexp.tex", "Background kept")
    if regimes["cv2pop"] is not None:
        table_main(regimes["cv2pop"], a.tables, "tab_main_cv2pop.tex", "Background kept")
    for key, ttl in [("popdb", "PopDB-supported variants vs. background"),
                     ("cvexp", "ClinVar expert-panel variants vs. background"),
                     ("cv2", "ClinVar variants (≥2 review stars) vs. background")]:
        if regimes[key] is not None:
            table_auc_simple(regimes[key], a.tables, f"tab_auc_{key}.tex")
            table_tranche_selected(regimes[key], a.tables, f"tab_trsel_{key}.tex")
            fig_heatmap_single(regimes[key], ttl, a.figures, f"fig_heatmap_{key}")
    regs = [("(a) Population-catalogue concordance", regimes["popdb"]),
            ("(b) ClinVar curated variants (≥2 stars)", regimes["cv2"])]
    fig_auc_dots(regs, a.figures)
    fig_delta_heatmaps(regs, a.figures)
    if a.clinvar_records and regimes["popdb"] is not None:
        table_plp(a.clinvar_records, regimes["popdb"][2], a.tables)


if __name__ == "__main__":
    main()
