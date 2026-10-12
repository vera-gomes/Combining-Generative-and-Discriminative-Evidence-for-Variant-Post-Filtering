#!/usr/bin/env python3
"""
make_supp_extra.py — Supplementary Tables S3-S6 of the revised paper:
stacked hybrid on the benchmark (S3) and on NA06985 (S4), in-distribution vs.
after-transfer operating points (S5), and all calls vs. primary chromosomes (S6).

  python3 scripts/make_supp_extra.py --results results --tables tables

Reads results/external_validation (primary chromosomes, main analysis),
results/external_validation_allcalls, and results/stacked_hybrid.
"""
import argparse
import os
import re

import pandas as pd

NM = {"GM": "GM", "BGM": "BGM", "LogReg": "LogReg", "RandForest": "RF", "LGB": "LGB", "LGB_Bayes": "LGB\\_Bayes",
      "LGB_MultiObj": "LGB\\_MultiObj", "Hybrid_LGB_alpha08": "Hybrid\\_0.8", "Hybrid_LGB_alpha07": "Hybrid\\_0.7",
      "Stack_LGB": "Stack\\_LGB", "Stack_LGB_Bayes": "Stack\\_LGB\\_Bayes"}
ORDER = ["GM", "BGM", "LogReg", "RandForest", "LGB", "LGB_Bayes", "LGB_MultiObj", "Hybrid_LGB_alpha08", "Hybrid_LGB_alpha07"]


def m(k):
    return f"\\mdl{{{NM[k]}}}"


def n(x):
    return f"{int(x):,}"


def p(x):
    if x < 1e-300:
        return "$<10^{-300}$"
    if x < 1e-3:
        mant, exp = f"{x:.1e}".split("e")
        return f"${mant}\\times10^{{{int(exp)}}}$"
    return f"{x:.3f}"


def fp(t, mod, s):
    return t[(t.Model == mod) & (t.Target_sensitivity == s)].FP.iloc[0]


def auc(d, reg, mod):
    return pd.read_csv(os.path.join(d, f"{reg}_summary.csv")).set_index("Model").AUC[mod]


def write(path, lines):
    t = "\n".join(lines) + "\n"
    t = re.sub(r"(?<=[ \[,])-0\.", "$-$0.", t)             # math minus sign
    t = t.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}").rstrip() + "}\n"
    open(path, "w").write(t)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--tables", default="tables")
    a = ap.parse_args()
    os.makedirs(a.tables, exist_ok=True)
    P = os.path.join(a.results, "external_validation")
    A = os.path.join(a.results, "external_validation_allcalls")
    S = os.path.join(a.results, "stacked_hybrid")
    R = os.path.join(S, "reeval")

    # S6: all calls vs primary chromosomes
    ta, tp = pd.read_csv(f"{A}/popdb_tranches.csv"), pd.read_csv(f"{P}/popdb_tranches.csv")
    L = ["\\begin{tabular}{@{}lrrrrrrrr@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{\\textbf{PopDB AUC}} & \\multicolumn{2}{c}{\\textbf{Bkg.\\ kept, sens.\\ 0.90}} & \\multicolumn{2}{c}{\\textbf{Bkg.\\ kept, sens.\\ 0.99}} & \\multicolumn{2}{c}{\\textbf{ClinVar $\\geq$2$\\star$ AUC}} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}",
         "\\textbf{Model} & All & Primary & All & Primary & All & Primary & All & Primary \\\\", "\\midrule"]
    for k in ORDER:
        L.append(f"{m(k)} & {auc(A,'popdb',k):.4f} & {auc(P,'popdb',k):.4f} & {n(fp(ta,k,.9))} & {n(fp(tp,k,.9))} & "
                 f"{n(fp(ta,k,.99))} & {n(fp(tp,k,.99))} & {auc(A,'clinvar_2star_B_vs_bg',k):.4f} & {auc(P,'clinvar_2star_B_vs_bg',k):.4f} \\\\")
    write(os.path.join(a.tables, "tab_allcalls.tex"), L + ["\\bottomrule", "\\end{tabular}"])

    # S3: stacked hybrid, benchmark
    pf = pd.read_csv(f"{S}/stacked_cv_per_fold.csv")
    te = pd.read_csv(f"{S}/stacked_cv_tests.csv")
    oof = pd.read_csv(f"{S}/stacked_cv_oof_summary.csv").set_index("Model")
    imp = {k: pd.read_csv(f"{S}/importance_{k}.csv").set_index("feature").gain_share for k in ("Stack_LGB", "Stack_LGB_Bayes")}
    L = ["\\setlength{\\tabcolsep}{3.5pt}", "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
         "\\textbf{Model} & \\textbf{AUC (mean $\\pm$ SD)} & \\textbf{AP (mean $\\pm$ SD)} & \\textbf{Pooled AUC} & \\textbf{$\\Delta$ vs.\\ base} & \\textbf{DeLong $p$} & \\textbf{Folds better} & \\textbf{GM gain share} \\\\",
         "\\midrule"]
    for k, base in [("LGB", None), ("Stack_LGB", "LGB"), ("LGB_Bayes", None), ("Stack_LGB_Bayes", "LGB_Bayes")]:
        g = pf[pf.model == k]
        row = f"{m(k)} & {g.AUC.mean():.4f} $\\pm$ {g.AUC.std():.4f} & {g.AP.mean():.4f} $\\pm$ {g.AP.std():.4f} & {oof.pooled_AUC[k]:.4f}"
        if base:
            t = te[(te.model_A == k) & (te.model_B == base)].iloc[0]
            row += f" & {t.delta:+.5f} & {p(t.DeLong_p)} & {t.folds_A_better} of 5 & {100*imp[k]['GM_VQSLOD']:.1f}\\% \\\\"
        else:
            row += " & -- & -- & -- & -- \\\\"
        L.append(row)
    write(os.path.join(a.tables, "tab_stack_cv.tex"), L + ["\\bottomrule", "\\end{tabular}"])

    # S4: stacked hybrid, NA06985
    sm = pd.read_csv(f"{R}/popdb_summary.csv").set_index("Model")
    tr = pd.read_csv(f"{R}/popdb_tranches.csv")
    pw = pd.read_csv(f"{R}/popdb_pairwise.csv")
    L = ["\\begin{tabular}{@{}lccrrcc@{}}", "\\toprule",
         " & \\multicolumn{2}{c}{\\textbf{PopDB-supported vs.\\ background}} & \\multicolumn{2}{c}{\\textbf{Background kept}} & \\multicolumn{2}{c}{\\textbf{ClinVar benign vs.\\ background, AUC}} \\\\",
         "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}",
         "\\textbf{Model} & \\textbf{AUC} & \\textbf{$\\Delta$ AUC vs.\\ \\mdl{LGB} [95\\% CI]} & \\textbf{Sens.\\ 0.90} & \\textbf{Sens.\\ 0.99} & \\textbf{Expert panel} & \\textbf{$\\geq$2 stars} \\\\",
         "\\midrule"]
    for k in ["LGB", "Hybrid_LGB_alpha08", "Stack_LGB", "LGB_Bayes", "Stack_LGB_Bayes", "LGB_MultiObj"]:
        if k == "LGB":
            d = "--"
        else:
            r = pw[(pw.Model_A == "LGB") & (pw.Model_B == k)].iloc[0]
            d = f"{-r.Delta_AUC:+.4f} [{-r.Delta_AUC_bootstrap_CI_high:+.4f}, {-r.Delta_AUC_bootstrap_CI_low:+.4f}]"
        L.append(f"{m(k)} & {sm.AUC[k]:.4f} & {d} & {n(fp(tr,k,.9))} & {n(fp(tr,k,.99))} & "
                 f"{auc(R,'clinvar_expert_B_vs_bg',k):.4f} & {auc(R,'clinvar_2star_B_vs_bg',k):.4f} \\\\")
    write(os.path.join(a.tables, "tab_stack_ext.tex"), L + ["\\bottomrule", "\\end{tabular}"])

    # S5: in distribution vs after transfer
    L = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
         " & \\multicolumn{3}{c}{\\textbf{NA12878, cross-validated (in distribution)}} & \\multicolumn{3}{c}{\\textbf{NA06985 (after transfer)}} \\\\",
         "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
         "\\textbf{Model} & \\textbf{AUC} & \\textbf{Sens.\\ 0.90} & \\textbf{Sens.\\ 0.99} & \\textbf{AUC} & \\textbf{Sens.\\ 0.90} & \\textbf{Sens.\\ 0.99} \\\\",
         "\\midrule"]
    for k, src, d in [("GM", tp, P), ("LGB", tp, P), ("LGB_Bayes", tp, P), ("Hybrid_LGB_alpha08", tp, P),
                      ("Stack_LGB", tr, R), ("Stack_LGB_Bayes", tr, R)]:
        o = oof.loc[k]
        L.append(f"{m(k)} & {o.pooled_AUC:.4f} & {n(o['false_kept_sens0.90'])} & {n(o['false_kept_sens0.99'])} & "
                 f"{auc(d,'popdb',k):.4f} & {n(fp(src,k,.9))} & {n(fp(src,k,.99))} \\\\")
    L += ["\\midrule", f"Negatives & \\multicolumn{{3}}{{c}}{{{n(oof.N_false.iloc[0])} false calls}} & "
          f"\\multicolumn{{3}}{{c}}{{{n(sm.N_neg.iloc[0])} background calls}} \\\\"]
    write(os.path.join(a.tables, "tab_reversal.tex"), L + ["\\bottomrule", "\\end{tabular}"])


if __name__ == "__main__":
    main()
