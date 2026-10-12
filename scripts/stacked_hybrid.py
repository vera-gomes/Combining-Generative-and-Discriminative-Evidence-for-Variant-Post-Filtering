#!/usr/bin/env python3
"""
stacked_hybrid.py — feature-level hybrid ("stacked hybrid"), suggested by
Reviewer 3: the Gaussian-mixture log-likelihood ratio (VQSLOD) is given to
LightGBM as a ninth input feature, instead of blending the two scores with a
fixed weight. LightGBM can then learn where the generative evidence helps.

Two models are defined in advance and both are reported:
  Stack_LGB        default LightGBM settings (as LGB) + GM feature
  Stack_LGB_Bayes  the hyperparameters selected for LGB_Bayes (not re-tuned) + GM feature

Leakage control. The GM feature of a training call must not come from mixtures
fitted on that same call, or LightGBM would over-trust it. Training values are
therefore cross-fitted: within the training data, 1 Mb genomic blocks are split
into 3 folds and each call gets the VQSLOD of mixtures fitted on the other two.
Held-out and NA06985 calls get the VQSLOD of mixtures fitted on all training
calls. Imputation and scaling are fitted on training calls only.

Steps (all in one run):
  1. Benchmark CV on NA12878: 5-fold block CV (as in the pipeline); in every
     fold LGB, LGB_Bayes, Stack_LGB and Stack_LGB_Bayes are fitted on the same
     split, so the comparison is paired.
  2. Final fit on all NA12878 calls.
  3. Apply to NA06985: annotations are re-extracted from the VCF, scored, and
     the two new score columns are appended to a copy of the apply-mode scores
     file, which can then go through scripts/reeval_external.py unchanged.

  python scripts/stacked_hybrid.py \
      --annotations ~/Documents/deep_v2_train_20260919_193843/annotations_labeled.csv \
      --tuning-json ~/Documents/deep_v2_train_20260919_193843/lightgbm_tuning_summary.json \
      --apply-vcf ~/Documents/NA06985.grch38.g.vcf.gz \
      --scores-in ~/Documents/deep_revision/scored_variants_with_external_validation.csv \
      --out results/stacked_hybrid

  Quick check first: add --max-blocks 300 --skip-apply
"""
import argparse
import json
import logging
import os
import time

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.mixture import GaussianMixture

from postfilter_pipeline.data_prep import build_genomic_groups, extract_annotations, make_block_splitter
from postfilter_pipeline.models import LeakageAwareImputer, LeakageAwareScaler, compute_vqslod
from postfilter_pipeline.stats import delong_roc_test

FEATURES = ["MQ", "QD", "FS", "SOR", "MQRankSum", "ReadPosRankSum", "BaseQRankSum", "DP"]
LGB_DEFAULT = dict(objective="binary", boosting_type="gbdt", n_estimators=100, learning_rate=0.05,
                   max_depth=7, num_leaves=31, min_child_samples=20, subsample=0.8,
                   colsample_bytree=0.8, reg_alpha=0.0, reg_lambda=0.0, verbose=-1, force_row_wise=True)
SEED = 42
log = logging.getLogger("stacked")


def fit_gm_pair(Xs, y, seed):
    kw = dict(n_components=5, max_iter=500, random_state=seed)
    return GaussianMixture(**kw).fit(Xs[y == 1]), GaussianMixture(**kw).fit(Xs[y == 0])


def gm_features(X_tr, y_tr, g_tr, X_eval_list, seed):
    """Cross-fitted VQSLOD for the training rows, full-fit VQSLOD for each eval matrix.
    Imputer and scaler are fitted on the training rows only."""
    imp = LeakageAwareImputer().fit(X_tr)
    Xi = imp.transform(X_tr)
    sc = LeakageAwareScaler().fit(Xi)
    Xs = sc.transform(Xi)
    cross = np.full(len(y_tr), np.nan)
    inner, _ = make_block_splitter(3, random_state=seed)
    for k, (a, b) in enumerate(inner.split(Xs, y_tr, g_tr)):
        good, bad = fit_gm_pair(Xs[a], y_tr[a], seed + k)
        cross[b] = compute_vqslod(good, bad, Xs[b])
    good, bad = fit_gm_pair(Xs, y_tr, seed)
    evals = [compute_vqslod(good, bad, sc.transform(imp.transform(Xe))) for Xe in X_eval_list]
    return Xi, cross, evals, (imp, sc, good, bad)


def with_gm(X_imp, v):
    return np.column_stack([X_imp, v])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--tuning-json", required=True, help="lightgbm_tuning_summary.json of the train run")
    ap.add_argument("--apply-vcf")
    ap.add_argument("--scores-in", help="apply-mode scores CSV to which the new columns are appended")
    ap.add_argument("--out", default="results/stacked_hybrid")
    ap.add_argument("--max-blocks", type=int, default=0)
    ap.add_argument("--skip-apply", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(a.out, "stacked.log"))])

    bayes = json.load(open(a.tuning_json))["LGB_Bayes"]["best_params"]
    bayes = {k: v for k, v in bayes.items() if k not in ("n_jobs", "random_state")}
    models = {"LGB": (LGB_DEFAULT, False), "LGB_Bayes": (bayes, False),
              "Stack_LGB": (LGB_DEFAULT, True), "Stack_LGB_Bayes": (bayes, True)}

    d = pd.read_csv(a.annotations, usecols=lambda c: c in FEATURES + ["CHROM", "POS", "TruthLabel"])
    groups = build_genomic_groups(d, block_size_bp=1_000_000)
    if a.max_blocks:
        keep = np.random.default_rng(0).choice(np.unique(groups), size=a.max_blocks, replace=False)
        m = np.isin(groups, keep)
        d, groups = d[m].reset_index(drop=True), groups[m]
    X, y = d[FEATURES].to_numpy(float), d["TruthLabel"].to_numpy(int)
    log.info(f"benchmark: {len(y):,} calls, {y.mean():.3f} true, {len(np.unique(groups)):,} blocks")

    # ---------------- 1. paired benchmark CV ----------------
    oof = {m: np.full(len(y), np.nan) for m in models}
    oof["GM"] = np.full(len(y), np.nan)
    rows = []
    outer, _ = make_block_splitter(5, random_state=SEED)
    for fold, (tr, va) in enumerate(outer.split(X, y, groups)):
        t0 = time.time()
        Xi_tr, gm_tr, (gm_va,), (imp, _, _, _) = gm_features(X[tr], y[tr], groups[tr], [X[va]], SEED + fold)
        Xi_va = imp.transform(X[va])
        oof["GM"][va] = gm_va
        for name, (params, stacked) in models.items():
            Xa, Xb = (with_gm(Xi_tr, gm_tr), with_gm(Xi_va, gm_va)) if stacked else (Xi_tr, Xi_va)
            clf = LGBMClassifier(**params, random_state=SEED + fold, n_jobs=a.n_jobs).fit(Xa, y[tr])
            p = clf.predict_proba(Xb)[:, 1]
            oof[name][va] = p
            rows.append((fold, name, roc_auc_score(y[va], p), average_precision_score(y[va], p)))
        log.info(f"fold {fold}: " + "  ".join(f"{r[1]}={r[2]:.4f}" for r in rows[-len(models):])
                 + f"  ({time.time() - t0:.0f}s)")
    per_fold = pd.DataFrame(rows, columns=["fold", "model", "AUC", "AP"])
    per_fold.to_csv(os.path.join(a.out, "stacked_cv_per_fold.csv"), index=False)
    summ = per_fold.groupby("model")[["AUC", "AP"]].agg(["mean", "std"])
    summ.to_csv(os.path.join(a.out, "stacked_cv_summary.csv"))
    pd.DataFrame({"y": y, **oof}).to_csv(os.path.join(a.out, "stacked_cv_oof.csv.gz"), index=False)
    tests = []
    for s, b in [("Stack_LGB", "LGB"), ("Stack_LGB_Bayes", "LGB_Bayes"), ("Stack_LGB_Bayes", "LGB")]:
        r = delong_roc_test(y, oof[s], oof[b])
        wins = int((per_fold[per_fold.model == s].AUC.values > per_fold[per_fold.model == b].AUC.values).sum())
        tests.append((s, b, r["auc1"], r["auc2"], r["delta"], r["p_value"], wins))
    tests = pd.DataFrame(tests, columns=["model_A", "model_B", "AUC_A_oof", "AUC_B_oof", "delta", "DeLong_p", "folds_A_better"])
    tests.to_csv(os.path.join(a.out, "stacked_cv_tests.csv"), index=False)
    log.info("\n" + summ.round(4).to_string() + "\n" + tests.round(5).to_string())

    # ---------------- 2. final fit on all benchmark calls ----------------
    log.info("final fit on all benchmark calls")
    Xi_all, gm_all, _, (imp, sc, good, bad) = gm_features(X, y, groups, [], SEED)
    final = {}
    for name in ("Stack_LGB", "Stack_LGB_Bayes"):
        params, _ = models[name]
        final[name] = LGBMClassifier(**params, random_state=SEED, n_jobs=a.n_jobs).fit(with_gm(Xi_all, gm_all), y)
        gain = final[name].booster_.feature_importance(importance_type="gain")
        imp_df = pd.DataFrame({"feature": FEATURES + ["GM_VQSLOD"], "gain": gain, "gain_share": gain / gain.sum()})
        imp_df.sort_values("gain", ascending=False).to_csv(os.path.join(a.out, f"importance_{name}.csv"), index=False)
        log.info(f"{name}: share of gain from GM_VQSLOD = {imp_df.set_index('feature').gain_share['GM_VQSLOD']:.3f}")
    joblib.dump({"imputer": imp, "scaler": sc, "gm_good": good, "gm_bad": bad, **final},
                os.path.join(a.out, "stacked_models.joblib"))

    # ---------------- 3. apply to NA06985 ----------------
    if a.skip_apply or not a.apply_vcf:
        return
    log.info("extracting NA06985 annotations")
    ann = extract_annotations(a.apply_vcf, FEATURES)
    Xa = ann[FEATURES].to_numpy(float)
    Xa_imp = imp.transform(Xa)
    gm_apply = compute_vqslod(good, bad, sc.transform(Xa_imp))
    new = ann[["CHROM", "POS", "REF", "ALT"]].copy()
    for name in final:
        new[f"{name}_score"] = final[name].predict_proba(with_gm(Xa_imp, gm_apply))[:, 1]
    if a.scores_in:
        base = pd.read_csv(a.scores_in)
        same = (len(base) == len(new)) and all(
            (base[c].astype(str).to_numpy() == new[c].astype(str).to_numpy()).all() for c in ["CHROM", "POS", "REF", "ALT"])
        if same:
            for name in final:
                base[f"{name}_score"] = new[f"{name}_score"].to_numpy()
        else:
            log.warning("row order differs from the scores file; merging on CHROM, POS, REF, ALT")
            base = base.merge(new.drop_duplicates(["CHROM", "POS", "REF", "ALT"]),
                              on=["CHROM", "POS", "REF", "ALT"], how="left")
        out_csv = os.path.join(a.out, "scored_variants_with_stacked.csv")
        base.to_csv(out_csv, index=False)
        log.info(f"wrote {out_csv} ({len(base):,} rows); missing stacked scores: "
                 f"{int(base['Stack_LGB_score'].isna().sum())}")
    else:
        new.to_csv(os.path.join(a.out, "stacked_scores_NA06985.csv"), index=False)


if __name__ == "__main__":
    main()
