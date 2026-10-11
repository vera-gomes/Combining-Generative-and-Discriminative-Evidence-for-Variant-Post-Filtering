#!/usr/bin/env python3
"""
reeval_external.py — re-evaluation of the NA06985 apply-mode scores for the
Frontiers revision. No model is retrained; only the external-validation
statistics are recomputed from the per-variant scores.

What it fixes relative to try_live28_corrected.py (apply mode):
  1. ClinVar positives were "any expert-reviewed ClinVar record" (P/LP *and*
     B/LB). Here P/LP and B/LB are separated, and several contrasts are run
     (P vs background, B vs background, P vs B, and the original P+B pooling
     for reproducibility).
  2. The pipeline's bootstrap subsampled to 30,000 variants, which left ~2
     ClinVar positives per replicate. Here the bootstrap keeps every variant:
     paired, stratified weight bootstrap (Poisson(1) weights = standard
     bootstrap; Exp(1) weights = Bayesian bootstrap), computed exactly with
     cumulative-sum tricks so it runs on all ~5 M variants.
  3. Duplicate models (hybrids that selected alpha = 1.0 and are numerically
     identical to their base model) are detected and dropped, so Holm's
     correction runs over the distinct model pairs only.
  4. DeLong tests are exact (full structural components, both classes).

Input: the apply-mode output file scored_variants_with_external_validation.csv
       (CHROM, POS, REF, ALT, IsExternalVariant, <Model>_score ...), or
       real_data_scores.csv (then PopDB contrasts are skipped).

Usage:
  python3 reeval_external.py \
      --scores output/apply/scored_variants_with_external_validation.csv \
      --clinvar /path/to/clinvar.vcf.gz \
      --out reeval_out --n-boot 2000

  Quick test first:  add  --n-boot 50

Outputs (all small CSV/JSON):
  run_info.json                      counts, ClinVar file date, model identity checks
  clinvar_records_in_callset.csv     every ClinVar P/LP or B/LB record matched in the callset, with scores
  <contrast>_summary.csv             per-model AUC/AP/F1/LogLoss + CIs + P(best)
  <contrast>_pairwise.csv            all distinct pairs: DeLong + Holm, bootstrap CIs of deltas
  <contrast>_tranches.csv            operating points at target sensitivities
"""

import argparse
import itertools
import json
import logging
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    import pysam
except ImportError:  # only needed for ClinVar
    pysam = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("reeval")

# Order decides which copy is kept when two score columns are identical.
PREFERRED_ORDER = [
    "GM", "BGM", "LogReg", "RandForest", "LGB", "LGB_Bayes", "LGB_MultiObj",
    "Hybrid_LGB_alpha08", "Hybrid_LGB_alpha07",
    "Hybrid_LGB", "Hybrid_LGB_Bayes", "Hybrid_LGB_MultiObj", "Hybrid_LGB_alpha1",
]
EPS = 1e-15
TRANCHE_SENS = [1.0, 0.999, 0.99, 0.90]
MIN_CLASS_N = 10


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_scores(path, popdb_col):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    score_cols = [c for c in header if c.endswith("_score")]
    if not score_cols:
        sys.exit(f"No *_score columns in {path}")
    use = ["CHROM", "POS", "REF", "ALT"] + score_cols
    has_pop = popdb_col in header
    if has_pop:
        use.append(popdb_col)
    dtypes = {c: "float64" for c in score_cols}
    dtypes.update({"CHROM": "str", "REF": "str", "ALT": "str"})
    log.info(f"Reading {path} ({len(score_cols)} score columns; PopDB column present: {has_pop})")
    df = pd.read_csv(path, usecols=use, dtype=dtypes)
    df["CHROM"] = df["CHROM"].str.replace("^chr", "", regex=True)
    df["POS"] = pd.to_numeric(df["POS"], errors="coerce").astype("Int64")
    log.info(f"Loaded {len(df):,} variants")
    return df, score_cols, has_pop


def dedupe_models(df, score_cols):
    names = [c[:-len("_score")] for c in score_cols]
    order = [n for n in PREFERRED_ORDER if n in names] + [n for n in names if n not in PREFERRED_ORDER]
    kept, identical_to = [], {}
    for n in order:
        s = df[f"{n}_score"].to_numpy()
        dup = None
        for k in kept:
            if np.array_equal(s, df[f"{k}_score"].to_numpy(), equal_nan=True):
                dup = k
                break
        if dup is None:
            kept.append(n)
        else:
            identical_to[n] = dup
    log.info(f"Distinct models ({len(kept)}): {kept}")
    for n, k in identical_to.items():
        log.info(f"  {n} is numerically identical to {k} -> dropped")
    return kept, identical_to


_P_TOK = {"pathogenic", "likely_pathogenic"}
_B_TOK = {"benign", "likely_benign"}
_BLOCK_TOK = {"uncertain_significance", "conflicting_classifications_of_pathogenicity",
              "conflicting_interpretations_of_pathogenicity", "not_provided"}


def _as_str(v):
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ",".join(map(str, v))
    return str(v)


def classify_strict(clnsig):
    toks = {t.strip("_ ").lower() for t in re.split(r"[/,|]", clnsig) if t.strip("_ ")}
    has_p, has_b = bool(toks & _P_TOK), bool(toks & _B_TOK)
    if toks & _BLOCK_TOK:
        return None
    if has_p and not has_b:
        return "P"
    if has_b and not has_p:
        return "B"
    return None


def classify_legacy(clnsig_upper):
    """Exactly the rule in try_live28_corrected.load_clinvar_dict."""
    is_p = any(x in clnsig_upper for x in ["PATHOGENIC", "LIKELY_PATHOGENIC"])
    is_b = any(x in clnsig_upper for x in ["BENIGN", "LIKELY_BENIGN"])
    if is_p and not is_b:
        return "P"
    if is_b and not is_p:
        return "B"
    return None


def review_tier(clnrev_lower):
    if "practice_guideline" in clnrev_lower or "reviewed_by_expert_panel" in clnrev_lower:
        return "expert"          # 3-4 stars (the tier used in the original run)
    if "multiple_submitters" in clnrev_lower and "no_conflicts" in clnrev_lower:
        return "multi"           # 2 stars
    return None


def load_clinvar(path):
    if pysam is None:
        sys.exit("pysam is required to read the ClinVar VCF")
    log.info(f"Reading ClinVar VCF {path} (takes a few minutes)")
    vcf = pysam.VariantFile(path)
    file_date = None
    for rec in vcf.header.records:
        if rec.key == "fileDate":
            file_date = rec.value
    rows = []
    for rec in vcf:
        if not rec.alts:
            continue
        clnsig = _as_str(rec.info.get("CLNSIG"))
        clnrev = _as_str(rec.info.get("CLNREVSTAT"))
        if not clnsig:
            continue
        tier = review_tier(clnrev.lower())
        if tier is None:
            continue
        strict = classify_strict(clnsig)
        legacy = classify_legacy(clnsig.upper())
        if strict is None and legacy is None:
            continue
        gene = _as_str(rec.info.get("GENEINFO"))
        vc = _as_str(rec.info.get("CLNVC"))
        c = str(rec.chrom).replace("chr", "")
        for alt in rec.alts:
            rows.append((c, int(rec.pos), str(rec.ref), str(alt), clnsig, clnrev, tier,
                         strict, legacy, gene, vc, rec.id))
    cv = pd.DataFrame(rows, columns=["CHROM", "POS", "REF", "ALT", "CLNSIG", "CLNREVSTAT", "Tier",
                                     "Class", "LegacyClass", "GENEINFO", "CLNVC", "ClinVarID"])
    cv["POS"] = cv["POS"].astype("Int64")
    cv = cv.drop_duplicates(subset=["CHROM", "POS", "REF", "ALT"])
    log.info(f"ClinVar records kept (>=2 stars, P/B): {len(cv):,}; fileDate={file_date}")
    return cv, file_date


# ----------------------------------------------------------------------------
# Exact AUC / AP / F1 / LogLoss under per-variant weights
# ----------------------------------------------------------------------------

class ModelPrep:
    """Pre-sorted structures for one model on one contrast."""

    def __init__(self, x_pos, y_neg):
        self.x = x_pos
        self.y = y_neg
        self.neg_order = np.argsort(y_neg, kind="mergesort").astype(np.int32)
        ys = y_neg[self.neg_order]
        self.nL = np.searchsorted(ys, x_pos, "left").astype(np.int32)    # negatives  < x_i
        self.nR = np.searchsorted(ys, x_pos, "right").astype(np.int32)   # negatives <= x_i
        self.pos_order = np.argsort(x_pos, kind="mergesort").astype(np.int32)
        xs = x_pos[self.pos_order]
        self.pL = np.searchsorted(xs, x_pos, "left").astype(np.int32)    # positives  < x_i
        self.pos_hit = (x_pos >= 0.5)
        self.neg_hit = (y_neg >= 0.5)
        xp = np.clip(x_pos, EPS, 1 - EPS)
        yn = np.clip(y_neg, EPS, 1 - EPS)
        self.loss_pos = -np.log(xp)
        self.loss_neg = -np.log(1 - yn)

    def metrics(self, w, v):
        """w: weights of positives, v: weights of negatives (both >= 0)."""
        W, V = w.sum(), v.sum()
        cv = np.concatenate(([0.0], np.cumsum(v[self.neg_order])))
        below = cv[self.nL] + 0.5 * (cv[self.nR] - cv[self.nL])
        auc = float(np.dot(w, below) / (W * V))
        cw = np.concatenate(([0.0], np.cumsum(w[self.pos_order])))
        pos_ge = W - cw[self.pL]
        neg_ge = V - cv[self.nL]
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = pos_ge / (pos_ge + neg_ge)
        ap = float(np.dot(w, np.nan_to_num(prec)) / W)
        tp = float(np.dot(w, self.pos_hit))
        fp = float(np.dot(v, self.neg_hit))
        fn = W - tp
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        ll = float((np.dot(w, self.loss_pos) + np.dot(v, self.loss_neg)) / (W + V))
        return auc, ap, f1, ll


def delong(preps):
    """Exact DeLong covariance for k correlated AUCs."""
    k = len(preps)
    m, n = len(preps[0].x), len(preps[0].y)
    v10 = np.empty((m, k))
    v01 = np.empty((n, k))
    aucs = np.empty(k)
    for s, p in enumerate(preps):
        v10[:, s] = (p.nL + 0.5 * (p.nR - p.nL)) / n
        xs = p.x[p.pos_order]
        gL = np.searchsorted(xs, p.y, "left")
        gR = np.searchsorted(xs, p.y, "right")
        v01[:, s] = ((m - gR) + 0.5 * (gR - gL)) / m       # P(x > y_j) + 0.5 P(x == y_j)
        aucs[s] = v10[:, s].mean()
    S = np.cov(v10, rowvar=False) / m + np.cov(v01, rowvar=False) / n
    return aucs, np.atleast_2d(S)


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    M = len(order)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (M - r) * p[i]))
        out[i] = run
    return out


def threshold_at_sens(x_pos, target):
    s = np.sort(x_pos)[::-1]
    k = max(0, min(int(np.ceil(target * len(s))) - 1, len(s) - 1))
    return float(s[k])


# ----------------------------------------------------------------------------
# One contrast
# ----------------------------------------------------------------------------

def run_contrast(name, df, pos_mask, neg_mask, models, args, out):
    m, n = int(pos_mask.sum()), int(neg_mask.sum())
    log.info(f"[{name}] positives={m:,} negatives={n:,}")
    info = {"contrast": name, "n_pos": m, "n_neg": n}
    if min(m, n) < MIN_CLASS_N:
        log.warning(f"[{name}] skipped: fewer than {MIN_CLASS_N} in a class")
        info["skipped"] = True
        return info

    preps = [ModelPrep(df.loc[pos_mask, f"{mm}_score"].to_numpy(),
                       df.loc[neg_mask, f"{mm}_score"].to_numpy()) for mm in models]
    k = len(models)

    # point estimates + sanity check against sklearn on the full data
    point = np.array([p.metrics(np.ones(m), np.ones(n)) for p in preps])   # k x 4
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        yv = np.r_[np.ones(m), np.zeros(n)]
        for i, p in enumerate(preps[:2]):
            sv = np.r_[p.x, p.y]
            a, b = roc_auc_score(yv, sv), average_precision_score(yv, sv)
            if abs(a - point[i, 0]) > 1e-9 or abs(b - point[i, 1]) > 1e-9:
                log.error(f"[{name}] self-check FAILED for {models[i]}: {a} vs {point[i,0]}, {b} vs {point[i,1]}")
                info["self_check"] = "FAILED"
        info.setdefault("self_check", "ok")
    except Exception as e:  # sklearn missing is fine
        info["self_check"] = f"not run ({e})"

    aucs, S = delong(preps)

    # bootstrap (paired: same weights for every model in a replicate)
    rng = np.random.default_rng(args.seed)
    schemes = {"bootstrap": lambda size: rng.poisson(1.0, size).astype(float),
               "bayesboot": lambda size: rng.exponential(1.0, size)}
    draws = {}
    for sch, gen in schemes.items():
        arr = np.empty((args.n_boot, k, 4))
        t0 = time.time()
        for b in range(args.n_boot):
            w, v = gen(m), gen(n)
            if w.sum() == 0 or v.sum() == 0:
                arr[b] = np.nan
                continue
            for i, p in enumerate(preps):
                arr[b, i] = p.metrics(w, v)
            if (b + 1) % max(1, args.n_boot // 10) == 0:
                log.info(f"[{name}] {sch} {b+1}/{args.n_boot} ({time.time()-t0:.0f}s)")
        draws[sch] = arr

    metric_names = ["AUC", "AP", "F1_at_0.5", "LogLoss"]
    rows = []
    for i, mm in enumerate(models):
        r = {"Model": mm, "N_pos": m, "N_neg": n,
             "AUC": point[i, 0], "AUC_DeLong_SE": float(np.sqrt(S[i, i])),
             "AUC_DeLong_CI_low": point[i, 0] - 1.96 * np.sqrt(S[i, i]),
             "AUC_DeLong_CI_high": point[i, 0] + 1.96 * np.sqrt(S[i, i])}
        for j, mn in enumerate(metric_names):
            r[mn] = point[i, j]
            for sch, arr in draws.items():
                lo, hi = np.nanpercentile(arr[:, i, j], [2.5, 97.5])
                r[f"{mn}_{sch}_CI_low"], r[f"{mn}_{sch}_CI_high"] = lo, hi
        r["Median_pos_score_percentile"] = float(
            pd.Series(np.r_[preps[i].x, preps[i].y]).rank(pct=True).iloc[:m].median())
        rows.append(r)
    summ = pd.DataFrame(rows)
    for sch, arr in draws.items():
        for j, mn in enumerate(metric_names):
            vals = arr[:, :, j]
            ok = np.all(np.isfinite(vals), axis=1)
            best = (np.argmin if mn == "LogLoss" else np.argmax)(vals[ok], axis=1)
            summ[f"Pbest_{mn}_{sch}"] = np.bincount(best, minlength=k) / max(ok.sum(), 1)
    summ.to_csv(os.path.join(out, f"{name}_summary.csv"), index=False)

    # pairwise
    prs = []
    for a, b in itertools.combinations(range(k), 2):
        d = aucs[a] - aucs[b]
        var = S[a, a] + S[b, b] - 2 * S[a, b]
        se = float(np.sqrt(var)) if var > 0 else np.nan
        z = d / se if se and np.isfinite(se) else np.nan
        p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
        r = {"Model_A": models[a], "Model_B": models[b], "Delta_AUC": d,
             "DeLong_SE": se, "DeLong_z": z, "DeLong_p": p}
        for sch, arr in draws.items():
            for j, mn in enumerate(metric_names):
                dd = arr[:, a, j] - arr[:, b, j]
                r[f"Delta_{mn}"] = point[a, j] - point[b, j]
                r[f"Delta_{mn}_{sch}_CI_low"], r[f"Delta_{mn}_{sch}_CI_high"] = np.nanpercentile(dd, [2.5, 97.5])
                r[f"P_Delta_{mn}_gt0_{sch}"] = float(np.nanmean(dd > 0))
        prs.append(r)
    pw = pd.DataFrame(prs)
    pw["DeLong_p_Holm"] = holm(pw["DeLong_p"])
    # Holm-adjusted simultaneous CI for Delta AUC (step-down approximated by Bonferroni width)
    zc = norm.isf(0.025 / len(pw))
    pw["Delta_AUC_Bonferroni_CI_low"] = pw["Delta_AUC"] - zc * pw["DeLong_SE"]
    pw["Delta_AUC_Bonferroni_CI_high"] = pw["Delta_AUC"] + zc * pw["DeLong_SE"]
    pw.to_csv(os.path.join(out, f"{name}_pairwise.csv"), index=False)

    # tranches
    tr = []
    N = m + n
    for i, mm in enumerate(models):
        for s in TRANCHE_SENS:
            thr = threshold_at_sens(preps[i].x, s)
            tp = int((preps[i].x >= thr).sum())
            fp = int((preps[i].y >= thr).sum())
            prec = tp / (tp + fp) if tp + fp else np.nan
            rec = tp / m
            tr.append({"Model": mm, "Target_sensitivity": s, "Threshold": thr,
                       "TP": tp, "FP": fp, "Precision": prec, "Recall": rec,
                       "F1": 2 * prec * rec / (prec + rec) if prec and rec else np.nan,
                       "Retained_fraction": (tp + fp) / N})
    pd.DataFrame(tr).to_csv(os.path.join(out, f"{name}_tranches.csv"), index=False)
    log.info(f"[{name}] done")
    return info


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--clinvar", help="ClinVar VCF (local path preferred; URL works with pysam)")
    ap.add_argument("--out", default="reeval_out")
    ap.add_argument("--popdb-col", default="IsExternalVariant")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--exclude-models", nargs="*", default=[],
                    help="e.g. Hybrid_LGB_alpha07 to reproduce the 8-model set")
    ap.add_argument("--skip-popdb", action="store_true")
    ap.add_argument("--only", nargs="*", default=[],
                    help="run only these contrasts, e.g. popdb clinvar_2star_B_vs_bg")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, score_cols, has_pop = load_scores(args.scores, args.popdb_col)
    models, identical = dedupe_models(df, score_cols)
    models = [mm for mm in models if mm not in set(args.exclude_models)]
    run_info = {"n_variants": int(len(df)), "models": models, "identical_models": identical,
                "n_boot": args.n_boot, "seed": args.seed, "contrasts": []}

    nan_rows = df[[f"{mm}_score" for mm in models]].isna().any(axis=1)
    if nan_rows.any():
        log.warning(f"Dropping {int(nan_rows.sum())} variants with a missing score")
        df = df.loc[~nan_rows].reset_index(drop=True)

    # ---------------- ClinVar ----------------
    if args.clinvar:
        cv, file_date = load_clinvar(args.clinvar)
        run_info["clinvar_fileDate"] = file_date
        merged = df[["CHROM", "POS", "REF", "ALT"]].reset_index().merge(
            cv, on=["CHROM", "POS", "REF", "ALT"], how="inner")
        idx = merged["index"].to_numpy()
        detail = merged.drop(columns="index").copy()
        for mm in models:
            detail[f"{mm}_score"] = df.loc[idx, f"{mm}_score"].to_numpy()
        if has_pop:
            detail["PopDB_supported"] = df.loc[idx, args.popdb_col].to_numpy()
        detail.to_csv(os.path.join(args.out, "clinvar_records_in_callset.csv"), index=False)

        counts = merged.groupby(["Tier", "Class"], dropna=False).size().reset_index(name="n")
        counts["Class"] = counts["Class"].fillna("unclassified_strict")
        run_info["clinvar_counts_by_tier_class"] = counts.to_dict(orient="records")
        legacy_expert = merged[(merged["Tier"] == "expert") & merged["LegacyClass"].notna()]
        run_info["clinvar_legacy_expert_PB_n"] = int(len(legacy_expert))
        log.info(f"Legacy rule, expert tier (should reproduce the original 363): {len(legacy_expert)}")

        N = len(df)

        def mask_of(rows):
            mk = np.zeros(N, dtype=bool)
            mk[rows["index"].to_numpy()] = True
            return mk

        any_cv = mask_of(merged)
        contrasts = []
        # reproduction of the original analysis
        contrasts.append(("clinvar_legacy_expertPB_vs_bg", mask_of(legacy_expert), ~mask_of(legacy_expert)))
        for tier_name, tiers in [("expert", ["expert"]), ("2star", ["expert", "multi"])]:
            sub = merged[merged["Tier"].isin(tiers)]
            P = mask_of(sub[sub["Class"] == "P"])
            B = mask_of(sub[sub["Class"] == "B"])
            contrasts += [
                (f"clinvar_{tier_name}_P_vs_bg", P, ~any_cv),
                (f"clinvar_{tier_name}_B_vs_bg", B, ~any_cv),
                (f"clinvar_{tier_name}_P_vs_B", P, B),
            ]
        # Control for the population-database confound: every expert-reviewed
        # ClinVar record found in this callset is also PopDB-supported, so compare
        # ClinVar benign variants against PopDB-supported non-ClinVar variants only.
        if has_pop:
            popdb = pd.to_numeric(df[args.popdb_col], errors="coerce").fillna(0).astype(int).to_numpy() == 1
            for tier_name, tiers in [("expert", ["expert"]), ("2star", ["expert", "multi"])]:
                sub = merged[merged["Tier"].isin(tiers)]
                B = mask_of(sub[sub["Class"] == "B"])
                contrasts.append((f"clinvar_{tier_name}_B_vs_popdbbg", B, popdb & ~any_cv))

        if args.only:
            contrasts = [c for c in contrasts if c[0] in set(args.only)]
        for name, pm, nm in contrasts:
            if min(int(pm.sum()), int(nm.sum())) < MIN_CLASS_N:
                log.info(f"[{name}] skipped: positives={int(pm.sum())}, negatives={int(nm.sum())}")
                run_info["contrasts"].append({"contrast": name, "n_pos": int(pm.sum()),
                                              "n_neg": int(nm.sum()), "skipped": True})
                continue
            run_info["contrasts"].append(run_contrast(name, df, pm, nm, models, args, args.out))

    # ---------------- PopDB ----------------
    if has_pop and not args.skip_popdb and (not args.only or "popdb" in args.only):
        y = pd.to_numeric(df[args.popdb_col], errors="coerce").fillna(0).astype(int).to_numpy() == 1
        run_info["contrasts"].append(run_contrast("popdb", df, y, ~y, models, args, args.out))
    elif not has_pop:
        log.warning("No PopDB label column; PopDB contrast skipped "
                    "(use scored_variants_with_external_validation.csv)")

    with open(os.path.join(args.out, "run_info.json"), "w") as f:
        json.dump(run_info, f, indent=2, default=str)
    log.info(f"All done. Send back the folder: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
