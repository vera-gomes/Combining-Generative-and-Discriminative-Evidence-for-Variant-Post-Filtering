#!/usr/bin/env python3
"""
count_clinvar_pathogenic.py — list every ClinVar Pathogenic / Likely pathogenic
record (ANY review status, 0-4 stars) that is present in the NA06985 callset,
with its review level, PopDB support and all model scores.

Usage:
  python3 count_clinvar_pathogenic.py \
      --scores output/apply/scored_variants_with_external_validation.csv \
      --clinvar https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz \
      --out clinvar_PLP_in_callset.csv
"""
import argparse
import re

import pandas as pd
import pysam

P_TOK = {"pathogenic", "likely_pathogenic"}
BLOCK = {"benign", "likely_benign", "uncertain_significance",
         "conflicting_classifications_of_pathogenicity",
         "conflicting_interpretations_of_pathogenicity"}


def s(v):
    return ",".join(map(str, v)) if isinstance(v, (list, tuple)) else ("" if v is None else str(v))


def info(rec, key):
    try:
        return s(rec.info.get(key))
    except ValueError:  # tag not in header
        return ""


def stars(rev):
    r = rev.lower()
    if "practice_guideline" in r:
        return 4
    if "reviewed_by_expert_panel" in r:
        return 3
    if "multiple_submitters" in r and "no_conflicts" in r:
        return 2
    if "criteria_provided" in r and "conflicting" not in r:
        return 1
    return 0


ap = argparse.ArgumentParser()
ap.add_argument("--scores", required=True)
ap.add_argument("--clinvar", required=True)
ap.add_argument("--out", default="clinvar_PLP_in_callset.csv")
a = ap.parse_args()

rows = []
for rec in pysam.VariantFile(a.clinvar):
    if not rec.alts:
        continue
    sig = info(rec, "CLNSIG")
    toks = {t.strip("_ ").lower() for t in re.split(r"[/,|]", sig) if t.strip("_ ")}
    if not (toks & P_TOK) or (toks & BLOCK):
        continue
    rev = info(rec, "CLNREVSTAT")
    for alt in rec.alts:
        rows.append((str(rec.chrom).replace("chr", ""), int(rec.pos), str(rec.ref), str(alt),
                     sig, rev, stars(rev), info(rec, "GENEINFO"), info(rec, "CLNDN"),
                     info(rec, "MC"), rec.id))
cv = pd.DataFrame(rows, columns=["CHROM", "POS", "REF", "ALT", "CLNSIG", "CLNREVSTAT", "Stars",
                                 "GENEINFO", "CLNDN", "MC", "ClinVarID"])
print(f"ClinVar P/LP records (all review levels): {len(cv):,}")

header = pd.read_csv(a.scores, nrows=0).columns.tolist()
use = [c for c in header if c in ("CHROM", "POS", "REF", "ALT", "IsExternalVariant") or c.endswith("_score")]
df = pd.read_csv(a.scores, usecols=use, dtype={"CHROM": str, "REF": str, "ALT": str})
df["CHROM"] = df["CHROM"].str.replace("^chr", "", regex=True)
hit = df.merge(cv, on=["CHROM", "POS", "REF", "ALT"], how="inner")
hit.to_csv(a.out, index=False)

print(f"P/LP records present in the callset: {len(hit)}")
print(hit["Stars"].value_counts().sort_index().rename("n").to_string())
cols = ["CHROM", "POS", "REF", "ALT", "Stars", "GENEINFO", "CLNSIG"] + \
       [c for c in ["IsExternalVariant", "LGB_Bayes_score", "GM_score"] if c in hit.columns]
print(hit[cols].sort_values("Stars", ascending=False).to_string(index=False))
print(f"Saved: {a.out}")
