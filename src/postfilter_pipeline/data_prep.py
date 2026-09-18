"""VCF annotation extraction, truth-set labelling and genomic-block grouping for CV."""

import gc
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pysam
from sklearn.model_selection import GroupKFold


def extract_annotations(vcf_path: str, desired_keys: List[str], batch_size: int = 100000) -> pd.DataFrame:
    logging.info(f"Reading VCF: {vcf_path}")
    vcf = pysam.VariantFile(vcf_path)
    header_info_keys = set(vcf.header.info)
    usable_keys = [k for k in desired_keys if k in header_info_keys]
    missing_keys = sorted(set(desired_keys) - set(usable_keys))
    if missing_keys:
        logging.warning(f"Skipping INFO tags not in header: {missing_keys}")

    try:
        import psutil
        MEMORY_MONITOR = True
    except ImportError:
        MEMORY_MONITOR = False

    all_batches = []
    batch = []
    for i, rec in enumerate(vcf):
        row: Dict[str, Any] = {
            "CHROM": rec.chrom,
            "POS": rec.pos,
            "REF": str(rec.ref),
            "ALT": str(rec.alts[0]) if rec.alts else "N",
        }
        for key in usable_keys:
            val = rec.info.get(key)
            if val is None:
                row[key] = np.nan
            else:
                if isinstance(val, (list, tuple)) and len(val) == 1:
                    val = val[0]
                try:
                    row[key] = float(val)
                except (TypeError, ValueError):
                    row[key] = np.nan
        for key in missing_keys:
            row[key] = np.nan
        batch.append(row)
        if len(batch) >= batch_size:
            all_batches.append(pd.DataFrame(batch))
            batch = []
            if MEMORY_MONITOR and i % (batch_size * 2) == 0:
                mem_mb = psutil.Process().memory_info().rss / 1024 ** 2
                logging.info(f"Processed {i + 1} variants, memory: {mem_mb:.1f}MB")
            gc.collect()

    if batch:
        all_batches.append(pd.DataFrame(batch))
    df = pd.concat(all_batches, ignore_index=True) if all_batches else pd.DataFrame(
        columns=["CHROM", "POS", "REF", "ALT"] + usable_keys + missing_keys)
    gc.collect()
    logging.info(f"Extracted {len(df)} variants with {len(usable_keys)} features")
    return df


def extract_truth_positions(vcf_path: str) -> set:
    logging.info(f"Extracting truth positions from: {vcf_path}")
    truth_set = set()
    for rec in pysam.VariantFile(vcf_path):
        truth_set.add((rec.chrom.replace("chr", ""), rec.pos))
    logging.info(f"Extracted {len(truth_set)} truth positions")
    return truth_set


def label_variants(df: pd.DataFrame, truth_set: set, label_col: str = "TruthLabel") -> pd.DataFrame:
    """Label each variant as 1 if (chrom_norm, pos) is in truth_set, else 0 — vectorised."""
    df = df.copy()
    chrom_norm = df["CHROM"].astype(str).str.replace("^chr", "", regex=True)
    pos = pd.to_numeric(df["POS"], errors="coerce").fillna(-1).astype(int)
    df[label_col] = [1 if (c, p) in truth_set else 0
                     for c, p in zip(chrom_norm, pos)]
    return df


def label_variants_with_external_set(df: pd.DataFrame, external_variant_set: set,
                                     label_col: str = "IsExternalVariant") -> pd.DataFrame:
    """Label each variant as 1 if (chrom_norm, pos, ref, alt) in external_set — vectorised."""
    df = df.copy()
    chrom_norm = df["CHROM"].astype(str).str.replace("^chr", "", regex=True)
    pos = pd.to_numeric(df["POS"], errors="coerce").fillna(-1).astype(int)
    ref = df["REF"].astype(str)
    alt = df["ALT"].astype(str)
    df[label_col] = [1 if (c, p, r, a) in external_variant_set else 0
                     for c, p, r, a in zip(chrom_norm, pos, ref, alt)]
    n = int(df[label_col].sum())
    logging.info(f"Labelled variants: {n} external, {len(df) - n} novel")
    return df


def build_genomic_groups(df: pd.DataFrame, block_size_bp: int = 1_000_000) -> Optional[np.ndarray]:
    if df is None or df.empty:
        return None
    chrom_norm = df["CHROM"].astype(str).str.replace("^chr", "", regex=True)
    pos = pd.to_numeric(df["POS"], errors="coerce").fillna(-1).astype(int)
    block = pos // int(block_size_bp)
    return (chrom_norm + ":" + block.astype(str)).to_numpy()


def make_block_splitter(n_splits: int, random_state: int = 42):
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state), "StratifiedGroupKFold"
    except ImportError:
        return GroupKFold(n_splits=n_splits), "GroupKFold"


# ============================================================================
# LIGHTGBM HELPERS
# ============================================================================


