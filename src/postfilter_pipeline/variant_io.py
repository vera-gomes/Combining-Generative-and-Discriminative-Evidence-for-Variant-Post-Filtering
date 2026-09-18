"""VCF I/O helpers: remote/local index handling, population DB scanning, ClinVar loading."""

import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import pysam

from .config import PipelineConfig


def _is_remote_path(p: str) -> bool:
    return isinstance(p, str) and p.startswith(("http://", "https://", "ftp://"))


def _find_local_index_for_remote(remote_path: str, search_dirs=None) -> Optional[str]:
    if not _is_remote_path(remote_path):
        return None
    if search_dirs is None:
        search_dirs = [os.getcwd()]
    base = os.path.basename(urlparse(remote_path).path)
    for ext in (".tbi", ".csi"):
        for d in search_dirs:
            cand = os.path.join(d, base + ext)
            if os.path.exists(cand):
                return cand
    return None


def open_variantfile_with_optional_local_index(vcf_path: str, search_dirs=None) -> pysam.VariantFile:
    idx = _find_local_index_for_remote(vcf_path, search_dirs=search_dirs)
    if idx:
        logging.info(f"Using existing local index for remote VCF: {idx}")
        return pysam.VariantFile(vcf_path, index_filename=idx)
    return pysam.VariantFile(vcf_path)


_CHROM_RE = re.compile(r'\.?chr(\d{1,2}|X|Y)\b')


def _parse_chrom_from_popdb_path(path: str) -> Optional[str]:
    """Extract chromosome token from a PopDB file path using regex."""
    if not isinstance(path, str):
        return None
    m = _CHROM_RE.search(path)
    return m.group(1) if m else None


def _choose_db_contig_name(vcf: pysam.VariantFile, chrom_norm: str) -> Optional[str]:
    contigs = set(vcf.header.contigs)
    for cand in (f"chr{chrom_norm}", chrom_norm):
        if cand in contigs:
            return cand
    return None


def _iter_fetch_with_retry(vcf: pysam.VariantFile, vcf_path: str, contig: str,
                           start0: int, end0: int, max_attempts: int = 3):
    for attempt in range(1, max_attempts + 1):
        try:
            yield from vcf.fetch(contig, start0, end0)
            return
        except Exception as e:
            logging.warning(
                f"[PopDB fetch fail] {vcf_path} {contig}:{start0}-{end0} "
                f"(attempt {attempt}/{max_attempts}): {e}"
            )
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 5))


# ============================================================================
# POPULATION DATABASE MANAGER
# ============================================================================


class PopulationDatabaseManager:
    def __init__(self, config: PipelineConfig, search_dirs: Optional[List[str]] = None):
        self.config = config
        self.population_dbs = config._raw_get('evaluation.external_validation.population_dbs', []) or []
        self.chunk_bp = int(config._raw_get('evaluation.external_validation.popdb_chunk_bp', 1_000_000))
        self.fetch_retries = int(config._raw_get('evaluation.external_validation.popdb_fetch_retries', 3))
        self.search_dirs = search_dirs or [os.getcwd(), config._raw_get('paths.output_dir', 'output')]

    def load_population_variants_for_chromosomes(self, df_variants: pd.DataFrame) -> set:
        if df_variants is None or df_variants.empty:
            logging.warning("PopulationDatabaseManager: input variants are empty.")
            return set()
        if not self.population_dbs:
            logging.info("PopulationDatabaseManager: no population DB VCFs configured.")
            return set()

        df = df_variants.copy()
        df["CHROM_NORM"] = df["CHROM"].astype(str).str.replace("^chr", "", regex=True)
        df["POS"] = pd.to_numeric(df["POS"], errors="coerce").astype("Int64")

        # Vectorised key set
        valid = df[["CHROM_NORM", "POS", "REF", "ALT"]].dropna()
        sample_keys = set(
            zip(valid["CHROM_NORM"].astype(str),
                valid["POS"].astype(int),
                valid["REF"].astype(str),
                valid["ALT"].astype(str))
        )
        if not sample_keys:
            logging.warning("PopulationDatabaseManager: no valid sample keys after normalisation.")
            return set()

        needed_chroms = {k[0] for k in sample_keys} & {str(i) for i in range(1, 23)} | {"X", "Y"}
        max_pos_by_chrom = (
            df.dropna(subset=["CHROM_NORM", "POS"])
            .groupby("CHROM_NORM")["POS"].max().to_dict()
        )

        db_paths = [
            p for p in self.population_dbs
            if (ct := _parse_chrom_from_popdb_path(p)) is None or ct in needed_chroms
        ]
        if not db_paths:
            logging.info("PopulationDatabaseManager: no matching PopDB paths to query.")
            return set()

        found: set = set()
        for vcf_path in db_paths:
            chrom_token = _parse_chrom_from_popdb_path(vcf_path)
            if chrom_token is None or chrom_token not in max_pos_by_chrom:
                continue
            max_pos = int(max_pos_by_chrom[chrom_token])
            if max_pos <= 0:
                continue
            try:
                vcf = open_variantfile_with_optional_local_index(vcf_path, search_dirs=self.search_dirs)
            except Exception as e:
                logging.warning(f"Could not open {vcf_path}: {e}")
                continue
            contig = _choose_db_contig_name(vcf, chrom_token)
            if contig is None:
                logging.warning(f"DB header has no contig for chrom {chrom_token}; skipping {vcf_path}")
                continue
            consecutive_failures = 0
            failed_regions: list = []
            start0 = 0
            vcf_ok = True
            while start0 < max_pos and vcf_ok:
                end0 = min(start0 + self.chunk_bp, max_pos)
                chunk_ok = False
                for attempt in range(1, self.fetch_retries + 1):
                    try:
                        for rec in vcf.fetch(contig, start0, end0):
                            if rec.alts is None:
                                continue
                            c = str(rec.chrom).replace("chr", "")
                            for alt in rec.alts:
                                key = (c, int(rec.pos), str(rec.ref), str(alt))
                                if key in sample_keys:
                                    found.add(key)
                        chunk_ok = True
                        break
                    except Exception as e:
                        logging.warning(
                            f"[PopDB fetch fail] {vcf_path} {contig}:{start0}-{end0}"
                            f" (attempt {attempt}/{self.fetch_retries}): {e}"
                        )
                        if attempt < self.fetch_retries:
                            time.sleep(min(2 ** (attempt - 1), 5))
                if chunk_ok:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    failed_regions.append(f"{contig}:{start0}-{end0}")
                    if consecutive_failures >= 3:
                        logging.warning(
                            f"Aborting {vcf_path} after {consecutive_failures} consecutive "
                            f"chunk failures — file is likely truncated or corrupted. "
                            f"Data lost from position {start0} to end of chromosome."
                        )
                        vcf_ok = False
                start0 = end0
            if failed_regions:
                logging.warning(
                    f"{vcf_path}: {len(failed_regions)} chunk(s) failed permanently: "
                    + ", ".join(failed_regions[:5])
                    + (" ..." if len(failed_regions) > 5 else "")
                )
            try:
                vcf.close()
            except Exception as close_err:
                logging.warning(f"Could not close {vcf_path} cleanly (likely truncated): {close_err}")

        logging.info(f"PopulationDatabaseManager: found {len(found)} sample variants in population DBs.")
        return found


# ============================================================================
# CLINVAR HELPERS
# ============================================================================


def load_clinvar_dict(clinvar_vcf: str) -> Dict[Tuple[str, int, str, str], str]:
    logging.info(f"Loading ClinVar VCF: {clinvar_vcf}")
    try:
        vcf = open_variantfile_with_optional_local_index(clinvar_vcf, search_dirs=[os.getcwd()])
    except Exception:
        vcf = pysam.VariantFile(clinvar_vcf)

    has_clnsig = "CLNSIG" in set(vcf.header.info)
    has_clnrev = "CLNREVSTAT" in set(vcf.header.info)
    if not has_clnsig:
        logging.warning("ClinVar VCF has no CLNSIG INFO field; cannot filter.")
        return {}

    def classify_record(rec) -> Optional[str]:
        clnsig = rec.info.get("CLNSIG", [])
        if not isinstance(clnsig, (list, tuple)):
            clnsig = [clnsig]
        sig_str = ",".join(map(str, clnsig)).upper()
        if has_clnrev:
            clnrev = rec.info.get("CLNREVSTAT", [])
            if not isinstance(clnrev, (list, tuple)):
                clnrev = [clnrev]
            rev_str = ",".join(map(str, clnrev)).upper()
            if not any(x in rev_str for x in ["PRACTICE_GUIDELINE", "REVIEWED_BY_EXPERT_PANEL"]):
                return None
        is_path = any(x in sig_str for x in ["PATHOGENIC", "LIKELY_PATHOGENIC"])
        is_benign = any(x in sig_str for x in ["BENIGN", "LIKELY_BENIGN"])
        if is_path and not is_benign:
            return "P"
        if is_benign and not is_path:
            return "B"
        return None

    cv_dict: Dict[Tuple[str, int, str, str], str] = {}
    for rec in vcf:
        label = classify_record(rec)
        if label is None or rec.alts is None:
            continue
        c = str(rec.chrom).replace("chr", "")
        for alt in rec.alts:
            cv_dict[(c, int(rec.pos), str(rec.ref), str(alt))] = label

    logging.info(f"Loaded {len(cv_dict)} high-confidence ClinVar variants (P/B).")
    return cv_dict


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================


