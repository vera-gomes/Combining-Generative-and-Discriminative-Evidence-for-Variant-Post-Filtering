"""End-to-end smoke test: train on a small synthetic benchmark, then apply the
saved models to a second synthetic sample. Checks that both modes run and that
apply mode writes one score column per model. Takes about a minute."""
import glob
import os
import subprocess
import sys

import numpy as np
import pysam
import pytest
import yaml

FEATURES = ["MQ", "QD", "FS", "SOR", "MQRankSum", "ReadPosRankSum", "BaseQRankSum", "DP"]


def _write_vcf(path, n, seed, truth_path=None):
    rng = np.random.default_rng(seed)
    header = pysam.VariantHeader()
    for c, length in (("chr1", 50_000_000), ("chr2", 50_000_000)):
        header.add_line(f"##contig=<ID={c},length={length}>")
    for f in FEATURES:
        t = "Integer" if f == "DP" else "Float"
        header.add_line(f'##INFO=<ID={f},Number=1,Type={t},Description="{f}">')
    header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')
    header.add_sample("S1")
    true = rng.random(n) < 0.75
    pos = np.sort(rng.choice(np.arange(1_000, 49_000_000), n, replace=False))
    chroms = np.where(np.arange(n) < n // 2, "chr1", "chr2")
    recs = []
    with pysam.VariantFile(path, "w", header=header) as out:
        for i in range(n):
            r = out.new_record(contig=chroms[i], start=int(pos[i]) - 1, alleles=("A", "G"))
            good = true[i]
            vals = {"MQ": rng.normal(59 if good else 45, 3), "QD": rng.normal(20 if good else 5, 4),
                    "FS": abs(rng.normal(2 if good else 15, 4)), "SOR": abs(rng.normal(0.7 if good else 2.5, 0.5)),
                    "MQRankSum": rng.normal(0 if good else -3, 1), "ReadPosRankSum": rng.normal(0, 1),
                    "BaseQRankSum": rng.normal(0, 1), "DP": int(rng.integers(15, 60))}
            for k, v in vals.items():
                if k in ("MQRankSum", "ReadPosRankSum", "BaseQRankSum") and rng.random() < 0.1:
                    continue  # missing annotation, as at homozygous sites
                r.info[k] = v
            r.samples["S1"]["GT"] = (0, 1)
            out.write(r)
            recs.append((chroms[i], int(pos[i]), good))
    pysam.tabix_index(path, preset="vcf", force=True)
    if truth_path:
        with pysam.VariantFile(truth_path, "w", header=header) as out:
            for c, p, good in recs:
                if good:
                    r = out.new_record(contig=c, start=p - 1, alleles=("A", "G"))
                    r.samples["S1"]["GT"] = (0, 1)
                    out.write(r)
        pysam.tabix_index(truth_path, preset="vcf", force=True)


def _run(cfg_path, cwd):
    res = subprocess.run([sys.executable, "-m", "postfilter_pipeline.cli", "--config", cfg_path],
                         cwd=cwd, capture_output=True, text=True, timeout=1800)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]


@pytest.mark.slow
def test_train_then_apply(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_vcf, truth_vcf = str(tmp_path / "train.vcf"), str(tmp_path / "truth.vcf")
    apply_vcf = str(tmp_path / "apply.vcf")
    _write_vcf(train_vcf, 3000, 1, truth_vcf)
    _write_vcf(apply_vcf, 1500, 2)

    cfg = yaml.safe_load(open(os.path.join(root, "configs", "train.example.yaml")))
    cfg["paths"] = {"train_input_vcf": train_vcf + ".gz", "truth_vcf": truth_vcf + ".gz",
                    "output_dir": str(tmp_path / "train_out")}
    cfg["models"]["lightgbm_optuna"]["n_trials"] = 3
    cfg["cv"]["n_splits"] = 3
    cfg["statistics"]["pairwise_bootstrap_n"] = 20
    with open(tmp_path / "train.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh)
    _run(str(tmp_path / "train.yaml"), str(tmp_path))
    train_dirs = sorted(glob.glob(str(tmp_path / "train_out*")))
    assert train_dirs, "train mode wrote no output directory"

    acfg = yaml.safe_load(open(os.path.join(root, "configs", "apply.example.yaml")))
    acfg["paths"] = {"apply_input_vcf": apply_vcf + ".gz", "trained_model_dir": train_dirs[-1],
                     "output_dir": str(tmp_path / "apply_out")}
    acfg["evaluation"]["external_validation"]["enabled"] = False
    acfg["evaluation"]["external_validation"]["clinvar"]["enabled"] = False
    acfg["evaluation"]["external_validation"]["population_dbs"] = []
    acfg["statistics"]["pairwise_bootstrap_n"] = 20
    acfg["statistics"]["bayes_bootstrap"]["n_draws"] = 20
    with open(tmp_path / "apply.yaml", "w") as fh:
        yaml.safe_dump(acfg, fh)
    _run(str(tmp_path / "apply.yaml"), str(tmp_path))

    score_files = glob.glob(str(tmp_path / "apply_out*" / "**" / "*scores*.csv"), recursive=True)
    assert score_files, "apply mode wrote no score file"
    import pandas as pd
    cols = pd.read_csv(score_files[0], nrows=5).columns
    for m in ["GM", "BGM", "LogReg", "RandForest", "LGB", "LGB_Bayes", "LGB_MultiObj",
              "Hybrid_LGB_alpha08", "Hybrid_LGB_alpha07"]:
        assert any(c.startswith(m) for c in cols), f"no score column for {m}"
