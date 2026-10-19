"""Milestone 4: TopoAgent tools - topological feature generation from interface PDB files.

Run from the project root:

    python -m tdq_adapters.step4 [--per-cell 10] [--extremes 5] [--pfpd-per-complex 2] [--workers 8]

Steps (no LLM involved; the run guard adds the protected-directory check and a
structured report even on failure):
  1. Sample SinglePPD structures that have published features: every split (train,
     valid, test) x every DockQ quality class (<0.23, 0.23-0.49, 0.49-0.8, >0.8),
     ``--per-cell`` each (seed 0), plus the ``--extremes`` smallest and largest
     interface files. Extract only those files from singlePPD_interface_files.tar.gz
     into this run's folder.
  2. Generate the 2754 raw features with both engines through feature_gen:
       original = authors' main.pyc, unchanged, Python 3.8 + gudhi 3.8;
       compat   = tdq_adapters.topo_features, Python 3.13 + gudhi 3.13.
     Both use the paper's --bins/--filtration and the authors' 03 conversion script.
  3. Compare every feature of every sample with the published feature table:
     per-sample and per-feature absolute and relative differences and mismatch
     counts (tolerance below), plus a breakdown by feature type.
  4. p-DockQ from generated features vs from published features (preprocessing package).
  5. PFPD: original vs compat on the authors' published PFPD interfaces
     (no published PFPD features exist, so this checks engine agreement only).
Writes runs/step4-<timestamp>/step4_report.json, per_feature_differences.csv,
per_sample_differences.csv and the generated feature tables.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import feature_gen, preprocessing
from .paths import AUTHORS_REPO, EXTERNAL_DIR, FEATURE38_PYTHON, RUNS_DIR, ZENODO_DIR, to_project_relative
from .provenance import sha256
from .runner import run_guarded

INTERFACE_TAR = ZENODO_DIR / "singlePPD_interface_files.tar.gz"
PFPD_DIR = EXTERNAL_DIR / "PFPD_AF2_models"
EXAMPLE_IFACE = AUTHORS_REPO / "data" / "interface_files" / "4k38_ranked_44_sp_interface.pdb"
EXAMPLE_NPY = AUTHORS_REPO / "feature" / "feature_4k38_ranked_44_sp_interface.npy"
SPLITS = ("train", "valid", "test")
DOCKQ_CLASSES = [("incorrect", 0.0, 0.23), ("acceptable", 0.23, 0.49), ("medium", 0.49, 0.80), ("high", 0.80, 1.01)]
RTOL, ATOL = 1e-9, 1e-9          # published table stores full double precision
PDOCKQ_TOL = 1e-6
KEY = ["pdb_id", "af_model_id"]

SPECTRA_STATS = ["eig_sum", "eig_min", "eig_max", "eig_mean", "eig_std", "eig_var", "zero_eig_count"]
ALPHA_STATS = ["pers_mean", "pers_std", "pers_max", "pers_min", "pers_sum", "longest_birth", "longest_death",
               "birth_mean", "birth_std", "birth_max", "birth_min", "birth_sum",
               "death_mean", "death_std", "death_max", "death_min", "death_sum", "bar_count"]
RIPS_STATS = ["death_mean", "death_std", "death_max", "death_min", "death_sum", "death_count"]
PAIRS = [a + b for a in "CNO" for b in "CNO"]


def describe_column(col: str) -> dict[str, str]:
    """Map a column of the 03 CSV layout to (element pair, feature family, statistic)."""
    if col.startswith("persistent_"):
        _, filt, k = col.split("_")
        k = int(k) - 1
        pair, within = PAIRS[k // 8], k % 8
        if within == 0:
            return {"pair": pair, "family": "rips_H0_death_bin_count", "stat": f"bin_ending_{filt}"}
        return {"pair": pair, "family": "rips_laplacian_spectrum", "stat": f"{SPECTRA_STATS[within - 1]}@{filt}"}
    j = int(col.split("_")[1]) - 1
    pair, within = PAIRS[j // 42], j % 42
    if within < 6:
        return {"pair": pair, "family": "rips_H0_death_stats", "stat": RIPS_STATS[within]}
    if within < 24:
        return {"pair": pair, "family": "alpha_H1_barcode_stats", "stat": ALPHA_STATS[within - 6]}
    return {"pair": pair, "family": "alpha_H2_barcode_stats", "stat": ALPHA_STATS[within - 24]}


# ------------------------------------------------------------------ sampling
def dockq_class(v: float) -> str:
    return next(name for name, lo, hi in DOCKQ_CLASSES if lo <= v < hi)


def select_sample(ids: pd.DataFrame, member_sizes: dict[str, int], per_cell: int, extremes: int,
                  seed: int = 0) -> pd.DataFrame:
    ids = ids.copy()
    ids["file"] = [f"{p}_ranked_{m}_sp_interface.pdb" for p, m in zip(ids.pdb_id, ids.af_model_id)]
    ids = ids[ids["file"].isin(member_sizes)].copy()
    ids["file_bytes"] = ids["file"].map(member_sizes)
    ids["dockq_class"] = ids["pdb2sql_DockQ"].map(dockq_class)
    picks = []
    for split in SPLITS:
        for cls, _, _ in DOCKQ_CLASSES:
            cell = ids[(ids.data_class == split) & (ids.dockq_class == cls)]
            picks.append(cell.sample(n=min(per_cell, len(cell)), random_state=seed).assign(reason=f"{split}/{cls}"))
    by_size = ids.sort_values("file_bytes")
    picks.append(by_size.head(extremes).assign(reason="smallest interface"))
    picks.append(by_size.tail(extremes).assign(reason="largest interface"))
    return pd.concat(picks).drop_duplicates(KEY).reset_index(drop=True)


def read_tar_index(tar_path: Path) -> dict[str, int]:
    with tarfile.open(tar_path, "r:gz") as t:
        return {Path(m.name).name: m.size for m in t.getmembers() if m.isfile()}


def extract_members(tar_path: Path, names: set[str], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    got = []
    with tarfile.open(tar_path, "r:gz") as t:
        for m in t:
            if m.isfile() and Path(m.name).name in names:
                dst = out_dir / Path(m.name).name
                with t.extractfile(m) as src, open(dst, "wb") as f:
                    shutil.copyfileobj(src, f)
                got.append(dst)
    return got


def published_rows(keys: pd.DataFrame) -> pd.DataFrame:
    wanted = set(map(tuple, keys[KEY].values))
    parts = []
    for chunk in pd.read_csv(preprocessing.FEATURES_CSV, chunksize=5000):
        mask = [(p, m) in wanted for p, m in zip(chunk.pdb_id, chunk.af_model_id)]
        if any(mask):
            parts.append(chunk[mask])
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------ comparison
class KeyMismatchError(ValueError):
    """The two tables do not describe exactly the same set of (pdb_id, af_model_id) samples."""


def check_keys(ours: pd.DataFrame, ref: pd.DataFrame, label: str) -> None:
    """Both tables must have unique keys and exactly the same key set before any merge."""
    for name, t in (("generated", ours), ("reference", ref)):
        dup = int(t.duplicated(KEY).sum())
        if dup:
            raise KeyMismatchError(f"{label}: {dup} duplicate (pdb_id, af_model_id) keys in the {name} table")
    a, b = set(map(tuple, ours[KEY].values)), set(map(tuple, ref[KEY].values))
    if a != b:
        raise KeyMismatchError(f"{label}: key sets differ; only generated {sorted(a - b)[:5]}, "
                               f"only reference {sorted(b - a)[:5]}")


def compare_tables(ours: pd.DataFrame, ref: pd.DataFrame, cols: list[str], label: str) -> dict:
    check_keys(ours, ref, label)
    m = ours.merge(ref, on=KEY, suffixes=("_ours", "_ref"), validate="one_to_one")
    a = m[[c + "_ours" for c in cols]].to_numpy(float)
    b = m[[c + "_ref" for c in cols]].to_numpy(float)
    absd = np.abs(a - b)
    scale = np.abs(b)
    reld = np.where(scale > ATOL, absd / np.where(scale > 0, scale, 1), np.nan)   # relative only where meaningful
    mismatch = ~np.isclose(a, b, rtol=RTOL, atol=ATOL)
    per_sample = pd.DataFrame({"pdb_id": m.pdb_id, "af_model_id": m.af_model_id, "comparison": label,
                               "max_abs_diff": absd.max(axis=1), "max_rel_diff": np.nanmax(np.where(np.isnan(reld), 0, reld), axis=1),
                               "mismatched_features": mismatch.sum(axis=1)})
    per_feature = pd.DataFrame({"column": cols, "comparison": label, "max_abs_diff": absd.max(axis=0),
                                "mean_abs_diff": absd.mean(axis=0),
                                "max_rel_diff": np.nanmax(np.where(np.isnan(reld), 0, reld), axis=0),
                                "mismatched_samples": mismatch.sum(axis=0),
                                "max_abs_reference": np.abs(b).max(axis=0)})
    desc = pd.DataFrame([describe_column(c) for c in cols])
    per_feature = pd.concat([per_feature, desc], axis=1)
    by_family = per_feature.groupby("family").agg(features=("column", "size"), max_abs_diff=("max_abs_diff", "max"),
                                                  max_rel_diff=("max_rel_diff", "max"),
                                                  features_with_mismatch=("mismatched_samples", lambda s: int((s > 0).sum())),
                                                  mismatched_values=("mismatched_samples", "sum")).reset_index()
    summary = {"samples_compared": int(len(m)), "samples_missing": int(len(ours) - len(m)),
               "values_compared": int(a.size), "mismatched_values": int(mismatch.sum()),
               "samples_with_any_mismatch": int((mismatch.sum(axis=1) > 0).sum()),
               "features_with_any_mismatch": int((mismatch.sum(axis=0) > 0).sum()),
               "max_abs_diff": float(absd.max()) if absd.size else None,
               "max_rel_diff": float(np.nanmax(reld)) if np.isfinite(reld).any() else 0.0,
               "worst_features": per_feature.sort_values("max_abs_diff", ascending=False)
               .head(10)[["column", "family", "pair", "stat", "max_abs_diff", "max_rel_diff", "mismatched_samples"]]
               .to_dict("records"),
               "by_family": by_family.to_dict("records")}
    return {"summary": summary, "per_sample": per_sample, "per_feature": per_feature}


# ------------------------------------------------------------------ milestone body
def body(report: dict, run_dir: Path, per_cell: int, extremes: int, pfpd_per_complex: int, workers: int) -> dict:
    report["tolerance"] = {"rtol": RTOL, "atol": ATOL, "pdockq": PDOCKQ_TOL,
                           "relative_difference": "reported only where |reference| > atol"}
    report["engines"] = {"original": {"python": to_project_relative(FEATURE38_PYTHON),
                                      "code": to_project_relative(AUTHORS_REPO / "main.pyc"),
                                      "main_pyc_sha256": sha256(AUTHORS_REPO / "main.pyc")},
                         "compat": {"module": "tdq_adapters.topo_features",
                                    "sha256": sha256(Path(__file__).with_name("topo_features.py"))}}
    checks: dict[str, bool] = {}

    print("[1/6] 4k38 example, both engines vs the authors' published .npy")
    ref = np.load(EXAMPLE_NPY)
    ex = {}
    for engine in feature_gen.ENGINES:
        job = feature_gen.generate_features([feature_gen.FeatureSample("4k38_ranked_44", "4k38", 44, str(EXAMPLE_IFACE))],
                                            run_dir / f"example_{engine}", engine=engine, workers=1)
        rec = job["records"][0]
        arr = np.load(rec["npy_file"]) if rec["status"] == "ok" else None
        ex[engine] = {"status": rec["status"], "error": rec.get("error"),
                      "max_abs_diff": float(np.abs(arr - ref).max()) if arr is not None else None,
                      "mismatched_values": int((~np.isclose(arr, ref, rtol=RTOL, atol=ATOL)).sum()) if arr is not None else None}
        checks[f"example_{engine}_matches_published"] = ex[engine]["mismatched_values"] == 0
    report["example_4k38"] = ex

    print("[2/6] sampling SinglePPD structures and extracting their interface files")
    index = read_tar_index(INTERFACE_TAR)
    ids = pd.read_csv(preprocessing.FEATURES_CSV, usecols=KEY + ["pdb2sql_DockQ", "data_class"])
    sample = select_sample(ids, index, per_cell, extremes)
    iface_dir = run_dir / "singleppd_interfaces"
    extracted = extract_members(INTERFACE_TAR, set(sample["file"]), iface_dir)
    sample.to_csv(run_dir / "sample.csv", index=False)
    cells = sample.groupby(["data_class", "dockq_class"]).size()
    report["sample"] = {"archive": str(INTERFACE_TAR), "archive_sha256": sha256(INTERFACE_TAR),
                        "archive_files": len(index), "published_rows_with_interface_file": int(ids.merge(
                            pd.DataFrame({"file": list(index)}).assign(
                                pdb_id=lambda d: d.file.str.split("_ranked_").str[0],
                                af_model_id=lambda d: d.file.str.split("_ranked_").str[1].str.split("_").str[0].astype(int)),
                            on=KEY).shape[0]),
                        "selected": int(len(sample)), "extracted": len(extracted),
                        "complexes": int(sample.pdb_id.nunique()),
                        "cells": {f"{s}/{c}": int(n) for (s, c), n in cells.items()},
                        "interface_bytes_range": [int(sample.file_bytes.min()), int(sample.file_bytes.max())],
                        "seed": 0, "per_cell": per_cell, "extremes": extremes}
    checks["sample_covers_every_split_and_dockq_class"] = len(cells) == len(SPLITS) * len(DOCKQ_CLASSES)
    checks["all_sampled_files_extracted"] = len(extracted) == len(sample)

    print(f"[3/6] generating features for {len(sample)} structures with both engines")
    samples = [feature_gen.FeatureSample(f"{p}_ranked_{m}", p, int(m), str(iface_dir / f))
               for p, m, f in zip(sample.pdb_id, sample.af_model_id, sample.file)]
    tables, jobs = {}, {}
    for engine in feature_gen.ENGINES:
        job = feature_gen.generate_features(samples, run_dir / f"singleppd_{engine}", engine=engine, workers=workers)
        jobs[engine] = job["summary"]
        tables[engine] = job["table"]
        checks[f"singleppd_{engine}_no_failures"] = job["summary"]["failed"] == 0
    report["generation"] = jobs

    print("[4/6] comparing every feature with the published table")
    pub = published_rows(sample)
    cols = [c for c in pub.columns if c.startswith(("persistent_", "static_"))]
    report["comparison"] = {}
    per_sample_all, per_feature_all = [], []
    for label, ours, ref_t in (("original_vs_published", tables["original"], pub),
                               ("compat_vs_published", tables["compat"], pub),
                               ("compat_vs_original", tables["compat"], tables["original"])):
        res = compare_tables(ours, ref_t, cols, label)
        report["comparison"][label] = res["summary"]
        per_sample_all.append(res["per_sample"])
        per_feature_all.append(res["per_feature"])
        checks[f"{label}_all_features_match"] = (res["summary"]["mismatched_values"] == 0
                                                 and res["summary"]["samples_compared"] == len(sample))
    pd.concat(per_sample_all).to_csv(run_dir / "per_sample_differences.csv", index=False)
    pd.concat(per_feature_all).to_csv(run_dir / "per_feature_differences.csv", index=False)

    print("[5/6] p-DockQ from generated vs published features")
    preds = pub[KEY + ["pdb2sql_DockQ", "data_class"]].copy()
    preds["p_dockq_published_features"] = preprocessing.predict_frame(pub, device="cpu")["predictions"]
    for engine in feature_gen.ENGINES:
        t = tables[engine].merge(pub[KEY], on=KEY)
        p = pd.DataFrame({"pdb_id": t.pdb_id, "af_model_id": t.af_model_id,
                          f"p_dockq_{engine}_features": preprocessing.predict_frame(t, device="cpu")["predictions"]})
        preds = preds.merge(p, on=KEY)
    preds.to_csv(run_dir / "pdockq_comparison.csv", index=False)
    report["pdockq"] = {}
    for engine in feature_gen.ENGINES:
        d = float((preds[f"p_dockq_{engine}_features"] - preds["p_dockq_published_features"]).abs().max())
        report["pdockq"][engine] = {"rows": int(len(preds)), "max_abs_diff_vs_published_features": d}
        checks[f"pdockq_{engine}_matches_published_features"] = d <= PDOCKQ_TOL and len(preds) == len(sample)

    print(f"[6/6] PFPD: original vs compat on {pfpd_per_complex} model(s) per complex")
    pfpd_dir = run_dir / "pfpd_interfaces"
    pfpd_dir.mkdir()
    pfpd_samples = []
    for cdir in sorted(p for p in PFPD_DIR.iterdir() if p.is_dir()):
        for m in range(pfpd_per_complex):
            src = cdir / f"ranked_{m}_sp_interface.pdb"
            pdb_id = cdir.name.lower()
            dst = pfpd_dir / f"{pdb_id}_ranked_{m}_sp_interface.pdb"
            shutil.copyfile(src, dst)                               # authors' name prefix added; content unchanged
            pfpd_samples.append(feature_gen.FeatureSample(f"{pdb_id}_ranked_{m}", pdb_id, m, str(dst)))
    pt = {}
    for engine in feature_gen.ENGINES:
        job = feature_gen.generate_features(pfpd_samples, run_dir / f"pfpd_{engine}", engine=engine, workers=workers)
        pt[engine] = job["table"]
        checks[f"pfpd_{engine}_no_failures"] = job["summary"]["failed"] == 0
        report.setdefault("pfpd_generation", {})[engine] = job["summary"]
    res = compare_tables(pt["compat"], pt["original"], cols, "pfpd_compat_vs_original")
    report["comparison"]["pfpd_compat_vs_original"] = res["summary"]
    res["per_feature"].to_csv(run_dir / "pfpd_per_feature_differences.csv", index=False)
    checks["pfpd_compat_matches_original"] = (res["summary"]["mismatched_values"] == 0
                                              and res["summary"]["samples_compared"] == len(pfpd_samples))

    for label, s in report["comparison"].items():
        print(f"  {label:28s} samples={s['samples_compared']:4d} values={s['values_compared']:7d} "
              f"mismatched={s['mismatched_values']} max_abs={s['max_abs_diff']:.3e} max_rel={s['max_rel_diff']:.3e}")
    print("  p-DockQ:", report["pdockq"])
    return checks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--extremes", type=int, default=5)
    ap.add_argument("--pfpd-per-complex", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    run_dir = args.run_dir or RUNS_DIR / f"step4-{time.strftime('%Y%m%d-%H%M%S')}"
    code, _ = run_guarded("step4", run_dir, lambda rep, rd: body(rep, rd, args.per_cell, args.extremes,
                                                                    args.pfpd_per_complex, args.workers))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
