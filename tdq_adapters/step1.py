"""Milestone 1: reproduce the published 4k38 p-DockQ and build the preprocessing package.

Run from the project root:

    python -m tdq_adapters.step1

Steps (no LLM involved; the run guard in runner.py adds the protected-directory
check before and after, and writes a structured report even if a step fails):
  1. Stage the authors' repository into runs/step1-<timestamp>/workdir.
  2. Authors' 03 script, unchanged: 4k38 .npy -> feature/example_features.csv.
     Compare with the authors' own shipped example_features.csv.
  3. Authors' 04 script, unchanged: the baseline prediction.
  4. Build the preprocessing package (mirror of 04's selection + scaler).
  5. Predict with the package on the same CSV, on the baseline device and on CPU.
  6. Compare everything with the published value and with each other.
Writes runs/step1-<timestamp>/step1_report.json.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import author_scripts, preprocessing
from .paths import AUTHORS_REPO, PREPROCESSING_DIR, PUBLISHED_4K38_PDOCKQ, RUNS_DIR, TRAINED_MODEL
from .provenance import sha256
from .runner import run_guarded
from .staging import stage_author_repo

NPY_REL = "./feature/feature_4k38_ranked_44_sp_interface.npy"
CSV_REL = "./feature/example_features.csv"
TOLERANCE_VS_PUBLISHED = 5e-7     # published value is printed as float32 with 8 significant digits
TOLERANCE_PACKAGE_VS_SCRIPT = 1e-6


def body(report: dict, run_dir: Path) -> dict[str, bool]:
    workdir, log_dir = run_dir / "workdir", run_dir / "logs"

    print(f"[1/6] staging authors' repository -> {workdir}")
    report["staging"] = stage_author_repo(workdir)
    report["inputs"] = {
        "npy": {"path": str(AUTHORS_REPO / NPY_REL), "sha256": sha256(AUTHORS_REPO / NPY_REL)},
        "authors_example_features_csv": {"path": str(AUTHORS_REPO / CSV_REL), "sha256": sha256(AUTHORS_REPO / CSV_REL)},
        "checkpoint": {"path": str(TRAINED_MODEL), "sha256": sha256(TRAINED_MODEL)},
        "script_03": sha256(AUTHORS_REPO / "03_extract_features_from_npy_to_csv.py"),
        "script_04": sha256(AUTHORS_REPO / "04_inference_from_generated_csv.py"),
        "src_model_py": sha256(AUTHORS_REPO / "src" / "model.py"),
    }

    print("[2/6] authors' 03 script: npy -> csv")
    report["run_03"] = author_scripts.npy_to_csv(workdir, NPY_REL, CSV_REL, log_dir)
    ours = pd.read_csv(workdir / CSV_REL)
    theirs = pd.read_csv(AUTHORS_REPO / CSV_REL)
    same_cols = list(ours.columns) == list(theirs.columns)
    report["csv_check"] = {
        "same_columns_in_order": same_cols,
        "shape": list(ours.shape),
        "max_abs_diff_vs_authors_csv": float(np.abs(ours.values - theirs[ours.columns].values).max()) if same_cols else None,
        "regenerated_sha256": sha256(workdir / CSV_REL),
    }

    print("[3/6] authors' 04 script, unchanged (baseline; loads the 1.7 GB table)")
    report["run_04"], baseline = author_scripts.infer_from_generated_csv(workdir, log_dir)

    print("[4/6] building preprocessing package")
    pkg = preprocessing.build_package(PREPROCESSING_DIR / preprocessing.PACKAGE_ID)
    report["package"] = {k: pkg[k] for k in ("package_dir", "n_dropped_columns", "n_feature_columns", "rows",
                                             "checkpoint", "files", "build_seconds")}

    print("[5/6] predicting with the package (verifies package integrity first)")
    base_dev = report["device"]["device"]
    pkg_same = preprocessing.predict(workdir / CSV_REL, device=base_dev)
    pkg_cpu = preprocessing.predict(workdir / CSV_REL, device="cpu")

    print("[6/6] comparing")
    b, s, c = baseline[0], pkg_same["predictions"][0], pkg_cpu["predictions"][0]
    r = report["results"] = {
        "published": PUBLISHED_4K38_PDOCKQ,
        "baseline_04_script": b,
        "package_same_device": {"device": pkg_same["device"], "value": s},
        "package_cpu": {"device": "cpu", "value": c},
        "abs_diff_baseline_vs_published": abs(b - PUBLISHED_4K38_PDOCKQ),
        "abs_diff_package_vs_baseline_same_device": abs(s - b),
        "abs_diff_package_cpu_vs_baseline": abs(c - b),
        "tolerances": {"vs_published": TOLERANCE_VS_PUBLISHED, "package_vs_script": TOLERANCE_PACKAGE_VS_SCRIPT},
    }
    print({k: v for k, v in r.items() if k != "tolerances"})
    return {
        "regenerated_csv_matches_authors": same_cols and report["csv_check"]["max_abs_diff_vs_authors_csv"] == 0.0,
        "baseline_matches_published": r["abs_diff_baseline_vs_published"] <= TOLERANCE_VS_PUBLISHED,
        "package_matches_baseline": r["abs_diff_package_vs_baseline_same_device"] <= TOLERANCE_PACKAGE_VS_SCRIPT,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    run_dir = args.run_dir or RUNS_DIR / f"step1-{time.strftime('%Y%m%d-%H%M%S')}"
    code, _ = run_guarded("step1", run_dir, body)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
