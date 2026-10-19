"""Reusable preprocessing package for the published TopoDockQ checkpoint.

The authors' ``04_inference_from_generated_csv.py`` rebuilds the feature
selection and the StandardScaler from the 1.7 GB training table on every call.
``build_package`` runs that same logic once and saves its result:

  feature_columns.json   the 2646 selected column names, in model input order
  dropped_columns.json   columns removed as constant (authors' rule)
  scaler.npz             StandardScaler mean_, scale_, var_, n_samples_seen_
  manifest.json          checkpoint sha256, source data hashes, row counts,
                         model hyperparameters, library versions

``predict`` uses the package plus the authors' own ``TopoDockQ`` class from
``src/model.py``. It must always agree with the unmodified 04 script; Step 1
checks that.

Every statement in ``build_package`` marked "04:Lnn" is a line-for-line copy of
the authors' code (line numbers refer to 04_inference_from_generated_csv.py).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .paths import (AUTHORS_REPO, PREPROCESSING_DIR, PROCESSED_DATA_DIR, TRAINED_MODEL, from_project_relative,
                    to_project_relative)
from .provenance import library_versions, sha256

PACKAGE_FILES = ("feature_columns.json", "dropped_columns.json", "scaler.npz")
FEATURE_SCHEMA = "topodockq-v1"          # 2754 raw columns as written by 03_extract_features_from_npy_to_csv.py
PACKAGE_ID = "published-model-v1"
AUTHORS_04 = AUTHORS_REPO / "04_inference_from_generated_csv.py"

# Hyperparameters hard-coded in the authors' 04 script (04:L54-59).
MODEL_HPARAMS = {"input_dim": 2646, "neurons1": 2048, "neurons2": 2048, "neurons3": 2048,
                 "neurons4": 2048, "dropout": 0.0}


def author_feature_name_list() -> list[str]:
    """The ordered candidate column list built by the authors (04:L112-125)."""
    filtration_values = [f"{x:.1f}" if x.is_integer() else f"{x:.2f}".rstrip("0")
                         for x in np.arange(2.0, 10.25, 0.25)]                                      # 04:L112
    persistent_features = []                                                                       # 04:L115
    for filtration in filtration_values:                                                           # 04:L116
        persistent_features += [f"persistent_{filtration}_{str(i+1).zfill(2)}" for i in range(72)]  # 04:L117
    static_features = [f"static_{str(i+1).zfill(2)}" for i in range(378)]                          # 04:L120
    combined_feature_name_list = persistent_features + static_features                             # 04:L123
    return ['pdb2sql_DockQ'] + combined_feature_name_list                                          # 04:L125


FEATURES_CSV = PROCESSED_DATA_DIR / "singlePPD_full_bins_features.csv"
LABELS_CSV = PROCESSED_DATA_DIR / "singlePPD_DockQ.csv"


def author_merged_table(features_csv: Path = FEATURES_CSV,
                        labels_csv: Path = LABELS_CSV) -> tuple[pd.DataFrame, pd.Index]:
    """The authors' merged table ``df1`` and the constant columns they drop.

    Identical in 02_tutorial_inference.py (L62-82) and 04_inference_from_generated_csv.py (L63-83).
    Row order of the returned table is the order in which the authors' scripts emit predictions.
    """
    df1 = pd.read_csv(features_csv)                                                     # 04:L63
    df_train = df1[df1['data_class'] != 'test']                                         # 04:L66
    identical_columns = df_train.columns[df_train.nunique() == 1]                       # 04:L72
    df1 = df1.drop(columns=identical_columns)                                           # 04:L76
    df2 = pd.read_csv(labels_csv)                                                       # 04:L79
    df2 = df2[['pdb_id', 'af_model_id', 'af_confidence', 'pdb2sql_DockQ', 'data_class']]  # 04:L80
    df1 = pd.merge(df1, df2, on=['pdb_id', 'af_model_id', 'pdb2sql_DockQ', 'data_class'], how='inner')  # 04:L83
    return df1, identical_columns


def build_package(out_dir: Path | None = None,
                  features_csv: Path = FEATURES_CSV,
                  labels_csv: Path = LABELS_CSV,
                  checkpoint: Path = TRAINED_MODEL) -> dict[str, Any]:
    out_dir = Path(out_dir or PREPROCESSING_DIR / PACKAGE_ID)
    t0 = time.time()

    df1, identical_columns = author_merged_table(features_csv, labels_csv)              # 04:L63-83
    df_train = df1[df1['data_class'] == 'train']                                        # 04:L92
    feature_name_list = author_feature_name_list()                                      # 04:L112-125
    valid_columns = [col for col in feature_name_list if col in df_train.columns]       # 04:L128
    df_train = df_train[valid_columns]                                                  # 04:L131
    X_train = df_train.drop(columns=['pdb2sql_DockQ']).values                           # 04:L140 (get_features_and_target)
    scaler = StandardScaler()                                                           # 04:L152
    scaler.fit_transform(X_train)                                                       # 04:L153

    feature_columns = [c for c in valid_columns if c != 'pdb2sql_DockQ']
    if len(feature_columns) != MODEL_HPARAMS["input_dim"]:
        raise ValueError(f"expected {MODEL_HPARAMS['input_dim']} selected features, got {len(feature_columns)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feature_columns.json").write_text(json.dumps(feature_columns, indent=0), encoding="utf-8")
    (out_dir / "dropped_columns.json").write_text(json.dumps([str(c) for c in identical_columns], indent=0),
                                                  encoding="utf-8")
    np.savez(out_dir / "scaler.npz", mean_=scaler.mean_, scale_=scaler.scale_, var_=scaler.var_,
             n_samples_seen_=np.asarray(scaler.n_samples_seen_))
    manifest = {
        "package_id": PACKAGE_ID,
        "feature_schema": FEATURE_SCHEMA,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mirrors_script": {"path": to_project_relative(AUTHORS_04), "sha256": sha256(AUTHORS_04),
                           "lines": "63-153 (selection and scaler fit)"},
        "checkpoint": {"path": to_project_relative(checkpoint), "sha256": sha256(checkpoint)},
        "sources": {to_project_relative(p): sha256(p) for p in (features_csv, labels_csv)},
        "rows": {"features_csv": int(len(pd.read_csv(features_csv, usecols=['data_class']))),
                 "after_merge": int(len(df1)), "scaler_fit_rows_train": int(X_train.shape[0])},
        "n_raw_feature_columns": 2754,
        "n_dropped_columns": int(len(identical_columns)),
        "n_feature_columns": len(feature_columns),
        "model_hparams": MODEL_HPARAMS,
        "model_class": {"path": to_project_relative(AUTHORS_REPO / "src" / "model.py"),
                        "sha256": sha256(AUTHORS_REPO / "src" / "model.py"), "name": "TopoDockQ"},
        "library_versions": library_versions(),
        "build_seconds": round(time.time() - t0, 1),
        # Integrity: load_package refuses the package if any of these files changes.
        "files": {name: sha256(out_dir / name) for name in PACKAGE_FILES},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"package_dir": str(out_dir), **manifest}


def load_author_model_class():
    """Import ``TopoDockQ`` from the authors' src/model.py without writing bytecode into authors_code/."""
    path = AUTHORS_REPO / "src" / "model.py"
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("topodockq_author_model", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module.TopoDockQ


class PackageIntegrityError(ValueError):
    """The preprocessing package is incomplete, altered, or internally inconsistent."""


class FeatureInputError(ValueError):
    """The feature table cannot be fed to the model as given."""


def load_package(package_dir: Path | None = None) -> dict[str, Any]:
    """Load and verify the package: file hashes, column list, scaler shapes and values."""
    package_dir = Path(package_dir or PREPROCESSING_DIR / PACKAGE_ID)
    manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
    recorded = manifest.get("files")
    if not recorded:
        raise PackageIntegrityError("manifest has no file hashes; rebuild the package")
    for name, digest in recorded.items():
        path = package_dir / name
        if not path.is_file() or sha256(path) != digest:
            raise PackageIntegrityError(f"{name} is missing or does not match its recorded sha256")
    columns = json.loads((package_dir / "feature_columns.json").read_text(encoding="utf-8"))
    with np.load(package_dir / "scaler.npz") as z:
        scaler = {k: z[k] for k in z.files}

    n = manifest["model_hparams"]["input_dim"]
    if len(columns) != n or len(set(columns)) != n:
        raise PackageIntegrityError(f"expected {n} unique feature columns, got {len(columns)} "
                                    f"({len(set(columns))} unique)")
    for key in ("mean_", "scale_", "var_"):
        arr = scaler.get(key)
        if arr is None or arr.shape != (n,) or not np.all(np.isfinite(arr)):
            raise PackageIntegrityError(f"scaler {key} must be {n} finite values")
    if np.any(scaler["scale_"] <= 0):
        raise PackageIntegrityError("scaler scale_ must be positive")
    if int(scaler["n_samples_seen_"]) <= 0:
        raise PackageIntegrityError("scaler n_samples_seen_ must be positive")
    return {"dir": package_dir, "manifest": manifest, "columns": columns, "scaler": scaler}


def validate_features(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Return the model input matrix, or raise if any value is missing, non-numeric or non-finite."""
    if df.empty:
        raise FeatureInputError("feature table is empty")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise FeatureInputError(f"{len(missing)} required feature columns missing, e.g. {missing[:5]}")
    sub = df[columns]
    non_numeric = [c for c in columns if pd.api.types.is_bool_dtype(sub[c]) or not pd.api.types.is_numeric_dtype(sub[c])]
    if non_numeric:
        raise FeatureInputError(f"{len(non_numeric)} feature columns are not numeric, e.g. {non_numeric[:5]}")
    X = sub.astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    bad = ~np.isfinite(X)
    if bad.any():
        cols = [columns[j] for j in np.where(bad.any(axis=0))[0][:5]]
        raise FeatureInputError(f"{int(bad.sum())} missing or non-finite feature values in "
                                f"{int(bad.any(axis=1).sum())} rows, e.g. columns {cols}")
    return X


def predict(features_csv: Path, package_dir: Path | None = None, device: str = "auto") -> dict[str, Any]:
    """Predict p-DockQ for every row of a 2754-column feature CSV (03 script output format)."""
    return predict_frame(pd.read_csv(features_csv), package_dir, device)


def predict_frame(df: pd.DataFrame, package_dir: Path | None = None, device: str = "auto") -> dict[str, Any]:
    """Predict p-DockQ for every row of a table that contains the package's feature columns."""
    import torch

    pkg = load_package(package_dir)
    ckpt = from_project_relative(pkg["manifest"]["checkpoint"]["path"])
    if not ckpt.is_file() or sha256(ckpt) != pkg["manifest"]["checkpoint"]["sha256"]:
        raise PackageIntegrityError(f"checkpoint {ckpt} is missing or does not match the package's recorded sha256")

    X = validate_features(df, pkg["columns"])
    # Equivalent to sklearn StandardScaler.transform: (X - mean_) / scale_
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_, scaler.var_ = pkg["scaler"]["mean_"], pkg["scaler"]["scale_"], pkg["scaler"]["var_"]
    scaler.n_samples_seen_ = int(pkg["scaler"]["n_samples_seen_"])
    scaler.n_features_in_ = X.shape[1]
    Xs = scaler.transform(X)

    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else
                       (device if device != "auto" else "cpu"))
    TopoDockQ = load_author_model_class()
    hp = pkg["manifest"]["model_hparams"]
    model = TopoDockQ(hp["input_dim"], hp["neurons1"], hp["neurons2"], hp["neurons3"], hp["neurons4"],
                      hp["dropout"]).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(Xs, dtype=torch.float32).to(dev)).cpu().numpy().ravel()
    return {"predictions": [float(v) for v in out], "device": str(dev), "n_rows": int(len(df))}
