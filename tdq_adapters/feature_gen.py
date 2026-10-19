"""Job-level feature generation: the deterministic tools the TopoAgent will call.

For each interface file named ``{pdb_id}_ranked_{model_id}_sp_interface.pdb``:
  1. run the feature generator with the authors' command-line contract
     (--pdb_id --model_id --bins --filtration --file_path --saving_path), using
       engine "original": the authors' main.pyc, unchanged, in the Python 3.8
                          environment, from a staged copy of authors_code/TopoDockQ;
       engine "compat":   tdq_adapters.topo_features in the main environment;
  2. check the .npy: shape (1, 2754) for the paper settings, all values finite;
  3. convert it with the authors' 03_extract_features_from_npy_to_csv.py, unchanged;
  4. collect the rows into one feature table with pdb_id and af_model_id.
A thread pool launches one subprocess per structure; no LLM is involved per sample.
Before any work starts, identifiers are validated; duplicate (pdb_id, model_id)
identities, duplicate sample IDs and colliding output paths are refused; existing
output files and staging directories are never reused (ids.py).

Profiles. "paper" (default) locks --bins and --filtration to the published
settings; only then are rows converted with the authors' 03 script, labelled with
the published feature schema ("topodockq-v1"), and usable by the published model.
"experiment" allows other settings, but its features get a separate schema id,
generic raw_NNNN column names and no 03 conversion, so the published preprocessing
package refuses them (missing columns) instead of silently mislabelling them.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import ids
from .author_scripts import _ENV_OVERRIDES, _text
from .paths import AUTHORS_REPO, FEATURE38_PYTHON, PROJECT_ROOT
from .provenance import sha256
from .preprocessing import FEATURE_SCHEMA
from .topo_features import PAPER_BINS, PAPER_FILTRATION

ENGINES = ("original", "compat")
PROFILES = ("paper", "experiment")
EXPECTED_RAW_FEATURES = 2754             # 9 element pairs x 306 for the paper settings
FEATURE_TIMEOUT_S = 900                  # one structure takes ~15-30 s
CONVERT_TIMEOUT_S = 120
# One BLAS thread per worker process: parallelism comes from running structures side by side.
_THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _list_arg(values: list[float]) -> str:
    return "[" + ", ".join(repr(float(v)) for v in values) + "]"


@dataclass
class FeatureSample:
    sample_id: str
    pdb_id: str
    model_id: int
    interface_file: str


# Not needed to run main.pyc: macOS metadata, the README figure (4.7 MB) and the tutorial notebooks.
FEATURE_STAGING_EXCLUDE = (".DS_Store", "image", "*.ipynb")


def stage_feature_repo(job_dir: Path) -> Path:
    """Copy the authors' repository so main.pyc (with src/*.pyc) runs from a directory we own."""
    dst = Path(job_dir) / "feature_repo"
    if dst.exists():
        raise ids.StaleOutputError(f"feature repository staging directory already exists: {dst}")
    shutil.copytree(AUTHORS_REPO, dst, ignore=shutil.ignore_patterns(*FEATURE_STAGING_EXCLUDE))
    return dst


def feature_schema(bins, filtration) -> str:
    """Schema id: the published one only for the exact published settings."""
    if [float(b) for b in bins] == PAPER_BINS and [float(f) for f in filtration] == PAPER_FILTRATION:
        return FEATURE_SCHEMA
    import hashlib
    digest = hashlib.sha256(repr(([float(b) for b in bins], [float(f) for f in filtration])).encode()).hexdigest()
    return f"custom-{digest[:12]}"


def _feature_command(engine: str, s: FeatureSample, saving_path: Path, bins, filtration) -> tuple[list[str], Path]:
    iface = Path(s.interface_file).resolve()
    expected = f"{s.pdb_id}_ranked_{s.model_id}_sp_interface.pdb"
    if iface.name != expected:
        raise ValueError(f"interface file must be named {expected}, got {iface.name}")
    args = ["--pdb_id", s.pdb_id, "--model_id", str(s.model_id), "--bins", _list_arg(bins),
            "--filtration", _list_arg(filtration), "--file_path", str(iface.parent),
            "--saving_path", str(Path(saving_path).resolve())]
    if engine == "original":
        return [str(FEATURE38_PYTHON), "-m", "main", *args], saving_path.parent / "feature_repo"
    if engine == "compat":
        return [sys.executable, "-m", "tdq_adapters.topo_features", *args], PROJECT_ROOT
    raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")


def generate_one(engine: str, s: FeatureSample, job_dir: Path, bins=PAPER_BINS,
                 filtration=PAPER_FILTRATION) -> dict[str, Any]:
    schema = feature_schema(bins, filtration)
    npy_dir, csv_dir = job_dir / "npy", job_dir / "csv"
    npy_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    rec: dict[str, Any] = {"sample_id": s.sample_id, "pdb_id": s.pdb_id, "model_id": s.model_id,
                           "engine": engine, "interface_file": s.interface_file}
    try:
        cmd, cwd = _feature_command(engine, s, npy_dir, bins, filtration)
    except ValueError as exc:
        return {**rec, "status": "failed", "error": str(exc)}
    rec["interface_sha256"] = sha256(s.interface_file)
    rec["command"] = cmd
    env = {**os.environ, **_ENV_OVERRIDES, **_THREAD_ENV}
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              env=env, timeout=FEATURE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        return {**rec, "status": "failed", "error": f"timed out after {FEATURE_TIMEOUT_S} s",
                "stdout": _text(exc.stdout)[-500:]}
    rec["seconds"] = round(time.time() - t0, 2)
    npy = npy_dir / f"feature_{s.pdb_id}_ranked_{s.model_id}_sp_interface.npy"
    if proc.returncode != 0 or not npy.is_file():
        return {**rec, "status": "failed", "error": f"exit {proc.returncode}", "stderr": proc.stderr[-1500:]}

    arr = np.load(npy)
    expected_len = 9 * (len(bins) - 1 + 6 + 7 * len(filtration) + 36)
    rec.update(npy_file=str(npy), npy_sha256=sha256(npy), shape=list(arr.shape))
    if arr.shape != (1, expected_len) or not np.all(np.isfinite(arr)):
        return {**rec, "status": "failed",
                "error": f"expected shape (1, {expected_len}) with finite values, got {arr.shape}, "
                         f"{int((~np.isfinite(arr)).sum())} non-finite"}

    rec["feature_schema"] = schema
    if schema == FEATURE_SCHEMA:                 # published settings, not merely 2754 values
        csv = csv_dir / f"{s.sample_id}.csv"
        conv = subprocess.run([sys.executable, str(AUTHORS_REPO / "03_extract_features_from_npy_to_csv.py"),
                               "--npy_file", str(npy), "--output_file", str(csv)],
                              cwd=job_dir, capture_output=True, text=True, encoding="utf-8", errors="replace",
                              env={**os.environ, **_ENV_OVERRIDES}, timeout=CONVERT_TIMEOUT_S)
        if conv.returncode != 0 or not csv.is_file():
            return {**rec, "status": "failed", "error": "03 conversion failed", "stderr": conv.stderr[-1500:]}
        rec["csv_file"] = str(csv)
    return {**rec, "status": "ok"}


def check_batch(samples: list[FeatureSample], job_dir: Path) -> None:
    """Refuse the whole batch before any work if it is unsafe, ambiguous or would reuse old outputs."""
    for s in samples:
        ids.validate_pdb_id(s.pdb_id)
        ids.validate_model_id(s.model_id)
        ids.validate_pdb_id(s.sample_id)
    ids.check_unique((s.pdb_id, s.model_id) for s in samples)
    ids.check_unique((s.sample_id, 0) for s in samples)
    outs = [ids.contained_path(job_dir / "npy", f"feature_{s.pdb_id}_ranked_{s.model_id}_sp_interface.npy")
            for s in samples] + [ids.contained_path(job_dir / "csv", f"{s.sample_id}.csv") for s in samples]
    if len(set(outs)) != len(outs):               # e.g. case-insensitive file systems
        raise ids.DuplicateOutputError("two samples would write the same output file")
    ids.check_not_existing(outs + [job_dir / "features.csv", job_dir / "feature_repo"])


def generate_features(samples: list[FeatureSample], job_dir: Path, engine: str = "compat", workers: int = 8,
                      bins=PAPER_BINS, filtration=PAPER_FILTRATION, profile: str = "paper") -> dict[str, Any]:
    """Run one engine over many structures and assemble a feature table (pdb_id, af_model_id, features)."""
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; expected one of {ENGINES}")
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    schema = feature_schema(bins, filtration)
    if profile == "paper" and schema != FEATURE_SCHEMA:
        raise ValueError("the paper profile locks --bins and --filtration to the published settings; "
                         "use profile='experiment' for other settings")
    job_dir = Path(job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    check_batch(samples, job_dir)
    if engine == "original":
        stage_feature_repo(job_dir)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda s: generate_one(engine, s, job_dir, bins, filtration), samples))
    ok = [r for r in records if r["status"] == "ok"]
    table = None
    if ok:
        rows = []
        for r in ok:
            if schema == FEATURE_SCHEMA:
                row = pd.read_csv(r["csv_file"])          # authors' 03 column names and order
            else:
                arr = np.load(r["npy_file"])
                row = pd.DataFrame(arr, columns=[f"raw_{i:04d}" for i in range(arr.shape[1])])
            key = pd.DataFrame({"pdb_id": [r["pdb_id"]] * len(row), "af_model_id": [r["model_id"]] * len(row)})
            rows.append(pd.concat([key, row], axis=1))
        table = pd.concat(rows, ignore_index=True)
        table.to_csv(job_dir / "features.csv", index=False)
    summary = {"engine": engine, "profile": profile, "feature_schema": schema,
               "bins": [float(b) for b in bins], "filtration": [float(f) for f in filtration],
               "samples": len(records), "ok": len(ok), "failed": len(records) - len(ok),
               "failures": [{k: r.get(k) for k in ("sample_id", "error")} for r in records if r["status"] != "ok"],
               "seconds": round(time.time() - t0, 1), "workers": workers,
               "feature_table": str(job_dir / "features.csv") if table is not None else None}
    return {"summary": summary, "records": records, "table": table}
