"""Run the authors' scripts unchanged, inside a staged working directory.

Each call records the exact command, working directory, return code, duration,
and full stdout/stderr (saved to ``logs/``).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# The authors' scripts print emoji; force UTF-8 so Windows consoles do not crash on them.
_ENV_OVERRIDES = {"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


class ScriptFailed(RuntimeError):
    pass


DEFAULT_SCRIPT_TIMEOUT_S = 1800      # 02/04 load a 1.7 GB table (~40 s); 30 min means something is wrong
EXTRACT_INTERFACE_TIMEOUT_S = 300    # one structure takes ~1-2 s


def run_script(workdir: Path, script: str, args: list[str], log_dir: Path, tag: str,
               timeout: float = DEFAULT_SCRIPT_TIMEOUT_S) -> dict[str, Any]:
    workdir, log_dir = Path(workdir), Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, script, *args]
    env = {**os.environ, **_ENV_OVERRIDES}
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        (log_dir / f"{tag}.stdout.txt").write_text(_text(exc.stdout), encoding="utf-8")
        (log_dir / f"{tag}.stderr.txt").write_text(_text(exc.stderr), encoding="utf-8")
        raise ScriptFailed(f"{script} timed out after {timeout} s") from exc
    record = {
        "tag": tag,
        "command": cmd,
        "cwd": str(workdir),
        "env_overrides": _ENV_OVERRIDES,
        "returncode": proc.returncode,
        "seconds": round(time.time() - t0, 2),
        "stdout_log": str(log_dir / f"{tag}.stdout.txt"),
        "stderr_log": str(log_dir / f"{tag}.stderr.txt"),
    }
    (log_dir / f"{tag}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
    (log_dir / f"{tag}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise ScriptFailed(f"{script} exited with {proc.returncode}:\n{tail}")
    return record


def npy_to_csv(workdir: Path, npy_file: str, output_file: str, log_dir: Path) -> dict[str, Any]:
    """Authors' 03_extract_features_from_npy_to_csv.py with its own --npy_file / --output_file arguments."""
    return run_script(workdir, "03_extract_features_from_npy_to_csv.py",
                      ["--npy_file", npy_file, "--output_file", output_file], log_dir, "03_npy_to_csv")


def extract_interface(pdb: Path, chain1: str, chain2: str, distance: float, out: Path,
                      python: Path | None = None, timeout: float = EXTRACT_INTERFACE_TIMEOUT_S) -> dict[str, Any]:
    """Authors' src/extract_interface.py, unchanged, with its own CLI arguments.

    Runs the script in place from authors_code/ (it only reads its --pdb input and
    writes --out) using the separate ProDy 2.4.1 environment. Never raises on a
    script failure: the return code and output are returned for the caller to record.
    """
    from .paths import AUTHORS_EXTRACT_INTERFACE, PRODY_PYTHON

    python = Path(python or PRODY_PYTHON)
    pdb, out = Path(pdb).resolve(), Path(out).resolve()
    cmd = [str(python), "-W", "ignore", str(AUTHORS_EXTRACT_INTERFACE), "--pdb", str(pdb), "--chain1", chain1,
           "--chain2", chain2, "--distance", str(distance), "--out", str(out)]
    env = {**os.environ, **_ENV_OVERRIDES}
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=Path(out).parent, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return {"command": cmd, "returncode": None, "timed_out": True, "seconds": round(time.time() - t0, 2),
                "stdout": _text(exc.stdout)[-2000:], "stderr": f"timed out after {timeout} s"}
    return {"command": cmd, "returncode": proc.returncode, "timed_out": False,
            "seconds": round(time.time() - t0, 2),
            "stdout": proc.stdout.strip()[-2000:], "stderr": proc.stderr.strip()[-2000:]}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def tutorial_inference(workdir: Path, log_dir: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Authors' 02_tutorial_inference.py. It takes no arguments.

    It reads ./data/processed_data/*.csv and ./models/best_model.pth, predicts the
    validation and test sets, and writes ./val_inference_results.csv and
    ./test_inference_results.csv with columns True_DockQ and Predicted_DockQ(p-DockQ).
    Those files carry no IDs; rows follow the order of the authors' merged table.
    """
    record = run_script(workdir, "02_tutorial_inference.py", [], log_dir, "02_tutorial_inference")
    out = {split: pd.read_csv(Path(workdir) / f"{split}_inference_results.csv") for split in ("val", "test")}
    record["results_csv"] = {split: str(Path(workdir) / f"{split}_inference_results.csv") for split in out}
    return record, out


def infer_from_generated_csv(workdir: Path, log_dir: Path) -> tuple[dict[str, Any], list[float]]:
    """Authors' 04_inference_from_generated_csv.py. It takes no arguments.

    It reads ./feature/example_features.csv, ./data/processed_data/*.csv and
    ./models/best_model.pth, and writes ./example_inference_results.csv.
    """
    record = run_script(workdir, "04_inference_from_generated_csv.py", [], log_dir, "04_inference")
    results = Path(workdir) / "example_inference_results.csv"
    preds = pd.read_csv(results)["Predicted_DockQ(p-DockQ)"].astype(float).tolist()
    record["results_csv"] = str(results)
    return record, preds
