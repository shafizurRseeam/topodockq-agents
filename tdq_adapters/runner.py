"""Shared run guard for milestone scripts.

Every milestone run, successful or not, ends with:
  * the protected-directory integrity check (fingerprints before vs after), and
  * a structured JSON report with status "passed", "failed_checks" or "error".
An exception inside the milestone body is captured (type, message, traceback)
instead of leaving no record.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from .paths import PROTECTED_DIRS
from .provenance import device_info, diff_fingerprints, library_versions, protected_fingerprints

Body = Callable[[dict, Path], dict]   # (report, run_dir) -> checks {name: bool}


def run_guarded(milestone: str, run_dir: Path, body: Body,
                protected_dirs: tuple[Path, ...] = PROTECTED_DIRS) -> tuple[int, dict[str, Any]]:
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {"milestone": milestone, "run_dir": str(run_dir),
                              "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                              "library_versions": library_versions(), "device": device_info()}
    out = run_dir / f"{milestone}_report.json"
    checks: dict[str, bool] = {}
    before = None
    try:
        print("[guard] fingerprinting protected directories")
        before = protected_fingerprints(protected_dirs)
        checks = dict(body(report, run_dir) or {})
        report["status"] = "passed" if all(checks.values()) else "failed_checks"
    except BaseException as exc:  # noqa: BLE001 - record every failure, including KeyboardInterrupt
        report["status"] = "error"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc),
                             "traceback": traceback.format_exc()}
        print(f"[guard] run failed: {type(exc).__name__}: {exc}")
    finally:
        print("[guard] re-fingerprinting protected directories")
        if before is None:
            report["protected_dirs"] = {"unchanged": None, "note": "fingerprinting failed before the run started"}
            checks["protected_dirs_unchanged"] = False
        else:
            changes = diff_fingerprints(before, protected_fingerprints(protected_dirs))
            report["protected_dirs"] = {"dirs": [str(d) for d in protected_dirs], "unchanged": not changes,
                                        "changes": changes}
            checks["protected_dirs_unchanged"] = not changes
        if report.get("status") == "passed" and not checks["protected_dirs_unchanged"]:
            report["status"] = "failed_checks"
        report["checks"] = checks
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"[guard] status: {report['status']}  checks: {checks}\n[guard] report: {out}")
    return (0 if report["status"] == "passed" else 1), report


def fmt(value: float | None, digits: int = 4) -> str:
    """Format a metric that may be undefined (None) without raising."""
    return "undefined" if value is None else f"{value:.{digits}f}"
