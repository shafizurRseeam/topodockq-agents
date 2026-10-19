"""Hashes, library versions, device info, and protected-directory fingerprints."""
from __future__ import annotations

import hashlib
import importlib
import platform
import sys
from pathlib import Path
from typing import Any

LIBRARIES = ("numpy", "pandas", "sklearn", "scipy", "torch", "gudhi", "matplotlib")


def sha256(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def library_versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in LIBRARIES:
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception as exc:  # noqa: BLE001 - record, never fail on provenance
            out[name] = f"unavailable ({type(exc).__name__})"
    return out


def device_info() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {"cuda_available": torch.cuda.is_available(), "torch_cuda": torch.version.cuda}
    if torch.cuda.is_available():
        info["device"] = "cuda"
        info["gpu"] = torch.cuda.get_device_name(0)
    else:
        info["device"] = "cpu"
    return info


def tree_fingerprint(root: Path) -> dict[str, str]:
    """Map every file under ``root`` (relative path) to its sha256."""
    root = Path(root)
    if not root.exists():
        return {}
    return {str(p.relative_to(root)).replace("\\", "/"): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def protected_fingerprints(dirs: tuple[Path, ...]) -> dict[str, dict[str, str]]:
    return {str(d): tree_fingerprint(d) for d in dirs}


def diff_fingerprints(before: dict[str, dict[str, str]], after: dict[str, dict[str, str]]) -> list[str]:
    """Human-readable list of added, removed or changed files. Empty means untouched."""
    changes: list[str] = []
    for d in sorted(set(before) | set(after)):
        b, a = before.get(d, {}), after.get(d, {})
        changes += [f"added   {d}/{f}" for f in sorted(set(a) - set(b))]
        changes += [f"removed {d}/{f}" for f in sorted(set(b) - set(a))]
        changes += [f"changed {d}/{f}" for f in sorted(set(a) & set(b)) if a[f] != b[f]]
    return changes
