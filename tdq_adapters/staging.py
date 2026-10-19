"""Build a per-run working directory laid out the way the authors' scripts expect.

The authors' scripts use paths relative to the repository root, for example
``./data/processed_data/...``, ``./models/best_model.pth`` and
``./feature/example_features.csv``, and import ``src.model``. Instead of editing
those paths, we copy the repository into ``<run>/workdir`` and place the
downloaded files where the scripts look for them. Large files are hard-linked
(no copy, no admin rights needed on NTFS); the scripts only read them, and the
caller verifies afterwards that the originals are unchanged.
"""
from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from .paths import AUTHORS_REPO, PROCESSED_DATA_DIR, PROCESSED_FILES, TRAINED_MODEL

# Files left out of the staged copy (the originals in authors_code/ are untouched):
#   .DS_Store      macOS folder metadata.
#   __init__.pyc   src/__init__.pyc is Python 3.8 bytecode with no matching .py.
#                  Python 3.13 tries to load it for `from src.model import *` and
#                  fails with "bad magic number". Its only content is a module
#                  docstring (author/date header), so omitting it changes no
#                  behaviour: `src` becomes a namespace package and src/model.py
#                  and src/train.py import exactly as written.
EXCLUDE_PATTERNS: tuple[str, ...] = (".DS_Store", "__init__.pyc")


def _readonly(path: Path) -> None:
    os.chmod(path, stat.S_IREAD)


def protect_originals() -> list[str]:
    """Mark the downloaded data files and checkpoint read-only.

    A hard link shares one file with the original, so a script writing to a staged
    link would change the original. Read-only blocks that write at the OS level.
    This changes only the read-only attribute, never the content.
    """
    targets = [PROCESSED_DATA_DIR / n for n in PROCESSED_FILES] + [TRAINED_MODEL]
    for t in targets:
        _readonly(t)
    return [str(t) for t in targets]


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        os.chmod(dst, stat.S_IWRITE)
        dst.unlink()
    try:
        os.link(src, dst)
        mode = "hardlink (read-only)"
    except OSError:
        shutil.copy2(src, dst)
        mode = "copy (read-only)"
    _readonly(dst)
    return mode


def remove_staged_workdir(workdir: Path) -> None:
    """Delete a staged working directory, including its read-only hard links.

    Clearing read-only on a hard link clears it on the original too, so the
    originals are re-protected afterwards.
    """
    def _onerror(func, path, _exc):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    shutil.rmtree(workdir, onexc=lambda f, p, e: _onerror(f, p, e))
    protect_originals()


def stage_author_repo(workdir: Path, exclude: tuple[str, ...] = EXCLUDE_PATTERNS) -> dict[str, str]:
    """Copy the authors' repository to ``workdir`` and place data + checkpoint inside it.

    Returns a mapping of staged path -> how it was placed, for the run record.
    """
    workdir = Path(workdir)
    if workdir.exists():
        raise FileExistsError(f"staging directory already exists: {workdir}")
    protected = protect_originals()
    shutil.copytree(AUTHORS_REPO, workdir, ignore=shutil.ignore_patterns(*exclude))
    placed: dict[str, str] = {"<repo>": f"copytree from {AUTHORS_REPO} excluding {list(exclude)}",
                              "<originals marked read-only>": ", ".join(protected)}
    for name in PROCESSED_FILES:
        rel = Path("data") / "processed_data" / name
        placed[str(rel)] = _link_or_copy(PROCESSED_DATA_DIR / name, workdir / rel)
    rel = Path("models") / "best_model.pth"
    placed[str(rel)] = _link_or_copy(TRAINED_MODEL, workdir / rel)
    return placed
