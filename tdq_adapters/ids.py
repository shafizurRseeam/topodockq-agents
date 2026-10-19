"""Identifier and output-path safety for adapter tools.

Complex and model identifiers become file names, and later they may come from
user requests, manifests or LLM tool arguments. They are validated here before
any file is written:
  * pdb_id: 1-64 characters from [A-Za-z0-9_-] (no path separators, no dots, so no
    traversal); model_id: a non-negative integer.
  * every output path must resolve inside its designated directory;
  * a batch may not contain the same output identity twice, and may not reuse an
    output file that already exists (stale results are never mistaken for new ones).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

PDB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class UnsafeIdentifierError(ValueError):
    """An identifier could escape its directory or is not a valid ID."""


class DuplicateOutputError(ValueError):
    """Two samples in a batch would write the same output file."""


class StaleOutputError(ValueError):
    """An output file already exists from an earlier run."""


def validate_pdb_id(pdb_id: object) -> str:
    if not isinstance(pdb_id, str) or not PDB_ID_RE.fullmatch(pdb_id):
        raise UnsafeIdentifierError(f"invalid pdb_id {pdb_id!r}: use 1-64 characters from A-Z, a-z, 0-9, '_' or '-'")
    return pdb_id


def validate_model_id(model_id: object) -> int:
    if isinstance(model_id, bool) or not isinstance(model_id, int) or model_id < 0:
        raise UnsafeIdentifierError(f"invalid model_id {model_id!r}: must be a non-negative integer")
    return model_id


def contained_path(base: Path, filename: str) -> Path:
    """``base / filename``, guaranteed to resolve inside ``base``."""
    base_r = Path(base).resolve()
    out = (base_r / filename).resolve()
    if out.parent != base_r:
        raise UnsafeIdentifierError(f"output {filename!r} would be written outside {base_r}")
    return out


def check_unique(identities: Iterable[tuple[str, int]]) -> None:
    dups = [k for k, n in Counter(identities).items() if n > 1]
    if dups:
        raise DuplicateOutputError(f"{len(dups)} (pdb_id, model_id) identities appear more than once, "
                                   f"e.g. {dups[:5]}; each would overwrite the other's output")


def check_not_existing(paths: Iterable[Path]) -> None:
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing:
        raise StaleOutputError(f"{len(existing)} output files already exist, e.g. {existing[:3]}; "
                               f"use a new output directory")
