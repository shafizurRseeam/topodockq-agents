"""Input discovery: turn a user-supplied source into a validated list of structures (no LLM).

Supported layouts:
  manifest_csv  A CSV with columns pdb_id, af_model_id (or model_id) and input_pdb; optional
                sample_id, protein_chain, peptide_chain. Relative input_pdb paths resolve
                against the CSV's folder.
  complex_dirs  <source>/<COMPLEX>/ranked_<N>.pdb or ranked_<N>_sp.pdb: AlphaFold output, one
                folder per complex, as in the authors' Zenodo archives. When both exist for a
                model, the superposed ranked_<N>_sp.pdb is used, as the authors did. Other files
                (interfaces, DockQ inputs) are ignored and counted.
  complex_dir   One such complex folder on its own (<source>/ranked_<N>[_sp].pdb); the folder
                name is the complex ID.
  flat_files    Every *.pdb directly in <source>. Names {pdb_id}_ranked_{N}*.pdb are parsed;
                any other name becomes pdb_id = its sanitized stem, model 0.

Selection (complexes, max_complexes, models, max_models_per_complex) only narrows what the
user asked for. A listed structure whose file is missing is kept, so the fixed validation
rules exclude it with a recorded reason instead of it silently disappearing.

Chain roles count as confirmed for a manifest row that states protein_chain and
peptide_chain explicitly, or for every structure when the caller passes
chains_confirmed=True because the user's request states the roles.
"""
from __future__ import annotations

import csv
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import ids
from .data_prep import Sample
from .paths import from_project_relative, to_project_relative

LAYOUTS = ("auto", "manifest_csv", "complex_dirs", "complex_dir", "flat_files")
_RANKED_RE = re.compile(r"^ranked_(\d+)(_sp)?\.pdb$", re.IGNORECASE)
_FLAT_RE = re.compile(r"^(?P<pdb>[A-Za-z0-9_-]+?)_ranked_(?P<model>\d+)(?:_[A-Za-z0-9_]*)?\.pdb$", re.IGNORECASE)
MANIFEST_COLUMNS = ("sample_id", "pdb_id", "af_model_id", "input_pdb", "protein_chain", "peptide_chain",
                    "chains_confirmed")


class InputError(ValueError):
    """The input source cannot be turned into a structure list."""


def _sanitize(stem: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_")[:64]
    if not s:
        raise InputError(f"cannot derive a valid pdb_id from file name {stem!r}")
    return s


def detect_layout(source: Path) -> str:
    if source.is_file():
        if source.suffix.lower() == ".csv":
            return "manifest_csv"
        raise InputError(f"unsupported input file {source.name}: expected a .csv manifest or a folder")
    if not source.is_dir():
        raise InputError(f"input source does not exist: {source}")
    if any(p.is_file() and _RANKED_RE.match(p.name) for p in source.iterdir()):
        return "complex_dir"
    if any(p.is_dir() and any(_RANKED_RE.match(f.name) for f in p.iterdir()) for p in source.iterdir()):
        return "complex_dirs"
    if any(p.suffix.lower() == ".pdb" for p in source.iterdir() if p.is_file()):
        return "flat_files"
    raise InputError(f"no PDB files found in {source} (looked for <complex>/ranked_<N>[_sp].pdb and *.pdb)")


def _scan_complex_dir(cdir: Path) -> tuple[list[dict[str, Any]], int, int, int]:
    """Models of one complex folder; prefers ranked_<N>_sp.pdb over ranked_<N>.pdb."""
    rows, ignored, used_sp, used_raw = [], 0, 0, 0
    variants: dict[int, dict[bool, Path]] = {}
    for f in cdir.iterdir():
        m = _RANKED_RE.match(f.name)
        if not m:
            ignored += f.is_file()
            continue
        variants.setdefault(int(m.group(1)), {})[bool(m.group(2))] = f
    for model_id, v in sorted(variants.items()):
        chosen = v.get(True) or v[False]
        used_sp += chosen is v.get(True)
        used_raw += chosen is not v.get(True)
        rows.append({"pdb_id": _sanitize(cdir.name).lower(), "complex_name": cdir.name, "af_model_id": model_id,
                     "input_pdb": chosen})
    return rows, ignored, used_sp, used_raw


def _from_complex_dirs(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, ignored, used_sp, used_raw = [], 0, 0, 0
    for cdir in sorted(p for p in source.iterdir() if p.is_dir()):
        r, i, s, w = _scan_complex_dir(cdir)
        rows += r
        ignored, used_sp, used_raw = ignored + i, used_sp + s, used_raw + w
    return rows, {"files_ignored": ignored, "superposed_sp_files_used": used_sp, "raw_files_used": used_raw}


def _from_complex_dir(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, ignored, used_sp, used_raw = _scan_complex_dir(source)
    return rows, {"files_ignored": ignored, "superposed_sp_files_used": used_sp, "raw_files_used": used_raw}


def _from_flat_files(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for f in sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() == ".pdb"):
        m = _FLAT_RE.match(f.name)
        if m:
            rows.append({"pdb_id": m["pdb"], "complex_name": m["pdb"], "af_model_id": int(m["model"]), "input_pdb": f})
        else:
            pid = _sanitize(f.stem)
            rows.append({"pdb_id": pid, "complex_name": f.stem, "af_model_id": 0, "input_pdb": f})
    return rows, {}


def _from_manifest_csv(source: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with open(source, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        cols = set(reader.fieldnames or [])
        model_col = "af_model_id" if "af_model_id" in cols else "model_id" if "model_id" in cols else None
        missing = [c for c in ("pdb_id", "input_pdb") if c not in cols] + ([] if model_col else ["af_model_id"])
        if missing:
            raise InputError(f"manifest {source.name} lacks columns {missing}")
        rows = []
        for i, r in enumerate(reader, start=2):
            try:
                model_id = int(str(r[model_col]).strip())
            except ValueError:
                raise InputError(f"manifest line {i}: model id {r[model_col]!r} is not an integer") from None
            p = Path(str(r["input_pdb"]).strip())
            rows.append({"pdb_id": str(r["pdb_id"]).strip(), "complex_name": str(r["pdb_id"]).strip(),
                         "af_model_id": model_id, "input_pdb": p if p.is_absolute() else source.parent / p,
                         "sample_id": (r.get("sample_id") or "").strip() or None,
                         "protein_chain": (r.get("protein_chain") or "").strip() or None,
                         "peptide_chain": (r.get("peptide_chain") or "").strip() or None})
    return rows, {}


def discover(source: str | Path, layout: str = "auto", complexes: list[str] | None = None,
             max_complexes: int | None = None, models: list[int] | None = None,
             max_models_per_complex: int | None = None, protein_chain: str = "A",
             peptide_chain: str = "B", chains_confirmed: bool = False) -> tuple[list[Sample], dict[str, Any]]:
    """Return validated samples and a discovery report. Raises InputError on unusable input."""
    if layout not in LAYOUTS:
        raise InputError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")
    src = from_project_relative(str(source))
    layout_used = detect_layout(src) if layout == "auto" else layout
    if layout_used == "manifest_csv" and not src.is_file():
        raise InputError(f"manifest not found: {src}")
    if layout_used != "manifest_csv" and not src.is_dir():
        raise InputError(f"input folder not found: {src}")
    rows, extra = {"manifest_csv": _from_manifest_csv, "complex_dirs": _from_complex_dirs,
                   "complex_dir": _from_complex_dir, "flat_files": _from_flat_files}[layout_used](src)
    found_complexes = sorted({r["complex_name"] for r in rows})

    if complexes:
        wanted = {c.lower() for c in complexes}
        unknown = sorted(c for c in complexes if c.lower() not in {r["complex_name"].lower() for r in rows}
                         and c.lower() not in {r["pdb_id"].lower() for r in rows})
        if unknown:
            raise InputError(f"requested complexes not found in {src}: {unknown}; available e.g. {found_complexes[:10]}")
        rows = [r for r in rows if r["complex_name"].lower() in wanted or r["pdb_id"].lower() in wanted]
    if max_complexes is not None:
        if max_complexes < 1:
            raise InputError("max_complexes must be at least 1")
        keep = sorted({r["pdb_id"] for r in rows})[:max_complexes]
        rows = [r for r in rows if r["pdb_id"] in keep]
    if models is not None:
        if any((not isinstance(m, int)) or isinstance(m, bool) or m < 0 for m in models):
            raise InputError(f"models must be non-negative integers, got {models}")
        rows = [r for r in rows if r["af_model_id"] in set(models)]
    if max_models_per_complex is not None:
        if max_models_per_complex < 1:
            raise InputError("max_models_per_complex must be at least 1")
        by: dict[str, list[dict[str, Any]]] = {}
        for r in sorted(rows, key=lambda r: (r["pdb_id"], r["af_model_id"])):
            by.setdefault(r["pdb_id"], []).append(r)
        rows = [r for group in by.values() for r in group[:max_models_per_complex]]
    if not rows:
        raise InputError(f"the selection matched no structures in {src}")

    samples = []
    for r in sorted(rows, key=lambda r: (r["pdb_id"], r["af_model_id"])):
        pdb_id = ids.validate_pdb_id(r["pdb_id"])
        model_id = ids.validate_model_id(int(r["af_model_id"]))
        sample_id = ids.validate_pdb_id(r.get("sample_id") or f"{pdb_id}_ranked_{model_id}")
        row_declares_chains = bool(r.get("protein_chain") and r.get("peptide_chain"))
        samples.append(Sample(sample_id=sample_id, pdb_id=pdb_id, model_id=model_id, input_pdb=str(r["input_pdb"]),
                              protein_chain=r.get("protein_chain") or protein_chain,
                              peptide_chain=r.get("peptide_chain") or peptide_chain,
                              chains_confirmed=chains_confirmed or row_declares_chains))
    ids.check_unique((s.pdb_id, s.model_id) for s in samples)
    ids.check_unique((s.sample_id, 0) for s in samples)
    report = {"source": to_project_relative(src), "layout": layout_used, "complexes_found": len(found_complexes),
              "structures_selected": len(samples), "complexes_selected": len({s.pdb_id for s in samples}),
              "selection": {"complexes": complexes, "max_complexes": max_complexes, "models": models,
                            "max_models_per_complex": max_models_per_complex},
              "chains_confirmed": sum(s.chains_confirmed for s in samples),
              "missing_files": sum(1 for s in samples if not Path(s.input_pdb).is_file()), **extra}
    return samples, report


def write_manifest(samples: list[Sample], path: Path) -> Path:
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        for s in samples:
            d = asdict(s)
            w.writerow({"sample_id": d["sample_id"], "pdb_id": d["pdb_id"], "af_model_id": d["model_id"],
                        "input_pdb": to_project_relative(d["input_pdb"]), "protein_chain": d["protein_chain"],
                        "peptide_chain": d["peptide_chain"], "chains_confirmed": int(d["chains_confirmed"])})
    return path


def read_manifest(path: Path) -> list[Sample]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [Sample(sample_id=r["sample_id"], pdb_id=r["pdb_id"], model_id=int(r["af_model_id"]),
                       input_pdb=str(from_project_relative(r["input_pdb"])), protein_chain=r["protein_chain"],
                       peptide_chain=r["peptide_chain"],
                       chains_confirmed=str(r.get("chains_confirmed", "0")).strip() in ("1", "True", "true"))
                for r in csv.DictReader(fh)]
