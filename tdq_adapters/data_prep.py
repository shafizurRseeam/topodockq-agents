"""Job-level data preparation: the deterministic tools the DataAgent will call.

``prepare_interfaces`` takes a list of samples and, for each one:
  1. validates the input structure (structures.validate_structure, fixed rules);
  2. runs the authors' extract_interface.py unchanged (10 A cutoff by default);
  3. validates the extracted interface;
  4. names it ``{pdb_id}_ranked_{model_id}_sp_interface.pdb``, the file name the
     authors' feature code (main.pyc) builds from --pdb_id and --model_id.
It returns one record per sample plus a summary. Samples with an ERROR are
excluded with the rule code recorded; nothing else is ever dropped.

Before any work starts, the batch is checked: identifiers must be safe file-name
parts (ids.py), every output must stay inside ``out_dir``, no two samples may share
an output identity, and no output file may already exist.

Profiles: "paper" (default) fixes the interface cutoff at 10 A, as in the paper
and the authors' README; "experiment" allows another cutoff and must be chosen
explicitly. Parallelism: a thread pool launches one subprocess of the authors'
script per structure; no LLM is involved per sample.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import author_scripts, ids, structures
from .provenance import sha256

INTERFACE_CUTOFF_A = 10.0   # paper and authors' README: residues within 10 A of the other chain
PROFILES = ("paper", "experiment")


@dataclass
class Sample:
    sample_id: str
    pdb_id: str
    model_id: int
    input_pdb: str
    protein_chain: str = "A"
    peptide_chain: str = "B"
    chains_confirmed: bool = False      # the user confirmed which chain is the protein and which the peptide


def interface_filename(pdb_id: str, model_id: int) -> str:
    """The authors' feature code opens f"{file_path}/{pdb_id}_ranked_{model_id}_sp_interface.pdb"."""
    return f"{ids.validate_pdb_id(pdb_id)}_ranked_{ids.validate_model_id(model_id)}_sp_interface.pdb"


def check_batch(samples: list[Sample], out_dir: Path, profile: str, distance: float) -> list[Path]:
    """Refuse the whole batch before any extraction if it is unsafe or ambiguous."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    if profile == "paper" and distance != INTERFACE_CUTOFF_A:
        raise ValueError(f"the paper profile fixes the interface cutoff at {INTERFACE_CUTOFF_A} A; "
                         f"use profile='experiment' to change it")
    outs = [ids.contained_path(out_dir, interface_filename(s.pdb_id, s.model_id)) for s in samples]
    ids.check_unique((s.pdb_id, s.model_id) for s in samples)
    ids.check_not_existing(outs)
    return outs


def prepare_one(s: Sample, out_dir: Path, distance: float = INTERFACE_CUTOFF_A) -> dict[str, Any]:
    rec: dict[str, Any] = {"sample_id": s.sample_id, "pdb_id": s.pdb_id, "model_id": s.model_id,
                           "input_pdb": s.input_pdb, "cutoff_A": distance,
                           "chain_assignment": {"protein": s.protein_chain, "peptide": s.peptide_chain,
                                                "confirmed": s.chains_confirmed,
                                                "basis": "confirmed by the user" if s.chains_confirmed else
                                                         "not confirmed; chain IDs alone do not prove which chain "
                                                         "is the protein and which the peptide"}}
    v_in = structures.validate_structure(s.input_pdb, s.protein_chain, s.peptide_chain, stage="input",
                                         chains_confirmed=s.chains_confirmed)
    rec["input_sha256"] = sha256(s.input_pdb) if Path(s.input_pdb).is_file() else None
    rec["input_residues_by_chain"] = v_in["stats"].get("residues_by_chain", {})
    rec["issues"] = list(v_in["issues"])
    if not v_in["ok"]:
        return {**rec, "status": "excluded", "interface_file": None}

    out = ids.contained_path(out_dir, interface_filename(s.pdb_id, s.model_id))
    run = author_scripts.extract_interface(Path(s.input_pdb), s.protein_chain, s.peptide_chain, distance, out)
    rec["extraction"] = run
    if run["returncode"] != 0 or run.get("timed_out") or not out.is_file():
        rec["issues"].append(structures.issue("E_EXTRACTION_FAILED", stderr=run["stderr"][-500:]))
        return {**rec, "status": "excluded", "interface_file": None}

    v_out = structures.validate_structure(out, s.protein_chain, s.peptide_chain, stage="interface")
    rec["issues"] += [i for i in v_out["issues"] if i["code"] not in {x["code"] for x in rec["issues"]}]
    rec["interface_stats"] = v_out["stats"]
    rec["interface_file"] = str(out)
    rec["interface_sha256"] = sha256(out)
    rec["status"] = "ready_for_features" if v_out["ok"] else "excluded"
    return rec


def prepare_interfaces(samples: list[Sample], out_dir: Path, workers: int = 8,
                       distance: float = INTERFACE_CUTOFF_A, manifest_path: Path | None = None,
                       profile: str = "paper") -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_batch(samples, out_dir, profile, distance)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(pool.map(lambda s: prepare_one(s, out_dir, distance), samples))
    excluded = [r for r in records if r["status"] == "excluded"]
    warning_counts: dict[str, int] = {}
    for r in records:
        for i in r["issues"]:
            if i["severity"] == "WARNING":
                warning_counts[i["code"]] = warning_counts.get(i["code"], 0) + 1
    summary = {
        "samples": len(records),
        "ready_for_features": len(records) - len(excluded),
        "excluded": len(excluded),
        "exclusions_by_rule": _count_errors(excluded),
        "samples_with_warnings_by_rule": warning_counts,
        "cutoff_A": distance,
        "profile": profile,
        "out_dir": str(out_dir),
    }
    if manifest_path:
        with open(manifest_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, default=str) + "\n")
        summary["manifest"] = str(manifest_path)
    return {"summary": summary, "records": records}


def _count_errors(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        for i in r["issues"]:
            if i["severity"] == "ERROR":
                counts[i["code"]] = counts.get(i["code"], 0) + 1
    return counts
