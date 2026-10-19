"""DataAgent tools: input discovery, structure validation and interface extraction.

Plain functions taking the shared PipelineState first. They call the verified,
deterministic code in ``tdq_adapters`` (inputs, structures, data_prep: the authors'
extract_interface.py, 10 A, paper profile) and register artifacts. No LLM here.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tdq_adapters import data_prep, inputs, structures
from tdq_adapters.paths import to_project_relative

from ..state import Keys, PipelineState

AGENT = "DataAgent"
INTERFACE_COLUMNS = ("sample_id", "pdb_id", "af_model_id", "input_pdb", "status", "interface_file", "errors",
                     "warnings")


def next_attempt(state: PipelineState, prefix: str) -> int:
    return 1 + sum(1 for p in state.artifacts_dir.glob(f"{prefix}_a*") if p.is_dir())


def build_structure_manifest(state: PipelineState, source: str, layout: str = "auto",
                             complexes: list[str] | None = None, max_complexes: int | None = None,
                             models: list[int] | None = None, max_models_per_complex: int | None = None,
                             protein_chain: str = "A", peptide_chain: str = "B",
                             chains_confirmed: bool = False) -> dict[str, Any]:
    """Discover the input structures, validate their IDs, and register data.structures."""
    samples, report = inputs.discover(source, layout=layout, complexes=complexes, max_complexes=max_complexes,
                                      models=models, max_models_per_complex=max_models_per_complex,
                                      protein_chain=protein_chain, peptide_chain=peptide_chain,
                                      chains_confirmed=chains_confirmed)
    out = inputs.write_manifest(samples, state.path_for("structures.csv"))
    per_complex: dict[str, int] = {}
    for s in samples:
        per_complex[s.pdb_id] = per_complex.get(s.pdb_id, 0) + 1
    meta = {**report, "protein_chain": protein_chain, "peptide_chain": peptide_chain,
            "chains_confirmed_by_request": chains_confirmed,
            "structures_per_complex": dict(sorted(per_complex.items())[:30]),
            "complexes": sorted({s.pdb_id for s in samples})[:30],
            "first_samples": [s.sample_id for s in samples[:3]]}
    state.register(Keys.STRUCTURES, out, "csv", AGENT, **meta)
    return meta


def check_structures(state: PipelineState) -> dict[str, Any]:
    """Preflight: apply the fixed validation rules to every input structure (no extraction, no exclusion)."""
    samples = inputs.read_manifest(Path(state.require(Keys.STRUCTURES).path))
    errors, warnings = Counter(), Counter()
    examples: dict[str, list[str]] = {}
    failing = 0
    for s in samples:
        v = structures.validate_structure(s.input_pdb, s.protein_chain, s.peptide_chain, stage="input")
        failing += not v["ok"]
        for i in v["issues"]:
            (errors if i["severity"] == "ERROR" else warnings)[i["code"]] += 1
            examples.setdefault(i["code"], [])
            if len(examples[i["code"]]) < 3:
                examples[i["code"]].append(s.sample_id)
    seen = set(errors) | set(warnings)
    return {"structures": len(samples), "would_be_excluded": failing, "errors_by_rule": dict(errors),
            "warnings_by_rule": dict(warnings), "examples": examples,
            "rule_meanings": {c: structures.RULES[c][1] for c in sorted(seen)}}


def prepare_interfaces(state: PipelineState, workers: int = 8) -> dict[str, Any]:
    """Run validation + the authors' interface extraction on every structure; register data.interfaces."""
    samples = inputs.read_manifest(Path(state.require(Keys.STRUCTURES).path))
    n = next_attempt(state, "interfaces")
    out_dir = state.artifacts_dir / f"interfaces_a{n}"
    job = data_prep.prepare_interfaces(samples, out_dir, workers=workers, profile="paper",
                                       manifest_path=state.artifacts_dir / f"interfaces_a{n}_records.jsonl")
    table = state.path_for(f"interfaces_a{n}.csv")
    excluded_examples = []
    with open(table, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=INTERFACE_COLUMNS)
        w.writeheader()
        for r in job["records"]:
            errs = [i["code"] for i in r["issues"] if i["severity"] == "ERROR"]
            warns = [i["code"] for i in r["issues"] if i["severity"] == "WARNING"]
            w.writerow({"sample_id": r["sample_id"], "pdb_id": r["pdb_id"], "af_model_id": r["model_id"],
                        "input_pdb": to_project_relative(r["input_pdb"]), "status": r["status"],
                        "interface_file": to_project_relative(r["interface_file"]) if r.get("interface_file") else "",
                        "errors": ";".join(errs), "warnings": ";".join(warns)})
            if r["status"] != "ready_for_features" and len(excluded_examples) < 10:
                excluded_examples.append({"sample_id": r["sample_id"], "rules": errs})
    s = job["summary"]
    meta = {"attempt": n, "structures": s["samples"], "ready_for_features": s["ready_for_features"],
            "excluded": s["excluded"], "exclusions_by_rule": s["exclusions_by_rule"],
            "samples_with_warnings_by_rule": s["samples_with_warnings_by_rule"], "cutoff_A": s["cutoff_A"],
            "profile": s["profile"], "excluded_examples": excluded_examples,
            "interface_dir": to_project_relative(out_dir)}
    state.register(Keys.INTERFACES, table, "csv", AGENT, **meta)
    return meta


def write_data_report(state: PipelineState, notes: str = "") -> dict[str, Any]:
    """Write data.report: what was discovered, what is ready, what was excluded and why."""
    report = {"structures": state.require(Keys.STRUCTURES).meta, "interfaces": state.require(Keys.INTERFACES).meta,
              "notes": notes}
    out = state.path_for("data_report.json")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    state.register(Keys.DATA_REPORT, out, "json", AGENT, notes=notes[:300])
    return {"written": to_project_relative(out), "ready_for_features": report["interfaces"]["ready_for_features"],
            "excluded": report["interfaces"]["excluded"]}
