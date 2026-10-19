"""Independent verification of an agent run against the deterministic pipeline.

After the agents finish, the structure list the DataAgent built is run through
``tdq_adapters.pipeline.score_structures`` (no LLM, compat feature engine, CPU) into
``runs/<id>/verification/`` and compared with the agents' outputs.

1. Predictions (compare_predictions). Before any join, both tables must have unique
   (pdb_id, af_model_id) keys, the same number of rows and exactly the same structures.
   Then every structure must have the same status, the same exclusion rule codes and the
   same validation warnings, every scored structure a finite p-DockQ, and the two
   p-DockQ values may differ by at most 1e-6.
2. Ranking (compare_rankings). Each ranking must have exactly one row per complex that
   has at least one scored structure, and both must agree on every complex's top-1
   candidate and ranking status (strict mode).
3. Input selection (audit_selection). The manifest must equal a fresh deterministic
   re-discovery of the input with the selection the DataAgent recorded, every manifest
   structure must appear in the predictions, and, when the user stated how many
   structures to expect (--expect-structures), the count must match. The audit also
   reports, per complex, how many models were available and which were selected.

What this cannot do: decide from free text whether the selection matches the user's
intent. It makes the selection explicit and checkable instead; the MainAgent is asked
to confirm it against the request, and --expect-structures makes it machine-checked.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tdq_adapters import inputs, pipeline

from .config import Settings
from .state import Keys, PipelineState

TOL = 1e-6
KEY = ["pdb_id", "af_model_id"]


def _codes(value: Any) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        return ()
    return tuple(sorted(part.split(":")[0].strip() for part in value.split(";") if part.strip()))


def _keys(t: pd.DataFrame) -> list[tuple[str, int]]:
    return sorted(zip(t["pdb_id"].astype(str), t["af_model_id"].astype(int)))


def _ranges(ids: list[int]) -> str:
    """[0, 1, 2, 5] -> "0-2, 5"."""
    ids = sorted(set(ids))
    out, start = [], None
    for i, v in enumerate(ids):
        if start is None:
            start = v
        if i == len(ids) - 1 or ids[i + 1] != v + 1:
            out.append(f"{start}-{v}" if v != start else f"{v}")
            start = None
    return ", ".join(out)


def compare_predictions(agent: pd.DataFrame, det: pd.DataFrame) -> tuple[dict[str, bool], dict[str, Any]]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    for name, t in (("agent", agent), ("deterministic", det)):
        dup = int(t.duplicated(KEY).sum())
        checks[f"unique_keys_{name}"] = dup == 0
        details[f"duplicate_keys_{name}"] = dup
        scored = pd.to_numeric(t.loc[t["status"] == "scored", "p_dockq"], errors="coerce")
        checks[f"finite_pdockq_{name}"] = bool(np.isfinite(scored).all())
    checks["same_row_count"] = len(agent) == len(det)
    checks["same_structures"] = _keys(agent) == _keys(det)        # multiset equality: duplicates count
    if not all(checks.values()):
        details["note"] = "rows not compared: keys are duplicated or the structure lists differ"
        return checks, details
    m = agent.merge(det, on=KEY, suffixes=("_agent", "_det"), validate="one_to_one")
    status_diff = m[m.status_agent != m.status_det]
    reason_diff = m[[_codes(x) != _codes(y) for x, y in zip(m.reason_agent, m.reason_det)]]
    warn_a = m["warnings_agent"] if "warnings_agent" in m else pd.Series([""] * len(m))
    warn_d = m["warnings_det"] if "warnings_det" in m else pd.Series([""] * len(m))
    warn_diff = m[[_codes(x) != _codes(y) for x, y in zip(warn_a, warn_d)]]
    both = m[(m.status_agent == "scored") & (m.status_det == "scored")]
    max_diff = float(np.abs(both.p_dockq_agent.astype(float) - both.p_dockq_det.astype(float)).max()) if len(both) else 0.0
    checks.update({"same_status": status_diff.empty, "same_exclusion_rules": reason_diff.empty,
                   "same_warnings": warn_diff.empty, "pdockq_within_tolerance": max_diff <= TOL})
    details.update({"structures": int(len(m)), "scored_both": int(len(both)), "max_abs_pdockq_diff": max_diff,
                    "status_differences": status_diff[KEY + ["status_agent", "status_det"]].head(10).to_dict("records"),
                    "reason_differences": reason_diff[KEY + ["reason_agent", "reason_det"]].head(10).to_dict("records")})
    return checks, details


def compare_rankings(agent_rank: pd.DataFrame, det_rank: pd.DataFrame,
                     predictions: pd.DataFrame) -> tuple[dict[str, bool], dict[str, Any]]:
    expected = set(predictions.loc[predictions["status"] == "scored", "pdb_id"].astype(str))
    checks: dict[str, bool] = {}
    for name, r in (("agent", agent_rank), ("deterministic", det_rank)):
        ids = list(r["pdb_id"].astype(str))
        checks[f"one_ranking_row_per_scored_complex_{name}"] = len(ids) == len(set(ids)) and set(ids) == expected
    details: dict[str, Any] = {"complexes_with_scored_candidates": len(expected)}
    if not all(checks.values()):
        details["note"] = "rankings not compared: missing, extra or duplicated complexes"
        return checks, details
    m = agent_rank.merge(det_rank, on="pdb_id", suffixes=("_agent", "_det"), validate="one_to_one")
    top_diff = m[m.top1_af_model_id_agent.astype(int) != m.top1_af_model_id_det.astype(int)]
    status_a = m["ranking_status_agent"] if "ranking_status_agent" in m else pd.Series(["accepted"] * len(m))
    status_d = m["ranking_status_det"] if "ranking_status_det" in m else pd.Series(["accepted"] * len(m))
    checks["same_top1_per_complex"] = top_diff.empty
    checks["same_ranking_status"] = bool((status_a.values == status_d.values).all())
    details["top1_differences"] = top_diff[["pdb_id", "top1_af_model_id_agent", "top1_af_model_id_det"]].head(10).to_dict("records")
    return checks, details


def audit_selection(state: PipelineState, predictions: pd.DataFrame,
                    expected_structures: int | None) -> tuple[dict[str, bool], dict[str, Any]]:
    meta = state.require(Keys.STRUCTURES).meta
    manifest = inputs.read_manifest(Path(state.require(Keys.STRUCTURES).path))
    sel = meta.get("selection") or {}
    redo, _ = inputs.discover(meta["source"], layout=meta["layout"], complexes=sel.get("complexes"),
                              max_complexes=sel.get("max_complexes"), models=sel.get("models"),
                              max_models_per_complex=sel.get("max_models_per_complex"),
                              protein_chain=meta.get("protein_chain", "A"), peptide_chain=meta.get("peptide_chain", "B"),
                              chains_confirmed=bool(meta.get("chains_confirmed_by_request", False)))
    ident = lambda s: (s.sample_id, s.pdb_id, s.model_id, str(Path(s.input_pdb).resolve()),  # noqa: E731
                       s.protein_chain, s.peptide_chain, s.chains_confirmed)
    manifest_keys = sorted((s.pdb_id, s.model_id) for s in manifest)
    checks = {
        "manifest_unique": len(manifest_keys) == len(set(manifest_keys)),
        "manifest_matches_rediscovery": sorted(map(ident, manifest)) == sorted(map(ident, redo)),
        "predictions_cover_manifest": _keys(predictions) == manifest_keys,
    }
    if expected_structures is not None:
        checks["expected_structure_count"] = len(manifest) == expected_structures
    selected: dict[str, list[int]] = {}
    for s in manifest:
        selected.setdefault(s.pdb_id, []).append(s.model_id)
    available: dict[str, int] = {}
    if meta["layout"] != "manifest_csv":
        everything, _ = inputs.discover(meta["source"], layout=meta["layout"], complexes=sorted(selected))
        for s in everything:
            available[s.pdb_id] = available.get(s.pdb_id, 0) + 1
    details = {"source": meta["source"], "layout": meta["layout"], "selection": sel,
               "structures_selected": len(manifest), "complexes_selected": len(selected),
               "complexes_found_in_source": meta.get("complexes_found"), "expected_structures": expected_structures,
               "per_complex": [{"pdb_id": p, "available": available.get(p, len(m)), "selected": len(m),
                                "selected_ids": _ranges(m)} for p, m in sorted(selected.items())]}
    return checks, details


def verify_against_deterministic(state: PipelineState, settings: Settings) -> dict[str, Any]:
    if not (state.has(Keys.STRUCTURES) and state.has(Keys.PREDICTIONS) and state.has(Keys.RANKING)):
        result = {"status": "skipped", "reason": "the agents did not produce structures, predictions and ranking"}
        state.record("verification", None, **result)
        return result
    t0 = time.time()
    samples = inputs.read_manifest(Path(state.require(Keys.STRUCTURES).path))
    det = pipeline.score_structures(samples, state.run_dir / "verification", engine="compat",
                                    workers=settings.workers, device="cpu", strict=settings.strict)
    agent_pred = pd.read_csv(state.require(Keys.PREDICTIONS).path, keep_default_na=False,
                             converters={"p_dockq": lambda v: float(v) if v not in ("", None) else np.nan})
    det_pred = det["predictions"].copy()
    for col in ("reason", "warnings"):
        if col in det_pred:
            det_pred[col] = det_pred[col].fillna("")
    agent_rank = pd.read_csv(state.require(Keys.RANKING).path, keep_default_na=False)
    pred_checks, pred_details = compare_predictions(agent_pred, det_pred)
    rank_checks, rank_details = compare_rankings(agent_rank, det["ranking"], agent_pred)
    sel_checks, sel_details = audit_selection(state, agent_pred, settings.expected_structures)
    checks = {**pred_checks, **rank_checks, **sel_checks}
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "structures": pred_details.get("structures"), "scored_both": pred_details.get("scored_both"),
              "max_abs_pdockq_diff": pred_details.get("max_abs_pdockq_diff"), "tolerance": TOL,
              "predictions": pred_details, "ranking": rank_details, "selection": sel_details,
              "strict": settings.strict, "deterministic_run": str(state.run_dir / "verification"),
              "seconds": round(time.time() - t0, 1)}
    (state.run_dir / "verification.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    state.record("verification", None, **result)
    return result
