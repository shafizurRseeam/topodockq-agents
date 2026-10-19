"""MLAgent tools: p-DockQ prediction with the published model, and candidate ranking.

Plain functions taking the shared PipelineState first. Prediction uses the saved,
integrity-checked preprocessing package (the authors' column selection and scaler)
and the published checkpoint through the authors' own model class. Ranking is the
paper's top-1 rule with the 0.82 threshold. No training, no LLM here.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pandas as pd

from tdq_adapters import evaluation, pipeline, preprocessing
from tdq_adapters.paths import to_project_relative

from ..state import Keys, PipelineState

AGENT = "MLAgent"
KEY = ["pdb_id", "af_model_id"]


def check_model_package(state: PipelineState) -> dict[str, Any]:
    """Load and verify the preprocessing package and checkpoint (file hashes, columns, scaler)."""
    pkg = preprocessing.load_package()
    m = pkg["manifest"]
    return {"package_id": m["package_id"], "feature_schema": m["feature_schema"],
            "input_features": m["n_feature_columns"], "dropped_columns": m["n_dropped_columns"],
            "scaler_fit_rows": m["rows"]["scaler_fit_rows_train"], "checkpoint": m["checkpoint"]["path"],
            "checkpoint_sha256": m["checkpoint"]["sha256"], "integrity": "verified"}


def predict_pdockq(state: PipelineState, device: str = "cpu") -> dict[str, Any]:
    """Predict p-DockQ for every structure with features; carry every exclusion through with its rule code."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    feats = pd.read_csv(state.require(Keys.FEATURES).path)
    res = preprocessing.predict_frame(feats, device=device)
    scores = dict(zip(zip(feats.pdb_id, feats.af_model_id), res["predictions"]))
    iface = pd.read_csv(state.require(Keys.INTERFACES).path, keep_default_na=False)
    rows = []
    for r in iface.itertuples(index=False):
        k = (r.pdb_id, int(r.af_model_id))
        base = {"sample_id": r.sample_id, "pdb_id": r.pdb_id, "af_model_id": int(r.af_model_id),
                "input_pdb": r.input_pdb, "warnings": r.warnings}
        if r.status != "ready_for_features":
            rows.append({**base, "status": "excluded", "reason": r.errors, "p_dockq": None})
        elif k not in scores:
            rows.append({**base, "status": "excluded", "reason": "E_FEATURES_FAILED", "p_dockq": None})
        else:
            rows.append({**base, "status": "scored", "reason": "", "p_dockq": scores[k]})
    pred = pd.DataFrame(rows).sort_values(KEY).reset_index(drop=True)
    out = state.path_for("predictions.csv")
    pred.to_csv(out, index=False)
    scored = pred[pred.status == "scored"]
    meta = {"structures": int(len(pred)), "scored": int(len(scored)), "excluded": int(len(pred) - len(scored)),
            "exclusions_by_reason": dict(Counter(pred.loc[pred.status == "excluded", "reason"])),
            "p_dockq_min": float(scored.p_dockq.min()) if len(scored) else None,
            "p_dockq_median": float(scored.p_dockq.median()) if len(scored) else None,
            "p_dockq_max": float(scored.p_dockq.max()) if len(scored) else None,
            "device": res["device"], "package_id": preprocessing.PACKAGE_ID}
    state.register(Keys.PREDICTIONS, out, "csv", AGENT, **meta)
    return meta


def rank_candidates(state: PipelineState, strict: bool = False) -> dict[str, Any]:
    """Pick the top-1 candidate per complex by p-DockQ and flag it high quality if p-DockQ > 0.82.

    ``strict`` comes from the run settings, never from the model: in strict mode a complex whose top-1
    candidate has unconfirmed, implausible chain roles gets ranking_status "needs_confirmation".
    """
    pred = pd.read_csv(state.require(Keys.PREDICTIONS).path)        # excluded rows: p_dockq NaN
    pred["reason"] = pred["reason"].fillna("")
    pred["warnings"] = pred["warnings"].fillna("") if "warnings" in pred.columns else ""
    ranking = pipeline.rank_candidates(pred, strict=strict)
    out = state.path_for("ranking.csv")
    ranking.to_csv(out, index=False)
    top = [{"pdb_id": r.pdb_id, "top1": r.top1_sample_id, "p_dockq": round(float(r.top1_p_dockq), 4),
            "predicted_high_quality": bool(r.predicted_high_quality), "warnings": r.top1_warnings,
            "ranking_status": r.ranking_status, "scored": int(r.candidates_scored),
            "excluded": int(r.candidates_excluded)} for r in ranking.itertuples(index=False)]
    unranked = sorted(set(pred.pdb_id) - set(ranking.pdb_id))
    meta = {"complexes_ranked": int(len(ranking)), "complexes_without_scored_candidates": unranked,
            "predicted_high_quality": int(ranking.predicted_high_quality.sum()) if len(ranking) else 0,
            "strict": strict,
            "needs_confirmation": [r["pdb_id"] for r in top if r["ranking_status"] == "needs_confirmation"],
            "top1_with_warnings": [r["pdb_id"] for r in top if r["warnings"]],
            "threshold": evaluation.SCORE_THRESHOLD, "top1": top[:30]}
    state.register(Keys.RANKING, out, "csv", AGENT, **{k: v for k, v in meta.items() if k != "top1"},
                   top1_preview=top[:10])
    return meta


def write_ml_report(state: PipelineState, notes: str = "") -> dict[str, Any]:
    """Write ml.report: model package identity, prediction counts and the ranking summary."""
    report = {"model_package": check_model_package(state), "predictions": state.require(Keys.PREDICTIONS).meta,
              "ranking": state.require(Keys.RANKING).meta, "notes": notes}
    out = state.path_for("ml_report.json")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    state.register(Keys.ML_REPORT, out, "json", AGENT, notes=notes[:300])
    return {"written": to_project_relative(out)}
