"""Deterministic TopoDockQ inference pipeline: full structures in, ranked p-DockQ out.

This is the independent, LLM-free execution path. The agents call the same
verified tools stage by stage; their results are checked against this function.

    full PDB  --data_prep.prepare_interfaces-->  interface PDB      (authors' extract_interface.py)
              --feature_gen.generate_features-->  2754 raw features  (authors' main.pyc or compat)
              --preprocessing.predict_frame-->    p-DockQ            (published scaler + checkpoint)
              --rank_candidates-->                top-1 per complex  (paper rule R1, threshold R2)

Excluded structures (fixed rules in structures.py) stay in the output with their
rule codes; they are never silently dropped.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from . import data_prep, evaluation, feature_gen, preprocessing
from .paths import to_project_relative

KEY = ["pdb_id", "af_model_id"]


RANKING_COLUMNS = ["pdb_id", "top1_sample_id", "top1_af_model_id", "top1_p_dockq", "predicted_high_quality",
                   "top1_warnings", "ranking_status", "candidates_scored", "candidates_excluded"]
NEEDS_CONFIRMATION_WARNINGS = {"W_CHAIN_ROLE_SUSPICIOUS"}     # strict mode: such a top-1 needs confirmation


def _codes(issues: list[dict[str, Any]], severity: str) -> str:
    return ";".join(i["code"] for i in issues if i["severity"] == severity)


def rank_candidates(pred: pd.DataFrame, score_col: str = "p_dockq", strict: bool = False) -> pd.DataFrame:
    """Per complex: the top-1 candidate by p-DockQ (paper R1) and whether it is predicted high quality (R2).

    ``top1_warnings`` lists the validation warnings of the top-1 candidate. ``ranking_status`` is
    "accepted", or in strict mode "needs_confirmation" when the top-1 candidate carries a warning
    that questions its input representation (unconfirmed, implausible chain roles).
    """
    scored = pred[pred["status"] == "scored"]
    if scored.empty:
        return pd.DataFrame(columns=RANKING_COLUMNS)
    top, _ = evaluation.select_top1(scored, score_col)
    counts = pred.groupby("pdb_id")["status"].value_counts().unstack(fill_value=0)
    warnings = (top["warnings"].fillna("").astype(str) if "warnings" in top.columns
                else pd.Series([""] * len(top), index=top.index))
    needs = warnings.map(lambda w: bool(NEEDS_CONFIRMATION_WARNINGS & set(filter(None, w.split(";")))))
    out = pd.DataFrame({
        "pdb_id": top["pdb_id"].values,
        "top1_sample_id": top["sample_id"].values,
        "top1_af_model_id": top["af_model_id"].values,
        "top1_p_dockq": top[score_col].values,
        "predicted_high_quality": (top[score_col] > evaluation.SCORE_THRESHOLD).values,
        "top1_warnings": warnings.values,
        "ranking_status": ["needs_confirmation" if (strict and n) else "accepted" for n in needs],
    })
    out["candidates_scored"] = out["pdb_id"].map(counts.get("scored", pd.Series(dtype=int))).fillna(0).astype(int)
    out["candidates_excluded"] = out["pdb_id"].map(counts.get("excluded", pd.Series(dtype=int))).fillna(0).astype(int)
    return out[RANKING_COLUMNS].sort_values("pdb_id").reset_index(drop=True)


def score_structures(structures: list[data_prep.Sample], run_dir: Path, engine: str = "compat",
                     workers: int = 8, device: str = "cpu", strict: bool = False) -> dict[str, Any]:
    """Run the whole chain for a batch of full structures. All outputs go under ``run_dir``."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    prep = data_prep.prepare_interfaces(structures, run_dir / "interfaces", workers=workers,
                                        manifest_path=run_dir / "interfaces_manifest.jsonl")
    ready = [r for r in prep["records"] if r["status"] == "ready_for_features"]
    warnings = {r["sample_id"]: _codes(r["issues"], "WARNING") for r in prep["records"]}
    rows = [{"sample_id": r["sample_id"], "pdb_id": r["pdb_id"], "af_model_id": r["model_id"],
             "input_pdb": to_project_relative(r["input_pdb"]), "status": "excluded",
             "reason": _codes(r["issues"], "ERROR"), "warnings": warnings[r["sample_id"]]}
            for r in prep["records"] if r["status"] != "ready_for_features"]

    feat_summary = None
    if ready:
        fs = [feature_gen.FeatureSample(r["sample_id"], r["pdb_id"], r["model_id"], r["interface_file"]) for r in ready]
        feats = feature_gen.generate_features(fs, run_dir / f"features_{engine}", engine=engine, workers=workers)
        feat_summary = feats["summary"]
        failed = {r["sample_id"]: r.get("error") for r in feats["records"] if r["status"] != "ok"}
        if feats["table"] is not None:
            res = preprocessing.predict_frame(feats["table"], device=device)
            scores = dict(zip(zip(feats["table"].pdb_id, feats["table"].af_model_id), res["predictions"]))
        else:
            scores = {}
        for r in ready:
            k = (r["pdb_id"], r["model_id"])
            if r["sample_id"] in failed or k not in scores:
                rows.append({"sample_id": r["sample_id"], "pdb_id": r["pdb_id"], "af_model_id": r["model_id"],
                             "input_pdb": to_project_relative(r["input_pdb"]), "status": "excluded",
                             "reason": f"E_FEATURES_FAILED: {failed.get(r['sample_id'])}",
                             "warnings": warnings[r["sample_id"]]})
            else:
                rows.append({"sample_id": r["sample_id"], "pdb_id": r["pdb_id"], "af_model_id": r["model_id"],
                             "input_pdb": to_project_relative(r["input_pdb"]),
                             "interface_file": to_project_relative(r["interface_file"]),
                             "status": "scored", "reason": "", "warnings": warnings[r["sample_id"]],
                             "p_dockq": scores[k]})

    pred = pd.DataFrame(rows).sort_values(KEY).reset_index(drop=True)
    ranking = rank_candidates(pred, strict=strict)
    pred.to_csv(run_dir / "predictions.csv", index=False)
    ranking.to_csv(run_dir / "ranking.csv", index=False)
    pkg = preprocessing.load_package()
    provenance = {
        "engine": engine, "device": device, "workers": workers, "strict": strict,
        "interface_job": prep["summary"], "feature_job": feat_summary,
        "preprocessing_package": {"id": pkg["manifest"]["package_id"], "files": pkg["manifest"]["files"],
                                  "checkpoint": pkg["manifest"]["checkpoint"]},
        "score_threshold": evaluation.SCORE_THRESHOLD,
        "counts": {"structures": len(structures), "scored": int((pred.status == "scored").sum()),
                   "excluded": int((pred.status == "excluded").sum()), "complexes_ranked": int(len(ranking))},
        "seconds": round(time.time() - t0, 1),
        "outputs": {"predictions": to_project_relative(run_dir / "predictions.csv"),
                    "ranking": to_project_relative(run_dir / "ranking.csv")},
    }
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    return {"predictions": pred, "ranking": ranking, "provenance": provenance}
