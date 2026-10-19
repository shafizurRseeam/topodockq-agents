"""TopoAgent tools: topological feature generation for every ready interface.

Plain functions taking the shared PipelineState first. Feature generation is
``tdq_adapters.feature_gen`` in the paper profile: published bins and filtration,
the authors' 03 conversion script, and one of two verified engines:
  "compat"   the Python 3.13 port (default; matched the published features exactly)
  "original" the authors' main.pyc, unchanged, in the Python 3.8 environment
No LLM here.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tdq_adapters import feature_gen, preprocessing
from tdq_adapters.paths import from_project_relative, to_project_relative

from ..state import Keys, PipelineState
from .data_tools import next_attempt

AGENT = "TopoAgent"
KEY = ["pdb_id", "af_model_id"]


def ready_samples(state: PipelineState) -> list[feature_gen.FeatureSample]:
    with open(state.require(Keys.INTERFACES).path, newline="", encoding="utf-8") as fh:
        return [feature_gen.FeatureSample(r["sample_id"], r["pdb_id"], int(r["af_model_id"]),
                                          str(from_project_relative(r["interface_file"])))
                for r in csv.DictReader(fh) if r["status"] == "ready_for_features"]


def generate_features(state: PipelineState, engine: str = "compat", workers: int = 8,
                      retry_failed_only: bool = False) -> dict[str, Any]:
    """Generate the 2754 features for every ready interface (or only those still missing) and register topo.features."""
    if engine not in feature_gen.ENGINES:
        raise ValueError(f"engine must be one of {feature_gen.ENGINES}")
    ready = ready_samples(state)
    current = None
    if retry_failed_only:
        current = pd.read_csv(state.require(Keys.FEATURES).path)
        done = set(zip(current.pdb_id, current.af_model_id))
        todo = [s for s in ready if (s.pdb_id, s.model_id) not in done]
        if not todo:
            return {"retried": 0, "message": "every ready structure already has features; nothing to retry"}
    else:
        todo = ready
    if not todo:
        raise ValueError("no structures are ready for feature generation (all were excluded by the DataAgent)")

    n = next_attempt(state, "features")
    job_dir = state.artifacts_dir / f"features_a{n}_{engine}"
    job = feature_gen.generate_features(todo, job_dir, engine=engine, workers=workers)
    s = job["summary"]
    table, path = job["table"], None
    if table is not None:
        if current is not None:
            table = pd.concat([current, table], ignore_index=True)
            if table.duplicated(KEY).any():
                raise RuntimeError("merged feature table has duplicate (pdb_id, af_model_id) keys")
        path = state.path_for(f"features_a{n}.csv")
        table.to_csv(path, index=False)
    result = {"attempt": n, "engine": engine, "feature_schema": s["feature_schema"], "retry_failed_only": retry_failed_only,
              "structures_attempted": s["samples"], "ok": s["ok"], "failed": s["failed"],
              "failures": s["failures"][:10], "seconds": s["seconds"],
              "rows_in_registered_table": int(len(table)) if table is not None else 0,
              "ready_structures": len(ready)}
    if path is not None:
        state.register(Keys.FEATURES, path, "csv", AGENT, **result)
    return result


def check_features(state: PipelineState) -> dict[str, Any]:
    """Check the registered feature table: one row per ready structure, published columns, finite values."""
    ready = {(s.pdb_id, s.model_id) for s in ready_samples(state)}
    table = pd.read_csv(state.require(Keys.FEATURES).path)
    keys = list(zip(table.pdb_id, table.af_model_id))
    dup = len(keys) - len(set(keys))
    pkg = preprocessing.load_package()
    issues = []
    try:
        preprocessing.validate_features(table, pkg["columns"])
    except preprocessing.FeatureInputError as e:
        issues.append(str(e))
    missing = sorted(ready - set(keys))
    extra = sorted(set(keys) - ready)
    ok = not dup and not missing and not extra and not issues
    raw = sum(1 for c in table.columns if c.startswith(("persistent_", "static_")))
    return {"ok": ok, "ready_structures": len(ready), "rows": len(table), "duplicate_keys": dup,
            "missing_structures": [f"{p}_ranked_{m}" for p, m in missing[:10]], "missing_count": len(missing),
            "unexpected_rows": len(extra), "schema_issues": issues,
            "raw_feature_columns": raw, "model_input_columns_checked": len(pkg["columns"]),
            "columns_note": "expected: the generator writes 2,754 raw features; the published model uses 2,646 of "
                            "them (the authors' preprocessing drops 108 that are constant in the training data)"}


def write_topo_report(state: PipelineState, notes: str = "") -> dict[str, Any]:
    """Write topo.report: engine, attempts, failures and the final check."""
    report = {"features": state.require(Keys.FEATURES).meta, "check": check_features(state), "notes": notes}
    out = state.path_for("topo_report.json")
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    state.register(Keys.TOPO_REPORT, out, "json", AGENT, check_ok=report["check"]["ok"], notes=notes[:300])
    return {"written": to_project_relative(out), "check_ok": report["check"]["ok"]}
