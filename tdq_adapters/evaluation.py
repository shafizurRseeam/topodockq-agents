"""Top-1 model-selection metrics as defined in the TopoDockQ paper.

The authors published no code for these metrics. Every rule below is taken from
the paper's Results text (Dai, Wang & Zhang, Commun. Chem. 8:347, 2025, p. 3-4):

  R1 "both AF2-M and AF3 confidence scores were used for ranking, and the top-1
     model is selected for further analysis."
     -> For each complex (pdb_id), keep the single candidate with the highest score.
  R2 "Models with a confidence score greater than 0.82 are considered positive
     predictions"
     -> Predicted positive  <=>  selected candidate's score > 0.82 (strict).
        The same threshold is applied to af_confidence and to p-DockQ.
  R3 "models with an actual DockQ score above 0.8 are classified as true
     high-quality structures"
     -> Actually positive  <=>  selected candidate's true DockQ > 0.8 (strict).
  R4 FPR, precision, recall (TPR) and F1 are computed over the selected
     candidates, one per complex, from the resulting confusion matrix.

Ties in R1 are broken by the first row in the authors' table order
(pandas idxmax); the number of complexes with a tied top score is reported.
Selection uses only the candidates present in the table; complexes with fewer
than 50 candidates are counted and reported (see data_audit.py), never dropped.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SCORE_THRESHOLD = 0.82      # R2
DOCKQ_HIGH_QUALITY = 0.80   # R3
KEY = ("pdb_id", "af_model_id")


class EvaluationInputError(ValueError):
    """The candidate table cannot be evaluated as given."""


def _as_finite_float(series: pd.Series, name: str) -> np.ndarray:
    """Require a real numeric column (not text, not boolean) with no NaN, pd.NA or inf."""
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        raise EvaluationInputError(f"column {name!r} must be numeric, got dtype {series.dtype}")
    values = series.astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    bad = int((~np.isfinite(values)).sum())
    if bad:
        raise EvaluationInputError(f"column {name!r} has {bad} missing or non-finite values")
    return values


def _validate_ids(df: pd.DataFrame) -> None:
    pdb = df["pdb_id"]
    bad_pdb = int((pdb.isna() | ~pdb.map(lambda v: isinstance(v, str) and v.strip() != "")).sum())
    if bad_pdb:
        raise EvaluationInputError(f"{bad_pdb} rows with a missing or invalid pdb_id")
    model = df["af_model_id"]
    if pd.api.types.is_bool_dtype(model) or not pd.api.types.is_numeric_dtype(model):
        raise EvaluationInputError(f"af_model_id must be numeric, got dtype {model.dtype}")
    values = model.astype("Float64").to_numpy(dtype=float, na_value=np.nan)
    bad_model = int((~np.isfinite(values) | (values < 0) | (values != np.round(values))).sum())
    if bad_model:
        raise EvaluationInputError(f"{bad_model} rows with a missing, negative or non-integer af_model_id")


def validate_candidates(df: pd.DataFrame, score_cols: list[str], dockq_col: str) -> None:
    """Fail loudly instead of letting bad IDs, NA, text or duplicates change the selection."""
    if df.empty:
        raise EvaluationInputError("candidate table is empty")
    missing = [c for c in (*KEY, dockq_col, *score_cols) if c not in df.columns]
    if missing:
        raise EvaluationInputError(f"missing columns: {missing}")
    _validate_ids(df)
    dup = int(df.duplicated(list(KEY)).sum())
    if dup:
        raise EvaluationInputError(f"{dup} duplicate (pdb_id, af_model_id) rows")
    for col in score_cols:
        _as_finite_float(df[col], col)
    dockq = _as_finite_float(df[dockq_col], dockq_col)
    out_of_range = int(((dockq < 0) | (dockq > 1)).sum())
    if out_of_range:
        raise EvaluationInputError(f"{out_of_range} DockQ values outside [0, 1]")


def select_top1(df: pd.DataFrame, score_col: str, group_col: str = "pdb_id") -> tuple[pd.DataFrame, int]:
    """R1: one row per complex, the candidate with the highest ``score_col``."""
    top = df.loc[df.groupby(group_col, sort=True)[score_col].idxmax()].copy()
    is_max = df.groupby(group_col)[score_col].transform("max") == df[score_col]
    tied = int(is_max.groupby(df[group_col]).sum().gt(1).sum())
    return top, tied


def confusion(top: pd.DataFrame, score_col: str, dockq_col: str = "pdb2sql_DockQ") -> dict[str, int]:
    pred = top[score_col] > SCORE_THRESHOLD          # R2
    true = top[dockq_col] > DOCKQ_HIGH_QUALITY       # R3
    return {"TP": int((pred & true).sum()), "FP": int((pred & ~true).sum()),
            "TN": int((~pred & ~true).sum()), "FN": int((~pred & true).sum())}


def _div(a: float, b: float) -> float | None:
    return a / b if b else None


def metrics(cm: dict[str, int]) -> dict[str, float | None]:
    """R4. None where a denominator is zero (reported, never silently treated as 0)."""
    tp, fp, tn, fn = cm["TP"], cm["FP"], cm["TN"], cm["FN"]
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    f1 = (_div(2 * precision * recall, precision + recall)
          if precision is not None and recall is not None else None)
    return {"FPR": _div(fp, fp + tn), "precision": precision, "recall": recall, "F1": f1}


def evaluate(df: pd.DataFrame, score_cols: dict[str, str], dockq_col: str = "pdb2sql_DockQ") -> dict[str, Any]:
    """Run R1-R4 for each scorer. ``score_cols`` maps a display name to a column of ``df``."""
    validate_candidates(df, list(score_cols.values()), dockq_col)
    per = df.groupby("pdb_id").size()
    out: dict[str, Any] = {"n_complexes": int(len(per)), "n_candidates": int(len(df)),
                           "complexes_with_fewer_than_50_candidates": int((per < 50).sum()),
                           "min_candidates_per_complex": int(per.min())}
    for name, col in score_cols.items():
        top, tied = select_top1(df, col)
        cm = confusion(top, col, dockq_col)
        if sum(cm.values()) != out["n_complexes"]:
            raise AssertionError(f"{name}: confusion matrix covers {sum(cm.values())} of {out['n_complexes']} complexes")
        out[name] = {"confusion": cm, **metrics(cm), "tied_top_scores": tied,
                     "selected": top[["pdb_id", "af_model_id", col, dockq_col]].to_dict("records")}
    return out
