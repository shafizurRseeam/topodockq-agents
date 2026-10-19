"""Unit tests for tdq_adapters.evaluation on small hand-built tables (no downloaded data needed)."""
import math

import pandas as pd
import pytest

from tdq_adapters import evaluation as ev


def _table():
    # Complex A: top by score is model 1 (score 0.90, DockQ 0.85) -> TP
    # Complex B: top is model 0 (score 0.95, DockQ 0.30)          -> FP
    # Complex C: top is model 1 (score 0.50, DockQ 0.90)          -> FN
    # Complex D: top is model 0 (score 0.82, DockQ 0.80)          -> TN (both thresholds are strict)
    return pd.DataFrame({
        "pdb_id":        ["A", "A", "B", "B", "C", "C", "D", "D"],
        "af_model_id":   [0, 1, 0, 1, 0, 1, 0, 1],
        "score":         [0.10, 0.90, 0.95, 0.20, 0.40, 0.50, 0.82, 0.10],
        "pdb2sql_DockQ": [0.99, 0.85, 0.30, 0.95, 0.10, 0.90, 0.80, 0.99],
    })


def test_hand_computed_confusion_and_metrics():
    out = ev.evaluate(_table(), {"s": "score"})
    assert out["s"]["confusion"] == {"TP": 1, "FP": 1, "TN": 1, "FN": 1}
    assert out["s"]["FPR"] == 0.5 and out["s"]["precision"] == 0.5 and out["s"]["recall"] == 0.5
    assert math.isclose(out["s"]["F1"], 0.5)
    assert [r["af_model_id"] for r in out["s"]["selected"]] == [1, 0, 1, 0]


def test_zero_denominators_are_none_not_zero():
    df = _table()
    df["score"] = 0.0                      # nothing predicted positive -> precision undefined
    out = ev.evaluate(df, {"s": "score"})
    assert out["s"]["precision"] is None and out["s"]["F1"] is None


def test_ties_are_counted_and_first_row_wins():
    df = _table()
    df.loc[df.pdb_id == "A", "score"] = 0.9
    out = ev.evaluate(df, {"s": "score"})
    assert out["s"]["tied_top_scores"] == 1
    assert out["s"]["selected"][0]["af_model_id"] == 0


@pytest.mark.parametrize("mutate, message", [
    (lambda d: d.iloc[0:0], "empty"),
    (lambda d: d.drop(columns=["score"]), "missing columns"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate"),
    (lambda d: d.assign(score=d.score.where(d.index != 3)), "missing or non-finite"),
    (lambda d: d.assign(pdb2sql_DockQ=d.pdb2sql_DockQ.where(d.index != 3)), "missing or non-finite"),
    (lambda d: d.assign(pdb2sql_DockQ=d.pdb2sql_DockQ + 1.0), "outside"),
])
def test_bad_inputs_are_rejected(mutate, message):
    with pytest.raises(ev.EvaluationInputError, match=message):
        ev.evaluate(mutate(_table()), {"s": "score"})
