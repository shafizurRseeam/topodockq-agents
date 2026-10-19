"""Regression tests for the robustness pass (no downloaded data, no ProDy, no GPU needed)."""
import json
import os
import stat
import sys

import numpy as np
import pandas as pd
import pytest

from tdq_adapters import author_scripts, data_audit, evaluation as ev, preprocessing as pp, runner, staging
from tdq_adapters.provenance import sha256


# ---------------------------------------------------------------- evaluation.validate_candidates
def _cands(**overrides):
    df = pd.DataFrame({"pdb_id": ["A", "A", "B", "B"], "af_model_id": [0, 1, 0, 1],
                       "score": [0.9, 0.1, 0.5, 0.95], "pdb2sql_DockQ": [0.85, 0.1, 0.2, 0.9]})
    for k, v in overrides.items():
        df[k] = v
    return df


@pytest.mark.parametrize("override, message", [
    ({"score": pd.array([0.9, pd.NA, 0.5, 0.95], dtype="Float64")}, "missing or non-finite"),
    ({"pdb2sql_DockQ": pd.array([0.85, 0.1, pd.NA, 0.9], dtype="Float64")}, "missing or non-finite"),
    ({"score": ["0.9", "0.1", "0.5", "0.95"]}, "must be numeric"),
    ({"pdb2sql_DockQ": ["0.85", "0.1", "0.2", "0.9"]}, "must be numeric"),
    ({"score": [True, False, True, False]}, "must be numeric"),
    ({"score": [0.9, np.inf, 0.5, 0.95]}, "missing or non-finite"),
    ({"pdb_id": ["A", None, "B", "B"]}, "pdb_id"),
    ({"pdb_id": ["A", "  ", "B", "B"]}, "pdb_id"),
    ({"pdb_id": ["A", 7, "B", "B"]}, "pdb_id"),
    ({"af_model_id": pd.array([0, pd.NA, 0, 1], dtype="Int64")}, "af_model_id"),
    ({"af_model_id": [0, -1, 0, 1]}, "af_model_id"),
    ({"af_model_id": [0, 1.5, 0, 1]}, "af_model_id"),
    ({"af_model_id": ["0", "1", "0", "1"]}, "af_model_id must be numeric"),
])
def test_validate_candidates_rejects(override, message):
    with pytest.raises(ev.EvaluationInputError, match=message):
        ev.evaluate(_cands(**override), {"s": "score"})


def test_validate_candidates_accepts_nullable_dtypes_without_missing_values():
    df = _cands(score=pd.array([0.9, 0.1, 0.5, 0.95], dtype="Float64"),
                af_model_id=pd.array([0, 1, 0, 1], dtype="Int64"))
    out = ev.evaluate(df, {"s": "score"})
    assert sum(out["s"]["confusion"].values()) == 2


# ---------------------------------------------------------------- data_audit.coverage
def _complex(pdb, ids):
    return pd.DataFrame({"pdb_id": pdb, "af_model_id": ids})


def test_coverage_complete_set():
    c = data_audit.coverage(_complex("X", list(range(50))))
    assert c["missing_candidates"] == 0 and c["complexes_with_incomplete_id_set"] == 0
    assert c["duplicate_rows"] == 0 and c["out_of_range_or_invalid_ids"] == 0


def test_coverage_detects_duplicates_even_when_row_count_is_50():
    ids = list(range(49)) + [0]                      # 50 rows, 49 unique: model 49 missing, model 0 duplicated
    c = data_audit.coverage(_complex("X", ids))
    assert c["rows"] == 50 and c["duplicate_rows"] == 1 and c["duplicates"] == {"X": [0]}
    assert c["missing_model_ids"] == {"X": [49]} and c["complexes_with_incomplete_id_set"] == 1


def test_coverage_detects_out_of_range_ids_even_when_row_count_is_50():
    ids = list(range(49)) + [50]                     # 50 rows, one ID outside 0-49
    c = data_audit.coverage(_complex("X", ids))
    assert c["out_of_range_or_invalid_ids"] == 1 and c["invalid_ids"] == {"X": ["50"]}
    assert c["missing_model_ids"] == {"X": [49]}


def test_coverage_flags_invalid_ids_and_complex_with_none_valid():
    df = pd.concat([_complex("X", list(range(50))), _complex("Y", [-1, 2.5, np.nan])])
    c = data_audit.coverage(df)
    assert c["out_of_range_or_invalid_ids"] == 3 and len(c["missing_model_ids"]["Y"]) == 50


def test_label_only_rows_are_classified_by_id_not_by_count():
    key = ["pdb_id", "af_model_id"]
    labels = pd.DataFrame({"pdb_id": ["A"] * 4, "af_model_id": [0, 1, 2, 3],
                           "pdb2sql_DockQ": [0.5, 0.6, np.nan, 0.7], "data_class": "train"})
    filtered = labels[labels.af_model_id.isin([0, 2, 3])]          # 1 was filtered out
    feats = labels[labels.af_model_id == 0][key + ["data_class"]]  # only 0 has features
    r = data_audit.classify_label_only_train_rows(labels, filtered, feats)
    assert (r["label_only_rows"], r["intentionally_filtered_rows"], r["missing_label_rows"],
            r["unexplained_rows"]) == (3, 1, 1, 1)
    assert r["unexplained_keys"] == [{"pdb_id": "A", "af_model_id": 3}]


# ---------------------------------------------------------------- runner
def test_fmt_handles_undefined_metrics():
    assert runner.fmt(None) == "undefined" and runner.fmt(0.123456) == "0.1235"


def test_failed_run_writes_structured_report_and_still_checks_protected_dirs(tmp_path):
    prot = tmp_path / "protected"
    prot.mkdir()
    (prot / "f.txt").write_text("x")

    def body(report, run_dir):
        raise RuntimeError("boom")

    code, report = runner.run_guarded("t", tmp_path / "run", body, protected_dirs=(prot,))
    saved = json.loads((tmp_path / "run" / "t_report.json").read_text())
    assert code == 1 and saved["status"] == "error" and saved["failure"]["type"] == "RuntimeError"
    assert "boom" in saved["failure"]["traceback"] and saved["protected_dirs"]["unchanged"] is True


def test_run_that_modifies_protected_file_fails(tmp_path):
    prot = tmp_path / "protected"
    prot.mkdir()
    (prot / "f.txt").write_text("x")

    def body(report, run_dir):
        (prot / "f.txt").write_text("changed")
        return {"ok": True}

    code, report = runner.run_guarded("t", tmp_path / "run", body, protected_dirs=(prot,))
    assert code == 1 and report["status"] == "failed_checks"
    assert report["protected_dirs"]["changes"] and report["checks"]["protected_dirs_unchanged"] is False


# ---------------------------------------------------------------- subprocess timeout
def test_script_timeout_raises(tmp_path):
    (tmp_path / "slow.py").write_text("import time; time.sleep(30)")
    with pytest.raises(author_scripts.ScriptFailed, match="timed out"):
        author_scripts.run_script(tmp_path, "slow.py", [], tmp_path / "logs", "slow", timeout=1)


# ---------------------------------------------------------------- read-only staged links
def test_staged_link_is_read_only_and_original_cannot_be_written_through_it(tmp_path):
    src = tmp_path / "orig.csv"
    src.write_text("data")
    dst = tmp_path / "stage" / "orig.csv"
    mode = staging._link_or_copy(src, dst)
    assert "read-only" in mode
    with pytest.raises(PermissionError):
        dst.write_text("overwrite")
    assert src.read_text() == "data"
    os.chmod(dst, stat.S_IWRITE)
    os.chmod(src, stat.S_IWRITE)


# ---------------------------------------------------------------- package integrity + feature validation
def _make_package(tmp_path, columns=("f1", "f2", "f3"), scale=(1.0, 2.0, 3.0)):
    d = tmp_path / "pkg"
    d.mkdir()
    ckpt = tmp_path / "model.pth"
    ckpt.write_bytes(b"weights")
    (d / "feature_columns.json").write_text(json.dumps(list(columns)))
    (d / "dropped_columns.json").write_text("[]")
    n = 3
    np.savez(d / "scaler.npz", mean_=np.zeros(n), scale_=np.array(scale, float), var_=np.array(scale, float) ** 2,
             n_samples_seen_=np.asarray(10))
    manifest = {"model_hparams": {"input_dim": n}, "checkpoint": {"path": str(ckpt), "sha256": sha256(ckpt)},
                "files": {name: sha256(d / name) for name in pp.PACKAGE_FILES}}
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d


def test_valid_package_loads(tmp_path):
    assert pp.load_package(_make_package(tmp_path))["columns"] == ["f1", "f2", "f3"]


def test_tampered_package_file_is_rejected(tmp_path):
    d = _make_package(tmp_path)
    np.savez(d / "scaler.npz", mean_=np.ones(3), scale_=np.ones(3), var_=np.ones(3), n_samples_seen_=np.asarray(10))
    with pytest.raises(pp.PackageIntegrityError, match="scaler.npz"):
        pp.load_package(d)


def test_manifest_without_file_hashes_is_rejected(tmp_path):
    d = _make_package(tmp_path)
    m = json.loads((d / "manifest.json").read_text())
    del m["files"]
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(pp.PackageIntegrityError, match="no file hashes"):
        pp.load_package(d)


@pytest.mark.parametrize("kwargs, message", [
    ({"columns": ("f1", "f1", "f3")}, "unique"),
    ({"scale": (1.0, 0.0, 3.0)}, "positive"),
    ({"scale": (1.0, np.nan, 3.0)}, "finite"),
])
def test_inconsistent_package_is_rejected(tmp_path, kwargs, message):
    with pytest.raises(pp.PackageIntegrityError, match=message):
        pp.load_package(_make_package(tmp_path, **kwargs))


def test_changed_checkpoint_is_rejected_before_prediction(tmp_path):
    d = _make_package(tmp_path)
    (tmp_path / "model.pth").write_bytes(b"other weights")
    with pytest.raises(pp.PackageIntegrityError, match="checkpoint"):
        pp.predict_frame(pd.DataFrame({"f1": [1.0], "f2": [1.0], "f3": [1.0]}), package_dir=d, device="cpu")


@pytest.mark.parametrize("frame, message", [
    (pd.DataFrame({"f1": [1.0], "f2": [1.0]}), "missing"),
    (pd.DataFrame({"f1": [1.0], "f2": [np.nan], "f3": [1.0]}), "non-finite"),
    (pd.DataFrame({"f1": [1.0], "f2": pd.array([pd.NA], dtype="Float64"), "f3": [1.0]}), "non-finite"),
    (pd.DataFrame({"f1": [1.0], "f2": ["1.0"], "f3": [1.0]}), "not numeric"),
    (pd.DataFrame({"f1": [1.0], "f2": [np.inf], "f3": [1.0]}), "non-finite"),
    (pd.DataFrame({"f1": [], "f2": [], "f3": []}), "empty"),
])
def test_feature_values_are_validated(frame, message):
    with pytest.raises(pp.FeatureInputError, match=message):
        pp.validate_features(frame, ["f1", "f2", "f3"])
