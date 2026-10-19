"""Tests for the Step 5 review fixes: verification completeness, selection audit, chain roles and strict
mode, Python-side stage checks, and the run guard. All offline and fast (no extraction, no API)."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdq_adapters import inputs, pipeline as det_pipeline, preprocessing, structures
from topodockq_agents import verify
from topodockq_agents.agents import MainAgent, TopoAgent
from topodockq_agents.config import Settings
from topodockq_agents.pipeline import run_pipeline
from topodockq_agents.state import Keys, PipelineState
from topodockq_agents.tools import data_tools


def _pred(rows):
    return pd.DataFrame(rows, columns=["sample_id", "pdb_id", "af_model_id", "status", "reason", "warnings", "p_dockq"])


GOOD = [("a_ranked_0", "a", 0, "scored", "", "", 0.5), ("a_ranked_1", "a", 1, "scored", "", "", 0.7),
        ("b_ranked_0", "b", 0, "excluded", "E_CHAIN_MISSING", "", np.nan)]


# ---------------------------------------------------------------- predictions
def test_identical_predictions_pass():
    checks, details = verify.compare_predictions(_pred(GOOD), _pred(GOOD))
    assert all(checks.values()) and details["max_abs_pdockq_diff"] == 0.0


def test_duplicate_agent_row_is_caught_even_though_the_key_sets_match():
    agent = _pred(GOOD + [GOOD[0]])
    checks, details = verify.compare_predictions(agent, _pred(GOOD))
    assert not checks["unique_keys_agent"] and not checks["same_row_count"] and not checks["same_structures"]
    assert details["duplicate_keys_agent"] == 1 and "not compared" in details["note"]


@pytest.mark.parametrize("change, failing", [
    (lambda r: r[:2], "same_structures"),                                                          # missing row
    (lambda r: [r[0], ("a_ranked_1", "a", 1, "scored", "", "", 0.7 + 1e-3), r[2]], "pdockq_within_tolerance"),
    (lambda r: [r[0], r[1], ("b_ranked_0", "b", 0, "excluded", "E_FILE_UNREADABLE", "", np.nan)], "same_exclusion_rules"),
    (lambda r: [r[0], ("a_ranked_1", "a", 1, "scored", "", "W_ALTLOC", 0.7), r[2]], "same_warnings"),
    (lambda r: [r[0], ("a_ranked_1", "a", 1, "scored", "", "", np.nan), r[2]], "finite_pdockq_agent"),
])
def test_prediction_differences_are_caught(change, failing):
    checks, _ = verify.compare_predictions(_pred(change(list(GOOD))), _pred(GOOD))
    assert checks[failing] is False


# ---------------------------------------------------------------- rankings
def _rank(rows):
    return pd.DataFrame(rows, columns=["pdb_id", "top1_af_model_id", "ranking_status"])


def test_rankings_need_one_row_per_scored_complex_and_agree():
    preds = _pred(GOOD)
    ok = _rank([("a", 1, "accepted")])
    assert all(verify.compare_rankings(ok, ok, preds)[0].values())
    dup = _rank([("a", 1, "accepted"), ("a", 1, "accepted")])
    assert not verify.compare_rankings(dup, ok, preds)[0]["one_ranking_row_per_scored_complex_agent"]
    extra = _rank([("a", 1, "accepted"), ("b", 0, "accepted")])          # b has no scored candidate
    assert not verify.compare_rankings(extra, ok, preds)[0]["one_ranking_row_per_scored_complex_agent"]
    other = _rank([("a", 0, "accepted")])
    assert not verify.compare_rankings(other, ok, preds)[0]["same_top1_per_complex"]
    status = _rank([("a", 1, "needs_confirmation")])
    assert not verify.compare_rankings(status, ok, preds)[0]["same_ranking_status"]


# ---------------------------------------------------------------- chain roles and strict mode
def test_strict_mode_marks_a_suspicious_top1_as_needing_confirmation():
    preds = _pred([("a_ranked_0", "a", 0, "scored", "", "W_CHAIN_ROLE_SUSPICIOUS", 0.9),
                   ("a_ranked_1", "a", 1, "scored", "", "", 0.4)])
    lenient = det_pipeline.rank_candidates(preds, strict=False)
    strict = det_pipeline.rank_candidates(preds, strict=True)
    assert lenient.loc[0, "ranking_status"] == "accepted" and lenient.loc[0, "top1_warnings"] == "W_CHAIN_ROLE_SUSPICIOUS"
    assert strict.loc[0, "ranking_status"] == "needs_confirmation" and strict.loc[0, "top1_af_model_id"] == 0


def _pdb(path: Path, protein_residues: int, peptide_residues: int) -> Path:
    lines, n = [], 1
    for chain, count in (("A", protein_residues), ("B", peptide_residues)):
        for r in range(1, count + 1):
            lines.append(f"ATOM  {n:>5}  CA  ALA {chain}{r:>4}    {float(n):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C  \n")
            n += 1
    path.write_text("".join(lines) + "END\n")
    return path


def test_confirmed_chain_roles_suppress_the_plausibility_warning(tmp_path):
    p = _pdb(tmp_path / "x.pdb", protein_residues=2, peptide_residues=6)
    codes = lambda r: {i["code"] for i in r["issues"]}  # noqa: E731
    assert "W_CHAIN_ROLE_SUSPICIOUS" in codes(structures.validate_structure(p))
    assert "W_CHAIN_ROLE_SUSPICIOUS" not in codes(structures.validate_structure(p, chains_confirmed=True))


def test_manifest_chain_columns_count_as_confirmation_and_survive_roundtrip(tmp_path):
    _pdb(tmp_path / "a.pdb", 3, 1)
    (tmp_path / "m.csv").write_text("pdb_id,af_model_id,input_pdb,protein_chain,peptide_chain\n"
                                    "x1,0,a.pdb,A,B\nx1,1,a.pdb,,\n")
    samples, report = inputs.discover(tmp_path / "m.csv")
    assert [s.chains_confirmed for s in samples] == [True, False] and report["chains_confirmed"] == 1
    back = inputs.read_manifest(inputs.write_manifest(samples, tmp_path / "s.csv"))
    assert [s.chains_confirmed for s in back] == [True, False]


# ---------------------------------------------------------------- selection audit
def _complex_dir(root: Path) -> Path:
    (root / "1ABC").mkdir(parents=True)
    for m in range(3):
        _pdb(root / "1ABC" / f"ranked_{m}.pdb", 3, 1)
    return root


def test_selection_audit_passes_and_reports_coverage(tmp_path):
    src = _complex_dir(tmp_path / "in")
    state = PipelineState.new(tmp_path / "runs", "t")
    data_tools.build_structure_manifest(state, str(src), models=[0, 1])
    preds = _pred([("1abc_ranked_0", "1abc", 0, "scored", "", "", 0.5), ("1abc_ranked_1", "1abc", 1, "scored", "", "", 0.6)])
    checks, details = verify.audit_selection(state, preds, expected_structures=2)
    assert all(checks.values()), checks
    assert details["per_complex"] == [{"pdb_id": "1abc", "available": 3, "selected": 2, "selected_ids": "0-1"}]


def test_selection_audit_catches_count_mismatch_and_edited_manifest(tmp_path):
    src = _complex_dir(tmp_path / "in")
    state = PipelineState.new(tmp_path / "runs", "t")
    data_tools.build_structure_manifest(state, str(src), models=[0, 1])
    preds = _pred([("1abc_ranked_0", "1abc", 0, "scored", "", "", 0.5), ("1abc_ranked_1", "1abc", 1, "scored", "", "", 0.6)])
    assert verify.audit_selection(state, preds, expected_structures=3)[0]["expected_structure_count"] is False
    manifest = Path(state.require(Keys.STRUCTURES).path)
    manifest.write_text("".join(manifest.read_text().splitlines(keepends=True)[:2]))       # drop one structure
    checks, _ = verify.audit_selection(state, preds, expected_structures=None)
    assert not checks["manifest_matches_rediscovery"] and not checks["predictions_cover_manifest"]


# ---------------------------------------------------------------- Python-side stage checks
def _topo_state(tmp_path, feature_rows: int, report_first: bool) -> PipelineState:
    state = PipelineState.new(tmp_path / "runs", "t")
    iface = state.path_for("interfaces.csv")
    pd.DataFrame({"sample_id": ["x_ranked_0", "x_ranked_1"], "pdb_id": ["x", "x"], "af_model_id": [0, 1],
                  "input_pdb": ["a", "b"], "status": ["ready_for_features"] * 2,
                  "interface_file": ["x_ranked_0_sp_interface.pdb", "x_ranked_1_sp_interface.pdb"],
                  "errors": ["", ""], "warnings": ["", ""]}).to_csv(iface, index=False)
    state.register(Keys.INTERFACES, iface, "csv", "test")
    cols = preprocessing.load_package()["columns"]
    feats = pd.DataFrame(np.zeros((feature_rows, len(cols))), columns=cols)
    feats.insert(0, "af_model_id", list(range(feature_rows)))
    feats.insert(0, "pdb_id", ["x"] * feature_rows)
    fpath, rpath = state.path_for("features.csv"), state.path_for("topo_report.json")
    feats.to_csv(fpath, index=False)
    rpath.write_text("{}")
    if report_first:
        state.register(Keys.TOPO_REPORT, rpath, "json", "test")
        time.sleep(0.02)
        state.register(Keys.FEATURES, fpath, "csv", "test")
    else:
        state.register(Keys.FEATURES, fpath, "csv", "test")
        time.sleep(0.02)
        state.register(Keys.TOPO_REPORT, rpath, "json", "test")
    return state


def test_topo_stage_is_partial_when_some_features_are_missing(tmp_path):
    agent = TopoAgent(Settings(runs_dir=tmp_path / "runs", dry_run=True, verbose=False), _topo_state(tmp_path, 1, False))
    status, problems, details = agent.stage_checks()
    assert status == "partial" and details["missing_count"] == 1 and "E_FEATURES_FAILED" in problems[0]


def test_topo_stage_is_incomplete_when_the_report_predates_the_features(tmp_path):
    agent = TopoAgent(Settings(runs_dir=tmp_path / "runs", dry_run=True, verbose=False), _topo_state(tmp_path, 2, True))
    status, problems, _ = agent.stage_checks()
    assert status == "incomplete" and "older than topo.features" in problems[0]


def test_outputs_from_an_earlier_attempt_do_not_count(tmp_path):
    state = _topo_state(tmp_path, 2, False)
    agent = TopoAgent(Settings(runs_dir=tmp_path / "runs", dry_run=True, verbose=False), state)
    assert agent.validate_stage(started_at=0.0)[0] == "done"
    status, problems, _ = agent.validate_stage(started_at=time.time() + 1)
    assert status == "incomplete" and all("not produced in this attempt" in p for p in problems)


# ---------------------------------------------------------------- run guard
def _nothing_ready(tmp_path) -> Settings:
    (tmp_path / "m.csv").write_text(f"pdb_id,af_model_id,input_pdb\nx1,0,{tmp_path / 'missing.pdb'}\n")
    return Settings(runs_dir=tmp_path / "runs", dry_run=True, verbose=False, input_source=str(tmp_path / "m.csv"))


def test_crashing_verifier_still_produces_a_report_and_a_failed_status(tmp_path, monkeypatch):
    def boom(state, settings):
        raise RuntimeError("verifier exploded")
    monkeypatch.setattr("topodockq_agents.pipeline.verify_against_deterministic", boom)
    st = run_pipeline("score", _nothing_ready(tmp_path), mode="sequential")
    events = [json.loads(l) for l in st.trace_path.read_text().splitlines()]
    ver = [e for e in events if e["event"] == "verification"][-1]
    end = [e for e in events if e["event"] == "run_end"][-1]
    assert ver["status"] == "error" and "verifier exploded" in ver["error"]
    assert end["run_status"] == "failed"
    assert "verifier exploded" in (st.run_dir / "report.md").read_text(encoding="utf-8")


def test_crash_outside_the_agents_is_recorded_and_reported(tmp_path, monkeypatch):
    def crash(self, task):
        raise MemoryError("simulated crash")
    monkeypatch.setattr(MainAgent, "run", crash)
    st = run_pipeline("score", _nothing_ready(tmp_path), mode="llm")
    events = [json.loads(l) for l in st.trace_path.read_text().splitlines()]
    assert any(e["event"] == "run_error" and e["type"] == "MemoryError" for e in events)
    assert [e for e in events if e["event"] == "run_end"][-1]["run_status"] == "failed"
    assert "simulated crash" in (st.run_dir / "report.md").read_text(encoding="utf-8")
