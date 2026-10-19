"""Offline tests of the agent layer (no API calls).

Two ways to drive the agents without the API:
  * dry-run mode: each agent calls its tool functions in a fixed order;
  * a scripted stand-in for the model (fake_runner): it calls the agents' real
    @beta_tool wrappers with JSON-style arguments, exactly as the tool runner would,
    and feeds fake messages to the trace hook. This covers the wrappers' argument
    schemas, delegation from MainAgent, error-then-retry, and LLM message tracing.
Integration tests use 3 real PFPD models plus 2 broken inputs and are skipped if the
PFPD data or the ProDy environment is missing.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from topodockq_agents import llm
from topodockq_agents.agents import DataAgent, MainAgent, MLAgent, TopoAgent
from topodockq_agents.config import Settings
from topodockq_agents.pipeline import run_pipeline
from topodockq_agents.state import STAGE_OUTPUTS, Keys, PipelineState
from tdq_adapters.paths import EXTERNAL_DIR, PRODY_PYTHON

PFPD = EXTERNAL_DIR / "PFPD_AF2_models"
needs_data = pytest.mark.skipif(not (PFPD.is_dir() and PRODY_PYTHON.is_file()),
                                reason="PFPD structures or the ProDy environment are not available")

EXPECTED_TOOLS = {
    "DataAgent": {"build_structure_manifest", "check_structures", "prepare_interfaces", "write_data_report"},
    "TopoAgent": {"generate_features", "check_features", "write_topo_report"},
    "MLAgent": {"check_model_package", "predict_pdockq", "rank_candidates", "write_ml_report"},
    "MainAgent": {"run_data_agent", "run_topo_agent", "run_ml_agent", "inspect_state"},
}


# ---------------------------------------------------------------- fixtures
def _write_manifest(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb_id", "af_model_id", "input_pdb"])
        w.writeheader()
        w.writerows(rows)
    return path


def _mixed_manifest(folder: Path) -> Path:
    """3 good PFPD models (2 complexes) + 1 missing file + 1 structure without the peptide chain."""
    folder.mkdir(parents=True, exist_ok=True)
    no_peptide = folder / "1eg4_no_peptide.pdb"
    no_peptide.write_text("".join(l for l in open(PFPD / "1EG4" / "ranked_0_sp.pdb")
                                  if not (l.startswith("ATOM") and l[21] == "B")))
    return _write_manifest(folder / "manifest.csv", [
        {"pdb_id": "1awr", "af_model_id": 0, "input_pdb": str(PFPD / "1AWR" / "ranked_0_sp.pdb")},
        {"pdb_id": "1awr", "af_model_id": 1, "input_pdb": str(PFPD / "1AWR" / "ranked_1_sp.pdb")},
        {"pdb_id": "1czy", "af_model_id": 0, "input_pdb": str(PFPD / "1CZY" / "ranked_0_sp.pdb")},
        {"pdb_id": "1eg4", "af_model_id": 0, "input_pdb": str(no_peptide)},
        {"pdb_id": "1eg4", "af_model_id": 1, "input_pdb": str(folder / "does_not_exist.pdb")},
    ])


def _settings(tmp: Path, **kw) -> Settings:
    return Settings(runs_dir=tmp / "runs", dry_run=True, verbose=False, workers=4, **kw)


def _events(state: PipelineState, kind: str) -> list[dict]:
    return [json.loads(l) for l in state.trace_path.read_text(encoding="utf-8").splitlines()
            if json.loads(l)["event"] == kind]


@pytest.fixture(scope="module")
def dry_run_state(tmp_path_factory):
    if not (PFPD.is_dir() and PRODY_PYTHON.is_file()):
        pytest.skip("PFPD structures or the ProDy environment are not available")
    tmp = tmp_path_factory.mktemp("dry")
    s = _settings(tmp, input_source=str(_mixed_manifest(tmp / "input")))
    return run_pipeline("dry run", s, mode="sequential")


# ---------------------------------------------------------------- scripted model stand-in
def _msg(blocks, stop, model="fake-model"):
    usage = SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    return SimpleNamespace(content=blocks, stop_reason=stop, model=model, usage=usage)


def fake_runner(scripts: dict[str, list[tuple[str, dict]]], seen: list):
    """Replace llm.run_tool_agent: run each agent's scripted tool calls through its real tool wrappers."""
    def run(client, cfg, *, system, user, tools, enable_fallbacks=True, on_message=None):
        agent = system.split(",")[0].replace("You are ", "").strip()
        by_name = {t.name: t for t in tools}
        calls, usage = [], {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0,
                            "cache_creation_input_tokens": 0}
        for i, (tool, args) in enumerate(scripts[agent]):
            on_message(_msg([SimpleNamespace(type="thinking", thinking=f"{agent} step {i}: use {tool}"),
                             SimpleNamespace(type="tool_use", name=tool, input=args)], "tool_use"))
            out = by_name[tool].call(args)
            try:
                parsed = json.loads(out)
            except json.JSONDecodeError:
                parsed = out                       # inspect_state returns plain text
            seen.append((agent, tool, parsed))
            calls.append(llm.ToolCall(tool, args))
            usage["input_tokens"] += 100
            usage["output_tokens"] += 20
        on_message(_msg([SimpleNamespace(type="text", text=f"{agent} finished")], "end_turn"))
        return llm.LLMRun(final_text=f"{agent} finished", stop_reason="end_turn", tool_calls=calls, usage=usage,
                          turns=len(calls) + 1)
    return run


# ---------------------------------------------------------------- tests
def test_every_agent_exposes_expected_tools(tmp_path):
    state = PipelineState.new(tmp_path / "runs", "t")
    for cls in (DataAgent, TopoAgent, MLAgent, MainAgent):
        tools = cls(_settings(tmp_path), state).build_tools()
        assert {t.name for t in tools} == EXPECTED_TOOLS[cls.__name__]
        for t in tools:
            assert t.to_dict()["input_schema"]["type"] == "object"


@needs_data
def test_dry_run_scores_ranks_and_verifies(dry_run_state):
    st = dry_run_state
    for agent, outs in STAGE_OUTPUTS.items():
        assert st.stages[agent] == "done", st.stages
        assert all(st.has(k) for k in outs)
    pred = pd.read_csv(st.require(Keys.PREDICTIONS).path)
    assert (pred.status == "scored").sum() == 3
    reasons = dict(zip(pred.sample_id, pred.reason.fillna("")))
    assert reasons["1eg4_ranked_0"] == "E_CHAIN_MISSING"
    assert reasons["1eg4_ranked_1"] == "E_FILE_UNREADABLE"
    ranking = pd.read_csv(st.require(Keys.RANKING).path)
    assert set(ranking.pdb_id) == {"1awr", "1czy"}
    ver = _events(st, "verification")[-1]
    assert ver["status"] == "passed" and ver["max_abs_pdockq_diff"] <= 1e-6, ver
    assert (st.run_dir / "report.md").exists() and (st.run_dir / "verification.json").exists()


@needs_data
def test_trace_records_every_tool_invocation(dry_run_state):
    inv = _events(dry_run_state, "tool_invocation")
    assert {e["tool"] for e in inv} == set().union(*(EXPECTED_TOOLS[a] for a in ("DataAgent", "TopoAgent", "MLAgent")))
    assert all(e["status"] == "ok" and "seconds" in e and "args" in e for e in inv)
    ends = _events(dry_run_state, "stage_end")
    assert [e["agent"] for e in ends] == ["DataAgent", "TopoAgent", "MLAgent"]
    assert all("summary" in e and "seconds" in e for e in ends)


@needs_data
def test_state_roundtrip(dry_run_state):
    loaded = PipelineState.load(dry_run_state.run_dir)
    assert set(loaded.artifacts) == set(dry_run_state.artifacts) and loaded.stages == dry_run_state.stages


def test_missing_input_fails_cleanly_and_stops(tmp_path):
    st = run_pipeline("score", _settings(tmp_path, input_source=str(tmp_path / "nope")), mode="sequential")
    assert st.stages == {"DataAgent": "failed"}
    assert "InputError" in _events(st, "stage_end")[0]["error"]
    inv = _events(st, "tool_invocation")
    assert inv[0]["tool"] == "build_structure_manifest" and inv[0]["status"] == "error"
    assert _events(st, "verification")[-1]["status"] == "skipped"


def test_main_agent_stops_when_nothing_is_ready(tmp_path):
    bad = _write_manifest(tmp_path / "m.csv", [{"pdb_id": "x1", "af_model_id": 0, "input_pdb": str(tmp_path / "missing.pdb")}])
    st = run_pipeline("score", _settings(tmp_path, input_source=str(bad)), mode="llm")
    assert st.stages["DataAgent"] == "done"
    assert "TopoAgent" not in st.stages and "MLAgent" not in st.stages
    assert st.require(Keys.INTERFACES).meta["exclusions_by_rule"] == {"E_FILE_UNREADABLE": 1}


def test_unverified_model_package_blocks_scoring(tmp_path, monkeypatch):
    from tdq_adapters import preprocessing

    def broken():
        raise preprocessing.PackageIntegrityError("scaler.npz does not match its recorded sha256")
    monkeypatch.setattr(preprocessing, "load_package", broken)
    state = PipelineState.new(tmp_path / "runs", "t")
    dummy = state.path_for("x.csv")
    dummy.write_text("a\n1\n")
    state.register(Keys.INTERFACES, dummy, "csv", "test")
    state.register(Keys.FEATURES, dummy, "csv", "test")
    res = MLAgent(_settings(tmp_path), state).run("score")
    assert res.status == "failed" and "scaler.npz" in res.error
    assert not state.has(Keys.PREDICTIONS)


@needs_data
def test_scripted_model_drives_real_tool_wrappers(tmp_path, monkeypatch):
    manifest = _mixed_manifest(tmp_path / "input")
    scripts = {
        "MainAgent": [("run_data_agent", {"instructions": f"Use {manifest}; complexes 1awr, 1czy, 1eg4."}),
                      ("inspect_state", {}),
                      ("run_topo_agent", {"instructions": "Use the compat engine."}),
                      ("run_ml_agent", {"instructions": "Score and rank."})],
        "DataAgent": [("build_structure_manifest", {"source": str(manifest), "complexes": ["NOPE"]}),   # error
                      ("build_structure_manifest", {"source": str(manifest), "complexes": ["1awr", "1czy", "1eg4"]}),
                      ("check_structures", {}),
                      ("prepare_interfaces", {"workers": 4}),
                      ("write_data_report", {"notes": "2 excluded by fixed rules"})],
        "TopoAgent": [("generate_features", {"engine": "compat", "workers": 4}),
                      ("check_features", {}),
                      ("write_topo_report", {"notes": "compat"})],
        "MLAgent": [("check_model_package", {}), ("predict_pdockq", {"device": "cpu"}),
                    ("rank_candidates", {}), ("write_ml_report", {"notes": ""})],
    }
    seen: list = []
    monkeypatch.setattr("topodockq_agents.agents.base.run_tool_agent", fake_runner(scripts, seen))
    s = Settings(runs_dir=tmp_path / "runs", dry_run=False, verbose=False, workers=4, api_key="sk-ant-test-offline")
    st = run_pipeline("Rank complexes 1awr, 1czy, 1eg4.", s, mode="llm")

    assert st.stages == {"MainAgent": "done", "DataAgent": "done", "TopoAgent": "done", "MLAgent": "done"}
    first = next(out for agent, tool, out in seen if tool == "build_structure_manifest")
    assert "error" in first and "NOPE" in first["error"]                       # error surfaced to the model ...
    inv = [e for e in _events(st, "tool_invocation") if e["tool"] == "build_structure_manifest"]
    assert [e["status"] for e in inv] == ["error", "ok"]                         # ... and the retry recorded
    assert [e["to_agent"] for e in _events(st, "delegation")] == ["DataAgent", "TopoAgent", "MLAgent"]
    msgs = _events(st, "llm_message")
    assert {m["agent"] for m in msgs} == {"MainAgent", "DataAgent", "TopoAgent", "MLAgent"}
    assert any(m["thinking"] for m in msgs) and any(m["tool_calls"] for m in msgs)
    assert st.usage["input_tokens"] == 100 * sum(len(v) for v in scripts.values())
    ver = _events(st, "verification")[-1]
    assert ver["status"] == "passed", ver
    pred = pd.read_csv(st.require(Keys.PREDICTIONS).path)
    assert (pred.status == "scored").sum() == 3 and (pred.status == "excluded").sum() == 2


def test_project_key_comes_only_from_dotenv(tmp_path, monkeypatch):
    from topodockq_agents.llm import MissingAPIKey, make_client

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-SYSTEM-KEY-MUST-NOT-BE-USED")
    monkeypatch.delenv("TOPODOCKQ_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TOPODOCKQ_MODEL", raising=False)
    s = Settings.from_env(project_root=tmp_path, runs_dir=tmp_path / "runs")
    assert s.api_key is None
    with pytest.raises(MissingAPIKey):
        make_client(s.api_key)
    with pytest.raises(MissingAPIKey):
        run_pipeline("x", Settings(runs_dir=tmp_path / "runs", dry_run=False, api_key=None), mode="sequential")

    (tmp_path / ".env").write_text('TOPODOCKQ_ANTHROPIC_API_KEY="sk-ant-PROJECT"\nTOPODOCKQ_MODEL=claude-sonnet-5-5\n'
                                   '# comment\n', encoding="utf-8")
    s = Settings.from_env(project_root=tmp_path, runs_dir=tmp_path / "runs")
    assert s.api_key == "sk-ant-PROJECT"
    assert s.main.model == s.data_agent.model == "claude-sonnet-5-5"
    assert make_client(s.api_key).api_key == "sk-ant-PROJECT"

    (tmp_path / ".env").write_bytes(b"\xef\xbb\xbfTOPODOCKQ_ANTHROPIC_API_KEY=sk-ant-BOM\r\n"
                                    b"TOPODOCKQ_MODEL=claude-sonnet-5-5\r\n")
    assert Settings.from_env(project_root=tmp_path, runs_dir=tmp_path / "runs").api_key == "sk-ant-BOM"


def test_legacy_bca_prefix_is_ignored(tmp_path, monkeypatch):
    for name in ("TOPODOCKQ_ANTHROPIC_API_KEY", "TOPODOCKQ_MODEL", "BCA_ANTHROPIC_API_KEY", "BCA_MODEL"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text("BCA_ANTHROPIC_API_KEY=sk-ant-OLD\nBCA_MODEL=claude-sonnet-5-5\n", encoding="utf-8")
    s = Settings.from_env(project_root=tmp_path, runs_dir=tmp_path / "runs")
    assert s.api_key is None and s.main.model == "claude-opus-5-5"
