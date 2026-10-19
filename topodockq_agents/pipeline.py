"""Entry point: run the TopoDockQ agent pipeline from Python or the command line.

    uv run topodockq --input data/external/PFPD_AF2_models --dry-run          # no API calls
    uv run topodockq --input <folder or manifest.csv> --task "..."              # MainAgent orchestrates
    uv run topodockq --input <...> --mode sequential                            # fixed order, each agent Claude-driven
    uv run topodockq --check                                                    # validate key and models (unbilled)

(``uv run topodockq`` is the same as ``python -m topodockq_agents.pipeline``.)

After the agents finish, the same structures are re-run through the deterministic
pipeline (tdq_adapters.pipeline) and compared (topodockq_agents.verify); disable with --no-verify.
Every run, even one that crashes, ends with a recorded run status, a saved trace and report.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

from .agents import DataAgent, MainAgent, MLAgent, TopoAgent
from .config import Settings
from .llm import make_client
from .state import Keys, PipelineState
from .verify import verify_against_deterministic

log = logging.getLogger("topodockq_agents")

DEFAULT_TASK = (
    "Score every AlphaFold candidate structure in the input with TopoDockQ, rank the candidates of each "
    "complex, and report the top-ranked candidate per complex with its p-DockQ."
)
OK_STATUSES = ("done", "partial")      # partial: valid outputs, some structures failed at that stage


def run_status(state: PipelineState, verification: dict) -> str:
    """"passed" only if every stage finished (done/partial), nothing crashed and verification passed or was off."""
    if any(e["event"] == "run_error" for e in state.events):
        return "failed"
    if not state.stages or any(s not in OK_STATUSES for s in state.stages.values()):
        return "failed"
    return "passed" if verification.get("status") in ("passed", "disabled") else "failed"


def run_pipeline(task: str = DEFAULT_TASK, settings: Settings | None = None, mode: str = "llm") -> PipelineState:
    """Run the pipeline and return the final state. ``mode`` is "llm" or "sequential"."""
    settings = settings or Settings.from_env()
    if mode not in ("llm", "sequential"):
        raise ValueError("mode must be 'llm' or 'sequential'")
    if not settings.dry_run:
        make_client(settings.api_key)  # fail fast with a clear message before creating a run
    state = PipelineState.new(settings.runs_dir, task)
    t0 = time.time()
    agents_seconds: float | None = None
    verification: dict = {"status": "not_run"}
    try:
        state.record("run_start", None, mode=mode, dry_run=settings.dry_run, input_source=settings.input_source,
                     workers=settings.workers, verify=settings.verify, strict=settings.strict,
                     expected_structures=settings.expected_structures,
                     models={"MainAgent": settings.main.model, "DataAgent": settings.data_agent.model,
                             "TopoAgent": settings.topo_agent.model, "MLAgent": settings.ml_agent.model})
        log.info("run %s -> %s (mode=%s, dry_run=%s)", state.run_id, state.run_dir, mode, settings.dry_run)
        if mode == "sequential":
            for cls in (DataAgent, TopoAgent, MLAgent):
                res = cls(settings, state).run(task)
                log.info("%s: %s", res.agent, res.status)
                if res.status not in OK_STATUSES:
                    break
        else:
            res = MainAgent(settings, state).run(task)
            state.record("final_report", "MainAgent", text=res.summary)
        agents_seconds = round(time.time() - t0, 1)
        if settings.verify:
            try:
                verification = verify_against_deterministic(state, settings)
            except Exception as e:  # noqa: BLE001 - a crashing verifier is recorded, never silently passed
                verification = {"status": "error", "error": f"{type(e).__name__}: {e}",
                                "traceback": traceback.format_exc()}
                state.record("verification", None, **verification)
        else:
            verification = {"status": "disabled"}
    except Exception as e:  # noqa: BLE001 - record any crash, then still finish the run below
        log.error("run failed: %s", e)
        state.record("run_error", None, type=type(e).__name__, message=str(e), traceback=traceback.format_exc())
    finally:
        if agents_seconds is None:
            agents_seconds = round(time.time() - t0, 1)
        status = run_status(state, verification)
        state.record("run_end", None, run_status=status, agents_seconds=agents_seconds,
                     verification=verification.get("status"), usage=state.usage)
        state.save()
        try:
            write_report(state, verification, agents_seconds)
        except Exception as e:  # noqa: BLE001
            state.record("report_error", None, error=f"{type(e).__name__}: {e}")
            (state.run_dir / "report.md").write_text(
                f"# Run {state.run_id}\n\n**Run status:** {status}\n\nThe full report could not be written "
                f"({type(e).__name__}: {e}). See trace.jsonl.\n", encoding="utf-8")
    return state


def _selection_lines(state: PipelineState, verification: dict) -> list[str]:
    if not state.has(Keys.STRUCTURES):
        return []
    m = state.require(Keys.STRUCTURES).meta
    lines = ["", "## Input selection", "",
             f"- source `{m.get('source')}` (layout {m.get('layout')}): {m.get('complexes_found')} complexes found; "
             f"selected {m.get('structures_selected')} structures from {m.get('complexes_selected')} complexes",
             f"- selection used: `{json.dumps(m.get('selection'))}`; chain roles confirmed for "
             f"{m.get('chains_confirmed', 0)} of {m.get('structures_selected')} structures"]
    audit = (verification or {}).get("selection") or {}
    if audit.get("per_complex"):
        lines += ["", "| complex | models available | models selected | selected model IDs |", "|---|---|---|---|"]
        for c in audit["per_complex"][:30]:
            lines.append(f"| {c['pdb_id']} | {c['available']} | {c['selected']} | {c['selected_ids']} |")
    return lines


def write_report(state: PipelineState, verification: dict | None = None, agents_seconds: float | None = None) -> Path:
    verification = verification or {}
    lines = [f"# Run {state.run_id}", "", f"**Request:** {state.task}", ""]
    start = next((e for e in state.events if e["event"] == "run_start"), {})
    end = next((e for e in reversed(state.events) if e["event"] == "run_end"), {})
    lines += [f"**Run status:** {end.get('run_status', 'unknown')}  **Input:** `{start.get('input_source')}`  "
              f"**Mode:** {start.get('mode')}  **Dry run:** {start.get('dry_run')}  **Strict:** {start.get('strict')}",
              "", "## Stages", ""]
    for e in state.events:
        if e["event"] == "stage_end":
            lines.append(f"- {e['agent']} (attempt {e.get('attempt')}): **{e['status']}** in {e.get('seconds')} s"
                         + (f" - error: {e['error']}" if e.get("error") else "")
                         + (f" - problems: {'; '.join(e['problems'][:5])}" if e.get("problems") else ""))
    for e in state.events:
        if e["event"] == "run_error":
            lines += ["", f"**Run error:** {e['type']}: {e['message']}"]
    lines += _selection_lines(state, verification)
    lines += ["", "## Artifacts", ""]
    for k, a in state.artifacts.items():
        lines.append(f"- `{k}` ({a.kind}, {a.produced_by}): `{a.path}`")
    if state.has(Keys.PREDICTIONS):
        m = state.require(Keys.PREDICTIONS).meta
        lines += ["", "## Predictions", "", f"- scored: {m['scored']}, excluded: {m['excluded']} "
                  f"{json.dumps(m['exclusions_by_reason'])}"]
    if state.has(Keys.RANKING):
        r = pd.read_csv(state.require(Keys.RANKING).path, keep_default_na=False)
        lines += ["", "## Top-ranked candidate per complex", "",
                  "| complex | top-1 | p-DockQ | predicted high quality (> 0.82) | warnings | ranking status | scored | excluded |",
                  "|---|---|---|---|---|---|---|---|"]
        for x in r.head(50).itertuples(index=False):
            lines.append(f"| {x.pdb_id} | {x.top1_sample_id} | {float(x.top1_p_dockq):.4f} | "
                         f"{x.predicted_high_quality} | {x.top1_warnings or '-'} | {x.ranking_status} | "
                         f"{x.candidates_scored} | {x.candidates_excluded} |")
    lines += ["", "## Independent deterministic verification", "", "```json",
              json.dumps({k: v for k, v in verification.items()
                          if k in ("status", "checks", "structures", "scored_both", "max_abs_pdockq_diff", "reason",
                                   "error")}, indent=2), "```"]
    final = next((e["text"] for e in reversed(state.events) if e["event"] == "final_report"), None)
    if final:
        lines += ["", "## Orchestrator report", "", final]
    n_llm = sum(1 for e in state.events if e["event"] == "llm_message")
    n_tools = sum(1 for e in state.events if e["event"] == "tool_invocation")
    lines += ["", "## Execution", "", f"- agents wall time: {agents_seconds} s; model messages: {n_llm}; "
              f"tool invocations: {n_tools}", "- token usage:", "", "```json", json.dumps(state.usage, indent=2),
              "```", "", f"Full trace: `{state.trace_path.name}` (one JSON event per line).", ""]
    out = state.run_dir / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def _mask(secret: str | None) -> str:
    if not secret:
        return "<missing>"
    return secret[:12] + "..." + secret[-4:] if len(secret) > 20 else "<set, too short to be valid>"


def check_setup(settings: Settings) -> int:
    """Print the effective configuration (key masked) and validate the key + model IDs against the API."""
    import anthropic

    env_file = settings.project_root / ".env"
    print(f".env file     : {env_file} ({'found' if env_file.exists() else 'NOT FOUND'})")
    print(f"API key       : {_mask(settings.api_key)}  (TOPODOCKQ_ANTHROPIC_API_KEY in .env)")
    for label, cfg in (("MainAgent", settings.main), ("DataAgent", settings.data_agent),
                       ("TopoAgent", settings.topo_agent), ("MLAgent", settings.ml_agent)):
        print(f"{label:<14}: model={cfg.model} effort={cfg.effort}")
    print(f"input source  : {settings.input_source}")
    print(f"runs dir      : {settings.runs_dir}")
    if not settings.api_key:
        print("\nRESULT: no API key. Add TOPODOCKQ_ANTHROPIC_API_KEY=... to .env (not .env.example); see README.")
        return 1
    try:
        client = make_client(settings.api_key)
        for model in sorted({settings.main.model, settings.data_agent.model, settings.topo_agent.model, settings.ml_agent.model}):
            info = client.models.retrieve(model)   # unbilled; validates auth + model id
            print(f"model check   : {model} -> OK ({info.display_name})")
    except anthropic.AuthenticationError:
        print("\nRESULT: the API rejected the key (401). Check for a copy/paste error or that the key is active.")
        return 1
    except anthropic.NotFoundError as e:
        print(f"\nRESULT: a model id is not recognised: {e.message}")
        return 1
    except anthropic.APIConnectionError:
        print("\nRESULT: could not reach api.anthropic.com (network / proxy).")
        return 1
    print("\nRESULT: setup OK - key accepted and all model ids valid.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="topodockq", description="TopoDockQ Agents: score and rank candidate structures")
    p.add_argument("--task", default=DEFAULT_TASK, help="natural-language request for the orchestrator")
    p.add_argument("--input", default=None, help="folder of structures or manifest .csv")
    p.add_argument("--mode", choices=["llm", "sequential"], default="llm")
    p.add_argument("--dry-run", action="store_true", help="no API calls; run tool functions in fixed order")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--strict", action="store_true",
                   help="a top-1 with unconfirmed, implausible chain roles needs confirmation instead of being accepted")
    p.add_argument("--expect-structures", type=int, default=None,
                   help="number of structures you expect to be scored; checked by the selection audit")
    p.add_argument("--no-verify", action="store_true", help="skip the deterministic re-run and comparison")
    p.add_argument("--runs-dir", default=None)
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--check", action="store_true", help="show effective config (key masked) and validate it against the API")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    overrides: dict = {"verbose": not args.quiet}
    if args.dry_run:
        overrides["dry_run"] = True
    if args.input:
        overrides["input_source"] = args.input
    if args.workers:
        overrides["workers"] = args.workers
    if args.strict:
        overrides["strict"] = True
    if args.expect_structures is not None:
        overrides["expected_structures"] = args.expect_structures
    if args.no_verify:
        overrides["verify"] = False
    if args.runs_dir:
        overrides["runs_dir"] = Path(args.runs_dir)
    settings = Settings.from_env(**overrides)

    if args.check:
        return check_setup(settings)

    state = run_pipeline(args.task, settings, mode=args.mode)
    end = next((e for e in reversed(state.events) if e["event"] == "run_end"), {})
    print(f"\nrun dir      : {state.run_dir}\nrun status   : {end.get('run_status')}\n"
          f"stages       : {json.dumps(state.stages)}\nverification : {end.get('verification')}\n"
          f"usage        : {json.dumps(state.usage)}\nreport       : {state.run_dir / 'report.md'}")
    return 0 if end.get("run_status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
