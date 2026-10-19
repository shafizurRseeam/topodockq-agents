"""MLAgent: score every structure with the published TopoDockQ model and rank candidates per complex."""
from __future__ import annotations

from typing import Any

from anthropic import beta_tool

from ..config import AgentConfig
from ..state import Keys
from ..tools import ml_tools as mt
from .base import COMMON_RULES, BaseAgent
from .data_agent import DOMAIN


class MLAgent(BaseAgent):
    name = "MLAgent"
    role = ("Verifies the published model package, predicts p-DockQ for every structure with features, and "
            "ranks the candidates of each complex; produces ml.predictions, ml.ranking, ml.report.")

    @property
    def cfg(self) -> AgentConfig:
        return self.settings.ml_agent

    def system_prompt(self) -> str:
        return f"""You are MLAgent, the final stage of TopoDockQ Agents.

{DOMAIN}
Your job: score every structure with the published TopoDockQ network and rank the candidates of each
complex. You never train, tune or change the model, its preprocessing or its threshold.

Workflow:
1. check_model_package. It verifies the saved column selection, scaler and checkpoint against their recorded
   hashes. If it reports an error, stop: never score with an unverified model.
2. predict_pdockq with device="cpu". Structures excluded earlier stay in the output with their rule codes.
3. rank_candidates. For each complex the top-1 candidate is the one with the highest p-DockQ (paper rule),
   flagged "predicted high quality" when p-DockQ > 0.82 (the paper's threshold; DockQ > 0.8 is high quality).
   Each top-1 carries its validation warnings and a ranking_status: "accepted", or "needs_confirmation" in
   strict mode when its chain roles were not confirmed and look implausible.
4. write_ml_report with short notes.

In your summary give: structures scored and excluded (by rule code), complexes ranked, the top-1 candidate
and p-DockQ for each complex (at most 30; otherwise point to ranking.csv), how many are predicted high
quality, and every top-1 that has warnings or needs confirmation. p-DockQ is a prediction, not a measured
DockQ.

Outputs you own: {Keys.PREDICTIONS}, {Keys.RANKING}, {Keys.ML_REPORT}.

{COMMON_RULES}"""

    def build_tools(self) -> list[Any]:
        agent = self

        @beta_tool
        def check_model_package() -> str:
            """Verify the published preprocessing package and checkpoint against their recorded hashes."""
            return agent.call(mt.check_model_package)

        @beta_tool
        def predict_pdockq(device: str = "cpu") -> str:
            """Predict p-DockQ for every structure with features; register ml.predictions (exclusions carried through).

            Args:
                device: "cpu" (default) or "cuda".
            """
            return agent.call(mt.predict_pdockq, device=device)

        @beta_tool
        def rank_candidates() -> str:
            """Rank candidates per complex (top-1 by p-DockQ, high quality if > 0.82); register ml.ranking."""
            return agent.call(mt.rank_candidates, strict=agent.settings.strict)

        @beta_tool
        def write_ml_report(notes: str = "") -> str:
            """Write ml.report with the model identity, prediction counts and ranking summary.

            Args:
                notes: Notes for the final report.
            """
            return agent.call(mt.write_ml_report, notes=notes)

        return [check_model_package, predict_pdockq, rank_candidates, write_ml_report]

    def stage_checks(self):
        """Every structure is in ml.predictions exactly once (scored with a finite p-DockQ, or excluded with
        a rule code), and ml.ranking has exactly one row per complex with a scored candidate."""
        import numpy as np
        import pandas as pd

        s = self.state
        it, ft, pr, rk, rp = (s.require(k) for k in (Keys.INTERFACES, Keys.FEATURES, Keys.PREDICTIONS,
                                                      Keys.RANKING, Keys.ML_REPORT))
        problems = []
        if pr.created_at < max(it.created_at, ft.created_at):
            problems.append("ml.predictions is older than its inputs")
        if rk.created_at < pr.created_at or rp.created_at < rk.created_at:
            problems.append("ml.ranking / ml.report were not produced after ml.predictions")
        iface = pd.read_csv(it.path, keep_default_na=False)
        pred = pd.read_csv(pr.path, keep_default_na=False)
        keys = lambda t: sorted(zip(t.pdb_id.astype(str), t.af_model_id.astype(int)))  # noqa: E731
        if keys(pred) != keys(iface):
            problems.append(f"ml.predictions lists {len(pred)} structures, data.interfaces {len(iface)}")
        scored = pred[pred.status == "scored"]
        p = pd.to_numeric(scored.p_dockq, errors="coerce")
        if not np.isfinite(p).all():
            problems.append(f"{int((~np.isfinite(p)).sum())} scored structures have no finite p-DockQ")
        no_rule = pred[(pred.status == "excluded") & (pred.reason == "")]
        if len(no_rule):
            problems.append(f"{len(no_rule)} excluded structures have no rule code")
        ranking = pd.read_csv(rk.path, keep_default_na=False)
        ids = list(ranking.pdb_id.astype(str))
        if len(ids) != len(set(ids)) or set(ids) != set(scored.pdb_id.astype(str)):
            problems.append("ml.ranking does not have exactly one row per complex with a scored candidate")
        details = {"structures": len(pred), "scored": len(scored), "complexes_ranked": len(ids),
                   "needs_confirmation": int((ranking.get("ranking_status", pd.Series(dtype=str)) == "needs_confirmation").sum())}
        return ("incomplete" if problems else "done"), problems, details

    def dry_run(self, task: str) -> str:
        import json

        def call(fn, **kw):
            out = json.loads(self.call(fn, **kw))
            if isinstance(out, dict) and "error" in out and len(out) == 1:
                raise RuntimeError(out["error"])
            return out

        call(mt.check_model_package)
        pred = call(mt.predict_pdockq, device="cpu")
        rank = call(mt.rank_candidates, strict=self.settings.strict)
        call(mt.write_ml_report, notes="dry run")
        return (f"[dry-run] scored {pred['scored']}, excluded {pred['excluded']}; ranked {rank['complexes_ranked']} "
                f"complexes, {rank['predicted_high_quality']} predicted high quality")
