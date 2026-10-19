"""TopoAgent: compute the 2,754 TopoDockQ topological features for every ready interface."""
from __future__ import annotations

from typing import Any

from anthropic import beta_tool

from ..config import AgentConfig
from ..state import Keys
from ..tools import topo_tools as tt
from .base import COMMON_RULES, BaseAgent
from .data_agent import DOMAIN


class TopoAgent(BaseAgent):
    name = "TopoAgent"
    role = ("Generates the 2,754 topological features (Rips/alpha complexes, persistent Laplacian spectra, "
            "barcodes) for every ready interface with a verified engine; produces topo.features, topo.report.")

    @property
    def cfg(self) -> AgentConfig:
        return self.settings.topo_agent

    def system_prompt(self) -> str:
        return f"""You are TopoAgent, the second stage of TopoDockQ Agents.

{DOMAIN}
Your job: generate the topological features of every interface the DataAgent marked ready, using the
authors' published settings (locked by the tools).

Two engines are available and were verified to give identical features (zero mismatches in 358,020
values against the published features):
  "compat"   Python 3.13 port of the authors' feature code; about 1.6 s per structure. Default.
  "original" the authors' own compiled code in its Python 3.8 environment; about 2.1 s per structure.
Use "original" only when the task asks for the authors' original code, or as the second attempt after
failures.

Workflow:
1. generate_features with the requested engine (default "compat") and workers={self.settings.workers}.
2. If some structures failed, call generate_features once more with the other engine and
   retry_failed_only=True. Do not retry more than once.
3. check_features. It must report ok. Structures still missing after the retry are excluded downstream
   with rule E_FEATURES_FAILED; name them in your summary.
4. write_topo_report with short notes (engine used, attempts, failures).

Outputs you own: {Keys.FEATURES}, {Keys.TOPO_REPORT}.

{COMMON_RULES}"""

    def build_tools(self) -> list[Any]:
        agent = self

        @beta_tool
        def generate_features(engine: str = "compat", workers: int = 8, retry_failed_only: bool = False) -> str:
            """Generate the 2,754 published-schema features for the ready interfaces; register topo.features.

            Args:
                engine: "compat" (Python 3.13 port, default) or "original" (the authors' code, Python 3.8).
                workers: Structures processed in parallel.
                retry_failed_only: Only structures still missing from the registered table; merge the results.
            """
            return agent.call(tt.generate_features, engine=engine, workers=workers,
                              retry_failed_only=retry_failed_only)

        @beta_tool
        def check_features() -> str:
            """Check topo.features: one row per ready structure, all published columns, finite values."""
            return agent.call(tt.check_features)

        @beta_tool
        def write_topo_report(notes: str = "") -> str:
            """Write topo.report with the engine, attempts, failures and the final check.

            Args:
                notes: Notes for the MLAgent and the final report.
            """
            return agent.call(tt.write_topo_report, notes=notes)

        return [generate_features, check_features, write_topo_report]

    def stage_checks(self):
        """The registered feature table passes check_features; missing rows make the stage partial."""
        feats, rep = self.state.require(Keys.FEATURES), self.state.require(Keys.TOPO_REPORT)
        chk = tt.check_features(self.state)
        problems = []
        if rep.created_at < feats.created_at:
            problems.append("topo.report is older than topo.features")
        if chk["duplicate_keys"]:
            problems.append(f"{chk['duplicate_keys']} duplicate feature rows")
        if chk["unexpected_rows"]:
            problems.append(f"{chk['unexpected_rows']} feature rows for structures that are not ready")
        problems += [f"schema: {s}" for s in chk["schema_issues"]]
        if problems:
            return "incomplete", problems, chk
        if chk["missing_count"]:
            return "partial", [f"{chk['missing_count']} ready structures have no features and will be excluded "
                               f"with E_FEATURES_FAILED: {chk['missing_structures']}"], chk
        return "done", [], chk

    def dry_run(self, task: str) -> str:
        import json

        def call(fn, **kw):
            out = json.loads(self.call(fn, **kw))
            if isinstance(out, dict) and "error" in out and len(out) == 1:
                raise RuntimeError(out["error"])
            return out

        gen = call(tt.generate_features, engine="compat", workers=self.settings.workers)
        if gen["failed"]:
            call(tt.generate_features, engine="original", workers=self.settings.workers, retry_failed_only=True)
        chk = call(tt.check_features)
        call(tt.write_topo_report, notes="dry run: compat engine")
        return f"[dry-run] features for {chk['rows']}/{chk['ready_structures']} ready structures; check ok={chk['ok']}"
