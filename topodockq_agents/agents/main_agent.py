"""MainAgent: the orchestrator. Its tools *are* the subagents."""
from __future__ import annotations

import json
from typing import Any

from anthropic import beta_tool

from ..config import AgentConfig
from ..state import STAGE_INPUTS, STAGE_OUTPUTS, Keys
from .base import COMMON_RULES, AgentResult, BaseAgent
from .data_agent import DOMAIN, DataAgent
from .ml_agent import MLAgent
from .topo_agent import TopoAgent


class MainAgent(BaseAgent):
    name = "MainAgent"
    role = "Plans the job, delegates DataAgent -> TopoAgent -> MLAgent, verifies hand-offs, handles failures, reports."

    SUBAGENTS = (DataAgent, TopoAgent, MLAgent)

    @property
    def cfg(self) -> AgentConfig:
        return self.settings.main

    def expected_outputs(self) -> list[str]:
        return [k for outs in STAGE_OUTPUTS.values() for k in outs]

    def system_prompt(self) -> str:
        roster = "\n".join(f"- {a.name}: {a.role}\n    needs {STAGE_INPUTS[a.name] or 'nothing'}; "
                           f"produces {STAGE_OUTPUTS[a.name]}" for a in self.SUBAGENTS)
        return f"""You are MainAgent, the orchestrator of TopoDockQ Agents.

{DOMAIN}
You do not touch structures, features or scores yourself. You delegate whole batches to three specialist
subagents, each exposed as a tool, and you check their hand-offs through the artifact registry.

Subagents:
{roster}

What this system can do: score and rank AlphaFold candidate structures with the published model
(inference). It cannot retrain the model, change the method, or compute true DockQ without reference
structures; if the user asks for that, say so in your report and do what is possible.

Orchestration rules:
- Read the request: which input, which complexes or models, whether the user states which chain is the
  protein and which the peptide, and whether the authors' original feature code is requested. Pass these to
  the subagents as specific instructions. Do not narrow or widen the request.
- Run the stages in order: DataAgent, then TopoAgent, then MLAgent. Each subagent call handles the whole batch.
- After DataAgent, confirm the selection against the request: the complexes and the number of structures
  per complex in data.structures (inspect_state). If they do not match what was asked, re-run DataAgent
  once with corrected instructions.
- Each stage result has a status checked by the system, not by the subagent: "done"; "partial" (valid
  outputs, but some structures failed at that stage, listed in its problems: continue and report them);
  "incomplete" or "failed" (read its problems and error).
- If DataAgent reports 0 structures ready for features, stop: do not run TopoAgent or MLAgent; report why,
  with the rule codes.
- If a stage is "failed" or "incomplete", re-run it once with corrected instructions if its problems point
  to a fix you can express (for example a wrong input path or selection that the request clarifies). If it
  fails again, stop and report the blocker and the exact problems.
- Some structures excluded by the fixed rules is normal: continue with the rest and carry the exclusions
  into the final report. Never ask a subagent to drop or re-include individual structures.

Final report for the user (plain text, specific and quantitative):
- input and selection (complexes, structures per complex); structures scored and excluded, with rule codes;
- per complex: the top-1 candidate, its p-DockQ, whether it is predicted high quality (> 0.82), its
  warnings and ranking status, for at most 30 complexes (otherwise point to the ranking file). Name every
  top-1 with the warning W_CHAIN_ROLE_SUSPICIOUS or the status needs_confirmation: its chain roles should
  be confirmed before its ranking is used;
- feature engine used, any retries, partial stages or failures, and limitations. p-DockQ is a predicted
  DockQ, not a measurement. An independent deterministic re-run checks your results after you finish; do
  not claim it.

{COMMON_RULES}"""

    def user_message(self, task: str) -> str:
        source = self.settings.input_source or "not given in the settings; take it from the request"
        return (f"USER REQUEST:\n{task}\n\nInput source provided by the user: {source}\n"
                f"Parallel workers available: {self.settings.workers}\n\n"
                f"CURRENT PIPELINE STATE:\n{self.state.summary()}")

    # ------------------------------------------------------------------ tools = subagents
    def _spawn(self, cls: type[BaseAgent]) -> BaseAgent:
        return cls(self.settings, self.state, client=self._client)

    def _run_sub(self, cls: type[BaseAgent], instructions: str) -> str:
        self.state.record("delegation", self.name, to_agent=cls.name, instructions=instructions)
        result = self._spawn(cls).run(instructions)
        return json.dumps(result.to_dict(), default=str)

    def build_tools(self) -> list[Any]:
        agent = self

        @beta_tool
        def run_data_agent(instructions: str) -> str:
            """Delegate to DataAgent: build the structure list from the input, validate it, extract interfaces.

            Args:
                instructions: Input source, selection (complexes, models) and chain assignment from the request.
            """
            return agent._run_sub(DataAgent, instructions)

        @beta_tool
        def run_topo_agent(instructions: str) -> str:
            """Delegate to TopoAgent: generate the topological features for every ready interface.

            Args:
                instructions: Feature engine to use ("compat" default, or "original" if the request asks for the
                    authors' original code) and anything the TopoAgent must know.
            """
            return agent._run_sub(TopoAgent, instructions)

        @beta_tool
        def run_ml_agent(instructions: str) -> str:
            """Delegate to MLAgent: verify the published model, predict p-DockQ, rank candidates per complex.

            Args:
                instructions: Anything the MLAgent must know (for example exclusions to mention in its summary).
            """
            return agent._run_sub(MLAgent, instructions)

        @beta_tool
        def inspect_state() -> str:
            """Show the current pipeline state: stage statuses and registered artifacts with their metadata."""
            return agent.state.summary()

        return [run_data_agent, run_topo_agent, run_ml_agent, inspect_state]

    # ------------------------------------------------------------------ dry run = fixed order
    def dry_run(self, task: str) -> str:
        results: list[AgentResult] = []
        for cls in self.SUBAGENTS:
            self.state.record("delegation", self.name, to_agent=cls.name, instructions=task)
            res = self._spawn(cls).run(task)
            results.append(res)
            if res.status not in ("done", "partial"):
                break
            if cls is DataAgent and self.state.require(Keys.INTERFACES).meta.get("ready_for_features", 0) == 0:
                results.append(AgentResult("MainAgent", "stopped", "no structures ready for features"))
                break
        return "\n".join(f"{r.agent}: {r.status} - {r.summary}" for r in results)

    def stage_checks(self):
        """The orchestration succeeded only if the latest attempt of every specialist is done or partial."""
        latest = {}
        for e in self.state.events:
            if e["event"] == "stage_end" and e["agent"] != self.name:
                latest[e["agent"]] = e["status"]
        problems = [f"{a.name} latest status: {latest.get(a.name, 'not run')}" for a in self.SUBAGENTS
                    if latest.get(a.name) not in ("done", "partial")]
        if problems:
            return "incomplete", problems, latest
        return ("partial" if "partial" in latest.values() else "done"), [], latest
