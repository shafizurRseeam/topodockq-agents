"""topodockq_agents - the agent layer of TopoDockQ Agents.

    MainAgent (orchestrator)
      |-- DataAgent  -> input structures, validation, protein-peptide interfaces
      |-- TopoAgent  -> 2,754 topological features per interface
      `-- MLAgent    -> p-DockQ with the published model, top-1 ranking per complex

Each agent is an Anthropic tool-runner loop with its own system prompt and a small
set of typed tools. The tools call the verified deterministic code in
``tdq_adapters``; agents exchange artifacts (files registered in PipelineState),
never feature arrays, so the LLM context stays small.
"""
from .config import DEFAULT_MODEL, AgentConfig, Settings
from .state import Keys, MissingArtifact, PipelineState

__all__ = ["Settings", "AgentConfig", "DEFAULT_MODEL", "PipelineState", "Keys", "MissingArtifact"]
