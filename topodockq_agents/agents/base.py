"""BaseAgent: the common shape of every agent in the pipeline.

An agent is: a system prompt, a set of typed tools bound to the shared
PipelineState, and a ``run(task)`` that drives a Claude tool-runner loop. In
dry-run mode ``run`` calls ``dry_run`` instead, which executes the same tool
functions in a fixed order with no API traffic.

Everything an agent does is recorded in the run trace (state.record ->
trace.jsonl): stage start and end with the instructions and final summary, every
tool invocation with its arguments, duration, outcome and a result preview, and
every model message with its reasoning summary, text and tool calls.
"""
from __future__ import annotations

import json
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import anthropic

from ..config import AgentConfig, Settings
from ..llm import make_client, run_tool_agent
from ..state import STAGE_INPUTS, STAGE_OUTPUTS, MissingArtifact, PipelineState

log = logging.getLogger("topodockq_agents.agents")

RESULT_PREVIEW_CHARS = 1500

COMMON_RULES = """\
General rules for every agent in this pipeline:
- All numbers come from your tools. Never compute, estimate, or invent scores, counts, file paths or
  results yourself, and never describe a structure you have not been told about by a tool.
- The scientific method is fixed: the authors' interface extraction (10 A), their feature generator and
  the published model and preprocessing. You choose which tools to run and with which allowed options;
  you cannot change the method.
- Structures are excluded only by the fixed validation rules, which the tools apply and record with a
  rule code. You cannot exclude, drop, or re-include a structure yourself; report exclusions as they are.
- Work on the whole batch with one tool call per step; never loop over structures one by one.
- Artifacts are the contract between stages: tools register them under fixed keys. Make sure every
  expected output key exists before you finish.
- If a tool returns an error, read it. Retry only if a different allowed option could fix it, at most
  twice. Otherwise stop and state the blocker and the exact error.
- Finish with a short plain-text summary: what you did, the key counts, any exclusions or failures with
  their rule codes, and anything the next stage must know.
"""


def tool_call(fn: Any, *args: Any, **kwargs: Any) -> str:
    """Run a tool implementation and return a JSON string (errors become ``{"error": ...}``)."""
    try:
        return json.dumps(fn(*args, **kwargs), default=str)
    except Exception as e:  # noqa: BLE001
        log.warning("tool %s failed: %s", getattr(fn, "__name__", fn), e)
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


@dataclass
class AgentResult:
    agent: str
    status: str                       # done | partial | incomplete | failed
    summary: str
    artifacts: list[str] = field(default_factory=list)
    missing_outputs: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    attempt: int = 1
    seconds: float = 0.0
    problems: list[str] = field(default_factory=list)      # why the Python-side stage checks did not pass
    checks: dict[str, Any] = field(default_factory=dict)   # what the stage checks measured

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BaseAgent(ABC):
    name: str = "BaseAgent"
    role: str = ""                    # one-line description shown to the orchestrator

    def __init__(self, settings: Settings, state: PipelineState, client: anthropic.Anthropic | None = None):
        self.settings = settings
        self.state = state
        self._client = client
        self._turn = 0

    # ------------------------------------------------------------------ to implement
    @property
    @abstractmethod
    def cfg(self) -> AgentConfig: ...

    @abstractmethod
    def system_prompt(self) -> str: ...

    @abstractmethod
    def build_tools(self) -> list[Any]:
        """Return @beta_tool-wrapped closures bound to ``self.state``."""

    @abstractmethod
    def dry_run(self, task: str) -> str:
        """Execute the tool functions directly in a sensible fixed order; return a summary."""

    # ------------------------------------------------------------------ shared behaviour
    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = make_client(self.settings.api_key)
        return self._client

    def required_inputs(self) -> list[str]:
        return STAGE_INPUTS.get(self.name, [])

    def expected_outputs(self) -> list[str]:
        return STAGE_OUTPUTS.get(self.name, [])

    def check_inputs(self) -> None:
        missing = [k for k in self.required_inputs() if not self.state.has(k)]
        if missing:
            raise MissingArtifact(f"{self.name} cannot start; missing artifacts: {missing}")

    def validate_stage(self, started_at: float) -> tuple[str, list[str], dict[str, Any]]:
        """Python-side success criteria, independent of what the model says.

        Every expected output must exist and have been produced during this attempt (so a retry
        cannot pass on a previous attempt's files); then the stage-specific checks decide between
        done, partial (valid outputs, some structures failed at this stage) and incomplete.
        """
        problems = [f"missing {k}" for k in self.expected_outputs() if not self.state.has(k)]
        problems += [f"{k} was not produced in this attempt" for k in self.expected_outputs()
                     if self.state.has(k) and self.state.require(k).created_at < started_at]
        if problems:
            return "incomplete", problems, {}
        try:
            return self.stage_checks()
        except Exception as e:  # noqa: BLE001 - a check that cannot run is a failed check
            return "incomplete", [f"stage checks could not run: {type(e).__name__}: {e}"], {}

    def stage_checks(self) -> tuple[str, list[str], dict[str, Any]]:
        """Stage-specific checks of the outputs' contents; overridden by each agent."""
        return "done", [], {}

    def user_message(self, task: str) -> str:
        return (
            f"TASK FROM ORCHESTRATOR:\n{task}\n\n"
            f"CURRENT PIPELINE STATE:\n{self.state.summary()}\n\n"
            f"Inputs you may rely on: {self.required_inputs() or 'none'}\n"
            f"Outputs you must produce (artifact keys): {self.expected_outputs()}"
        )

    def call(self, fn: Callable[..., Any], **kwargs: Any) -> str:
        """Run a tool function bound to the shared state, record the invocation, return JSON for the model."""
        t0 = time.time()
        args = {k: v for k, v in kwargs.items()}
        try:
            result = fn(self.state, **kwargs)
            text = json.dumps(result, default=str)
            self.state.record("tool_invocation", self.name, tool=fn.__name__, args=args, status="ok",
                              seconds=round(time.time() - t0, 2), result_preview=text[:RESULT_PREVIEW_CHARS])
            return text
        except Exception as e:  # noqa: BLE001 - surfaced to the model as data
            err = f"{type(e).__name__}: {e}"
            log.warning("[%s] tool %s failed: %s", self.name, fn.__name__, err)
            self.state.record("tool_invocation", self.name, tool=fn.__name__, args=args, status="error",
                              seconds=round(time.time() - t0, 2), error=err)
            return json.dumps({"error": err})

    def _attempt(self) -> int:
        return 1 + sum(1 for e in self.state.events if e["event"] == "stage_start" and e["agent"] == self.name)

    def run(self, task: str) -> AgentResult:
        attempt = self._attempt()
        t0 = time.time()
        self.state.set_stage(self.name, "running")
        self.state.record("stage_start", self.name, task=task, attempt=attempt, dry_run=self.settings.dry_run,
                          model=None if self.settings.dry_run else self.cfg.model,
                          effort=None if self.settings.dry_run else self.cfg.effort)
        before = {k: a.path for k, a in self.state.artifacts.items()}
        usage: dict[str, int] = {}
        tool_calls: list[str] = []
        try:
            self.check_inputs()
            if self.settings.dry_run:
                summary = self.dry_run(task)
            else:
                run = run_tool_agent(
                    self.client, self.cfg,
                    system=self.system_prompt(), user=self.user_message(task), tools=self.build_tools(),
                    enable_fallbacks=self.settings.enable_fallbacks, on_message=self._on_message,
                )
                summary, usage, tool_calls = run.final_text, run.usage, [tc.name for tc in run.tool_calls]
                self.state.add_usage(usage)
            produced = [k for k, a in self.state.artifacts.items() if before.get(k) != a.path]
            missing = [k for k in self.expected_outputs() if not self.state.has(k)]
            status, problems, checks = self.validate_stage(t0)
            result = AgentResult(self.name, status, summary, produced, missing, tool_calls, usage,
                                 attempt=attempt, seconds=round(time.time() - t0, 1), problems=problems,
                                 checks=checks)
        except Exception as e:  # noqa: BLE001 - surfaced to the orchestrator as data
            log.error("%s failed: %s", self.name, e)
            log.debug(traceback.format_exc())
            produced = [k for k, a in self.state.artifacts.items() if before.get(k) != a.path]
            result = AgentResult(self.name, "failed", f"{self.name} failed: {e}", produced,
                                 self.expected_outputs(), tool_calls, usage, error=f"{type(e).__name__}: {e}",
                                 attempt=attempt, seconds=round(time.time() - t0, 1))
        self.state.set_stage(self.name, result.status)
        self.state.record("stage_end", self.name, status=result.status, attempt=attempt, artifacts=result.artifacts,
                          missing_outputs=result.missing_outputs, tool_calls=result.tool_calls, usage=result.usage,
                          seconds=result.seconds, error=result.error, problems=result.problems,
                          checks=result.checks, summary=result.summary)
        self.state.save()
        return result

    def _on_message(self, message: Any) -> None:
        """Record each model message: reasoning summary, text, tool calls, stop reason and usage."""
        self._turn += 1
        thinking, texts, calls = [], [], []
        for block in message.content:
            if block.type == "thinking" and getattr(block, "thinking", ""):
                thinking.append(block.thinking)
            elif block.type == "text" and block.text.strip():
                texts.append(block.text.strip())
            elif block.type == "tool_use":
                calls.append({"name": block.name, "input": dict(block.input)})
        u = message.usage
        self.state.record("llm_message", self.name, turn=self._turn, model=getattr(message, "model", None),
                          stop_reason=message.stop_reason, thinking=thinking, text=texts, tool_calls=calls,
                          usage={k: int(getattr(u, k, 0) or 0) for k in ("input_tokens", "output_tokens",
                                                                          "cache_read_input_tokens",
                                                                          "cache_creation_input_tokens")})
        if self.settings.verbose:
            for c in calls:
                log.info("[%s] -> %s(%s)", self.name, c["name"], json.dumps(c["input"], default=str)[:200])
            for t in texts:
                log.info("[%s] %s", self.name, t[:500])
