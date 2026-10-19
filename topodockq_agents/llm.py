"""Thin wrapper around the Anthropic SDK tool runner used by every agent.

One function, ``run_tool_agent``, drives a complete agentic loop: Claude reads the
system prompt + task, calls the typed tools you pass in, sees their results, and
loops until it stops calling tools. Token usage and the sequence of tool calls
are returned for logging and cost tracking.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

from .config import AgentConfig

log = logging.getLogger("topodockq_agents.llm")

# Server-side refusal fallback (beta). On a policy refusal the API re-runs the request
# on a substitute model inside the same call. Disable with Settings.enable_fallbacks=False.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AgentRefused(RuntimeError):
    """The model (and any fallback) declined the request."""


class MissingAPIKey(RuntimeError):
    """No project-specific API key was configured."""


@dataclass
class ToolCall:
    name: str
    input: dict[str, Any]


@dataclass
class LLMRun:
    final_text: str
    stop_reason: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    turns: int = 0


def make_client(api_key: str | None, **kwargs: Any) -> anthropic.Anthropic:
    """Build a client with the project-specific key only.

    The key is passed explicitly so the SDK never falls back to a system-wide
    ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or `ant auth login` profile.
    """
    if not api_key:
        raise MissingAPIKey(
            "No project API key configured. Create <project>/.env with a line "
            "TOPODOCKQ_ANTHROPIC_API_KEY=sk-ant-... (see .env.example and the README). "
            "The system-wide ANTHROPIC_API_KEY is deliberately ignored."
        )
    return anthropic.Anthropic(api_key=api_key, **kwargs)


def run_tool_agent(
    client: anthropic.Anthropic,
    cfg: AgentConfig,
    *,
    system: str,
    user: str,
    tools: list[Any],
    enable_fallbacks: bool = True,
    on_message: Callable[[Any], None] | None = None,
) -> LLMRun:
    kwargs: dict[str, Any] = dict(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        max_iterations=cfg.max_iterations,
        # Static system prompt first with a cache breakpoint; the task goes in the user turn.
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user}],
        tools=tools,
        # Summarized thinking: billed the same as omitted, but the summary is recorded in the
        # run trace as the agent's reasoning for each decision.
        thinking={"type": "adaptive", "display": "summarized"},
        output_config={"effort": cfg.effort},
    )
    if enable_fallbacks:
        kwargs["betas"] = [FALLBACK_BETA]
        kwargs["fallbacks"] = "default"

    runner = client.beta.messages.tool_runner(**kwargs)

    usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    tool_calls: list[ToolCall] = []
    final_text = ""
    stop_reason: str | None = None
    turns = 0

    for message in runner:
        turns += 1
        u = message.usage
        for k in usage:
            usage[k] += int(getattr(u, k, 0) or 0)

        texts: list[str] = []
        for block in message.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(name=block.name, input=dict(block.input)))
        if texts:
            final_text = "\n".join(texts)
        stop_reason = message.stop_reason

        if on_message is not None:
            on_message(message)

        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise AgentRefused(f"model refused the request (model={cfg.model}, details={details})")
        if message.stop_reason == "max_tokens":
            log.warning("Response hit max_tokens=%s; consider raising AgentConfig.max_tokens", cfg.max_tokens)

    if turns >= cfg.max_iterations and stop_reason == "tool_use":
        log.warning("Tool runner stopped at max_iterations=%s before the agent finished", cfg.max_iterations)

    return LLMRun(final_text=final_text, stop_reason=stop_reason, tool_calls=tool_calls, usage=usage, turns=turns)
