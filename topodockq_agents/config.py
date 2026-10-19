"""Runtime settings for the pipeline.

Everything can be overridden via environment variables (prefix ``TOPODOCKQ_``) or
programmatically through ``Settings(...)``. Model IDs default to
Claude Opus 5.5; the project's .env sets claude-sonnet-5-5 for development.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5-5"
VALID_EFFORT = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class AgentConfig:
    """Per-agent LLM parameters."""

    model: str = DEFAULT_MODEL
    effort: str = "medium"          # low | medium | high | xhigh | max
    max_tokens: int = 16000
    max_iterations: int = 30        # tool-use round trips before the runner stops

    def __post_init__(self) -> None:
        if self.effort not in VALID_EFFORT:
            raise ValueError(f"effort must be one of {VALID_EFFORT}, got {self.effort!r}")


def load_dotenv(path: Path, override: bool = True) -> dict[str, str]:
    """Minimal ``.env`` loader (KEY=VALUE lines, ``#`` comments, optional quotes). No extra dependency.

    Values from the project ``.env`` override anything already in the process environment, so a
    project-specific key always wins over a system-wide one.
    """
    values: dict[str, str] = {}
    if not path.exists():
        return values
    # utf-8-sig strips the byte-order mark that PowerShell's `Set-Content -Encoding utf8` writes;
    # without this the first variable name silently gets a BOM prefix and is never found.
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        values[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return values


ENV_PREFIX = "TOPODOCKQ_"


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"{ENV_PREFIX}{name}") or default


def _agent_from_env(prefix: str, default_effort: str) -> AgentConfig:
    model = _env(f"{prefix}_MODEL") or _env("MODEL") or DEFAULT_MODEL
    effort = _env(f"{prefix}_EFFORT") or default_effort
    return AgentConfig(model=model, effort=effort)


@dataclass
class Settings:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    runs_dir: Path | None = None
    data_dir: Path | None = None
    # Structures to score: a folder (one sub-folder of ranked_<N>[_sp].pdb per complex, or flat
    # *.pdb files) or a manifest .csv. Shown to the agents; the DataAgent decides the selection.
    input_source: str | None = None
    workers: int = 8                # parallel processes for interface extraction / feature generation
    verify: bool = True             # after the agents finish, rerun the deterministic pipeline and compare
    # Strict mode: a complex whose top-1 candidate has unconfirmed, implausible chain roles is reported as
    # "needs_confirmation" instead of "accepted". Set by the user, never by an agent.
    strict: bool = False
    # Optional: the number of structures the user expects to be scored; the selection audit checks it.
    expected_structures: int | None = None

    # Project-specific credential. Read ONLY from TOPODOCKQ_ANTHROPIC_API_KEY, normally via <project>/.env.
    # The system-wide ANTHROPIC_API_KEY is deliberately never used.
    api_key: str | None = None

    main: AgentConfig = field(default_factory=lambda: AgentConfig(effort="high"))
    data_agent: AgentConfig = field(default_factory=lambda: AgentConfig(effort="medium"))
    topo_agent: AgentConfig = field(default_factory=lambda: AgentConfig(effort="medium"))
    ml_agent: AgentConfig = field(default_factory=lambda: AgentConfig(effort="medium"))

    dry_run: bool = False           # True -> never call the API; run tool functions in a fixed order
    enable_fallbacks: bool = True   # server-side refusal fallbacks (beta)
    verbose: bool = True

    def __post_init__(self) -> None:
        self.runs_dir = Path(self.runs_dir) if self.runs_dir else self.project_root / "runs"
        self.data_dir = Path(self.data_dir) if self.data_dir else self.project_root / "data"

    @classmethod
    def from_env(cls, project_root: Path | None = None, **overrides) -> "Settings":
        root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        load_dotenv(root / ".env")
        s = cls(
            project_root=root,
            api_key=_env("ANTHROPIC_API_KEY"),
            main=_agent_from_env("MAIN", "high"),
            data_agent=_agent_from_env("DATA", "medium"),
            topo_agent=_agent_from_env("TOPO", "medium"),
            ml_agent=_agent_from_env("ML", "medium"),
            dry_run=(_env("DRY_RUN", "0") or "0").lower() in ("1", "true", "yes"),
            enable_fallbacks=(_env("FALLBACKS", "1") or "1").lower() not in ("0", "false", "no"),
        )
        if _env("RUNS_DIR"):
            s.runs_dir = Path(_env("RUNS_DIR"))  # type: ignore[arg-type]
        if _env("DATA_DIR"):
            s.data_dir = Path(_env("DATA_DIR"))  # type: ignore[arg-type]
        if _env("INPUT"):
            s.input_source = _env("INPUT")
        if _env("WORKERS"):
            s.workers = int(_env("WORKERS"))  # type: ignore[arg-type]
        if _env("VERIFY"):
            s.verify = (_env("VERIFY") or "1").lower() not in ("0", "false", "no")
        if _env("STRICT"):
            s.strict = (_env("STRICT") or "0").lower() in ("1", "true", "yes")
        return replace(s, **overrides) if overrides else s
