"""Shared pipeline state and the artifact hand-off contract between agents.

Agents never pass raw data to each other through the LLM context. They write
files into ``runs/<run_id>/artifacts`` and register them here under well-known
keys. Downstream agents look artifacts up by key. Every event (stage starts and
ends, tool invocations, model messages, verification) is also appended to
``runs/<run_id>/trace.jsonl`` as it happens.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class Keys:
    """Artifact keys that form the contract between stages (TopoDockQ inference pipeline)."""

    STRUCTURES = "data.structures"      # DataAgent : structures.csv  one row per input structure (IDs, path, chains)
    INTERFACES = "data.interfaces"      # DataAgent : interfaces.csv  per structure: ready/excluded, interface file, rule codes
    DATA_REPORT = "data.report"         # DataAgent : .json notes and counts
    FEATURES = "topo.features"          # TopoAgent : features.csv    pdb_id, af_model_id + 2754 published-schema features
    TOPO_REPORT = "topo.report"         # TopoAgent : .json engine, failures, checks
    PREDICTIONS = "ml.predictions"      # MLAgent   : predictions.csv every structure: p_dockq, or excluded with rule code
    RANKING = "ml.ranking"              # MLAgent   : ranking.csv     top-1 candidate per complex
    ML_REPORT = "ml.report"             # MLAgent   : .json model package identity and summary


STAGE_INPUTS: dict[str, list[str]] = {
    "DataAgent": [],
    "TopoAgent": [Keys.INTERFACES],
    "MLAgent": [Keys.INTERFACES, Keys.FEATURES],
}
STAGE_OUTPUTS: dict[str, list[str]] = {
    "DataAgent": [Keys.STRUCTURES, Keys.INTERFACES, Keys.DATA_REPORT],
    "TopoAgent": [Keys.FEATURES, Keys.TOPO_REPORT],
    "MLAgent": [Keys.PREDICTIONS, Keys.RANKING, Keys.ML_REPORT],
}


class MissingArtifact(KeyError):
    """Raised when a stage needs an artifact that no earlier stage produced."""


@dataclass
class Artifact:
    key: str
    path: str
    kind: str
    produced_by: str
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


def _zero_usage() -> dict[str, int]:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }


@dataclass
class PipelineState:
    run_id: str
    run_dir: Path
    task: str
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    stages: dict[str, str] = field(default_factory=dict)        # agent -> running | done | incomplete | failed
    events: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=_zero_usage)

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def new(cls, runs_dir: Path, task: str) -> "PipelineState":
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        run_dir = Path(runs_dir) / run_id
        (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        state = cls(run_id=run_id, run_dir=run_dir, task=task)
        state.save()
        return state

    @classmethod
    def load(cls, run_dir: Path) -> "PipelineState":
        run_dir = Path(run_dir)
        data = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
        state = cls(
            run_id=data["run_id"],
            run_dir=run_dir,
            task=data["task"],
            stages=data.get("stages", {}),
            events=data.get("events", []),
            usage={**_zero_usage(), **data.get("usage", {})},
        )
        for key, a in data.get("artifacts", {}).items():
            state.artifacts[key] = Artifact(**a)
        return state

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "task": self.task,
            "artifacts": {k: asdict(a) for k, a in self.artifacts.items()},
            "stages": self.stages,
            "events": self.events,
            "usage": self.usage,
        }

    # ------------------------------------------------------------------ artifacts
    def register(self, key: str, path: Path | str, kind: str, produced_by: str, **meta: Any) -> Artifact:
        art = Artifact(key=key, path=str(path), kind=kind, produced_by=produced_by, meta=meta)
        self.artifacts[key] = art
        self.record("artifact_registered", produced_by, key=key, path=str(path))
        self.save()
        return art

    def get(self, key: str) -> Artifact | None:
        return self.artifacts.get(key)

    def has(self, *keys: str) -> bool:
        return all(k in self.artifacts for k in keys)

    def require(self, key: str) -> Artifact:
        art = self.artifacts.get(key)
        if art is None:
            producer = next((a for a, outs in STAGE_OUTPUTS.items() if key in outs), "an earlier stage")
            raise MissingArtifact(f"Artifact {key!r} is not available yet; it is produced by {producer}.")
        return art

    def path_for(self, filename: str) -> Path:
        return self.artifacts_dir / filename

    # ------------------------------------------------------------------ bookkeeping
    @property
    def trace_path(self) -> Path:
        return self.run_dir / "trace.jsonl"

    def record(self, event: str, agent: str | None = None, **data: Any) -> None:
        """Append an event to the in-memory log and, immediately, to trace.jsonl (crash-safe)."""
        entry = {"t": time.time(), "event": event, "agent": agent, **data}
        self.events.append(entry)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.trace_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

    def set_stage(self, agent: str, status: str) -> None:
        self.stages[agent] = status
        self.save()

    def add_usage(self, usage: dict[str, int]) -> None:
        for k, v in usage.items():
            self.usage[k] = self.usage.get(k, 0) + int(v or 0)

    def summary(self) -> str:
        """Compact, LLM-friendly description of what exists so far."""
        lines = [
            f"run_id: {self.run_id}",
            f"task: {self.task}",
            "stages: " + (json.dumps(self.stages) if self.stages else "{}"),
        ]
        if self.artifacts:
            lines.append("artifacts:")
            for k, a in self.artifacts.items():
                meta = json.dumps(a.meta, default=str)
                if len(meta) > 300:
                    meta = meta[:300] + "...}"
                lines.append(f"  - {k} ({a.kind}, by {a.produced_by}): {a.path} meta={meta}")
        else:
            lines.append("artifacts: none")
        return "\n".join(lines)
