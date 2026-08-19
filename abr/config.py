"""Small, explicit configuration objects for ABR experiments."""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class StreamingConfig:
    """Parameters matching the legacy PPO simulator by default."""

    episode_length: int = 50
    chunk_duration_seconds: float = 4.0
    bitrates_kbps: Sequence[int] = field(default_factory=lambda: (500, 1000, 2000))
    initial_bandwidth_kbps: float = 1000.0
    initial_buffer_seconds: float = 10.0
    max_buffer_seconds: float = 60.0
    seed: int | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration shared by evaluation scripts."""

    seed: int = 0
    episodes: int = 10
    deterministic: bool = True


def load_streaming_config(path: str | Path | None, *, seed: int | None = None) -> StreamingConfig:
    """Load a small JSON experiment configuration, rejecting unknown fields."""
    values: dict[str, Any] = {}
    if path is not None:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = set(StreamingConfig.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown streaming configuration fields: {sorted(unknown)}")
        if "bitrates_kbps" in values:
            values["bitrates_kbps"] = tuple(values["bitrates_kbps"])
    if seed is not None:
        values["seed"] = seed
    return StreamingConfig(**values)
