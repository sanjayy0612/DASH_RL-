"""Structured episode metrics for controller evaluation."""

from dataclasses import dataclass
from typing import Any


@dataclass
class EpisodeMetrics:
    episode_reward: float = 0.0
    selected_bitrates_kbps: list[int] | None = None
    total_rebuffering_seconds: float = 0.0
    rebuffer_events: int = 0
    bitrate_switches: int = 0
    buffer_levels_seconds: list[float] | None = None

    def __post_init__(self) -> None:
        self.selected_bitrates_kbps = self.selected_bitrates_kbps or []
        self.buffer_levels_seconds = self.buffer_levels_seconds or []

    def record(self, reward: float, info: dict[str, Any]) -> None:
        bitrate_kbps = int(info["bitrate_kbps"])
        if self.selected_bitrates_kbps and bitrate_kbps != self.selected_bitrates_kbps[-1]:
            self.bitrate_switches += 1
        self.episode_reward += float(reward)
        self.selected_bitrates_kbps.append(bitrate_kbps)
        rebuffer_seconds = float(info["rebuffer"])
        self.total_rebuffering_seconds += rebuffer_seconds
        self.rebuffer_events += int(rebuffer_seconds > 0.0)
        self.buffer_levels_seconds.append(float(info["buffer_seconds"]))

    def to_dict(self) -> dict[str, float | int]:
        return {
            "episode_reward": self.episode_reward,
            "average_selected_bitrate_kbps": sum(self.selected_bitrates_kbps) / len(self.selected_bitrates_kbps) if self.selected_bitrates_kbps else 0.0,
            "total_rebuffering_seconds": self.total_rebuffering_seconds,
            "rebuffer_events": self.rebuffer_events,
            "bitrate_switches": self.bitrate_switches,
            "average_buffer_seconds": sum(self.buffer_levels_seconds) / len(self.buffer_levels_seconds) if self.buffer_levels_seconds else 0.0,
        }
