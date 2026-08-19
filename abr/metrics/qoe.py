"""Legacy QoE calculation, isolated from streaming state transitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QoEOutcome:
    reward: float
    rebuffer_seconds: float
    buffer_seconds: float


def calculate_qoe(
    *,
    bitrate_kbps: int,
    previous_quality: int,
    new_quality: int,
    previous_buffer_seconds: float,
    bandwidth_kbps: float,
    chunk_duration_seconds: float,
    max_buffer_seconds: float,
) -> QoEOutcome:
    """Apply the legacy reward formula exactly.

    ``reward = bitrate_kbps / 1000 - 4 * rebuffer_seconds``. A fixed 0.5
    penalty applies only when quality changes by more than one level.
    """
    if bandwidth_kbps <= 0:
        raise ValueError("bandwidth_kbps must be positive")
    delay = bitrate_kbps * chunk_duration_seconds / bandwidth_kbps
    rebuffer_seconds = max(0.0, delay - previous_buffer_seconds)
    buffer_seconds = min(
        max_buffer_seconds,
        max(0.0, previous_buffer_seconds - delay + chunk_duration_seconds),
    )
    reward = bitrate_kbps / 1000.0 - 4.0 * rebuffer_seconds
    if abs(new_quality - previous_quality) > 1:
        reward -= 0.5
    return QoEOutcome(reward, rebuffer_seconds, buffer_seconds)
