"""Evaluate a PPO baseline and write per-episode JSON metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Permit both ``python experiments/evaluate.py`` and module-style execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from abr.config import EvaluationConfig, load_streaming_config
from abr.controllers import PPOController
from abr.environments import StreamingEnv
from abr.metrics.episode import EpisodeMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=None, help="JSON streaming configuration")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--stochastic", action="store_true", help="sample PPO actions instead of deterministic inference")
    parser.add_argument("--output", type=Path, default=Path("artifacts/results/ppo_evaluation.json"))
    return parser.parse_args()


def evaluate(
    config: EvaluationConfig, model_path: Path, streaming_config_path: Path | None = None
) -> list[dict[str, float | int]]:
    environment = StreamingEnv(load_streaming_config(streaming_config_path, seed=config.seed))
    controller = PPOController.from_path(
        model_path,
        action_count=environment.action_space.n,
        deterministic=config.deterministic,
    )
    results: list[dict[str, float | int]] = []
    for episode in range(config.episodes):
        observation, _ = environment.reset(seed=config.seed + episode)
        metrics = EpisodeMetrics()
        terminated = False
        while not terminated:
            action = controller.select_bitrate(observation)
            observation, reward, terminated, truncated, info = environment.step(action)
            metrics.record(reward, info)
            if truncated:
                break
        results.append({"episode": episode, **metrics.to_dict()})
    return results


def main() -> None:
    args = parse_args()
    config = EvaluationConfig(seed=args.seed, episodes=args.episodes, deterministic=not args.stochastic)
    results = evaluate(config, args.model_path, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Wrote evaluation metrics to {args.output}")


if __name__ == "__main__":
    main()
