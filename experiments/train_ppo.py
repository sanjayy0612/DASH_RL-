"""Train the legacy-compatible PPO baseline with explicit configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Permit both ``python experiments/train_ppo.py`` and module-style execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from abr.config import load_streaming_config
from abr.environments import StreamingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=None, help="JSON streaming configuration")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/models/ppo_baseline"))
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from stable_baselines3 import PPO

    env = StreamingEnv(load_streaming_config(args.config, seed=args.seed))
    if args.resume_from:
        model = PPO.load(str(args.resume_from), env=env)
        model.set_random_seed(args.seed)
    else:
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed)
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.model_path))
    print(f"Saved PPO baseline to {args.model_path}.zip")


if __name__ == "__main__":
    main()
