# DASH_RL: adaptive bitrate streaming research foundation

## Status

**Original project:** a PPO-based adaptive bitrate streaming prototype with a
Gymnasium simulator, Flask inference endpoint, and Dash.js player integration.

**Current `research` branch:** reproducible infrastructure for investigating
reliable and federated learned ABR. Federation, OOD detection, trust routing,
personalization, BOLA, and MPC are deliberately not implemented yet.

The research direction is a hypothesis, not a completed result. See
[`research/RESEARCH_DIRECTION.md`](research/RESEARCH_DIRECTION.md).

## Research baseline

The cleaned baseline preserves the legacy simulator defaults:

- observation: `[bandwidth_kbps, buffer_seconds, last_quality]`
- actions: `0`, `1`, `2` for 500, 1000, and 2000 kbps
- chunk duration: 4 seconds; episode length: 50 chunks
- synthetic network: ±20% multiplicative bandwidth random walk
- reward: quality reward minus rebuffering and two-level switching penalties

The original root-level scripts remain in place as historical code. New work
uses small modules in `abr/` and scripts in `experiments/`.

## Development

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the foundation tests:

```bash
python -m unittest discover -s tests -v
```

Train a reproducible PPO baseline (models are written to ignored `artifacts/`):

```bash
python experiments/train_ppo.py --config experiments/configs/legacy_synthetic.json --seed 0 --timesteps 20000
```

Evaluate a model and emit per-episode JSON metrics:

```bash
python experiments/evaluate.py --config experiments/configs/legacy_synthetic.json --model-path artifacts/models/ppo_baseline --seed 0 --episodes 10
```

To evaluate the historical checkpoint instead, pass
`--model-path ppo_video_streamer_2`.

## Legacy prototype

The original browser-connected prototype is retained unchanged:

- `video_streaming_env.py`: Gymnasium simulator
- `ppo_tra.py`: original training script
- `server.py`: Flask inference endpoint
- `video_project/index.html`: Dash.js player

See [`legacy/README.md`](legacy/README.md) for migration notes.
