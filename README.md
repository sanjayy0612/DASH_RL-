# Adaptive Bitrate Streaming using Reinforcement Learning

> ⚠️ **STATUS: IN DEVELOPMENT**

Using deep reinforcement learning (PPO) to optimize video quality selection in adaptive streaming environments. The system learns to balance video quality, buffering, and smoothness through experience.

## About

This project implements an AI agent that controls video bitrate selection in real-time. Instead of hardcoded rules, a neural network learns optimal decisions by observing network conditions and buffer states.

**Current Implementation:**
- Custom Gymnasium environment simulating video streaming
- PPO algorithm for quality adaptation
- DASH video format support
- Flask-based inference server

## Tech Stack

**Core:**
- Python 3.8+
- Gymnasium
- Stable-Baselines3
- FFmpeg

**Deployment:**
- Flask
- HTML5 Video API

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/adaptive-bitrate-rl.git
cd adaptive-bitrate-rl
python -m venv dash
source dash/bin/activate

# Install dependencies
pip install gymnasium stable-baselines3 numpy matplotlib flask

# Train the model
python train.py

# Run the server
python server.py
```

## Project Structure

```
├── video_streaming_env.py    # Streaming environment simulator
├── train.py                   # Training script
├── retrain.py                 # Continue training
├── server.py                  # Flask API + inference
├── video_project/             # DASH video files
│   ├── manifest.mpd
│   └── chunk-*.m4s
└── ppo_video_streamer.zip    # Model checkpoint
```
Project Stucture is Far ahead then the current Implementation:)
## Training

**Environment:**
- State: `[Network Speed, Buffer Level, Last Quality]`
- Actions: `[Low (500kbps), Medium (1000kbps), High (2000kbps)]`
- Reward: `Quality - (4.0 × Rebuffer) - Smoothness_Penalty`

**Results:**
```
Before Training: -149.84 (random policy)
After Training:  +13.00 (learned policy)
Rebuffer events: ~7.5s → 0.0s
```

## Video Preparation

Convert video to DASH format:

```bash
ffmpeg -i input.mp4 \
  -map 0:v -b:v:0 500k -s:v:0 640x360 -profile:v:0 main \
  -map 0:v -b:v:1 2000k -s:v:1 1280x720 -profile:v:1 main \
  -use_timeline 1 -use_template 1 -window_size 5 \
  -adaptation_sets "id=0,streams=v" \
  -f dash manifest.mpd
```

## Roadmap

- [x] Environment implementation
- [x] PPO training
- [x] Basic inference
- [ ] Web UI integration
- [ ] Real network traces
- [ ] Extended architecture (Pensieve-inspired)
- [ ] Benchmarking vs rule-based ABR

## References

- [Pensieve: Neural Adaptive Video Streaming](https://people.csail.mit.edu/hongzi/content/publications/Pensieve-Sigcomm17.pdf)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

## License

MIT

