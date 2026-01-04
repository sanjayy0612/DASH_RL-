# NeuralStream: AI-Powered Adaptive Bitrate Streaming



An end-to-end video streaming system that uses **Deep Reinforcement Learning (PPO)** to optimize video quality in real-time. The AI agent balances high resolution against buffering risks by observing network speed and buffer health, outperforming static rule-based algorithms.

![Project Demo](https://via.placeholder.com/800x400?text=Insert+Screenshot+of+Player+Here)

##  About

Traditional video players (like Netflix/YouTube) use rigid "If/Then" rules (e.g., "If speed < 5Mbps, drop to 720p"). These often fail in unstable networks, causing buffering or unnecessary quality drops.

**NeuroStream** replaces these rules with a **Neural Network Brain**.
* **Observes:** Network bandwidth, buffer level, and past decisions.
* **Decides:** The optimal quality for the *next* video chunk.
* **Result:** Zero buffering and maximized visual quality.

##  Tech Stack

**AI & Simulation:**
* **Python 3.8+**
* **Gymnasium** (Custom Network Environment)
* **Stable-Baselines3** (PPO Algorithm)
* **PyTorch**

**System Engineering:**
* **FFmpeg** (DASH Fragmentation & Transcoding)
* **Flask** (Inference Server with CORS)
* **Dash.js** (Frontend Player with Custom Rule Injection)

##  Project Structure
```text
DASH_RL/
├── __pycache__/              # Python cache files
├── dash/                     # Virtual environment
├── Training history/         # Saved model checkpoints
│   ├── ppo_video_streamer.zip
│   ├── ppo_video_streamerr.zip
│   └── ppo_video_streamerrrr.zip
├── video_project/            # The "Content": Video files
│   ├── ffmpeg/               # FFmpeg executable (if bundled)
│   ├── index.html            # Web Player Frontend
│   └── input.mp4             # Source video file
├── .gitignore                # Git ignore rules
├── ppo_tra.py                # Training script (alternate)
├── ppo_video_streamer_2.zip  # Primary trained AI model
├── README.md                 # This file
├── server.py                 # The "Brain": Flask API for the web player
└── video_streaming_env.py    # The "Game": Simulates network physics
```

## ⚡ Quick Start

### 1. Setup
```bash
# Clone the repo
git clone https://github.com/yourusername/neurostream.git
cd DASH_RL

# Create Virtual Env
python -m venv dash
source dash/bin/activate  # Windows: dash\Scripts\activate

# Install dependencies
pip install gymnasium stable-baselines3 numpy shimmy flask flask-cors
```

### 2. Prepare Video Data (DASH)

We need to chop a video into small chunks at different quality levels (360p, 480p, 720p).

**Prerequisite:** Install [FFmpeg](https://ffmpeg.org/).
```bash
cd video_project
# Download sample video (if not already present)
curl -o input.mp4 https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/Sintel.mp4

# Run Transcoding (Audio + Video + DASH Manifest)
ffmpeg -i input.mp4 \
  -map 0:v -b:v:0 500k -s:v:0 640x360 -profile:v:0 main \
  -map 0:v -b:v:1 1000k -s:v:1 854x480 -profile:v:1 main \
  -map 0:v -b:v:2 2000k -s:v:2 1280x720 -profile:v:2 main \
  -map 0:a -c:a aac -b:a 128k \
  -use_timeline 1 -use_template 1 \
  -adaptation_sets "id=0,streams=v id=1,streams=a" \
  -f dash manifest.mpd
```

### 3. Training the AI

Train the agent in the simulator before connecting it to the real player.
```bash
cd ..
python ppo_tra.py
```

*Results:*

* **Before Training:** Reward -149 (Frequent Buffering)
* **After Training:** Reward +13 (Smooth Streaming)

Trained models are automatically saved to the `Training history/` folder.

### 4. Launch the System

Start the server and watch the AI work in the browser.
```bash
python server.py
# Server runs on http://127.0.0.1:5001
```

1. Open your browser to `http://127.0.0.1:5001/index.html`
2. Open DevTools (F12) -> **Network Tab** -> Set throttling to **"Fast 3G"**.
3. Watch the "AI Decision" adapt in real-time!

##  Roadmap

* [x] **Phase 1:** Custom Gym Environment (Physics simulation)
* [x] **Phase 2:** Train PPO Agent (Stable-Baselines3)
* [x] **Phase 3:** DASH Video Pipeline (FFmpeg)
* [x] **Phase 4:** End-to-End Web Integration (Flask + Dash.js)
* [ ] **Phase 5:** Real-world Network Traces (FCC Dataset)
* [ ] **Phase 6:** Comparative Benchmarking vs. Standard ABR

## References

* [Pensieve: Neural Adaptive Video Streaming (MIT)](https://web.mit.edu/pensieve/)
* [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
* [Dash.js API](https://cdn.dashjs.org/latest/jsdoc/index.html)

## License

MIT
