import os
import numpy as np
from flask import Flask, send_from_directory, request, jsonify
from stable_baselines3 import PPO
from flask_cors import CORS  # <--- NEW IMPORT

app = Flask(__name__)
CORS(app)  # <--- THIS UNBLOCKS THE BROWSER

# Match your model name exactly
MODEL_NAME = "ppo_video_streamer_2" 
VIDEO_FOLDER = "video_project"

print(f"Loading AI Model: {MODEL_NAME}...")
try:
    model = PPO.load(MODEL_NAME)
    print("AI Model Loaded Successfully!")
except Exception as e:
    print(f"ERROR: Could not load model. Check file name! {e}")

@app.route('/<path:filename>')
def serve_files(filename):
    directory = os.path.join(os.getcwd(), VIDEO_FOLDER)
    return send_from_directory(directory, filename)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    speed_kbps = data.get('speed', 0)
    buffer_level = data.get('buffer', 0)
    last_quality = data.get('last_quality', 0)
    
    obs = np.array([float(speed_kbps), float(buffer_level), float(last_quality)])
    action, _ = model.predict(obs)

    # Print to terminal so you can see it working
    print(f"Stats: Speed={speed_kbps:.0f}kbps, Buf={buffer_level:.1f}s -> AI Action: {action}")
    
    return jsonify({'quality': int(action)})

if __name__ == '__main__':
    # We force Port 5001 to avoid the AirPlay conflict
    print(f"Server starting... Open http://127.0.0.1:5001/index.html")
    app.run(port=5001)