import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class VideoStreamingEnv(gym.Env):
    def __init__(self):
        super(VideoStreamingEnv, self).__init__()
        
        # --- CONFIGURATION ---
        self.CHUNK_DURATION = 4.0   # Each video chunk is 4 seconds
        self.BITRATES = [500, 1000, 2000] # Bitrates for Low(0), Med(1), High(2) in kbps
        
        # ACTIONS: 0=Low, 1=Medium, 2=High
        self.action_space = spaces.Discrete(3)
        
        # OBSERVATION: [Network Speed (kbps), Buffer Level (seconds), Last Quality (0-2)]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), 
            high=np.array([5000, 60, 2]), 
            dtype=np.float32
        )
        
        # Initial State
        self.state = np.array([1000, 10, 0], dtype=np.float32) # Start: 1000kbps, 10s buffer, Low Qual
        self.chunks_left = 50 # Video length

    def reset(self, seed=None):
        self.state = np.array([1000, 10, 0], dtype=np.float32)
        self.chunks_left = 50
        return self.state, {}

    def step(self, action):
        # 1. GET CURRENT STATE
        prev_speed = self.state[0]
        prev_buffer = self.state[1]
        
        # 2. SIMULATE NETWORK CHANGE (Random fluctuation for now)
        # Network speed fluctuates by +/- 20%
        new_speed = prev_speed * random.uniform(0.8, 1.2)
        
        # 3. CALCULATE DOWNLOAD
        chosen_bitrate = self.BITRATES[action] # e.g., 2000 kbps
        
        # Time to download = Size / Speed
        # Note: In real life size varies, here we approximate Size ~ Bitrate * Duration
        chunk_size = chosen_bitrate * self.CHUNK_DURATION 
        delay = chunk_size / new_speed  # seconds to download
        
        # 4. UPDATE BUFFER
        # Buffer grows by 4s (video added), but shrinks by download time (waiting)
        # However, buffer cannot be negative (that means stalling)
        rebuffer_time = max(0, delay - prev_buffer) # Did we freeze?
        new_buffer = max(0, prev_buffer - delay + self.CHUNK_DURATION)
        
        # Cap buffer at 60s
        new_buffer = min(60, new_buffer)

        # 5. CALCULATE REWARD
        # Reward = Quality - (Penalty * Rebuffer) - (Penalty * Smoothness)
        reward = (chosen_bitrate / 1000) - (4.0 * rebuffer_time) 
        
        # Penalty for changing quality too abruptly
        last_quality = self.state[2]
        if abs(action - last_quality) > 1:
            reward -= 0.5 

        # 6. UPDATE STATE
        self.state = np.array([new_speed, new_buffer, action], dtype=np.float32)
        self.chunks_left -= 1
        
        done = self.chunks_left <= 0
        
        return self.state, reward, done, False, {"rebuffer": rebuffer_time}

