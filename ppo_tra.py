import gymnasium as gym
from stable_baselines3 import PPO
from video_streaming_env import VideoStreamingEnv  
import os


env = VideoStreamingEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
print("--- TRAINING FINISHED ---")
model.save("ppo_video_streamer")


print("\n--- TESTING THE NEW BRAIN ---")
obs, _ = env.reset()
total_reward = 0

for _ in range(20):
  
    action, _states = model.predict(obs)
    
   
    obs, reward, done, _, info = env.step(action)
    
    speed = f"{obs[0]:.0f}kbps"
    qual = ["Low", "Med", "High"][int(obs[2])]
    print(f"AI Chose: {qual} | Speed: {speed} | Rebuffer: {info['rebuffer']:.2f}s | Reward: {reward:.2f}")
    
    total_reward += reward
    if done: break

print(f"Total Reward: {total_reward:.2f}")