# Legacy baseline

The root-level `ppo_tra.py`, `video_streaming_env.py`, and `server.py` are the
original prototype and remain unchanged for historical comparison.

The research modules under `abr/` preserve the simulator's default observation,
action, transition, and reward semantics while making trace generation, QoE
calculation, training, and evaluation explicit. The legacy browser/server
integration still loads `ppo_video_streamer_2.zip` directly and has not been
refactored in this foundation update.
