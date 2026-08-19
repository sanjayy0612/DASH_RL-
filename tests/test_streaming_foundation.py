import unittest
import random
import tempfile
from pathlib import Path

import numpy as np

from abr.config import StreamingConfig, load_streaming_config
from abr.controllers import PPOController
from abr.environments import StreamingEnv
from abr.metrics.episode import EpisodeMetrics
from abr.metrics.qoe import calculate_qoe
from abr.traces import SyntheticTrace
from video_streaming_env import VideoStreamingEnv


class FakePPO:
    def __init__(self, action: int):
        self.action = action

    def predict(self, observation, deterministic=True):
        return self.action, None


class StreamingFoundationTests(unittest.TestCase):
    def test_reset_has_legacy_observation_shape_and_initial_values(self):
        observation, info = StreamingEnv().reset(seed=7)
        np.testing.assert_array_equal(observation, np.array([1000, 10, 0], dtype=np.float32))
        self.assertEqual(observation.shape, (3,))
        self.assertEqual(info, {})

    def test_all_actions_are_valid_and_invalid_action_is_rejected(self):
        env = StreamingEnv()
        env.reset(seed=1)
        for action in range(env.action_space.n):
            _, _, _, _, info = env.step(action)
            self.assertEqual(info["bitrate_kbps"], (500, 1000, 2000)[action])
        with self.assertRaises(ValueError):
            env.step(3)

    def test_buffer_never_becomes_negative(self):
        env = StreamingEnv(StreamingConfig(initial_buffer_seconds=0.0, seed=3))
        env.reset()
        for _ in range(10):
            observation, _, _, _, _ = env.step(2)
            self.assertGreaterEqual(observation[1], 0.0)

    def test_episode_terminates_at_configured_length(self):
        env = StreamingEnv(StreamingConfig(episode_length=2, seed=4))
        env.reset()
        self.assertFalse(env.step(0)[2])
        self.assertTrue(env.step(0)[2])

    def test_reward_preserves_legacy_formula(self):
        outcome = calculate_qoe(
            bitrate_kbps=2000, previous_quality=0, new_quality=2,
            previous_buffer_seconds=1.0, bandwidth_kbps=1000.0,
            chunk_duration_seconds=4.0, max_buffer_seconds=60.0,
        )
        self.assertEqual(outcome.rebuffer_seconds, 7.0)
        self.assertEqual(outcome.buffer_seconds, 0.0)
        self.assertEqual(outcome.reward, -26.5)

    def test_synthetic_trace_is_repeatable_for_the_same_seed(self):
        left, right = SyntheticTrace(seed=9), SyntheticTrace(seed=9)
        self.assertEqual(
            [left.next_bandwidth(1000.0) for _ in range(3)],
            [right.next_bandwidth(1000.0) for _ in range(3)],
        )

    def test_environment_reset_seed_repeats_transition(self):
        env = StreamingEnv()
        env.reset(seed=42)
        first = env.step(1)[0]
        env.reset(seed=42)
        second = env.step(1)[0]
        np.testing.assert_array_equal(first, second)

    def test_default_environment_matches_legacy_for_a_seeded_action_sequence(self):
        actions = (0, 2, 1, 2)
        random.seed(21)
        legacy = VideoStreamingEnv()
        legacy_observation, _ = legacy.reset()
        cleaned = StreamingEnv()
        cleaned_observation, _ = cleaned.reset(seed=21)
        np.testing.assert_array_equal(legacy_observation, cleaned_observation)
        for action in actions:
            legacy_result = legacy.step(action)
            cleaned_result = cleaned.step(action)
            np.testing.assert_array_equal(legacy_result[0], cleaned_result[0])
            self.assertEqual(legacy_result[1], cleaned_result[1])
            self.assertEqual(legacy_result[2], cleaned_result[2])
            self.assertEqual(legacy_result[4]["rebuffer"], cleaned_result[4]["rebuffer"])

    def test_ppo_controller_returns_valid_action(self):
        controller = PPOController(FakePPO(1), action_count=3)
        self.assertEqual(controller.select_bitrate(np.array([1000, 10, 0])), 1)
        with self.assertRaises(ValueError):
            PPOController(FakePPO(3), action_count=3).select_bitrate(np.array([1000, 10, 0]))

    def test_json_config_accepts_baseline_fields_and_overrides_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text('{"episode_length": 3, "bitrates_kbps": [400, 800]}')
            config = load_streaming_config(config_path, seed=12)
        self.assertEqual(config.episode_length, 3)
        self.assertEqual(config.bitrates_kbps, (400, 800))
        self.assertEqual(config.seed, 12)

    def test_episode_metrics_include_all_required_evaluation_fields(self):
        metrics = EpisodeMetrics()
        metrics.record(1.0, {"bitrate_kbps": 500, "rebuffer": 0.0, "buffer_seconds": 8.0})
        metrics.record(-2.0, {"bitrate_kbps": 1000, "rebuffer": 1.5, "buffer_seconds": 4.0})
        self.assertEqual(
            metrics.to_dict(),
            {
                "episode_reward": -1.0,
                "average_selected_bitrate_kbps": 750.0,
                "total_rebuffering_seconds": 1.5,
                "rebuffer_events": 1,
                "bitrate_switches": 1,
                "average_buffer_seconds": 6.0,
            },
        )
