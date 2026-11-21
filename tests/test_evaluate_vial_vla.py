"""Native-ODE evaluation bookkeeping without loading weights or running MuJoCo."""
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts.evaluate_vial_vla import evaluate


class FakeEnv:
    def __init__(self, **kwargs):
        self.actions = []

    def observation(self):
        return {"object": np.zeros(11), **{key: np.zeros((64, 64, 3), np.uint8)
                for key in ("top1", "left_wrist", "right_wrist")}}

    def info(self):
        return dict(success=False, subgoal_dones=np.zeros(4), max_postcontrol_penetration_m=.001,
                    contact_certification="not_recorded_every_substep")

    def reset(self, **kwargs):
        return self.observation(), self.info()

    def step(self, action):
        self.actions.append(action)
        return self.observation(), 0., False, False, self.info()

    def close(self):
        pass


class FakePolicy:
    def __init__(self, *args, **kwargs):
        self.policy = SimpleNamespace(config=SimpleNamespace(chunk_size=50, n_action_steps=2))

    def reset(self):
        pass

    def act(self, obs):
        return np.array([2., -2.] + [0.] * 12, np.float32)


class VialEvaluationTests(unittest.TestCase):
    def test_clipping_is_explicit_and_partial_episode_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "data_manifest.json").write_text("{}")
            env = FakeEnv()
            with patch("chemlab_rl.vial_vla_data.VialVLABCPolicy", FakePolicy), \
                 patch("chemlab_rl.vial_vla_env.VialVLAEnv", return_value=env), \
                 patch("chemlab_rl.vial_vla_env.source_identity", return_value="source"):
                result = evaluate(checkpoint, root / "eval", max_steps=2, device="cpu", video=False)
            self.assertFalse(result["completed_evaluation"])
            self.assertFalse(result["success"])
            self.assertEqual(result["clipped_action_elements"], 4)
            self.assertEqual(result["action_elements"], 28)
            self.assertEqual(result["termination_reason"], "evaluation_budget")
            self.assertTrue(all(np.abs(a).max() <= 1 for a in env.actions))

    def test_budget_cannot_expand(self):
        with self.assertRaises(ValueError):
            evaluate("unused", "unused", max_steps=1601)


if __name__ == "__main__":
    unittest.main()
