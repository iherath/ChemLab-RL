"""Smoke checks for the lab, VaDar weights, the TD clip, and one training step."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from chemlab_rl.env import OBS_DIM, SUBGOALS, CVLab
from chemlab_rl.ogpo import group_advantages
from chemlab_rl.train import TrainConfig, run
from chemlab_rl.vadar import R_DONE, R_STEP, get_weighting, sparse_reward, td_target


class WeightingTest(unittest.TestCase):
    def test_table1_and_all_done(self) -> None:
        nothing = np.zeros(4, dtype=np.float32)
        np.testing.assert_array_equal(get_weighting(nothing, "current_myopic"), [1, 0, 0, 0])
        np.testing.assert_array_equal(get_weighting(nothing, "current_future"), [1, 1, 1, 1])
        np.testing.assert_array_equal(get_weighting(nothing, "next_myopic"), [0, 1, 0, 0])
        np.testing.assert_array_equal(get_weighting(nothing, "next_future"), [0, 1, 1, 1])

        weighed = np.array([1, 0, 0, 0], dtype=np.float32)
        np.testing.assert_array_equal(get_weighting(weighed, "current_myopic"), [0, 1, 0, 0])
        np.testing.assert_array_equal(get_weighting(weighed, "current_future"), [0, 1, 1, 1])
        np.testing.assert_array_equal(get_weighting(weighed, "next_myopic"), [0, 0, 1, 0])
        np.testing.assert_array_equal(get_weighting(weighed, "next_future"), [0, 0, 1, 1])

        last_open = np.array([1, 1, 1, 0], dtype=np.float32)
        np.testing.assert_array_equal(get_weighting(last_open, "next_future"), [0, 0, 0, 1])

        finished = np.ones((2, 4), dtype=np.float32)
        for mode in ("current_myopic", "current_future", "next_myopic", "next_future"):
            weights = get_weighting(finished, mode)
            np.testing.assert_array_equal(weights, [[0, 0, 0, 1], [0, 0, 0, 1]])

    def test_sparse_reward_matches_eq_3_2(self) -> None:
        reward = sparse_reward(np.array([0, 1, 0, 1], dtype=np.float32))
        np.testing.assert_array_equal(reward, [R_STEP, R_DONE, R_STEP, R_DONE])


class TdClipTest(unittest.TestCase):
    def test_completed_head_cannot_integrate(self) -> None:
        reward = torch.tensor([[R_DONE, R_STEP]], dtype=torch.float32)
        next_q = torch.tensor([[100.0, 0.0]], dtype=torch.float32)
        target = td_target(reward, next_q, gamma=0.99)
        self.assertAlmostEqual(target[0, 0].item(), R_DONE, places=5)
        self.assertAlmostEqual(target[0, 1].item(), R_STEP, places=5)

    def test_negative_backup_is_not_clipped(self) -> None:
        reward = torch.tensor([[R_STEP]], dtype=torch.float32)
        next_q = torch.tensor([[-10.0]], dtype=torch.float32)
        target = td_target(reward, next_q, gamma=0.99)
        self.assertAlmostEqual(target.item(), R_STEP + 0.99 * -10.0, places=5)


class AdvantageTest(unittest.TestCase):
    def test_group_mean_is_zero(self) -> None:
        q_agg = torch.tensor([[1.0, 3.0, 5.0], [2.0, 2.0, 8.0]])
        advantage = group_advantages(q_agg)
        self.assertTrue(torch.allclose(advantage.mean(dim=-1), torch.zeros(2), atol=1e-6))


class LabTest(unittest.TestCase):
    def test_scripted_solves_without_a_goal_input(self) -> None:
        for seed in (0, 1):
            env = CVLab(seed=seed)
            obs = env.reset(seed=seed)
            self.assertEqual(obs.shape, (OBS_DIM,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertEqual(len(SUBGOALS), 4)
            info = {"success": False, "done": env.done_vector(), "t": 0}
            for _ in range(env.horizon):
                # The demonstrator reads the simulator. The action itself is 8-D.
                action = env.scripted_action()
                self.assertEqual(action.shape, (env.action_dim,))
                obs, info = env.step(action)
                if info["terminated"] or info["truncated"]:
                    break
            self.assertTrue(
                info["success"],
                msg=f"seed {seed} stopped at t={info['t']} done={info['done']}",
            )
            np.testing.assert_array_equal(info["done"], np.ones(4, dtype=np.float32))

    def test_random_policy_does_not_finish(self) -> None:
        rng = np.random.default_rng(1)
        successes = 0
        for episode in range(5):
            env = CVLab(seed=episode)
            env.reset()
            info = {"success": False}
            for _ in range(env.horizon):
                action = rng.uniform(-1.0, 1.0, size=env.action_dim).astype(np.float32)
                _, info = env.step(action)
                if info["success"]:
                    successes += 1
                    break
        self.assertEqual(successes, 0)


class TrainStepTest(unittest.TestCase):
    def test_one_episode_runs(self) -> None:
        stats = run(
            TrainConfig(
                seed=0,
                n_demos=1,
                bc_epochs=1,
                online_episodes=1,
                batch_size=16,
                group_size=2,
                quiet=True,
            )
        )
        self.assertGreaterEqual(stats["env_steps"], 1)
        self.assertGreaterEqual(stats["updates"], 1)
        self.assertIsInstance(stats["bc_loss"], float)


if __name__ == "__main__":
    unittest.main()
