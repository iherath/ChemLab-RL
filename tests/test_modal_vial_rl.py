"""Offline comparison budgets, CLI compatibility and optimizer-smoke gate."""
import ast
import copy
import inspect
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from scripts import modal_vial_rl as m
from scripts.train_vial_ogpo import parser


class VialRLLauncherTests(unittest.TestCase):
    def request(self, mode="smoke", method="ogpo"):
        return dict(run_id="test-vial", checkpoint_run="bc", checkpoint_step=2000,
                    dataset="weigh_vial_grok_v2.hdf5", mode=mode, method=method, seed=0)

    def test_all_rl_profiles_match_the_actual_cli(self):
        for mode, method in (("smoke", "ogpo"), ("smoke", "vadar"), ("smoke", "bon"), ("baseline_stochastic", "ogpo"),
                             ("comparison", "bon"), ("comparison", "ogpo"), ("comparison", "vadar")):
            args = parser().parse_args(m.build_command(self.request(mode, method))[2:])
            self.assertEqual(args.flow_steps, 10)
            self.assertEqual(args.max_episode_steps, 1600)
            self.assertEqual(args.method, method)
            if mode == "smoke":
                self.assertEqual((args.batch_size, args.group_size, args.max_updates, args.max_environment_steps,
                                  args.eval_max_steps, args.critic_warmup_updates), (2, 2, 3, 4, 4, 1))
            if mode == "baseline_stochastic":
                self.assertTrue(args.baseline_only)

    def test_baselines_cannot_silently_use_a_learning_variant(self):
        for mode in ("baseline_ode", "baseline_stochastic"):
            with self.assertRaisesRegex(ValueError, "baselines"):
                m.build_command(self.request(mode, "vadar"))

    def test_ode_is_distinct_and_evaluation_seed_matches_stochastic(self):
        command = m.build_command(self.request("baseline_ode"))
        self.assertEqual(command[:2], ["-m", "scripts.evaluate_vial_vla"])
        self.assertEqual(command[command.index("--seed") + 1], "1000")
        self.assertEqual(command[command.index("--max-steps") + 1], "1600")

    def test_fixed_budgets_and_gpu_parallelism(self):
        tree = ast.parse(inspect.getsource(m))
        for name, timeout, containers in (("execute_smoke", 900, 1), ("execute_baseline", 1200, 2),
                                           ("execute_comparison", 4500, 3)):
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            kwargs = {k.arg: ast.literal_eval(k.value) for k in node.decorator_list[0].keywords
                      if k.arg in {"timeout", "max_containers", "retries", "cpu", "memory"}}
            self.assertEqual(kwargs, dict(timeout=timeout, max_containers=containers, retries=0,
                                          cpu=(8, 8), memory=(32768, 32768)))
        self.assertEqual(m.LIMITS["comparison"]["runner_seconds"], 3600)

    def test_smoke_requires_actual_updates_density_pass_and_identical_inputs(self):
        expected = dict(checkpoint_path="/runs/bc/checkpoint_002000", manifest_sha256="a", dataset_sha256="b",
                        simulation_source="c", algorithm_sources={"file": "d"})
        good = dict(mode="smoke", exit_code=0, timed_out=False, preflight=expected,
                    training=dict(actor_updates=2, critic_updates=3, density_check={"passed": True}))
        m.validate_smoke(good, expected)
        for change in (dict(training=None), dict(exit_code=1), dict(timed_out=True),
                       dict(training=dict(actor_updates=0, critic_updates=3, density_check={"passed": True})),
                       dict(training=dict(actor_updates=2, critic_updates=3, density_check={"passed": False}))):
            with self.assertRaises(ValueError):
                m.validate_smoke({**good, **change}, expected)
        for key in expected:
            changed = copy.deepcopy(expected)
            changed[key] = "different"
            with self.assertRaises(ValueError):
                m.validate_smoke(good, changed)

    def test_paths_and_unknown_modes_are_rejected(self):
        for key, value in (("checkpoint_run", "../x"), ("dataset", "../x.hdf5"), ("mode", "unbounded"),
                           ("checkpoint_step", 0), ("seed", -1), ("method", "ogpo_plus")):
            with self.assertRaises(ValueError):
                m.build_command({**self.request(), key: value})

    def test_short_smoke_rollout_is_not_a_complete_episode(self):
        self.assertFalse(m.evaluation_completed(dict(episodes=[dict(terminated=False, truncated=True, controls=4)])))
        self.assertTrue(m.evaluation_completed(dict(episodes=[dict(terminated=False, truncated=True, controls=1600)])))
        self.assertFalse(m.evaluation_completed(None))

    def test_failed_remote_process_preserves_result_and_fails_local_command(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m, "ROOT", Path(directory)):
            result = dict(exit_code=1, timed_out=False, status="incomplete")
            function = types.SimpleNamespace(remote=lambda *args: result)
            with self.assertRaisesRegex(RuntimeError, "failed"):
                m._single(self.request(), function)
            out = Path(directory) / "docs/pilot/test-vial"
            self.assertEqual(json.loads((out / "wrapper_result.json").read_text()), result)
            self.assertTrue((out / "wrapper_failure.json").is_file())


if __name__ == "__main__":
    unittest.main()
