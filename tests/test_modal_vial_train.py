"""Offline request limits for the separate vial Modal app; no cloud calls."""
import ast
import inspect
import unittest

from scripts import modal_vial_train as m


class VialLauncherTests(unittest.TestCase):
    def request(self):
        return dict(run_id="smolvla-vial-bc7-test", dataset="weigh_vial_grok_v2.hdf5", dataset_sha256="a" * 64,
                    steps=2000, batch_size=8, max_minutes=30, save_every=500, seed=0)

    def test_task_command_keeps_normalized_vial_module_and_fixed_chunks(self):
        command = m.build_command(self.request())
        self.assertEqual(command[:2], ["-m", "scripts.train_vial_vla_bc"])
        for key, value in (("--chunk-size", "50"), ("--n-action-steps", "2"), ("--max-seconds", "1620")):
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertNotIn("--heldout-powders", command)

    def test_deadline_and_work_limits_cannot_expand(self):
        for key, value in (("max_minutes", 61), ("max_minutes", 4), ("steps", 10001),
                           ("batch_size", 33), ("save_every", 2001)):
            request = self.request()
            request[key] = value
            with self.assertRaises(ValueError):
                m.build_command(request)

    def test_paths_and_hash_are_validated(self):
        for key, value in (("dataset", "../vial.hdf5"), ("run_id", "../overwrite"), ("dataset_sha256", "bad")):
            request = self.request()
            request[key] = value
            with self.assertRaises(ValueError):
                m.validate_request(request)

    def test_hybrid_split_is_explicit_and_forwarded(self):
        request = self.request()
        request["split"] = dict(train_episodes=[f"demo_{i}" for i in [*range(7), *range(8, 98)]],
                                heldout_episodes=["demo_7"], excluded_episodes=[])
        command = m.build_command(request)
        self.assertTrue(command[command.index("--split-json") + 1].endswith("/split.json"))
        request["split"]["heldout_episodes"] = ["demo_0"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            m.build_command(request)

    def test_remote_resource_cap_and_no_retry(self):
        tree = ast.parse(inspect.getsource(m))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "execute")
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in function.decorator_list[0].keywords
                  if kw.arg in {"gpu", "cpu", "memory", "timeout", "startup_timeout", "max_containers", "retries"}}
        self.assertEqual(kwargs, dict(gpu="A100-80GB", cpu=(8, 8), memory=(32768, 32768),
                                      timeout=3600, startup_timeout=120, max_containers=1, retries=0))


if __name__ == "__main__":
    unittest.main()
