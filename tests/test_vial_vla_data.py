"""Vial-only split, timing, quality, normalization and bounded-launch checks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from chemlab_rl.vial_vla_data import (
    ACTION_KEY, STATE_KEY, CAMERAS, VialHDFDataset, inspect_dataset, prepare_manifest,
    manifest_state_contract, observed_state,
    _validate_hybrid_provenance, episode_payload_sha256, ORIGINAL_DATASET_SHA256,
    _validate_native_audit,
)


def fixture(path):
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs.update(control_hz=20., effector="robotiq", env_args=json.dumps(dict(
            collector="grok_session_v2", env_kwargs=dict(control_hz=20., hand="robotiq", layout="weigh",
            clutter=False, normalized_actions=True, image_size=64))))
        for i in range(9):
            g = data.create_group(f"demo_{i}")
            n = 6
            g.attrs.update(success=1, num_samples=n)
            action = np.full((n, 14), i / 20, np.float32) + np.arange(n, dtype=np.float32)[:, None] / 100
            g.create_dataset("actions", data=action)
            g.create_dataset("actions_abs", data=action * 2)
            raw = np.zeros((n, 76))
            raw[:, 0] = np.arange(n) / 20
            raw[:, 1] = i
            g.create_dataset("states", data=raw)
            terminal = np.array([0, 0, 0, 1, 1, 1], np.uint8)
            g.create_dataset("dones", data=terminal)
            g.create_dataset("rewards", data=terminal.astype(np.float32))
            q = np.full((n + 1, 14), i * 10, np.float32) + np.arange(n + 1, dtype=np.float32)[:, None]
            obj = np.full((n + 1, 11), 9999, np.float32)
            obj[:, 7] = i + np.arange(n + 1, dtype=np.float32) / 10
            if i == 8:
                obj[:, 7] = 3600
            for group, selection in (("obs", slice(None, -1)), ("next_obs", slice(1, None))):
                g.create_dataset(group + "/joint_pos", data=q[selection])
                g.create_dataset(group + "/object", data=obj[selection])
                for cam in CAMERAS:
                    rgb = np.broadcast_to(np.arange(n + 1, dtype=np.uint8)[:, None, None, None], (n + 1, 64, 64, 3))
                    g.create_dataset(group + "/" + cam, data=rgb[selection], chunks=(1, 64, 64, 3), compression="gzip")


def native_audit(group, *, peak=None, duration=0.):
    n = len(group["actions"])
    if peak is None:
        peak = max(float(abs(group[key][..., 7]).max()) for key in ("obs/object", "next_obs/object"))
    group.attrs.update(source_version="sha256:" + "b" * 64, collector_sha256="c" * 64, physics_checks=n * 25)
    audit = dict(accepted=True, task_id="vial_placement_native_v1", samples=n, controls=n, physics_checks=n * 25,
                 physics_timestep_s=.002, control_hz=20., source_version=group.attrs["source_version"],
                 collector_sha256=group.attrs["collector_sha256"],
                 trajectory_sha256=hashlib.sha256(group["actions"][:].tobytes() + group["obs/joint_pos"][:].tobytes()).hexdigest(),
                 max_penetration_m=.0008, max_unintended_penetration_m=.00001, max_abs_scale_g=peak,
                 max_excessive_load_seconds=duration,
                 limits=dict(unintended_rigid_penetration_m=.00005, other_contact_penetration_m=.001,
                             sustained_scale_load_g=100., sustained_scale_duration_s=.1))
    group.attrs["physics_audit"] = json.dumps(audit)
    return audit


class VialDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "vial.hdf5"
        fixture(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def manifest(self):
        return prepare_manifest(self.path, self.root / "manifest")

    def test_first_terminal_inclusive_and_train_only_moments(self):
        m = self.manifest()
        self.assertEqual(m["episode_lengths"], {f"demo_{i}": 4 for i in range(8)})
        self.assertEqual(m["statistics"][STATE_KEY]["count"], [28])
        self.assertAlmostEqual(m["statistics"][STATE_KEY]["mean"][0], 31.5)
        self.assertAlmostEqual(m["statistics"][STATE_KEY]["max"][-1], 6.3, places=5)
        self.assertNotIn("powder_id", m)
        self.assertEqual(manifest_state_contract(m), (True, 15))
        self.assertGreater(m["audit"]["episodes"][8]["peak_scale_reading_g"], 100)

    def test_chunk_tail_masks_post_terminal_and_does_not_cross_episode(self):
        m = self.manifest()
        ds = VialHDFDataset(self.path, m, ["demo_0", "demo_1"], 3)
        self.assertEqual(len(ds), 8)
        np.testing.assert_array_equal(ds[3]["action_is_pad"], [False, True, True])
        np.testing.assert_allclose(ds[3][ACTION_KEY], .03)
        np.testing.assert_allclose(ds[4][ACTION_KEY][0], .05)
        np.testing.assert_allclose(ds[1][STATE_KEY][:14], 1)
        self.assertEqual(ds[0][next(iter(CAMERAS.values()))].shape, (3, 64, 64))
        ds.close()

    def test_actor_does_not_expose_object_pose_labels_or_operator(self):
        with h5py.File(self.path) as f:
            obs = {k: v[0] for k, v in f["data/demo_0/obs"].items()}
        baseline = observed_state(obs)
        obs["object"][:7] = -123456
        obs["object"][8:] = 888
        obs.update(hidden_mass=999, subgoal_dones=np.ones(3), goal_pos=np.ones(3))
        np.testing.assert_array_equal(observed_state(obs), baseline)
        m = self.manifest()
        ds = VialHDFDataset(self.path, m, ["demo_0"])
        self.assertEqual(set(ds[0]), {STATE_KEY, ACTION_KEY, "action_is_pad", "task", *CAMERAS.values()})
        ds.close()

    def test_additional_force_outlier_fails_closed(self):
        with h5py.File(self.path, "a") as f:
            f["data/demo_1/obs/object"][0, 7] = 101
        with self.assertRaisesRegex(ValueError, "excessive scale"):
            inspect_dataset(self.path)

    def test_nonterminal_or_misaligned_data_rejected(self):
        with h5py.File(self.path, "a") as f:
            f["data/demo_0/next_obs/joint_pos"][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "alignment"):
            inspect_dataset(self.path)

    def test_camera_alignment_rejected(self):
        with h5py.File(self.path, "a") as f:
            f["data/demo_0/next_obs/top1_image"][0, 0, 0, 0] = 255
        with self.assertRaisesRegex(ValueError, "image alignment"):
            inspect_dataset(self.path)

    def test_duplicate_and_action_range_rejected(self):
        with h5py.File(self.path, "a") as f:
            f["data/demo_0/actions"][0, 0] = 255
        with self.assertRaisesRegex(ValueError, "normalized"):
            inspect_dataset(self.path)

    def test_explicit_split_can_extend_and_cannot_overlap_or_ignore(self):
        with h5py.File(self.path, "a") as f:
            f.copy(f["data/demo_0"], f["data"], name="demo_9")
            f["data/demo_9/states"][:, 1] = 9
        m = prepare_manifest(self.path, self.root / "extended", train_episodes=[*[f"demo_{i}" for i in range(7)], "demo_9"],
                             heldout_episodes=["demo_7"], excluded_episodes=["demo_8"])
        self.assertEqual(len(m["train_episodes"]), 8)
        with self.assertRaisesRegex(ValueError, "explicit split"):
            inspect_dataset(self.path)
        m["heldout_episodes"] = ["demo_0"]
        with self.assertRaisesRegex(ValueError, "overlap"):
            manifest_state_contract(m)

    def test_manifest_task_state_semantics_and_immutability(self):
        m = self.manifest()
        for key, value in (("task_id", "powder"), ("state_dim", 14), ("action_semantics", "absolute_radians")):
            bad = copy.deepcopy(m)
            bad[key] = value
            with self.assertRaisesRegex(ValueError, "contract"):
                manifest_state_contract(bad)
        with h5py.File(self.path, "a") as f:
            f["data/demo_0/actions"][0, 0] += .001
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.manifest()

    def test_new_audited_collector_requires_physical_acceptance(self):
        with h5py.File(self.path, "a") as f:
            args = json.loads(f["data"].attrs["env_args"])
            args["collector"] = "audited_continuous_vial_v1"
            f["data"].attrs["env_args"] = json.dumps(args)
        with self.assertRaisesRegex(ValueError, "physical acceptance"):
            inspect_dataset(self.path)

    def test_hybrid98_requires_unchanged_old_eight_and_all_ninety_audited(self):
        with h5py.File(self.path, "a") as f:
            data = f["data"]
            del data["demo_8"]
            anchors = {f"demo_{i}": episode_payload_sha256(data[f"demo_{i}"]) for i in range(8)}
            for i in range(98):
                if i >= 8:
                    f.copy(data["demo_0"], data, name=f"demo_{i}")
                g = data[f"demo_{i}"]
                g.attrs.update(collector="grok_session_v2" if i < 8 else "audited_continuous_vial_v1",
                               source_dataset_sha256=ORIGINAL_DATASET_SHA256 if i < 8 else "a" * 64,
                               source_episode=f"demo_{i if i < 8 else i - 8}")
                if i >= 8:
                    g.attrs.update(physical_valid=1, physics_validation="native_vial_substep_contacts_v1")
                    native_audit(g)
            train = [f"demo_{i}" for i in [*range(7), *range(8, 98)]]
            with patch("chemlab_rl.vial_vla_data.HISTORICAL_EPISODE_SHA256", anchors):
                _validate_hybrid_provenance(data, train, ["demo_7"], [])
                data["demo_0/obs/top1_image"][0, 0, 0, 0] = 99
                with self.assertRaisesRegex(ValueError, "historical payload"):
                    _validate_hybrid_provenance(data, train, ["demo_7"], [])
                data["demo_0/obs/top1_image"][0, 0, 0, 0] = 0
                del data["demo_97"].attrs["physical_valid"]
                with self.assertRaisesRegex(ValueError, "physical provenance"):
                    _validate_hybrid_provenance(data, train, ["demo_7"], [])
                data["demo_97"].attrs["physical_valid"] = 1
                data["demo_97"].attrs["source_episode"] = "demo_0"
                with self.assertRaisesRegex(ValueError, "physical provenance"):
                    _validate_hybrid_provenance(data, train, ["demo_7"], [])
                with self.assertRaisesRegex(ValueError, "Hybrid98"):
                    _validate_hybrid_provenance(data, train, ["demo_7"], ["demo_8"])

    def test_brief_audited_force_is_preserved_and_sustained_force_rejected(self):
        with h5py.File(self.path, "a") as f:
            group = f["data/demo_0"]
            group["obs/object"][1, 7] = 250.
            group["next_obs/object"][0, 7] = 250.
            audit = native_audit(group, peak=250., duration=.02)
            _validate_native_audit(group)
            self.assertEqual(group["obs/object"][1, 7], 250.)  # Validation never clips the measured sensor.
            env_args = json.loads(f["data"].attrs["env_args"])
            env_args["collector"] = "audited_continuous_vial_v1"
            f["data"].attrs["env_args"] = json.dumps(env_args)
            for episode in f["data"].values():
                episode.attrs.update(physical_valid=1, physics_validation="native_vial_substep_contacts_v1")
            f.flush()
            inspected = inspect_dataset(self.path, check_images=False)
            self.assertEqual(inspected["episodes"][0]["peak_scale_reading_g"], 250.)
            self.assertTrue(inspected["episodes"][0]["contact_audit_verified"])
            audit["max_excessive_load_seconds"] = .102
            group.attrs["physics_audit"] = json.dumps(audit)
            with self.assertRaisesRegex(ValueError, "duration limits"):
                _validate_native_audit(group)
            audit["max_excessive_load_seconds"] = 0.
            group.attrs["physics_audit"] = json.dumps(audit)
            with self.assertRaisesRegex(ValueError, "duration are inconsistent"):
                _validate_native_audit(group)

    def test_audit_rejects_missing_coverage_changed_source_and_mismatched_reading(self):
        with h5py.File(self.path, "a") as f:
            group = f["data/demo_0"]
            original = native_audit(group)
            for key, value in (("physics_checks", 149), ("samples", 5), ("max_abs_scale_g", .1),
                               ("source_version", "sha256:" + "d" * 64), ("trajectory_sha256", "e" * 64),
                               ("max_unintended_penetration_m", .000051)):
                group.attrs["physics_audit"] = json.dumps({**original, key: value})
                with self.subTest(key=key), self.assertRaises(ValueError):
                    _validate_native_audit(group)
            del group.attrs["physics_audit"]
            with self.assertRaisesRegex(ValueError, "serialized physics_audit"):
                _validate_native_audit(group)


class VialHeldoutTests(unittest.TestCase):
    setUp = VialDataTests.setUp
    tearDown = VialDataTests.tearDown
    manifest = VialDataTests.manifest

    def test_heldout_rng_is_repeatable_and_restored(self):
        import torch
        from scripts.train_vial_vla_bc import heldout_loss
        class Policy(torch.nn.Module):
            def forward(self, batch):
                return batch[ACTION_KEY].mean() + torch.rand(()), {}
        m = self.manifest()
        ds = VialHDFDataset(self.path, m, m["heldout_episodes"], 50)
        p = Policy().train()
        torch.manual_seed(777)
        before = torch.get_rng_state().clone()
        a = heldout_loss(p, lambda b: b, ds, samples=4, batch_size=2, device="cpu")
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        b = heldout_loss(p, lambda b: b, ds, samples=4, batch_size=2, device="cpu")
        self.assertEqual(a, b)
        self.assertTrue(p.training)
        self.assertTrue(a["completed"])
        ds.close()


if __name__ == "__main__":
    unittest.main()
