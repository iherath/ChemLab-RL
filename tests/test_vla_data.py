"""Behavioral tests for split integrity, temporal alignment, and policy inputs."""
from pathlib import Path
import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from chemlab_rl.vla_bc import (
    ACTION_KEY, CAMERAS, STATE_KEY, ScrapeHDFDataset,
    fit_training_stats, inspect_dataset, prepare_manifest, manifest_state_contract,
    observed_state, train_main,
)
from chemlab_rl.granular_material import GRANULAR_MATERIALS
from scripts.merge_granular_demos import RGB_COPY_FRAMES, _copy_rgb_dataset, merge_demos


def make_file(path: Path, legacy: bool = False):
    with h5py.File(path, "w") as handle:
        if not legacy:
            handle.attrs["schema_version"] = "scrape_v2"
            handle.attrs["observation_timing"] = "pre_action"
            handle.attrs["action_semantics"] = "joint_position_radians_gripper_0_255"
            handle.attrs["powder_dynamics"] = "physical_grains_v1"
        for episode, powder in enumerate([0, 1, 5]):
            demo = handle.create_group(f"data/demo_{episode}")
            demo.attrs["powder_id"] = powder
            demo.attrs["target_g"] = 2.0 + episode
            demo.attrs["instruction"] = f"Weigh {2.0 + episode:.2f} g."
            length = 4
            if not legacy:
                demo.attrs.update(success=1, physical_valid=1, physics_validation="rigid_contact_v1",
                                  grasp_model="mujoco_friction_only", physics_steps_per_action=50,
                                  physics_checks=length * 50, physics_audit="{}")
                demo.create_dataset("physics/checks", data=np.arange(1, length + 1) * 50)
            # Heldout deliberately differs by orders of magnitude.
            offset = 10000 if powder == 5 else episode * 10
            actions = np.arange(length * 14, dtype=np.float32).reshape(length, 14) + offset
            demo.create_dataset("actions", data=actions)
            if not legacy:
                demo.create_dataset("obs/proprio", data=actions + 0.25)
                demo.create_dataset("obs/scale_reading_g", data=(offset + np.arange(length) * .1).reshape(-1, 1))
            low = np.arange(length * 20, dtype=np.float32).reshape(length, 20) + offset
            demo.create_dataset("obs/lowdim", data=low)
            for i, camera in enumerate(CAMERAS):
                pixels = np.full((length, 8, 8, 3), 40 + i, dtype=np.uint8)
                demo.create_dataset(f"obs/{camera}", data=pixels)
            demo.create_dataset("subgoal_dones", data=np.zeros((length, 3), np.float32))


class VLADataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / "demo.hdf5"
        make_file(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_holdout_is_by_powder_and_stats_exclude_it(self):
        audit = inspect_dataset(self.path)
        self.assertEqual(audit["train_episodes"], ["demo_0", "demo_1"])
        self.assertEqual(audit["heldout_episodes"], ["demo_2"])
        stats = fit_training_stats(self.path, audit["train_episodes"])
        with h5py.File(self.path, "r") as handle:
            expected = np.concatenate([handle["data/demo_0/actions"][:], handle["data/demo_1/actions"][:]])
        np.testing.assert_allclose(stats[ACTION_KEY]["mean"], expected.mean(0))
        self.assertLess(max(stats[ACTION_KEY]["max"]), 100)
        self.assertEqual(stats[ACTION_KEY]["count"], [8])

    def test_pre_action_alignment_and_tail_mask_never_cross_episode(self):
        dataset = ScrapeHDFDataset(self.path, ["demo_0", "demo_1"], chunk_size=3)
        first = dataset[0]
        np.testing.assert_array_equal(first[ACTION_KEY][0], np.arange(14))
        np.testing.assert_array_equal(first[STATE_KEY], np.r_[np.arange(14) + 0.25, 0.0])
        last = dataset[3]
        np.testing.assert_array_equal(last["action_is_pad"], [False, True, True])
        np.testing.assert_array_equal(last[ACTION_KEY][0], last[ACTION_KEY][2])
        np.testing.assert_array_equal(dataset[4][ACTION_KEY][0], np.arange(14) + 10)
        dataset.close()

    def test_policy_input_includes_measured_scale_but_no_hidden_state(self):
        dataset = ScrapeHDFDataset(self.path, ["demo_0"], chunk_size=2)
        item = dataset[0]
        self.assertEqual(set(item), {STATE_KEY, ACTION_KEY, "action_is_pad", "task", *CAMERAS.values()})
        self.assertEqual(item[STATE_KEY].shape, (15,))
        self.assertEqual(item["task"], "Weigh 2.00 g.")
        self.assertEqual(item[CAMERAS["top1"]].shape, (3, 8, 8))
        self.assertAlmostEqual(float(item[CAMERAS["top1"]].max()), 40 / 255, places=6)
        dataset.close()

    def test_scale_statistics_use_only_training_and_match_inference_selection(self):
        manifest = prepare_manifest(self.path, self.root / "scale")
        self.assertEqual(manifest_state_contract(manifest), (True, 15))
        self.assertEqual(manifest["state_features"][-1], "scale_reading_g")
        self.assertAlmostEqual(manifest["statistics"][STATE_KEY]["mean"][-1], 5.15, places=5)
        self.assertLess(manifest["statistics"][STATE_KEY]["max"][-1], 11)
        dataset = ScrapeHDFDataset(self.path, manifest["train_episodes"], 2)
        with h5py.File(self.path) as handle:
            observation = {key: value[1] for key, value in handle["data/demo_0/obs"].items()}
        observation.update(material=np.ones(8) * 999, mass_g=999, subgoal_dones=np.ones(3), target_g=999)
        np.testing.assert_array_equal(dataset[1][STATE_KEY], observed_state(observation, True))
        self.assertAlmostEqual(float(observed_state(observation, True)[-1]), .1, places=6)
        dataset.close()

    def test_no_scale_ablation_is_explicit_and_immutable(self):
        manifest = prepare_manifest(self.path, self.root / "ablation", include_measured_scale=False)
        self.assertEqual(manifest_state_contract(manifest), (False, 14))
        self.assertNotIn("scale_reading_g", manifest["state_features"])
        dataset = ScrapeHDFDataset(self.path, manifest["train_episodes"], 2, include_measured_scale=False)
        self.assertEqual(dataset[0][STATE_KEY].shape, (14,))
        self.assertEqual(dataset[0][ACTION_KEY].shape, (2, 14))
        dataset.close()
        with self.assertRaisesRegex(ValueError, "immutable"):
            prepare_manifest(self.path, self.root / "ablation", include_measured_scale=True)
        with contextlib.redirect_stdout(io.StringIO()):
            train_main(["--dataset", str(self.path), "--output", str(self.root / "cli_ablation"),
                        "--prepare-only", "--no-scale-reading"])
        saved = json.loads((self.root / "cli_ablation/data_manifest.json").read_text())
        self.assertEqual(manifest_state_contract(saved), (False, 14))

    def test_checkpoint_manifest_compatibility_is_fail_closed(self):
        old = prepare_manifest(self.path, self.root / "old_contract", include_measured_scale=False)
        old["manifest_version"] = 1
        for name in ("include_measured_scale", "state_dim", "state_features"):
            old.pop(name)
        self.assertEqual(manifest_state_contract(old), (False, 14))
        current = prepare_manifest(self.path, self.root / "new_contract")
        for field, value in (("state_dim", 14), ("include_measured_scale", False),
                             ("state_features", ["hidden_mass"] * 15)):
            bad = copy.deepcopy(current)
            bad[field] = value
            with self.assertRaisesRegex(ValueError, "contract"):
                manifest_state_contract(bad)
        bad = copy.deepcopy(current)
        bad["manifest_version"] = 1
        with self.assertRaisesRegex(ValueError, "14 finite"):
            manifest_state_contract(bad)

    def test_scale_missing_bad_shape_or_nonfinite_is_rejected(self):
        observation = {"proprio": np.zeros(14), "mass_g": 123.0}
        with self.assertRaisesRegex(ValueError, "scale_reading_g"):
            observed_state(observation, True)
        for value in (np.array(1.0), np.ones(2)):
            observation["scale_reading_g"] = value
            with self.assertRaises((ValueError, IndexError)):
                observed_state(observation, True)
        observation["scale_reading_g"] = np.array([np.nan])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            observed_state(observation, True)
        with h5py.File(self.path, "a") as handle:
            del handle["data/demo_0/obs/scale_reading_g"]
        with self.assertRaisesRegex(ValueError, "scale_reading_g"):
            inspect_dataset(self.path)

    def test_legacy_requires_explicit_diagnostic_and_shift(self):
        old = self.root / "old.hdf5"
        make_file(old, legacy=True)
        with self.assertRaisesRegex(ValueError, "diagnostic only"):
            prepare_manifest(old, self.root / "rejected")
        manifest = prepare_manifest(old, self.root / "accepted", allow_legacy_diagnostic=True)
        self.assertEqual(manifest["use"], "diagnostic_only")
        self.assertEqual(manifest_state_contract(manifest), (False, 14))
        with self.assertRaisesRegex(ValueError, "legacy diagnostics remain 14-D"):
            prepare_manifest(old, self.root / "legacy_scale", allow_legacy_diagnostic=True, include_measured_scale=True)
        dataset = ScrapeHDFDataset(old, ["demo_0"], 2, legacy=True)
        item = dataset[0]
        np.testing.assert_array_equal(item[ACTION_KEY][0], np.arange(14) + 14)
        np.testing.assert_array_equal(item[STATE_KEY], [0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13])
        self.assertEqual(len(dataset), 3)
        dataset.close()

    def test_manifest_is_immutable_including_dataset_contents(self):
        output = self.root / "run"
        first = prepare_manifest(self.path, output)
        self.assertEqual(first, prepare_manifest(self.path, output))
        with h5py.File(self.path, "a") as handle:
            handle["data/demo_2/actions"][0, 0] = -123
        # Heldout change must not silently reuse a report/checkpoint identity.
        with self.assertRaisesRegex(ValueError, "immutable"):
            prepare_manifest(self.path, output)

    def test_missing_powder_metadata_fails_instead_of_random_frame_split(self):
        with h5py.File(self.path, "a") as handle:
            del handle["data/demo_0"].attrs["powder_id"]
        with self.assertRaisesRegex(ValueError, "powder_id"):
            inspect_dataset(self.path)

    def test_nonfinite_input_fails_before_gpu_load(self):
        with h5py.File(self.path, "a") as handle:
            handle["data/demo_0/actions"][1, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            inspect_dataset(self.path)

    def test_hdf_handles_are_not_pickled_between_workers(self):
        dataset = ScrapeHDFDataset(self.path, ["demo_0"], 2)
        dataset[0]
        state = dataset.__getstate__()
        self.assertIsNone(state["_file"])
        self.assertIsNone(state["_pid"])
        dataset.close()

    def test_unvalidated_v2_data_cannot_be_used_even_with_legacy_override(self):
        with h5py.File(self.path, "a") as handle:
            del handle["data/demo_0"].attrs["physical_valid"]
        with self.assertRaisesRegex(ValueError, "physical validation"):
            prepare_manifest(self.path, self.root / "rejected", allow_legacy_diagnostic=True)

    def test_incomplete_physics_checks_are_rejected(self):
        with h5py.File(self.path, "a") as handle:
            handle["data/demo_0"].attrs["physics_checks"] = 4
        with self.assertRaisesRegex(ValueError, "substep validation"):
            inspect_dataset(self.path)

    def test_online_heldout_split_does_not_require_training_demos_for_unseen_powder(self):
        with h5py.File(self.path, "a") as handle:
            del handle["data/demo_2"]
            handle.attrs["held_out_powder_ids"] = "[5]"
        manifest = prepare_manifest(self.path, self.root / "train_only")
        self.assertEqual(manifest["heldout_powder_ids"], [5])
        self.assertEqual(manifest["heldout_episodes"], [])
        self.assertEqual(len(manifest["train_episodes"]), 2)

    def test_height_gate_cannot_be_mislabeled_as_corrected_training_data(self):
        with h5py.File(self.path, "a") as handle:
            handle.attrs["powder_dynamics"] = "height_gated_surrogate"
        with self.assertRaisesRegex(ValueError, "contact-driven physical grains"):
            prepare_manifest(self.path, self.root / "rejected", allow_legacy_diagnostic=True)


def make_physical_episode(path: Path, powder: int, seed: int = 0):
    """Small file with a complete transition/audit contract, without simulation."""
    material = GRANULAR_MATERIALS[powder]
    length, count = 4, 400
    mass = material.grain_mass_g * np.array([0, 2, 6, 14], dtype=float)
    balance = mass + np.array([0, -.002, -.003, .002])
    initial = count * material.grain_mass_g
    hashes = {"tests/fixture_source.py": "a" * 64}
    source_version = "sha256:" + hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    with h5py.File(path, "w") as handle:
        handle.attrs.update(schema_version="scrape_v2", observation_timing="pre_action",
                            action_semantics="joint_position_radians_gripper_0_255",
                            powder_dynamics="physical_grains_v1", grain_count=count, control_hz=10.,
                            mujoco_version="3.16.0", training_powder_ids="[0,1,2,3,4]", held_out_powder_ids="[5]",
                            powder_parameter_vectors=json.dumps([m.as_dict() for m in GRANULAR_MATERIALS]),
                            source_files_sha256=json.dumps(hashes, sort_keys=True), source_version=source_version)
        demo = handle.create_group("data/demo_0")
        audit = dict(material=material.as_dict(), intentional_contact_tolerance_m=.001,
                     forbidden_contact_tolerance_m=.00005, grain_contact_tolerance_m=.2 * material.radius_m,
                     mass_force_tolerance_g=.01, contact_minima_m={}, grasp_events=[])
        demo.attrs.update(success=1, physical_valid=1, physics_validation="rigid_contact_v1",
                          grasp_model="mujoco_friction_only", physics_steps_per_action=400,
                          physics_checks=1601, physics_audit=json.dumps(audit), num_samples=length,
                          physical_validation_scope="rigid_bodies_and_grains", powder_dynamics="physical_grains_v1",
                          scale_model="physical_pan_force_with_instrument_lag_and_noise",
                          grain_count=count, grain_mass_g=material.grain_mass_g, grain_radius_m=material.radius_m,
                          control_hz=10.,
                          powder_id=powder, seed=seed, target_g=float(mass[-1]), instruction=f"Weigh {mass[-1]:.3f} g.",
                          mass_g=mass[-1], reading_g=balance[-1], spilled_g=0.)
        # Same arm trajectory across materials is legitimate when physical
        # inventories and instrument readings differ.
        proprio = (np.arange((length + 1) * 14, dtype=np.float32).reshape(length + 1, 14) / 100
                   + seed + material.sliding_friction / 100)
        actions = proprio[:-1] + .1
        demo.create_dataset("actions", data=actions)
        for prefix, values, scale in (("obs", proprio[:-1], np.r_[0, balance[:-1]]),
                                     ("next_obs", proprio[1:], balance)):
            demo.create_dataset(f"{prefix}/proprio", data=values)
            demo.create_dataset(f"{prefix}/scale_reading_g", data=scale.astype(np.float32).reshape(-1, 1))
            for camera in CAMERAS:
                demo.create_dataset(f"{prefix}/{camera}", data=np.zeros((length, 8, 8, 3), np.uint8))
        demo.create_dataset("dones", data=np.array([False, False, False, True]))
        demo.create_dataset("timeouts", data=np.zeros(length, bool))
        demo.create_dataset("rewards", data=np.array([-1., -1., -1., 0.], np.float32))
        demo.create_dataset("material", data=np.array(list(material.as_dict().values()), np.float32))
        demo.create_dataset("subgoal_dones", data=[[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
        demo.create_dataset("physics/checks", data=np.arange(1, length + 1) * 400 + 1)
        demo.create_dataset("physics/qpos", data=proprio[1:])
        demo.create_dataset("physics/qvel", data=np.zeros((length, 14)))
        fields = dict(mass_g=mass, source_g=initial - mass, tool_g=np.zeros(length),
                      spilled_g=np.zeros(length), inflight_g=np.zeros(length),
                      raw_net_mass_g=balance, mass_force_error_g=balance - mass,
                      conservation_error_g=np.zeros(length), max_grain_contact_penetration_m=np.full(length, .0002),
                      balance_g=balance, elapsed_s=np.arange(1, length + 1) * .1)
        for key, value in fields.items():
            demo.create_dataset(f"physics/{key}", data=value)


class GranularMergeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = [self.root / f"powder_{i}.hdf5" for i in range(5)]
        for powder, path in enumerate(self.paths):
            make_physical_episode(path, powder)
        self.output = self.root / "merged.hdf5"

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_pilot_preserves_identity_and_passes_training_inspection(self):
        audit = merge_demos(self.paths, self.output, expected_episodes=5)
        self.assertEqual(audit["num_episodes"], 5)
        self.assertEqual(audit["heldout_episodes"], [])
        self.assertEqual({row["powder_id"] for row in audit["episodes"]}, set(range(5)))
        manifest = prepare_manifest(self.output, self.root / "run")
        self.assertEqual(manifest_state_contract(manifest), (True, 15))
        with h5py.File(self.output) as handle:
            self.assertEqual(handle.attrs["expected_episode_count"], 5)
            self.assertEqual(handle.attrs["merged_episode_count"], 5)
            self.assertEqual(len(json.loads(handle.attrs["merge_provenance"])), 5)
        with self.assertRaises(FileExistsError):
            merge_demos(self.paths, self.output, 5)

    def test_merge_rechunks_rgb_preserves_all_episode_bytes_attributes_and_sources(self):
        for index, path in enumerate(self.paths):
            with h5py.File(path, 'r+') as handle:
                demo = handle['data/demo_0']
                demo.attrs['custom_integer_vector'] = np.array([1, 2, 3], dtype=np.int16)
                for prefix in ('obs', 'next_obs'):
                    demo[prefix].attrs['sensor_group_note'] = 'measured observations'
                    for camera in CAMERAS:
                        name = f'{prefix}/{camera}'
                        del demo[name]
                        pixels = (np.arange(4 * 8 * 8 * 3).reshape(4, 8, 8, 3) + index).astype(np.uint8)
                        image = demo.create_dataset(name, data=pixels, chunks=(4, 4, 4, 3), compression='gzip', compression_opts=4)
                        image.attrs['color_space'] = 'RGB'
                        image.attrs['calibration'] = np.array([.25, .5], dtype=np.float32)
                    demo[f'{prefix}/proprio'].attrs['units'] = 'radians_and_gripper'
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in self.paths]
        audit = merge_demos(self.paths, self.output, 5)
        self.assertEqual(audit['num_episodes'], 5)
        with h5py.File(self.output) as merged:
            self.assertEqual(merged.attrs['rgb_storage_layout'], 'one_frame_per_chunk_gzip4')
            for index, path in enumerate(self.paths):
                with h5py.File(path) as original:
                    before, after = original['data/demo_0'], merged[f'data/demo_{index}']
                    objects = ['']
                    before.visit(objects.append)
                    for name in objects:
                        left, right = (before[name], after[name]) if name else (before, after)
                        self.assertEqual(set(left.attrs), set(right.attrs))
                        for attr in left.attrs:
                            np.testing.assert_array_equal(left.attrs[attr], right.attrs[attr])
                            self.assertEqual(left.attrs.get_id(attr).dtype, right.attrs.get_id(attr).dtype)
                            self.assertEqual(left.attrs.get_id(attr).shape, right.attrs.get_id(attr).shape)
                        if isinstance(left, h5py.Dataset):
                            self.assertEqual(left.dtype, right.dtype)
                            self.assertEqual(left.shape, right.shape)
                            np.testing.assert_array_equal(left[()], right[()])
                            if name.split('/')[-1] in CAMERAS:
                                self.assertEqual(right.chunks, (1, 8, 8, 3))
                                self.assertEqual((right.compression, right.compression_opts), ('gzip', 4))
                            else:
                                self.assertEqual(left.chunks, right.chunks)
                                self.assertEqual(left.compression, right.compression)
                    self.assertEqual(original.attrs['source_version'], merged.attrs['source_version'])
        self.assertEqual(hashes, [hashlib.sha256(path.read_bytes()).hexdigest() for path in self.paths])

    def test_rgb_rechunking_uses_bounded_slices_and_copies_terminal_tail(self):
        source_path = self.root / 'rgb_source.hdf5'
        target_path = self.root / 'rgb_target.hdf5'
        pixels = np.random.default_rng(42).integers(0, 256, (RGB_COPY_FRAMES * 2 + 3, 8, 8, 3), dtype=np.uint8)
        with h5py.File(source_path, 'w') as handle:
            handle.create_dataset('rgb', data=pixels, compression='gzip', compression_opts=4)
        original_get = h5py.Dataset.__getitem__
        reads = []
        def bounded_get(dataset, index):
            if dataset.name == '/rgb' and Path(dataset.file.filename) == source_path:
                self.assertIsInstance(index, slice)
                self.assertLessEqual(index.stop - index.start, RGB_COPY_FRAMES)
                reads.append((index.start, index.stop))
            return original_get(dataset, index)
        with h5py.File(source_path) as source, h5py.File(target_path, 'w') as target, \
                patch.object(h5py.Dataset, '__getitem__', bounded_get):
            _copy_rgb_dataset(source['rgb'], target, 'rgb')
            np.testing.assert_array_equal(target['rgb'][:], pixels)
        self.assertEqual(reads, [(0, 32), (32, 64), (64, 67)])

    def test_rechunk_failure_never_publishes_a_partial_merge(self):
        with patch('scripts.merge_granular_demos._copy_rgb_dataset', side_effect=RuntimeError('write failure')):
            with self.assertRaisesRegex(RuntimeError, 'write failure'):
                merge_demos(self.paths, self.output, 5)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('*.pending')))

    def test_default_requires_thirty_and_rejection_writes_nothing(self):
        with self.assertRaisesRegex(ValueError, "Expected 30 episodes"):
            merge_demos(self.paths, self.output)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob("*.pending")))

    def test_exact_thirty_episode_collection_is_accepted(self):
        paths = []
        for seed in range(6):
            for powder in range(5):
                path = self.root / f"episode_{powder}_{seed}.hdf5"
                make_physical_episode(path, powder, seed)
                paths.append(path)
        audit = merge_demos(paths, self.output)
        self.assertEqual(audit["num_episodes"], 30)
        self.assertEqual(len(audit["train_episodes"]), 30)

    def test_heldout_material_never_enters_the_training_aggregate(self):
        heldout = self.root / "heldout.hdf5"
        make_physical_episode(heldout, 5)
        with self.assertRaisesRegex(ValueError, "held-out"):
            merge_demos([*self.paths, heldout], self.output, 6)
        self.assertFalse(self.output.exists())

    def test_duplicate_paths_identity_and_disguised_trajectory_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate paths"):
            merge_demos([*self.paths, self.paths[0]], self.output, 6)
        duplicate = self.root / "duplicate.hdf5"
        make_physical_episode(duplicate, 0)
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            merge_demos([*self.paths, duplicate], self.output, 6)
        with h5py.File(duplicate, "a") as handle:
            demo = handle["data/demo_0"]
            demo.attrs["seed"] = 999
            # A renamed episode with only new instrument noise is still the
            # same physical trajectory and must not inflate demo count.
            changed = demo["physics/balance_g"][:] + .0001
            demo["physics/balance_g"][:] = changed
            demo["next_obs/scale_reading_g"][:, 0] = changed
            demo["obs/scale_reading_g"][1:, 0] = changed[:-1]
            demo.attrs["reading_g"] = changed[-1]
        with self.assertRaisesRegex(ValueError, "numerical trajectory"):
            merge_demos([*self.paths, duplicate], self.output, 6)

    def test_source_and_material_bank_changes_are_rejected(self):
        with h5py.File(self.paths[1], "a") as handle:
            handle.attrs["source_version"] = "sha256:" + "b" * 64
        with self.assertRaisesRegex(ValueError, "source_version"):
            merge_demos(self.paths, self.output, 5)
        make_physical_episode(self.paths[1], 1)
        with h5py.File(self.paths[1], "a") as handle:
            bank = json.loads(handle.attrs["powder_parameter_vectors"])
            bank[0]["sliding_friction"] += .1
            handle.attrs["powder_parameter_vectors"] = json.dumps(bank)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            merge_demos(self.paths, self.output, 5)

    def test_mixed_grain_count_and_split_are_rejected(self):
        with h5py.File(self.paths[1], "a") as handle:
            handle.attrs["grain_count"] = 399
        with self.assertRaisesRegex(ValueError, "incompatible"):
            merge_demos(self.paths, self.output, 5)
        make_physical_episode(self.paths[1], 1)
        with h5py.File(self.paths[1], "a") as handle:
            handle.attrs["held_out_powder_ids"] = "[4]"
        with self.assertRaisesRegex(ValueError, "Required split"):
            merge_demos(self.paths, self.output, 5)

    def test_mass_conservation_is_checked_before_the_final_frame(self):
        with h5py.File(self.paths[0], "a") as handle:
            handle["data/demo_0/physics/source_g"][1] -= .02
        with self.assertRaisesRegex(ValueError, "conservation"):
            merge_demos(self.paths, self.output, 5)
        self.assertFalse(self.output.exists())

    def test_penetration_at_any_recorded_step_rejects_episode(self):
        with h5py.File(self.paths[0], "a") as handle:
            handle["data/demo_0/physics/max_grain_contact_penetration_m"][1:] = .00031
        with self.assertRaisesRegex(ValueError, "penetration"):
            merge_demos(self.paths, self.output, 5)

    def test_raw_force_must_agree_even_if_displayed_reading_matches_target(self):
        with h5py.File(self.paths[0], "a") as handle:
            handle["data/demo_0/physics/raw_net_mass_g"][-1] += .02
            handle["data/demo_0/physics/mass_force_error_g"][-1] += .02
        with self.assertRaisesRegex(ValueError, "raw force"):
            merge_demos(self.paths, self.output, 5)

    def test_one_missing_substep_cannot_hide_behind_final_check_count(self):
        with h5py.File(self.paths[0], "a") as handle:
            handle["data/demo_0/physics/checks"][1] -= 1
        with self.assertRaisesRegex(ValueError, "per-action physics checks"):
            merge_demos(self.paths, self.output, 5)

    def test_incomplete_end_or_discontinuous_observations_reject(self):
        with h5py.File(self.paths[0], "a") as handle:
            handle["data/demo_0/dones"][-1] = False
        with self.assertRaisesRegex(ValueError, "complete successful episode"):
            merge_demos(self.paths, self.output, 5)
        make_physical_episode(self.paths[0], 0)
        with h5py.File(self.paths[0], "a") as handle:
            handle["data/demo_0/obs/scale_reading_g"][2, 0] += .01
        with self.assertRaisesRegex(ValueError, "discontinuous"):
            merge_demos(self.paths, self.output, 5)

    def test_all_declared_training_materials_are_required_for_small_pilot(self):
        with self.assertRaisesRegex(ValueError, "Every declared training material"):
            merge_demos(self.paths[:-1], self.output, 4)


if __name__ == "__main__":
    unittest.main()
