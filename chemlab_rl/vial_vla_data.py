"""Separate, audited SmolVLA contract for the fixed-scene vial placement task.

This module does not relax the physical powder dataset gates. The historical
vial demonstrations have native contact dynamics but no saved contact audit.
"""
from __future__ import annotations

import bisect
import json
import os
from pathlib import Path

import h5py
import numpy as np

from chemlab_rl.vla_bc import (
    ACTION_DIM, ACTION_KEY, STATE_KEY, LEROBOT_VERSION, PRETRAINED_MODEL,
    PRETRAINED_REVISION, _tensor_stats, file_sha256,
)

TASK_ID = "vial_placement_native_v1"
SCHEMA = "vial_weighing_robomimic_v1"
INSTRUCTION = "Open the top draft shield, place the vial on the balance pan, and close the draft shield."
CAMERAS = {
    "top1_image": "observation.images.camera1",
    "robot0_eye_in_hand_image": "observation.images.camera2",
    "robot1_eye_in_hand_image": "observation.images.camera3",
}
STATE_FEATURES = [*[f"left_joint_{i}" for i in range(6)], "left_gripper_driver_rad",
                  *[f"right_joint_{i}" for i in range(6)], "right_gripper_driver_rad", "scale_reading_g"]
TRAIN_EPISODES = [f"demo_{i}" for i in range(7)]
HELDOUT_EPISODES = ["demo_7"]
EXCLUDED_EPISODES = ["demo_8"]
ACTION_SEMANTICS = "normalized_absolute_actuator_targets_minus1_to1"
RESULT_SCOPE = "fixed_scene_vial_placement_bc_pilot"
ORIGINAL_DATASET_SHA256 = "1d2fc0e3a809b1bb00ceb12b4fa99bc8d7c39175e9e39b5a1592d20d7c942950"
# Full semantic payload hashes are filled from the immutable original HDF5.
HISTORICAL_EPISODE_SHA256 = {'demo_0': '926774c953558382d3201a7cf6ef6a21c6fb6d64c980f907478ce84215ecad54', 'demo_1': '626eddc79f1ce8f2460d705f83aa7edad7a128b38e26c321d24bbc1eafa5912f', 'demo_2': '7f51f6629a18ee44aaf8fe6f510256c3c3350a0ef1ec29469305034879f7cbae', 'demo_3': 'cec3bb2b33cfabfe4f5b20c88e42791ea96694749eaa601300abfd6f8b337a95', 'demo_4': 'b59e7b53d2ee9d420980a0764977f89dabe5c2e63d49daaba29fa22806eafc06', 'demo_5': 'a84f1f5e2a1daf2f5b4017fe4cabf8e888b66c705e2b8097287947ca4d75ffac', 'demo_6': '62c689eba47304020c490b65edb5239fce51c8dd1385779dd72dc2fabe123fe0', 'demo_7': '31849467b55b488d0fc2da4b1ea47d0d757637138be49845d99c83dc222da597'}


def episode_payload_sha256(group):
    """Hash every recorded dataset byte and original recorder metadata, independent of HDF packing."""
    import hashlib
    digest = hashlib.sha256()
    datasets = []
    group.visititems(lambda name, obj: datasets.append((name, obj)) if isinstance(obj, h5py.Dataset) else None)
    for name, dataset in sorted(datasets):
        digest.update(json.dumps([name, dataset.shape, dataset.dtype.str], separators=(",", ":")).encode())
        for start in range(0, len(dataset), 32):
            digest.update(np.ascontiguousarray(dataset[start:start + 32]).tobytes())
    metadata = {key: group.attrs[key].item() if isinstance(group.attrs[key], np.generic) else group.attrs[key]
                for key in ("num_samples", "success", "date", "duration_s") if key in group.attrs}
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def _validate_native_audit(group):
    """Check serialized substep evidence without clipping measured force inputs."""
    import hashlib
    try:
        audit = json.loads(group.attrs["physics_audit"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("New hybrid episodes require serialized physics_audit evidence") from error
    limits = dict(unintended_rigid_penetration_m=.00005, other_contact_penetration_m=.001,
                  sustained_scale_load_g=100., sustained_scale_duration_s=.1)
    if audit.get("accepted") is not True or audit.get("task_id") != TASK_ID or audit.get("limits") != limits:
        raise ValueError("Native physics audit acceptance/task/limits differ")
    n = len(group["actions"])
    for name in ("samples", "controls", "physics_checks"):
        if type(audit.get(name)) is not int:
            raise ValueError("Native physics audit counts must be integers")
    if (audit["samples"] != n or audit["controls"] < n or audit["physics_checks"] < audit["controls"] * 25
            or group.attrs.get("physics_checks") != audit["physics_checks"]
            or audit.get("physics_timestep_s") != .002 or audit.get("control_hz") != 20.):
        raise ValueError("Native physics audit does not cover every recorded control at25substeps")
    for name in ("source_version", "collector_sha256"):
        value = audit.get(name)
        digest = value.removeprefix("sha256:") if isinstance(value, str) else ""
        if (value != group.attrs.get(name) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("Native physics audit source identity differs from the episode")
    actual = hashlib.sha256(group["actions"][:].tobytes() + group["obs/joint_pos"][:].tobytes()).hexdigest()
    if audit.get("trajectory_sha256") != actual:
        raise ValueError("Native physics audit trajectory SHA differs from recorded commands/joints")
    fields = ("max_penetration_m", "max_unintended_penetration_m", "max_abs_scale_g", "max_excessive_load_seconds")
    values = {}
    for name in fields:
        value = audit.get(name)
        if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
            raise ValueError("Native physics audit requires finite nonnegative extrema")
        values[name] = value
    if (values["max_penetration_m"] > .001 or values["max_unintended_penetration_m"] > .00005
            or values["max_unintended_penetration_m"] > values["max_penetration_m"]
            or values["max_excessive_load_seconds"] > .1):
        raise ValueError("Native physics audit violates contact or sustained scale-duration limits")
    peak = max(float(np.abs(group[key][..., 7]).max()) for key in ("obs/object", "next_obs/object"))
    maximum = values["max_abs_scale_g"]
    tolerance = max(1e-4, maximum * 1e-6)  # HDF observations are float32, native audit is float64.
    if not np.isfinite(peak) or peak > maximum + tolerance:
        raise ValueError("Recorded scale reading exceeds the unfiltered substep audit maximum")
    if ((maximum > 100 and values["max_excessive_load_seconds"] < .002)
            or (maximum <= 100 and values["max_excessive_load_seconds"] != 0)):
        raise ValueError("Native physics audit peak and excessive-load duration are inconsistent")
    return audit


def _validate_hybrid_provenance(data, train, heldout, excluded):
    expected_train = [*[f"demo_{i}" for i in range(7)], *[f"demo_{i}" for i in range(8, 98)]]
    if train != expected_train or heldout != ["demo_7"] or excluded or set(data) != {f"demo_{i}" for i in range(98)}:
        raise ValueError("Hybrid98 requires historical0..7 plus90new demos, heldout7 and no current-file exclusions")
    new_hashes, new_source_episodes = set(), set()
    for i in range(98):
        name = f"demo_{i}"
        group = data[name]
        if i < 8:
            if (group.attrs.get("collector") != "grok_session_v2"
                    or group.attrs.get("source_dataset_sha256") != ORIGINAL_DATASET_SHA256
                    or group.attrs.get("source_episode") != name
                    or episode_payload_sha256(group) != HISTORICAL_EPISODE_SHA256.get(name)):
                raise ValueError(f"{name}: historical payload/provenance changed in hybrid98")
        else:
            source_sha = str(group.attrs.get("source_dataset_sha256", ""))
            source_episode = str(group.attrs.get("source_episode", ""))
            if (group.attrs.get("collector") != "audited_continuous_vial_v1"
                    or group.attrs.get("physical_valid") != 1
                    or group.attrs.get("physics_validation") != "native_vial_substep_contacts_v1"
                    or len(source_sha) != 64 or any(c not in "0123456789abcdef" for c in source_sha)
                    or source_sha == ORIGINAL_DATASET_SHA256 or source_episode in new_source_episodes):
                raise ValueError(f"{name}: new hybrid episode lacks unique native physical provenance")
            new_hashes.add(source_sha)
            new_source_episodes.add(source_episode)
            _validate_native_audit(group)
    if len(new_hashes) != 1 or new_source_episodes != {f"demo_{i}" for i in range(90)}:
        raise ValueError("Hybrid98 must reference all90 unique episodes of one audited new dataset")


def validate_split(train, heldout, excluded):
    groups = [list(train), list(heldout), list(excluded)]
    if not groups[0] or not groups[1] or any(len(set(g)) != len(g) for g in groups):
        raise ValueError("Need nonempty unique train/heldout episode lists")
    if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Episode split overlap is forbidden")
    if not all(isinstance(n, str) and n.startswith("demo_") and n[5:].isdigit() for g in groups for n in g):
        raise ValueError("Episode identifiers must be demo_<integer>")
    return groups


def observed_state(observation, selection=slice(None)) -> np.ndarray:
    """Allowlist only measured joints and instrument reading; omit object pose."""
    q = np.asarray(observation["joint_pos"][selection], dtype=np.float32)
    obj = np.asarray(observation["object"][selection], dtype=np.float32)
    if q.ndim < 1 or q.shape[-1] != 14 or obj.shape != (*q.shape[:-1], 11):
        raise ValueError("Vial actor requires joint_pos[...,14] and object[...,11]")
    state = np.concatenate([q, obj[..., 7:8]], axis=-1)
    if not np.isfinite(state).all():
        raise ValueError("Nonfinite measured vial actor state")
    return state


def _validate_stats(statistics):
    for key, dimension in ((STATE_KEY, 15), (ACTION_KEY, 14)):
        for field in ("mean", "std", "min", "max"):
            value = np.asarray(statistics[key][field], dtype=float)
            if value.shape != (dimension,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid vial statistics for {key}/{field}")
            if field == "std" and np.any(value <= 0):
                raise ValueError("Normalization std must be positive")


def manifest_state_contract(manifest: dict) -> tuple[bool, int]:
    expected = dict(manifest_version=1, task_id=TASK_ID, schema_version=SCHEMA,
                    state_dim=15, action_dim=14, include_measured_scale=True,
                    state_features=STATE_FEATURES, camera_mapping=CAMERAS,
                    action_semantics=ACTION_SEMANTICS, control_hz=20.0,
                    normalization_fitted_on="train_episodes_only",
                    terminal_handling="truncate_at_first_terminal_inclusive")
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Vial manifest contract mismatch: {key}")
    if not isinstance(manifest.get("dataset_sha256"), str) or len(manifest["dataset_sha256"]) != 64:
        raise ValueError("Missing vial dataset SHA256")
    lengths = manifest.get("episode_lengths", {})
    train, heldout, _ = validate_split(manifest["train_episodes"], manifest["heldout_episodes"], manifest["excluded_episodes"])
    if set(lengths) != set(train + heldout) or any(type(n) is not int or n < 1 for n in lengths.values()):
        raise ValueError("Invalid retained vial episode lengths")
    _validate_stats(manifest["statistics"])
    return True, 15


def inspect_dataset(path: str | Path, *, check_images: bool = True, train_episodes=TRAIN_EPISODES,
                    heldout_episodes=HELDOUT_EPISODES, excluded_episodes=EXCLUDED_EPISODES) -> dict:
    """Validate actual recorded alignment and retain the explicit quality filter."""
    import hashlib
    train, heldout, excluded = validate_split(train_episodes, heldout_episodes, excluded_episodes)
    rows, fingerprints, initial_states = [], set(), []
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        if set(data) != set(train + heldout + excluded):
            raise ValueError("Every dataset episode must appear in the explicit split")
        env_args = json.loads(data.attrs["env_args"])
        kwargs = env_args["env_kwargs"]
        if (float(data.attrs["control_hz"]) != 20.0 or data.attrs["effector"] != "robotiq"
                or env_args.get("collector") not in {"grok_session_v2", "audited_continuous_vial_v1", "hybrid_vial_98_v1"}
                or any(kwargs.get(k) != v for k, v in dict(control_hz=20.0, hand="robotiq", layout="weigh", clutter=False,
                                                           normalized_actions=True, image_size=64).items())):
            raise ValueError("Vial collector/environment contract differs")
        hybrid = env_args.get("collector") == "hybrid_vial_98_v1"
        if hybrid:
            _validate_hybrid_provenance(data, train, heldout, excluded)
        for name in sorted(data, key=lambda x: int(x.split("_")[-1])):
            g = data[name]
            if env_args.get("collector") == "audited_continuous_vial_v1" and (
                    g.attrs.get("physical_valid") != 1 or g.attrs.get("physics_validation") != "native_vial_substep_contacts_v1"):
                raise ValueError(f"{name}: new native collection requires its recorded physical acceptance")
            actions = np.asarray(g["actions"][:], dtype=np.float32)
            absolute = np.asarray(g["actions_abs"][:], dtype=np.float32)
            n = len(actions)
            state, next_state = observed_state(g["obs"]), observed_state(g["next_obs"])
            obj, next_obj = g["obs/object"][:], g["next_obs/object"][:]
            raw = g["states"][:]
            rewards, dones = g["rewards"][:], g["dones"][:]
            if actions.shape != (n, 14) or absolute.shape != actions.shape or state.shape != (n, 15) or raw.shape != (n, 76):
                raise ValueError(f"{name}: invalid dimensions")
            if not all(np.isfinite(v).all() for v in (actions, absolute, raw, obj, next_obj, rewards)):
                raise ValueError(f"{name}: nonfinite recorded arrays")
            if np.max(np.abs(actions)) > 1.000001:
                raise ValueError(f"{name}: actions are not normalized actuator targets")
            if (dones.shape != (n,) or rewards.shape != (n,) or not np.isin(dones, [0, 1]).all()
                    or not np.array_equal(dones, rewards) or not dones.any()):
                raise ValueError(f"{name}: invalid sparse terminal labels")
            terminal = int(np.flatnonzero(dones)[0])
            if not np.all(dones[terminal:] == 1) or g.attrs.get("success") != 1 or g.attrs.get("num_samples") != n:
                raise ValueError(f"{name}: inconsistent success/length metadata")
            if not np.allclose(np.diff(raw[:, 0]), .05, atol=1e-9, rtol=0):
                raise ValueError(f"{name}: recorded control time is not 20Hz")
            if not np.array_equal(next_state[:-1], state[1:]) or not np.array_equal(next_obj[:-1], obj[1:]):
                raise ValueError(f"{name}: adjacent observation alignment failed")
            for key in CAMERAS:
                left, right = g["obs"][key], g["next_obs"][key]
                if left.shape != (n, 64, 64, 3) or right.shape != left.shape or left.dtype != np.uint8 or right.dtype != np.uint8:
                    raise ValueError(f"{name}: invalid RGB camera {key}")
                if check_images:
                    for start in range(0, n - 1, 32):
                        end = min(start + 32, n - 1)
                        if not np.array_equal(right[start:end], left[start + 1:end + 1]):
                            raise ValueError(f"{name}: adjacent image alignment failed for {key}")
            fingerprint = hashlib.sha256(actions.tobytes() + raw.tobytes()).hexdigest()
            if fingerprint in fingerprints:
                raise ValueError(f"{name}: duplicate complete action/state trajectory")
            fingerprints.add(fingerprint)
            initial_states.append(raw[0])
            peak = float(max(obj[:, 7].max(), next_obj[:, 7].max()))
            accepted = name not in excluded
            audited_native = (hybrid and g.attrs.get("collector") == "audited_continuous_vial_v1")
            if env_args.get("collector") == "audited_continuous_vial_v1" and "physics_audit" in g.attrs:
                _validate_native_audit(g)
                audited_native = True
            if accepted and peak > 100 and not audited_native:
                raise ValueError(f"{name}: additional excessive scale force requires explicit quality review")
            rows.append(dict(name=name, original_samples=n, retained_samples=terminal + 1 if accepted else 0,
                             first_terminal=terminal, success_frames=int(dones.sum()),
                             peak_scale_reading_g=peak, action_state_sha256=fingerprint,
                             contact_audit_verified=audited_native,
                             split="train" if name in train else "heldout" if accepted else "excluded",
                             exclusion_reason=("excessive instrument force during placement" if peak > 100 else "explicit protocol exclusion") if not accepted else None))
    return dict(task_id=TASK_ID, env_args=env_args, episodes=rows, dataset_sha256=file_sha256(path),
                all_initial_states_identical=all(np.array_equal(initial_states[0], x) for x in initial_states),
                image_continuity_checked=check_images, observation_alignment="obs[t] -> actions[t] -> next_obs[t]",
                historical_contact_audit_available=False,
                audited_new_episode_count=sum(int(row["contact_audit_verified"]) for row in rows),
                excluded_original_episodes=[dict(source_dataset_sha256=ORIGINAL_DATASET_SHA256, source_episode="demo_8",
                    reason="excessive instrument force during placement; absent from hybrid98")] if hybrid else [],
                generalization_scope="fixed scene; held-out trajectory only")


def prepare_manifest(path: str | Path, output_dir: str | Path, *, train_episodes=TRAIN_EPISODES,
                     heldout_episodes=HELDOUT_EPISODES, excluded_episodes=EXCLUDED_EPISODES) -> dict:
    train, heldout, excluded = validate_split(train_episodes, heldout_episodes, excluded_episodes)
    audit = inspect_dataset(path, train_episodes=train, heldout_episodes=heldout, excluded_episodes=excluded)
    lengths = {r["name"]: r["retained_samples"] for r in audit["episodes"] if r["split"] != "excluded"}
    channels = {STATE_KEY: [], ACTION_KEY: []}
    with h5py.File(path, "r") as handle:
        for name in train:
            group = handle["data"][name]
            channels[STATE_KEY].append(observed_state(group["obs"], slice(0, lengths[name])))
            channels[ACTION_KEY].append(group["actions"][:lengths[name]])
    statistics = {}
    for key, values in channels.items():
        array = np.concatenate(values).astype(np.float64)
        statistics[key] = dict(mean=array.mean(0).tolist(), std=np.maximum(array.std(0), 1e-6).tolist(),
                               min=array.min(0).tolist(), max=array.max(0).tolist(), count=[len(array)])
    manifest = dict(manifest_version=1, task_id=TASK_ID, schema_version=SCHEMA, use=RESULT_SCOPE,
                    dataset_sha256=audit["dataset_sha256"], train_episodes=train,
                    heldout_episodes=heldout, excluded_episodes=excluded,
                    episode_lengths=lengths, terminal_handling="truncate_at_first_terminal_inclusive",
                    state_dim=15, action_dim=14, state_features=STATE_FEATURES, include_measured_scale=True,
                    camera_mapping=CAMERAS, image_shape=[64, 64, 3], instruction=INSTRUCTION,
                    action_semantics=ACTION_SEMANTICS, control_hz=20.0,
                    normalization_fitted_on="train_episodes_only", statistics=statistics,
                    collector_env_args=audit["env_args"], audit=audit)
    manifest_state_contract(manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("data_manifest.json", manifest), ("dataset_audit.json", audit)):
        target = output_dir / name
        if target.exists() and json.loads(target.read_text()) != payload:
            raise ValueError(f"Existing immutable {name} differs; choose a new output directory")
        target.write_text(json.dumps(payload, indent=2) + "\n")
    return manifest


class VialHDFDataset:
    def __init__(self, path, manifest, episodes, chunk_size=50):
        manifest_state_contract(manifest)
        if chunk_size < 1 or not episodes or len(set(episodes)) != len(episodes) or not set(episodes) <= set(manifest["episode_lengths"]):
            raise ValueError("Invalid episode selection/chunk size")
        self.path, self.manifest = str(path), manifest
        self.episodes, self.chunk_size = list(episodes), chunk_size
        self.lengths = [manifest["episode_lengths"][n] for n in episodes]
        self.ends = np.cumsum(self.lengths).tolist()
        self._file = self._pid = None

    def __len__(self):
        return self.ends[-1]

    def _handle(self):
        if self._file is None or self._pid != os.getpid():
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._file

    def __getstate__(self):
        return {**self.__dict__, "_file": None, "_pid": None}

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect.bisect_right(self.ends, index)
        t = index - (self.ends[episode - 1] if episode else 0)
        group = self._handle()["data"][self.episodes[episode]]
        actions = group["actions"][t:min(t + self.chunk_size, self.lengths[episode])].astype(np.float32)
        valid = len(actions)
        if valid < self.chunk_size:
            actions = np.concatenate([actions, np.repeat(actions[-1:], self.chunk_size - valid, axis=0)])
        result = {STATE_KEY: observed_state(group["obs"], t), ACTION_KEY: actions,
                  "action_is_pad": np.arange(self.chunk_size) >= valid, "task": INSTRUCTION}
        for source, destination in CAMERAS.items():
            result[destination] = group["obs"][source][t].transpose(2, 0, 1).astype(np.float32) / 255
        return result

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def load_smolvla(manifest, device, pretrained=PRETRAINED_MODEL, revision=None, chunk_size=None, n_action_steps=None):
    """Same pinned strict weight-loading path as powder BC, with a separate contract."""
    manifest_state_contract(manifest)
    import importlib.metadata
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    if importlib.metadata.version("lerobot") != LEROBOT_VERSION:
        raise RuntimeError(f"Expected tested lerobot=={LEROBOT_VERSION}")
    local_manifest = Path(pretrained) / "data_manifest.json"
    if local_manifest.is_file() and json.loads(local_manifest.read_text()) != manifest:
        raise ValueError("Checkpoint data contract differs from requested vial manifest")
    if pretrained == PRETRAINED_MODEL and revision is None:
        revision = PRETRAINED_REVISION
    config = SmolVLAConfig.from_pretrained(pretrained, revision=revision)
    config.device = device
    if config.max_state_dim != 32 or config.max_action_dim != 32:
        raise ValueError("Expected pretrained32-dimensional padded projections")
    config.input_features = {STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(15,)),
                             **{name: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for name in CAMERAS.values()}}
    config.output_features = {ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(14,))}
    if chunk_size is not None:
        config.chunk_size = chunk_size
    if n_action_steps is not None:
        config.n_action_steps = n_action_steps
    config.adapt_to_pi_aloha = False
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(pretrained, config=config, revision=revision, strict=True)
    policy.to(device=device, dtype=torch.float32)
    pre, post = make_smolvla_pre_post_processors(config, _tensor_stats(manifest["statistics"]))
    return policy, pre, post


def make_inference_batch(observation, instruction=INSTRUCTION):
    import torch
    state = observed_state(observation)
    if state.shape != (15,):
        raise ValueError("Expected a single vial observation")
    batch = {STATE_KEY: torch.from_numpy(state.copy()), "task": instruction}
    for source, destination in CAMERAS.items():
        image = np.asarray(observation[source])
        if image.shape != (64, 64, 3) or image.dtype != np.uint8:
            raise ValueError("Vial cameras must be native64x64 RGB uint8")
        batch[destination] = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255
    return batch


class VialVLABCPolicy:
    def __init__(self, checkpoint, device="cuda", bf16=True):
        checkpoint = Path(checkpoint)
        self.manifest = json.loads((checkpoint / "data_manifest.json").read_text())
        manifest_state_contract(self.manifest)
        self.policy, self.preprocessor, self.postprocessor = load_smolvla(self.manifest, device, str(checkpoint))
        self.policy.eval()
        self.bf16 = bf16 and device.startswith("cuda")

    def reset(self):
        self.policy.reset()

    def act(self, observation, instruction=INSTRUCTION):
        import torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            action = self.postprocessor(self.policy.select_action(self.preprocessor(make_inference_batch(observation, instruction))))
        return action[0].detach().cpu().numpy()
