"""Leakage-aware HDF5 data interface and pretrained SmolVLA behavior cloning.

The old scrape file is diagnostic only: snapshots follow their actions and the
collector supplies transfer/grasp side effects. ``scrape_v2`` records pre-action
observations. Neither an offline loss nor successful stored frames establish
closed-loop weighing success.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import h5py
import numpy as np

CAMERAS = {"top1": "observation.images.camera1", "left_wrist": "observation.images.camera2", "right_wrist": "observation.images.camera3"}
STATE_KEY = "observation.state"
ACTION_KEY = "action"
ACTION_DIM = 14
PROPRIO_DIM = 14
SCALE_READING_KEY = "scale_reading_g"
PROPRIO_FEATURES = [*[f"left_joint_{i}" for i in range(6)], "left_gripper",
                   *[f"right_joint_{i}" for i in range(6)], "right_gripper"]
LEROBOT_VERSION = "0.7.0"
PRETRAINED_MODEL = "lerobot/smolvla_base"
PRETRAINED_REVISION = "5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6"
# Old lowdim = left joints, right joints, left/right gripper commands,
# door positions, balance reading, target, error. Do not expose the last six.
LEGACY_PROPRIO_COLUMNS = [*range(6), 12, *range(6, 12), 13]


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _proprio(group: h5py.Group, selection: Any = slice(None)) -> np.ndarray:
    if "proprio" in group:
        result = np.asarray(group["proprio"][selection], dtype=np.float32)
    else:
        # Legacy gripper entries are commanded aperture, not measured joints.
        result = np.asarray(group["lowdim"][selection], dtype=np.float32)[..., LEGACY_PROPRIO_COLUMNS]
    if result.ndim < 1 or result.shape[-1] != PROPRIO_DIM:
        raise ValueError(f"Expected 14 proprio dimensions, got {result.shape}")
    return result


def observed_state(observation: Any, include_measured_scale: bool = False,
                   selection: Any = slice(None)) -> np.ndarray:
    """Select actor inputs by an allowlist; never read hidden mass or material.

    Works with an HDF5 observation group or an environment observation dict.
    The optional scalar is the measured instrument reading, in grams.
    """
    state = _proprio(observation, selection)
    if include_measured_scale:
        if SCALE_READING_KEY not in observation:
            raise ValueError("Measured obs/scale_reading_g is required by this observation contract")
        try:
            scale = np.asarray(observation[SCALE_READING_KEY][selection], dtype=np.float32)
        except (IndexError, TypeError) as error:
            raise ValueError("Measured scale must have an explicit trailing singleton dimension") from error
        expected = (*state.shape[:-1], 1)
        if scale.shape != expected:
            raise ValueError(f"Expected measured scale shape {expected}, got {scale.shape}")
        state = np.concatenate([state, scale], axis=-1)
    if not np.isfinite(state).all():
        raise ValueError("nonfinite measured policy state")
    return state


def _resolve_scale_selection(attrs: Any, legacy: bool, requested: bool | None) -> bool:
    physical = (_text(attrs.get("schema_version", "legacy")) == "scrape_v2"
                and _text(attrs.get("powder_dynamics", "")) == "physical_grains_v1"
                and not legacy)
    if requested and not physical:
        raise ValueError("Measured scale input is supported only for physical-grain data; legacy diagnostics remain 14-D")
    return physical if requested is None else bool(requested)


def manifest_state_contract(manifest: dict) -> tuple[bool, int]:
    """Validate new explicit contracts while retaining old 14-D checkpoints."""
    version = manifest.get("manifest_version", 1)
    if version not in (1, 2):
        raise ValueError(f"Unsupported data manifest version {version}")
    if version == 1:
        # Old checkpoints did not encode a scale channel. Never infer one from
        # a current environment or silently change an existing trained input.
        include_scale = False
        dimension = PROPRIO_DIM
    else:
        if type(manifest.get("include_measured_scale")) is not bool:
            raise ValueError("Manifest v2 requires explicit include_measured_scale")
        include_scale = manifest["include_measured_scale"]
        dimension = PROPRIO_DIM + int(include_scale)
        expected = PROPRIO_FEATURES + ([SCALE_READING_KEY] if include_scale else [])
        if manifest.get("state_dim") != dimension or manifest.get("state_features") != expected:
            raise ValueError("Manifest state_dim/state_features disagree with the observation contract")
        if include_scale and (manifest.get("legacy") or manifest.get("schema_version") != "scrape_v2"):
            raise ValueError("Legacy diagnostic manifests must retain 14-D state")
    for key, size in ((STATE_KEY, dimension), (ACTION_KEY, ACTION_DIM)):
        for statistic in ("mean", "std", "min", "max"):
            value = np.asarray(manifest["statistics"][key][statistic], dtype=float)
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"Manifest {key}/{statistic} must contain {size} finite values")
            if statistic == "std" and np.any(value <= 0):
                raise ValueError(f"Manifest {key}/std must be positive")
    return include_scale, dimension


def inspect_dataset(path: str | Path, heldout_powders: tuple[int, ...] = (5,)) -> dict:
    """Read metadata and numerical channels, without loading image movies."""
    rows = []
    reference = None
    identical = True
    with h5py.File(path, "r") as handle:
        schema = _text(handle.attrs.get("schema_version", "legacy"))
        timing = _text(handle.attrs.get("observation_timing", "post_action_snapshot"))
        if schema == "scrape_v2":
            if _text(handle.attrs.get("powder_dynamics", "")) != "physical_grains_v1":
                raise ValueError("Corrected training data requires contact-driven physical grains; height-gated surrogate demos are not accepted")
            if timing != "pre_action":
                raise ValueError("scrape_v2 must explicitly record pre_action observations")
            if _text(handle.attrs.get("action_semantics", "")) != "joint_position_radians_gripper_0_255":
                raise ValueError("scrape_v2 action semantics are missing or incompatible")
        for name in sorted(handle["data"], key=lambda k: int(k.rsplit("_", 1)[-1])):
            demo = handle["data"][name]
            actions = np.asarray(demo["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or len(actions) < 2:
                raise ValueError(f"{name}: actions must be [T>=2,14]")
            state = _proprio(demo["obs"])
            if schema == "scrape_v2" and "proprio" not in demo["obs"]:
                raise ValueError(f"{name}: scrape_v2 requires measured obs/proprio")
            if schema == "scrape_v2":
                observed_state(demo["obs"], include_measured_scale=True)
                if (not demo.attrs.get("physical_valid", False)
                        or _text(demo.attrs.get("physics_validation", "")) != "rigid_contact_v1"
                        or _text(demo.attrs.get("grasp_model", "")) != "mujoco_friction_only"
                        or not demo.attrs.get("success", False)):
                    raise ValueError(f"{name}: missing successful rigid-body physical validation")
                expected = len(actions) * int(demo.attrs.get("physics_steps_per_action", 0))
                if expected <= 0 or int(demo.attrs.get("physics_checks", 0)) < expected:
                    raise ValueError(f"{name}: incomplete physics substep validation")
                if "physics_audit" not in demo.attrs or "physics/checks" not in demo:
                    raise ValueError(f"{name}: missing recorded physics audit")
            if len(state) != len(actions):
                raise ValueError(f"{name}: observation/action lengths differ")
            if not np.isfinite(actions).all() or not np.isfinite(state).all():
                raise ValueError(f"{name}: nonfinite action or proprio")
            for camera in CAMERAS:
                if camera not in demo["obs"]:
                    raise ValueError(f"{name}: missing camera {camera}")
                image = demo["obs"][camera]
                if image.ndim != 4 or image.shape[0] != len(actions) or image.shape[-1] != 3 or image.dtype != np.uint8:
                    raise ValueError(f"{name}/{camera}: expected [T,H,W,3] uint8")
            if reference is None:
                reference = actions
            else:
                n = min(len(reference), len(actions))
                identical &= bool(np.array_equal(reference[:n], actions[:n]))
                if len(actions) > len(reference):
                    reference = actions
            if "powder_id" not in demo.attrs:
                raise ValueError(f"{name}: missing powder_id; cannot establish leakage-free split")
            powder = int(demo.attrs["powder_id"])
            dones = np.asarray(demo["subgoal_dones"]) if "subgoal_dones" in demo else None
            changes = np.flatnonzero(np.any(np.abs(np.diff(actions, axis=0)) > 1e-6, axis=1))
            row = {
                "name": name, "powder_id": powder, "samples": len(actions),
                "split": "heldout" if powder in heldout_powders else "train",
                "target_g": float(demo.attrs["target_g"]),
                "instruction": _text(demo.attrs["instruction"]),
                "last_action_change": int(changes[-1] + 1) if len(changes) else None,
                "subgoal_reversals": int((np.diff(dones, axis=0) < 0).sum()) if dones is not None else None,
                "proprio_source": "measured" if "proprio" in demo["obs"] else "legacy_joint_and_gripper_command",
            }
            for key in ("reading_g", "mass_g", "success", "seed"):
                if key in demo.attrs:
                    row[key] = _jsonable(demo.attrs[key])
            rows.append(row)
        attrs = _jsonable(dict(handle.attrs))
    train = [r["name"] for r in rows if r["split"] == "train"]
    heldout = [r["name"] for r in rows if r["split"] == "heldout"]
    return {
        "schema_version": schema, "observation_timing": timing,
        "dataset_attributes": attrs, "episodes": rows, "num_episodes": len(rows),
        "total_samples": sum(r["samples"] for r in rows),
        "train_episodes": train, "heldout_episodes": heldout,
        "heldout_powder_ids": list(heldout_powders),
        "identical_actions_until_termination": bool(identical),
        "use": "synthetic_coarse_grain_control_data" if schema == "scrape_v2" and timing == "pre_action" else "diagnostic_only",
    }


def fit_training_stats(path: str | Path, episodes: list[str], legacy: bool = False,
                       include_measured_scale: bool | None = None) -> dict:
    """Streaming moments from training episodes only, in raw physical units."""
    if not episodes:
        raise ValueError("Training split is empty")
    values = {STATE_KEY: [], ACTION_KEY: []}
    with h5py.File(path, "r") as handle:
        include_measured_scale = _resolve_scale_selection(handle.attrs, legacy, include_measured_scale)
        for name in episodes:
            demo = handle["data"][name]
            state = observed_state(demo["obs"], include_measured_scale)
            actions = np.asarray(demo["actions"], dtype=np.float32)
            # Approximate legacy alignment is explicit and restricted to diagnostics.
            values[STATE_KEY].append(state[:-1] if legacy else state)
            values[ACTION_KEY].append(actions[1:] if legacy else actions)
    stats = {}
    for key, arrays in values.items():
        array = np.concatenate(arrays).astype(np.float64)
        stats[key] = {
            "mean": array.mean(axis=0).tolist(),
            "std": np.maximum(array.std(axis=0), 1e-6).tolist(),
            "min": array.min(axis=0).tolist(), "max": array.max(axis=0).tolist(),
            "count": [len(array)],
        }
    return stats


def prepare_manifest(path: str | Path, output_dir: str | Path, heldout_powders: tuple[int, ...] = (5,),
                     allow_legacy_diagnostic: bool = False, include_measured_scale: bool | None = None) -> dict:
    audit = inspect_dataset(path, heldout_powders)
    legacy = audit["use"] == "diagnostic_only"
    if legacy and not allow_legacy_diagnostic:
        raise ValueError("Legacy scrape demos are diagnostic only; require --allow-legacy-diagnostic, or use validated scrape_v2 pre-action data.")
    if not audit["train_episodes"] or not heldout_powders:
        raise ValueError("Need a nonempty training split and declared held-out powders")
    if not audit["heldout_episodes"]:
        declared = json.loads(audit["dataset_attributes"].get("held_out_powder_ids", "[]"))
        if legacy or set(declared) != set(heldout_powders):
            raise ValueError("Absent held-out demos require an explicit online-evaluation powder split")
    include_measured_scale = _resolve_scale_selection(audit["dataset_attributes"], legacy, include_measured_scale)
    stats = fit_training_stats(path, audit["train_episodes"], legacy=legacy,
                               include_measured_scale=include_measured_scale)
    manifest = {
        "manifest_version": 2, "dataset_sha256": file_sha256(path),
        "schema_version": audit["schema_version"], "use": audit["use"],
        "heldout_powder_ids": audit["heldout_powder_ids"],
        "train_episodes": audit["train_episodes"], "heldout_episodes": audit["heldout_episodes"],
        "camera_mapping": CAMERAS,
        "state": "left6,left_gripper,right6,right_gripper" + (",scale_reading_g" if include_measured_scale else "") + "; target in instruction only",
        "include_measured_scale": include_measured_scale,
        "state_features": PROPRIO_FEATURES + ([SCALE_READING_KEY] if include_measured_scale else []),
        "state_dim": PROPRIO_DIM + int(include_measured_scale),
        "action_alignment": "obs[t] -> actions[t+1] (approximate, diagnostic only)" if legacy else "obs[t] -> actions[t] -> next_obs[t]",
        "legacy": legacy, "statistics": stats,
        "normalization_fitted_on": "train_episodes_only",
        "identical_actions_until_termination": audit["identical_actions_until_termination"],
        "episodes": audit["episodes"],
        "powder_parameter_vectors": audit["dataset_attributes"].get("powder_parameter_vectors"),
        "source_version": audit["dataset_attributes"].get("source_version"),
        "grain_count": audit["dataset_attributes"].get("grain_count"),
    }
    manifest_state_contract(manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "data_manifest.json"
    if target.exists():
        old = json.loads(target.read_text())
        if old != manifest:
            raise ValueError("Existing immutable data_manifest.json differs; use a new run directory")
    else:
        target.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class ScrapeHDFDataset:
    """Lazy per-process HDF reads. Supports torch DataLoader without importing torch."""
    def __init__(self, path: str | Path, episodes: list[str], chunk_size: int, legacy: bool = False,
                 max_samples: int | None = None, include_measured_scale: bool | None = None):
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.path = str(path)
        self.episodes = list(episodes)
        self.chunk_size = chunk_size
        self.legacy = legacy
        self._file = None
        self._pid = None
        self.ends = []
        count = 0
        with h5py.File(self.path, "r") as handle:
            self.include_measured_scale = _resolve_scale_selection(handle.attrs, legacy, include_measured_scale)
            for name in self.episodes:
                n = len(handle["data"][name]["actions"]) - int(legacy)
                count += n
                self.ends.append(count)
        self.size = min(count, max_samples) if max_samples else count
        if not self.size:
            raise ValueError("Empty dataset")

    def __len__(self):
        return self.size

    def _handle(self):
        if self._file is None or self._pid != os.getpid():
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._file

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_pid"] = None
        return state

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = bisect.bisect_right(self.ends, index)
        start = 0 if episode == 0 else self.ends[episode - 1]
        t = index - start
        demo = self._handle()["data"][self.episodes[episode]]
        action_start = t + int(self.legacy)
        action_end = min(action_start + self.chunk_size, len(demo["actions"]))
        actions = np.asarray(demo["actions"][action_start:action_end], dtype=np.float32)
        valid = len(actions)
        if valid < self.chunk_size:
            actions = np.concatenate([actions, np.repeat(actions[-1:], self.chunk_size - valid, axis=0)])
        result = {
            STATE_KEY: observed_state(demo["obs"], self.include_measured_scale, t),
            ACTION_KEY: actions,
            "action_is_pad": np.arange(self.chunk_size) >= valid,
            "task": _text(demo.attrs["instruction"]),
        }
        for source, target in CAMERAS.items():
            result[target] = np.asarray(demo["obs"][source][t]).transpose(2, 0, 1).astype(np.float32) / 255.0
        return result

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def _tensor_stats(stats: dict):
    import torch
    return {key: {name: torch.as_tensor(value, dtype=torch.float32) for name, value in channel.items()} for key, channel in stats.items()}


def load_smolvla(manifest: dict, device: str, pretrained: str = PRETRAINED_MODEL, revision: str | None = None, chunk_size: int | None = None, n_action_steps: int | None = None):
    """Load genuine base VLA weights, retaining pretrained 32-D projections."""
    _, state_dim = manifest_state_contract(manifest)
    import importlib.metadata
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    version = importlib.metadata.version("lerobot")
    if version != LEROBOT_VERSION:
        raise RuntimeError(f"Expected tested lerobot=={LEROBOT_VERSION}, found {version}")
    if pretrained == PRETRAINED_MODEL and revision is None:
        revision = PRETRAINED_REVISION
    config = SmolVLAConfig.from_pretrained(pretrained, revision=revision)
    config.device = device
    if config.max_state_dim != 32 or config.max_action_dim != 32:
        raise ValueError("Expected pretrained 32-D padded state and action projections")
    config.input_features = {STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)), **{name: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)) for name in CAMERAS.values()}}
    config.output_features = {ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    if chunk_size is not None:
        config.chunk_size = chunk_size
    if n_action_steps is not None:
        config.n_action_steps = n_action_steps
    config.adapt_to_pi_aloha = False
    config.freeze_vision_encoder = True
    config.train_expert_only = True
    # Avoid initializing a second pretrained trunk; complete policy weights follow.
    config.load_vlm_weights = False
    policy = SmolVLAPolicy.from_pretrained(pretrained, config=config, revision=revision, strict=True)
    policy.to(device=device, dtype=torch.float32)
    preprocessor, postprocessor = make_smolvla_pre_post_processors(config, _tensor_stats(manifest["statistics"]))
    return policy, preprocessor, postprocessor


def make_inference_batch(observation: dict, instruction: str, include_measured_scale: bool = False) -> dict:
    """Cameras, measured proprio, optional instrument reading, and language."""
    import torch
    state = observed_state(observation, include_measured_scale)
    if state.ndim != 1:
        raise ValueError("Inference requires one unbatched observation")
    batch = {STATE_KEY: torch.from_numpy(state), "task": instruction}
    for source, target in CAMERAS.items():
        batch[target] = torch.from_numpy(np.asarray(observation[source]).transpose(2, 0, 1).copy()).float() / 255.0
    return batch


class VLABCPolicy:
    """Checkpoint adapter usable by a Gymnasium evaluation loop."""
    def __init__(self, checkpoint: str | Path, device: str = "cuda", bf16: bool = True):
        checkpoint = Path(checkpoint)
        self.manifest = json.loads((checkpoint / "data_manifest.json").read_text())
        self.include_measured_scale, _ = manifest_state_contract(self.manifest)
        self.policy, self.preprocessor, self.postprocessor = load_smolvla(self.manifest, device, pretrained=str(checkpoint))
        self.policy.eval()
        self.bf16 = bool(bf16 and device.startswith("cuda"))

    def reset(self):
        self.policy.reset()

    def act(self, observation: dict, instruction: str) -> np.ndarray:
        import torch
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
            batch = self.preprocessor(make_inference_batch(observation, instruction, self.include_measured_scale))
            action = self.postprocessor(self.policy.select_action(batch))
        return action[0].detach().cpu().numpy()


def verify_checkpoint(checkpoint: str | Path, dataset_path: str | Path, device: str = "cuda") -> dict:
    """Check a full save/reload cycle using identical observations and flow noise.

    This is a serialization/inference test, not an environment success test.
    The model cache must retain the base VLM config and tokenizer assets.
    """
    import tempfile
    import torch
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / "data_manifest.json").read_text())
    if file_sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("Verification dataset differs from the checkpoint manifest")
    policy, pre, post = load_smolvla(manifest, device, str(checkpoint))
    policy.eval()
    include_measured_scale, _ = manifest_state_contract(manifest)
    dataset = ScrapeHDFDataset(dataset_path, manifest["train_episodes"][:1], policy.config.chunk_size,
                              legacy=manifest["legacy"], include_measured_scale=include_measured_scale)
    item = dataset[0]
    dataset.close()

    def batch():
        return {key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
                for key, value in item.items() if key not in {ACTION_KEY, "action_is_pad"}}

    generator = torch.Generator(device=device).manual_seed(1729)
    noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim), generator=generator, device=device)
    with torch.inference_mode():
        expected = post(policy.predict_action_chunk(pre(batch()), noise=noise.clone())).detach().cpu()
    with tempfile.TemporaryDirectory(prefix="chemlab-vla-roundtrip-") as temporary:
        policy.save_pretrained(temporary)
        reloaded, pre2, post2 = load_smolvla(manifest, device, temporary)
        reloaded.eval()
        with torch.inference_mode():
            actual = post2(reloaded.predict_action_chunk(pre2(batch()), noise=noise.clone())).detach().cpu()
    if tuple(actual.shape) != (1, policy.config.chunk_size, ACTION_DIM) or not torch.isfinite(actual).all():
        raise AssertionError(f"Invalid inference output: {tuple(actual.shape)}")
    error = float((expected - actual).abs().max())
    if not torch.allclose(expected, actual, atol=1e-4, rtol=1e-5):
        raise AssertionError(f"Checkpoint roundtrip changed actions (max absolute difference {error})")
    result = {"checkpoint": str(checkpoint), "roundtrip_passed": True, "max_action_difference": error,
              "action_shape": list(actual.shape), "closed_loop_evaluated": False}
    (checkpoint / "roundtrip_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _save_checkpoint(policy, preprocessor, postprocessor, optimizer, manifest, output: Path, step: int, elapsed: float, run_config: dict):
    import torch
    target = output / f"checkpoint_{step:06d}"
    target.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(target)
    preprocessor.save_pretrained(target, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(target, config_filename="policy_postprocessor.json")
    (target / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (target / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    torch.save({"optimizer": optimizer.state_dict(), "step": step, "elapsed_seconds": elapsed, "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(), "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, target / "training_state.pt")
    (output / "latest_checkpoint.txt").write_text(str(target.resolve()) + "\n")
    return target


def train_main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-checkpoint", type=Path)
    parser.add_argument("--heldout-powders", type=int, nargs="+", default=[5])
    parser.add_argument("--allow-legacy-diagnostic", action="store_true")
    parser.add_argument("--no-scale-reading", dest="include_measured_scale", action="store_false", default=None,
                        help="Ablation: omit numeric measured scale input; retain cameras, proprio and language")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--pretrained", default=PRETRAINED_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--save-initial", action="store_true")
    parser.add_argument("--overfit-samples", type=int, default=0)
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_checkpoint:
        print(json.dumps(verify_checkpoint(args.verify_checkpoint, args.dataset, args.device)), flush=True)
        return
    if args.output is None:
        parser.error("--output is required for training")
    if args.steps <= 0 or args.max_seconds <= 0 or args.batch_size <= 0 or args.save_every <= 0:
        parser.error("steps, max-seconds, batch-size, and save-every must be positive")
    if not 1 <= args.n_action_steps <= args.chunk_size:
        parser.error("n-action-steps must be between 1 and chunk-size")
    started = time.monotonic()
    # The remote wrapper supplies one absolute deadline, preserved across worker
    # restarts. Leave room for the full policy + optimizer checkpoint to flush.
    deadline_raw = os.environ.get("CHEMLAB_DEADLINE_UTC")
    deadline = float(deadline_raw) if deadline_raw else None
    if deadline is not None and deadline - time.time() <= 120:
        raise TimeoutError("Insufficient time before the experiment deadline to load/save a model")
    manifest = prepare_manifest(args.dataset, args.output, tuple(sorted(set(args.heldout_powders))),
                                args.allow_legacy_diagnostic, args.include_measured_scale)
    print(json.dumps({"event": "data_ready", "use": manifest["use"], "state_dim": manifest["state_dim"],
                      "include_measured_scale": manifest["include_measured_scale"],
                      "train_episodes": len(manifest["train_episodes"]), "heldout_episodes": len(manifest["heldout_episodes"])}), flush=True)
    if args.prepare_only:
        return
    if (args.output / "run_config.json").exists() and not args.resume:
        raise ValueError("This run directory already contains training; use --resume or a new --output")
    import torch
    from torch.utils.data import DataLoader
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.bf16 and (not args.device.startswith("cuda") or not torch.cuda.is_bf16_supported()):
        raise ValueError("--bf16 requires a CUDA device with bf16 support")
    pretrained = str(args.resume) if args.resume else args.pretrained
    if args.resume:
        prior_manifest = json.loads((args.resume / "data_manifest.json").read_text())
        if prior_manifest != manifest:
            raise ValueError("Resume checkpoint used a different dataset/split/normalization")
    policy, preprocessor, postprocessor = load_smolvla(manifest, args.device, pretrained, args.revision if not args.resume else None, args.chunk_size, args.n_action_steps)
    policy.train()
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    dataset = ScrapeHDFDataset(args.dataset, manifest["train_episodes"], args.chunk_size, legacy=manifest["legacy"],
                              max_samples=args.overfit_samples or None,
                              include_measured_scale=manifest["include_measured_scale"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=args.device.startswith("cuda"), drop_last=False, persistent_workers=args.workers > 0)
    step = 0
    if args.resume:
        # Only load our own trusted checkpoint, never arbitrary downloaded pickle.
        state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    run_config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    run_config.update(include_measured_scale=manifest["include_measured_scale"], state_dim=manifest["state_dim"])
    run_config["pretrained_revision"] = args.revision or (PRETRAINED_REVISION if args.pretrained == PRETRAINED_MODEL else None)
    run_config.update({"lerobot_version": LEROBOT_VERSION, "parameters": sum(p.numel() for p in policy.parameters()), "trainable_parameters": sum(p.numel() for p in trainable), "result_scope": manifest["use"], "loss": "supervised_flow_matching", "heldout_evaluated": False})
    (args.output / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    print(json.dumps({"event": "model_ready", **run_config}), flush=True)
    if args.save_initial and not args.resume:
        path = _save_checkpoint(policy, preprocessor, postprocessor, optimizer, manifest, args.output, 0, time.monotonic() - started, run_config)
        print(json.dumps({"event": "initial_checkpoint", "path": str(path)}), flush=True)
    iterator = iter(loader)
    loss_history = []
    stop_reason = "steps"
    try:
        while step < args.steps:
            if deadline is not None and time.time() >= deadline - 120:
                stop_reason = "absolute_deadline"
                break
            if time.monotonic() - started >= args.max_seconds:
                stop_reason = "time_limit"
                break
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = preprocessor(batch)
            optimizer.zero_grad(set_to_none=True)
            tick = time.monotonic()
            with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=args.bf16):
                loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(trainable, max_norm=10.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            row = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad), "seconds": time.monotonic() - started, "step_seconds": time.monotonic() - tick}
            loss_history.append(row["loss"])
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if step == 1 or step % 20 == 0:
                print(json.dumps(row), flush=True)
            if step % args.save_every == 0:
                path = _save_checkpoint(policy, preprocessor, postprocessor, optimizer, manifest, args.output, step, row["seconds"], run_config)
                print(json.dumps({"event": "checkpoint", "path": str(path)}), flush=True)
    finally:
        dataset.close()
    checkpoint = _save_checkpoint(policy, preprocessor, postprocessor, optimizer, manifest, args.output, step, time.monotonic() - started, run_config)
    result = {"steps": step, "stop_reason": stop_reason, "elapsed_seconds": time.monotonic() - started, "checkpoint": str(checkpoint), "result_scope": manifest["use"], "first_loss": loss_history[0] if loss_history else None, "last_loss": loss_history[-1] if loss_history else None, "closed_loop_evaluated": False, "heldout_evaluated": False}
    (args.output / "training_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    train_main()
