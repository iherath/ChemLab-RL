"""Experimental online SmolVLA OGPO–VaDar integration, with no training claim.

Uses the exact augmented-chain primitives in ``smolvla_ogpo`` and their pinned
paper/reference conventions. Replay actions train critics; actor updates draw
FRESH target-policy chains at replay states. All 50x32 latent coordinates enter
the likelihood, while critics/environment use the physical two-step prefix.

Deliberate engineering choices, not a reproduction of reference hyperparameters:
frozen contextual SmolVLM prefix mean-pooling plus proprio for critic features;
independent MLP vector critics; physical action clipping followed by bound-based
critic scaling; optional fixed-BC augmented-chain forward KL (OFF by default).
No latent or prefix KV cache is shared between actors. EMA targets update after
their corresponding complete optimization group. A legacy 14-D real checkpoint
passed forward/backward smoke; physical 15-D optimizer/robot evaluation remains
outstanding. Neither that diagnostic nor CPU tests establishes task improvement.

``run_online`` is opt-in and bounded. Importing this module starts no jobs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from chemlab_rl.smolvla_ogpo import (
    ChainConfig, ChainSample, chain_log_prob, clipped_chain_objective,
    conservative_group_advantages, full_chain_ratio, next_future_weights,
    sample_chain, vector_critic_loss, vector_td_target,
)
from chemlab_rl.vla_bc import CAMERAS, STATE_KEY


LANGUAGE_KEYS = ("observation.language.tokens", "observation.language.attention_mask")
POLICY_KEYS = (STATE_KEY, *CAMERAS.values(), *LANGUAGE_KEYS)
DEMONSTRATION_IMAGE_WINDOW = 32  # Even: h=2 chunks never cross an I/O window.


def actor_observation(observation: Mapping, instruction: str, state_dim: int = 15) -> dict:
    """Allowlist deployable inputs, excluding lowdim, material IDs and task labels.

    With the default ``state_dim=15``, append the environment's measured
    ``scale_reading_g`` to 14 robot coordinates through the same BC
    preprocessing contract. The true receiver mass is never a policy input.
    """
    state = np.asarray(observation["proprio"], dtype=np.float32)
    if state_dim not in (14, 15) or state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError("Actor proprio must be the declared finite sensor vector")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A task instruction is required")
    result = {"proprio": state.copy(), "task": instruction}
    if state_dim == 15:
        scale = np.asarray(observation["scale_reading_g"], dtype=np.float32)
        if scale.shape != (1,) or not np.isfinite(scale).all():
            raise ValueError("Measured scale channel must be finite shape [1]")
        result["scale_reading_g"] = scale.copy()
    for key in CAMERAS:
        value = np.asarray(observation[key])
        if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
            raise ValueError("Actor cameras must be uint8 HWC RGB arrays")
        result[key] = value.copy()
    return result


def prepare_policy_batch(observations: Sequence[Mapping], preprocessor: Callable,
                         device: str | torch.device) -> dict[str, Tensor]:
    """Use the BC tokenizer/normalizer, then strictly filter its model inputs."""
    if not observations:
        raise ValueError("Need at least one observation")
    states = [np.concatenate([o["proprio"], o["scale_reading_g"]]) if "scale_reading_g" in o else o["proprio"]
              for o in observations]
    batch = {STATE_KEY: torch.from_numpy(np.stack(states)),
             "task": [o["task"] for o in observations]}
    for raw, key in CAMERAS.items():
        batch[key] = torch.from_numpy(np.stack([o[raw] for o in observations])).permute(0, 3, 1, 2).float() / 255
    processed = preprocessor(batch)
    result = {key: processed[key].to(device) for key in POLICY_KEYS}
    if any(value.shape[0] != len(observations) for value in result.values()):
        raise ValueError("BC preprocessor changed the observation batch dimension")
    return result


@dataclass(frozen=True)
class Transition:
    observation: Mapping
    next_observation: Mapping  # Final observation BEFORE any reset, including time limits.
    actions: np.ndarray       # Physical commands [action_steps,14]; unused suffix ignored.
    dones: np.ndarray         # Latched flags before the chunk, never actor inputs.
    next_dones: np.ndarray    # Latched flags after the actually executed chunk.
    terminated: bool
    truncated: bool
    executed_steps: int
    powder_id: int            # Split audit only, never actor/critic features.


class PersistentReplay:
    """Bounded SQLite replay; arrays use NPZ without pickle, writes are atomic.

    Entries survive process exit. Sampling RNG is included in trainer checkpoints.
    A fixed metadata contract prevents reusing a DB with different action/state
    dimensions or an incompatible BC/preprocessing/material provenance string.
    """
    def __init__(self, path: str | Path, *, capacity=100_000, action_steps=2,
                 action_dim=14, state_dim=15, subgoals=3, heldout_ids=(5,),
                 provenance: str, seed=0):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1
               for x in (capacity, action_steps, action_dim, state_dim, subgoals)):
            raise ValueError("Replay sizes must be positive integers")
        if not provenance:
            raise ValueError("Replay requires explicit checkpoint/data provenance")
        self.contract = dict(version=1, capacity=capacity, action_steps=action_steps,
                             action_dim=action_dim, state_dim=state_dim, subgoals=subgoals,
                             heldout_ids=list(heldout_ids), provenance=provenance)
        self.rng = np.random.default_rng(seed)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS transitions (id INTEGER PRIMARY KEY AUTOINCREMENT, payload BLOB NOT NULL)")
        existing = self.db.execute("SELECT value FROM metadata WHERE key='contract'").fetchone()
        if existing and json.loads(existing[0]) != self.contract:
            self.db.close()
            raise ValueError("Existing replay contract/provenance differs")
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('contract', ?)", (json.dumps(self.contract),))

    def __len__(self):
        return int(self.db.execute("SELECT count(*) FROM transitions").fetchone()[0])

    def add(self, transition: Transition) -> int:
        payload = self._encode_transition(transition)
        with self.db:
            cursor = self.db.execute("INSERT INTO transitions(payload) VALUES (?)", (payload,))
            self.db.execute("DELETE FROM transitions WHERE id IN (SELECT id FROM transitions ORDER BY id DESC LIMIT -1 OFFSET ?)",
                            (self.contract["capacity"],))
        return int(cursor.lastrowid)

    def _encode_transition(self, transition: Transition) -> bytes:
        """Validate/encode without committing, allowing atomic streaming imports."""
        c, t = self.contract, transition
        if (isinstance(t.powder_id, bool) or not isinstance(t.powder_id, (int, np.integer))
                or not 0 <= int(t.powder_id) <= 5 or int(t.powder_id) in c["heldout_ids"]):
            raise ValueError("Held-out or invalid material cannot enter training replay")
        if (not isinstance(t.terminated, (bool, np.bool_)) or not isinstance(t.truncated, (bool, np.bool_))
                or isinstance(t.executed_steps, bool) or not isinstance(t.executed_steps, (int, np.integer))
                or not 1 <= t.executed_steps <= c["action_steps"]):
            raise ValueError("Invalid terminal flags or actual chunk horizon")
        if t.executed_steps < c["action_steps"] and not (t.terminated or t.truncated):
            raise ValueError("A shortened chunk must record its ending")
        arrays = {}
        for label, observation in (("obs", t.observation), ("next", t.next_observation)):
            clean = actor_observation(observation, observation["task"], c["state_dim"])
            arrays.update({f"{label}/{key}": value for key, value in clean.items()})
        actions = np.asarray(t.actions, dtype=np.float32)
        if actions.shape != (c["action_steps"], c["action_dim"]) or not np.isfinite(actions).all():
            raise ValueError("Replay action chunk has wrong shape or nonfinite commands")
        before, after = np.asarray(t.dones, np.float32), np.asarray(t.next_dones, np.float32)
        if before.shape != (c["subgoals"],) or after.shape != before.shape:
            raise ValueError("Replay completion vector has wrong shape")
        next_future_weights(torch.from_numpy(np.stack([before, after])))
        if np.any(after < before):
            raise ValueError("Subgoal completion must remain latched")
        arrays.update(actions=actions, dones=before, next_dones=after,
                      terminated=np.bool_(t.terminated), truncated=np.bool_(t.truncated),
                      executed_steps=np.int64(t.executed_steps), powder_id=np.int64(t.powder_id))
        payload = io.BytesIO()
        np.savez_compressed(payload, **arrays)
        return payload.getvalue()

    @staticmethod
    def _decode(payload: bytes) -> Transition:
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            def observation(prefix):
                keys = ("proprio", "task", *CAMERAS)
                if f"{prefix}/scale_reading_g" in arrays.files:
                    keys = (*keys, "scale_reading_g")
                return {key: (str(arrays[f"{prefix}/{key}"].item()) if key == "task"
                              else arrays[f"{prefix}/{key}"].copy()) for key in keys}
            return Transition(observation("obs"), observation("next"), arrays["actions"].copy(),
                              arrays["dones"].copy(), arrays["next_dones"].copy(),
                              bool(arrays["terminated"]), bool(arrays["truncated"]),
                              int(arrays["executed_steps"]), int(arrays["powder_id"]))

    def sample(self, count: int) -> list[Transition]:
        ids = np.array([row[0] for row in self.db.execute("SELECT id FROM transitions ORDER BY id")], dtype=np.int64)
        if count < 1 or count > len(ids):
            raise ValueError("Replay must contain at least the requested batch size")
        chosen = self.rng.choice(ids, size=count, replace=False)
        return [self._decode(self.db.execute("SELECT payload FROM transitions WHERE id=?", (int(i),)).fetchone()[0])
                for i in chosen]

    def checkpoint_state(self) -> dict:
        revision = self.db.execute("SELECT seq FROM sqlite_sequence WHERE name='transitions'").fetchone()
        return {"contract": self.contract, "revision": int(revision[0]) if revision else 0,
                "rng": copy.deepcopy(self.rng.bit_generator.state)}

    def restore_sampling(self, state: dict):
        current = self.checkpoint_state()
        if current["contract"] != state["contract"] or current["revision"] != state["revision"]:
            raise ValueError("Checkpoint replay revision/contract differs; exact resume is not possible")
        self.rng.bit_generator.state = state["rng"]

    def close(self):
        self.db.close()


def import_demonstrations(replay: PersistentReplay, path: str | Path, *, bc_manifest: Mapping,
                          expected_source_version: str, expected_source_hashes: Mapping[str, str],
                          grain_count: int, action_low, action_high, max_episode_steps=2600,
                          control_hz=10., mujoco_version: str, on_progress: Callable | None = None) -> dict:
    """Opt-in, atomic physical-demo replay seeding; never import behavior chains.

    Only the exact accepted five/30-demo dataset used by this 15-D BC checkpoint
    is eligible. Successful episodes must pass the collection/merge audit again
    and match current physical sources, materials and simulator version. Images
    use fixed 32-control read windows, checking every adjacent pair including
    window boundaries; numerical audits load at most one episode, explicitly
    bounded by ``max_episode_steps``. A late validation
    or serialization failure rolls back every inserted row and the audit marker.

    Two nonoverlapping controls form each transition. An odd terminal tail has
    actual horizon one and a masked zero suffix. Before-head flags come from the
    preceding control (000 at the full-task reset); next flags/observation are
    from the last executed control. Recorded HDF rewards are NOT used: the
    existing vector TD target reconstructs rewards from these labels/horizons.
    This is demonstration-seeded off-policy learning, not the no-demo baseline
    or a claim that TD warmup has calibrated a critic. Warmup is unchanged.
    """
    import h5py
    from chemlab_rl.granular_material import GRANULAR_MATERIALS
    from chemlab_rl.vla_bc import _text, file_sha256, inspect_dataset, manifest_state_contract
    from scripts.merge_granular_demos import _json_attribute, _metadata, _validate_episode

    started = time.monotonic()
    phase_seconds = dict(initial_sha256=0., metadata_and_numeric_validation=0.,
                         image_continuity_validation=0., image_read_encode_sql=0., final_sha256=0.)
    path = Path(path).resolve(strict=True)
    c = replay.contract
    if (c["action_steps"], c["action_dim"], c["state_dim"], c["subgoals"], c["heldout_ids"]) != (2, 14, 15, 3, [5]):
        raise ValueError("Demonstration import requires the physical h=2, 15-D, three-head, held-out-5 contract")
    if (manifest_state_contract(bc_manifest) != (True, 15) or bc_manifest.get("legacy")
            or bc_manifest.get("use") != "synthetic_coarse_grain_control_data"
            or bc_manifest.get("normalization_fitted_on") != "train_episodes_only"
            or bc_manifest.get("heldout_powder_ids") != [5] or bc_manifest.get("heldout_episodes")):
        raise ValueError("Demonstration import requires a physical training-only BC manifest")
    if (isinstance(max_episode_steps, bool) or not isinstance(max_episode_steps, int) or max_episode_steps < 2
            or isinstance(grain_count, bool) or not isinstance(grain_count, int) or not 1 <= grain_count <= 400
            or control_hz != 10. or not isinstance(mujoco_version, str) or not mujoco_version):
        raise ValueError("Require a bounded full-task horizon, valid grain count and 10-Hz simulator contract")
    low, high = np.asarray(action_low, np.float32), np.asarray(action_high, np.float32)
    if (low.shape != (14,) or high.shape != (14,) or not np.isfinite([low, high]).all()
            or np.any(high <= low)):
        raise ValueError("Need finite ordered 14-D robot action bounds")
    stamp = time.monotonic()
    digest = file_sha256(path)
    phase_seconds["initial_sha256"] = time.monotonic() - stamp
    if digest != bc_manifest.get("dataset_sha256"):
        raise ValueError("Demonstration dataset SHA256 differs from the BC manifest")
    request = dict(dataset_sha256=digest, bc_manifest_sha256=hashlib.sha256(
        json.dumps(dict(bc_manifest), sort_keys=True).encode()).hexdigest(),
        source_version=expected_source_version, source_files_sha256=dict(expected_source_hashes),
        grain_count=grain_count, max_episode_steps=max_episode_steps, control_hz=control_hz,
        mujoco_version=mujoco_version, action_low=low.tolist(), action_high=high.tolist())
    prior = replay.db.execute("SELECT value FROM metadata WHERE key='demonstration_import'").fetchone()
    if prior:
        report = json.loads(prior[0])
        if report["request"] != request:
            raise ValueError("Existing demonstration import has different dataset/source/BC provenance")
        return dict(report, already_imported=True)
    if len(replay):
        raise ValueError("First demonstration import requires an empty replay; refusing to mix prior history")

    stamp = time.monotonic()
    with h5py.File(path, "r") as handle:
        metadata = _metadata(handle)
        if (metadata["source_version"] != expected_source_version
                or metadata["source_files_sha256"] != dict(expected_source_hashes)
                or metadata["grain_count"] != grain_count or metadata["control_hz"] != control_hz
                or metadata["mujoco_version"] != mujoco_version
                or metadata["powder_parameter_vectors"] != [m.as_dict() for m in GRANULAR_MATERIALS]
                or bc_manifest.get("source_version") != expected_source_version
                or bc_manifest.get("grain_count") != grain_count
                or json.loads(bc_manifest.get("powder_parameter_vectors", "null")) != metadata["powder_parameter_vectors"]):
            raise ValueError("Demonstrations differ from current physical source/material/grain-count/simulator contract")
        count = handle.attrs.get("expected_episode_count", 0)
        if (count not in (5, 30) or handle.attrs.get("merged_episode_count") != count
                or _text(handle.attrs.get("merge_validation", "")) != "physical_grain_episode_audit_v1"
                or len(handle["data"]) != count):
            raise ValueError("Require an accepted five- or 30-episode physical merge")
        names = [f"demo_{i}" for i in range(count)]
        if set(handle["data"]) != set(names) or bc_manifest.get("train_episodes") != names:
            raise ValueError("Merged episodes differ from the BC training split")
        lengths = [len(handle["data"][name]["actions"]) for name in names]
        if any(not 2 <= length <= max_episode_steps for length in lengths):
            raise ValueError("Demonstration length exceeds the declared bounded full-task horizon")
        transitions = sum((length + 1) // 2 for length in lengths)
        if transitions > c["capacity"]:
            raise ValueError("Replay capacity cannot retain the entire demonstration seed")
        merged = _json_attribute(handle, "merge_provenance")
        if not isinstance(merged, list) or len(merged) != count:
            raise ValueError("Incomplete merge path/trajectory provenance")
        # This reads no image movies; numeric buffers are episode-bounded above.
        audit = inspect_dataset(path, (5,))
        if (audit["use"] != "synthetic_coarse_grain_control_data" or audit["heldout_episodes"]
                or audit["train_episodes"] != names
                or {row["powder_id"] for row in audit["episodes"]} != set(range(5))):
            raise ValueError("Require all training materials 0..4 and no held-out demonstrations")
        report = dict(mode="demonstration_seeded_off_policy", request=request, dataset_path=str(path),
                      physical_metadata=metadata, episodes=[], transition_count=transitions,
                      control_count=sum(lengths), chunk_stride=2, initial_subgoals=[0, 0, 0],
                      rewards="reconstructed_by_existing_vector_td_target", warmup_changed=False,
                      image_read_window_controls=DEMONSTRATION_IMAGE_WINDOW, max_image_window_bytes=0,
                      encoded_payload_bytes=0, phase_seconds=phase_seconds)
        phase_seconds["metadata_and_numeric_validation"] += time.monotonic() - stamp
        identities, trajectories = set(), set()
        with replay.db:
            # Explicitly acquire the transaction before validation/insertion so
            # no public add() can commit a partial dataset on this code path.
            replay.db.execute("BEGIN IMMEDIATE")
            if len(replay):
                raise ValueError("Replay changed before demonstration import")
            for name, length, origin in zip(names, lengths, merged, strict=True):
                stamp = time.monotonic()
                demo = handle["data"][name]
                identity, fingerprint = _validate_episode(demo, metadata)
                if identity in identities or fingerprint in trajectories:
                    raise ValueError("Duplicate demonstration identity or physical trajectory")
                identities.add(identity)
                trajectories.add(fingerprint)
                if (not isinstance(origin, dict) or origin.get("trajectory_sha256") != fingerprint
                        or origin.get("powder_id") != identity[0]
                        or any(not isinstance(origin.get(k), str) or not origin[k] for k in ("path", "episode"))
                        or not isinstance(origin.get("file_sha256"), str)
                        or len(origin["file_sha256"]) != 64
                        or any(ch not in "0123456789abcdef" for ch in origin["file_sha256"])):
                    raise ValueError(f"{name}: merge provenance does not match the validated trajectory")
                labels = np.asarray(demo["subgoal_dones"], np.float32)
                next_future_weights(torch.from_numpy(labels))  # Also enforces within-row ordered prefix.
                elapsed = np.asarray(demo["physics/elapsed_s"], np.float64)
                if not np.allclose(np.diff(np.r_[0., elapsed]), 1 / control_hz, rtol=0, atol=1e-6):
                    raise ValueError(f"{name}: controls do not have the declared discount time base")
                phase_seconds["metadata_and_numeric_validation"] += time.monotonic() - stamp
                # Retain Dataset handles (and their chunk caches). Fixed windows
                # avoid repeatedly decompressing temporally tiled gzip chunks.
                images = {prefix: {camera: demo[f"{prefix}/{camera}"] for camera in CAMERAS}
                          for prefix in ("obs", "next_obs")}
                stamp = time.monotonic()
                for camera in CAMERAS:
                    for start in range(0, length - 1, DEMONSTRATION_IMAGE_WINDOW):
                        stop = min(start + DEMONSTRATION_IMAGE_WINDOW, length - 1)
                        preceding = images["next_obs"][camera][start:stop]
                        following = images["obs"][camera][start+1:stop+1]
                        for local in range(stop-start):
                            if not np.array_equal(preceding[local], following[local]):
                                raise ValueError(f"{name}: discontinuous camera transition {camera}/{start+local}")
                # Do not retain validation windows while encoding six RGB streams.
                del preceding, following
                phase_seconds["image_continuity_validation"] += time.monotonic() - stamp
                instruction = _text(demo.attrs["instruction"])
                first_id = last_id = None
                stamp = time.monotonic()
                for block_start in range(0, length, DEMONSTRATION_IMAGE_WINDOW):
                    block_stop = min(block_start + DEMONSTRATION_IMAGE_WINDOW, length)
                    windows = {prefix: {camera: dataset[block_start:block_stop] for camera, dataset in values.items()}
                               for prefix, values in images.items()}
                    report["max_image_window_bytes"] = max(report["max_image_window_bytes"],
                        sum(value.nbytes for values in windows.values() for value in values.values()))
                    for start in range(block_start, block_stop, 2):
                        stop = min(start + 2, length)
                        commands = np.asarray(demo["actions"][start:stop], np.float32)
                        if not np.isfinite(commands).all() or np.any(commands < low) or np.any(commands > high):
                            raise ValueError(f"{name}: demonstration actions exceed physical robot bounds")
                        padded = np.zeros((2, 14), np.float32)
                        padded[:stop-start] = commands
                        def observed(prefix, index):
                            return {key: demo[f"{prefix}/{key}"][index] for key in ("proprio", "scale_reading_g")} | {
                                camera: window[index-block_start] for camera, window in windows[prefix].items()
                            } | {"task": instruction}
                        transition = Transition(observed("obs", start), observed("next_obs", stop-1), padded,
                                                labels[start-1] if start else np.zeros(3, np.float32), labels[stop-1],
                                                bool(demo["dones"][stop-1]), bool(demo["timeouts"][stop-1]),
                                                stop-start, identity[0])
                        payload = replay._encode_transition(transition)
                        report["encoded_payload_bytes"] += len(payload)
                        last_id = int(replay.db.execute("INSERT INTO transitions(payload) VALUES (?)", (payload,)).lastrowid)
                        if first_id is None:
                            first_id = last_id
                    del transition, windows
                phase_seconds["image_read_encode_sql"] += time.monotonic() - stamp
                report["episodes"].append(dict(name=name, powder_id=identity[0], seed=identity[1], target_g=identity[2],
                                               samples=length, trajectory_sha256=fingerprint, merge_origin=origin,
                                               replay_first_id=first_id, replay_last_id=last_id))
                if on_progress is not None:
                    on_progress(dict(event="demonstration_episode_encoded_pending_commit", episode=name,
                                     episodes=len(report["episodes"]), total_episodes=count,
                                     elapsed_seconds=time.monotonic()-started, phase_seconds=dict(phase_seconds)))
            stamp = time.monotonic()
            if file_sha256(path) != digest:
                raise ValueError("Demonstration file changed during replay import")
            phase_seconds["final_sha256"] = time.monotonic()-stamp
            report["elapsed_before_commit_seconds"] = time.monotonic()-started
            replay.db.execute("INSERT INTO metadata VALUES ('demonstration_import', ?)", (json.dumps(report, sort_keys=True),))
            commit_started = time.monotonic()
        return dict(report, already_imported=False, commit_seconds=time.monotonic()-commit_started,
                    elapsed_seconds=time.monotonic()-started)


class PhysicalActionAdapter:
    """Same BC unnormalizer for rollout and critic candidates, then robot limits.

    Clipping is a deterministic mapping of the FULL stochastic chain. It does
    not alter the scored chain or justify discarding its auxiliary coordinates.
    Critic commands are scaled by known robot limits, not by hidden material.
    """
    def __init__(self, postprocessor: Callable, low, high):
        self.postprocessor = postprocessor
        self.low, self.high = torch.as_tensor(low).float(), torch.as_tensor(high).float()
        if (self.low.ndim != 1 or self.low.shape != self.high.shape
                or not torch.isfinite(self.low).all() or not torch.isfinite(self.high).all()
                or not (self.high > self.low).all()):
            raise ValueError("Need finite ordered robot action limits")

    def physical(self, normalized_actions: Tensor) -> Tensor:
        actions = self.postprocessor(normalized_actions)
        if not isinstance(actions, Tensor) or actions.shape != normalized_actions.shape or not torch.isfinite(actions).all():
            raise ValueError("BC action postprocessor must preserve finite [B,h,action_dim] shape")
        # LeRobot's deployment postprocessor ends in to_cpu. Candidate Q values
        # are evaluated on the actor/critic device; rollout converts back to NumPy
        # explicitly at the environment boundary.
        actions = actions.to(device=normalized_actions.device, dtype=normalized_actions.dtype)
        return torch.clamp(actions, self.low.to(actions), self.high.to(actions))

    def critic(self, physical_actions: Tensor) -> Tensor:
        low, high = self.low.to(physical_actions), self.high.to(physical_actions)
        if not torch.isfinite(physical_actions).all() or ((physical_actions < low - 1e-5) | (physical_actions > high + 1e-5)).any():
            raise ValueError("Replay commands lie outside the declared robot action limits")
        return 2 * (physical_actions - low) / (high - low) - 1


class FrozenSmolVLAFeatures:
    """Reuse frozen SmolVLM, with a private frozen BC state-projection snapshot.

    LeRobot's ``train_expert_only`` freezes the VLM, while ``train_state_proj``
    independently controls the actor's small state projector. Preserve that
    actor setting: copy only the projector for stationary critic features.
    Prefix prefill uses a fresh temporary KV cache required by SmolVLA's cross-
    attention path; the cache is discarded, never persisted or shared with actors.
    Feature dimension is contextual hidden width plus declared sensor dimension.
    This pooled MLP critic architecture is a port choice, not the reference CNN.
    """
    def __init__(self, policy, state_dim=15, *, mask_builder=None):
        self.policy, self.state_dim, self.mask_builder = policy, state_dim, mask_builder
        if any(p.requires_grad for p in policy.model.vlm_with_expert.vlm.parameters()):
            raise ValueError("Critic prefix features require a frozen VLM")
        # A private module registry permits replacing this one submodule without
        # replacing anything in the actor's registry. All large frozen VLM
        # weights are shared, and no actor requires_grad flag is changed.
        self.prefix_model = copy.copy(policy.model)
        self.prefix_model._modules = policy.model._modules.copy()
        self.prefix_model._parameters = policy.model._parameters.copy()
        self.prefix_model._buffers = policy.model._buffers.copy()
        self.prefix_model.state_proj = copy.deepcopy(policy.model.state_proj).requires_grad_(False).eval()
        self.feature_dim = policy.model.vlm_with_expert.config.text_config.hidden_size + state_dim

    def state_dict(self):
        return {"state_projection": self.prefix_model.state_proj.state_dict()}

    def load_state_dict(self, state):
        if set(state) != {"state_projection"}:
            raise ValueError("Frozen critic feature checkpoint has incompatible state")
        self.prefix_model.state_proj.load_state_dict(state["state_projection"])

    @torch.no_grad()
    def __call__(self, batch):
        policy = self.policy
        policy.eval()
        model = self.prefix_model
        images, image_masks = policy.prepare_images(batch)
        prefix, padding, attention = model.embed_prefix(
            images, image_masks, batch[LANGUAGE_KEYS[0]], batch[LANGUAGE_KEYS[1]],
            state=policy.prepare_state(batch))
        if self.mask_builder is None:
            from lerobot.policies.common.vla_utils import make_att_2d_masks
            mask_builder = make_att_2d_masks
        else:
            mask_builder = self.mask_builder
        (context, _), _ = model.vlm_with_expert.forward(
            attention_mask=mask_builder(padding, attention),
            position_ids=padding.long().cumsum(dim=1) - 1,
            past_key_values=None, inputs_embeds=[prefix, None], use_cache=True)
        weights = padding.float().unsqueeze(-1)
        pooled = (context.float() * weights).sum(1) / weights.sum(1).clamp_min(1)
        return torch.cat([pooled, batch[STATE_KEY][:, :self.state_dim].float()], dim=-1).detach()


class VectorQEnsemble(nn.Module):
    def __init__(self, feature_dim: int, *, action_steps=2, action_dim=14, subgoals=3, members=2, hidden=256):
        super().__init__()
        self.feature_dim, self.action_steps, self.action_dim = feature_dim, action_steps, action_dim
        self.subgoals, self.members = subgoals, members
        width = feature_dim + action_steps * action_dim + action_steps
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.SiLU(),
                                                nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, subgoals))
                                    for _ in range(members)])

    def forward(self, features: Tensor, actions: Tensor, valid: Tensor) -> Tensor:
        if (features.ndim != 2 or features.shape[1] != self.feature_dim
                or actions.shape != (len(features), self.action_steps, self.action_dim)
                or valid.shape != actions.shape[:2]):
            raise ValueError("Critic feature/action/mask dimensions differ")
        inputs = torch.cat([features, (actions * valid[..., None]).flatten(1), valid], dim=-1)
        return torch.stack([head(inputs) for head in self.heads], dim=1)  # [B,M,G]


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, rate: float):
    """target <- (1-rate)*target + rate*online, including floating buffers."""
    if not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("EMA rate must lie in (0,1]")
    source, dest = online.state_dict(), target.state_dict()
    if source.keys() != dest.keys():
        raise ValueError("EMA module structures differ")
    for name, value in dest.items():
        if value.shape != source[name].shape:
            raise ValueError("EMA tensor shapes differ")
        if value.is_floating_point() or value.is_complex():
            value.lerp_(source[name], rate)
        else:
            value.copy_(source[name])
    target.requires_grad_(False).eval()


@dataclass(frozen=True)
class OGPOConfig:
    chunk_size: int = 50
    latent_dim: int = 32
    action_steps: int = 2
    action_dim: int = 14
    state_dim: int = 15
    subgoals: int = 3
    batch_size: int = 8
    group_size: int = 4
    actor_microbatch_size: int = 2
    discount: float = .99
    r_done: float = 5.0
    r_step: float = -1.0  # Pinned reference convention: once per executed chunk.
    value_cap: float = 15.0
    step_reward_per_control: float | None = None  # Explicit task-timescale deviation.
    clip_epsilon: float = .01
    actor_lr: float = 1e-6
    critic_lr: float = 3e-4
    actor_ema: float = .005
    critic_ema: float = .005
    grad_norm: float = 1.0
    critic_warmup_updates: int = 1000
    bc_anchor_weight: float = 0.0  # Optional fixed-BC path KL, not the default paper algorithm.
    chain: ChainConfig = field(default_factory=ChainConfig)

    def __post_init__(self):
        integers = (self.chunk_size, self.latent_dim, self.action_steps, self.action_dim,
                    self.state_dim, self.subgoals, self.batch_size, self.group_size, self.actor_microbatch_size)
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in integers):
            raise ValueError("Training sizes must be positive integers")
        if self.action_steps > self.chunk_size or self.action_dim > self.latent_dim or self.group_size < 2:
            raise ValueError("Invalid latent/action prefix or candidate group dimensions")
        if not isinstance(self.critic_warmup_updates, int) or self.critic_warmup_updates < 0:
            raise ValueError("Critic warmup must be nonnegative")
        numeric = (self.discount, self.clip_epsilon, self.actor_lr, self.critic_lr,
                   self.actor_ema, self.critic_ema, self.grad_norm, self.bc_anchor_weight)
        if not all(math.isfinite(x) for x in numeric) or min(numeric) < 0:
            raise ValueError("Training parameters must be finite and nonnegative")
        if not (0 <= self.discount <= 1 and 0 < self.clip_epsilon < 1
                and 0 < self.actor_ema <= 1 and 0 < self.critic_ema <= 1
                and min(self.actor_lr, self.critic_lr, self.grad_norm) > 0):
            raise ValueError("Invalid discount, optimizer, PPO or EMA settings")
        if (not all(math.isfinite(x) for x in (self.r_done, self.r_step, self.value_cap))
                or self.r_step > 0 or self.value_cap < self.r_done):
            raise ValueError("Task rewards need finite values, nonpositive step cost and cap >= done reward")
        if self.step_reward_per_control is not None and (not math.isfinite(self.step_reward_per_control)
                                                        or self.step_reward_per_control > 0):
            raise ValueError("Per-control step reward must be finite and nonpositive")


class OGPOTrainer:
    """One replay minibatch -> vector TD update -> optional fresh-chain actor group."""
    def __init__(self, actor: nn.Module, critic: VectorQEnsemble, *, preprocessor: Callable,
                 feature_fn: Callable, action_adapter: PhysicalActionAdapter,
                 config: OGPOConfig = OGPOConfig(), device="cpu", seed=0,
                 provenance: str):
        self.config, self.device = config, torch.device(device)
        self.actor, self.critic = actor.to(self.device), critic.to(self.device)
        if not provenance:
            raise ValueError("Record the BC checkpoint/preprocessing provenance")
        if (critic.action_steps != config.action_steps or critic.action_dim != config.action_dim
                or critic.subgoals != config.subgoals or action_adapter.low.numel() != config.action_dim):
            raise ValueError("Critic/action adapter/config contracts differ")
        self.provenance = provenance
        self.preprocessor, self.feature_fn, self.action_adapter = preprocessor, feature_fn, action_adapter
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False).eval()
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False).eval()
        self.bc_anchor = copy.deepcopy(self.actor).requires_grad_(False).eval() if config.bc_anchor_weight else None
        self.actor.eval()  # Recorded Gaussian chain noise is the only actor stochasticity.
        self.actor_opt = torch.optim.Adam([p for p in self.actor.parameters() if p.requires_grad], lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.critic_updates = self.actor_updates = self.environment_steps = 0

    def _inputs(self, observations):
        return prepare_policy_batch(observations, self.preprocessor, self.device)

    @torch.no_grad()
    def _features(self, batch):
        result = self.feature_fn(batch).detach().float()
        if result.shape != (len(batch[STATE_KEY]), self.critic.feature_dim) or not torch.isfinite(result).all():
            raise ValueError("Frozen critic encoder returned invalid features")
        return result

    def _sample(self, batch, actor=None) -> ChainSample:
        c = self.config
        draws = []
        for start in range(0, len(batch[STATE_KEY]), c.actor_microbatch_size):
            selected = {key: value[start:start+c.actor_microbatch_size] for key, value in batch.items()}
            draws.append(sample_chain(actor or self.target_actor, selected,
                                      (len(selected[STATE_KEY]), c.chunk_size, c.latent_dim),
                                      c.chain, device=self.device, generator=self.generator))
        return ChainSample(torch.cat([draw.latents for draw in draws]), torch.cat([draw.log_prob for draw in draws]))

    @torch.no_grad()
    def sample_actions(self, observations) -> Tensor:
        sample = self._sample(self._inputs(observations))
        return self.action_adapter.physical(sample.executed_actions(self.config.action_dim, self.config.action_steps))

    def _step_optimizer(self, optimizer, module, loss):
        optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite optimization loss; no optimizer step performed")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), self.config.grad_norm, error_if_nonfinite=True)
        optimizer.step()
        return float(norm)

    def update(self, replay: PersistentReplay) -> dict:
        c = self.config
        expected = dict(action_steps=c.action_steps, action_dim=c.action_dim, state_dim=c.state_dim,
                        subgoals=c.subgoals, provenance=self.provenance)
        if any(replay.contract[k] != value for k, value in expected.items()):
            raise ValueError("Replay and trainer contracts differ")
        rows = replay.sample(c.batch_size)
        batch, next_batch = self._inputs([r.observation for r in rows]), self._inputs([r.next_observation for r in rows])
        features, next_features = self._features(batch), self._features(next_batch)
        dones = torch.as_tensor(np.stack([r.dones for r in rows]), device=self.device)
        next_dones = torch.as_tensor(np.stack([r.next_dones for r in rows]), device=self.device)
        terminal = torch.tensor([r.terminated for r in rows], device=self.device)
        horizons = torch.tensor([r.executed_steps for r in rows], device=self.device)
        mask = (torch.arange(c.action_steps, device=self.device)[None] < horizons[:, None]).float()
        actions = torch.as_tensor(np.stack([r.actions for r in rows]), device=self.device)
        # Padded commands are ignored by Q, but canonical zero padding can be
        # outside a robot's positive-only range; replace only ignored positions.
        actions = torch.where(mask[..., None].bool(), actions, self.action_adapter.low.to(actions))
        actions = self.action_adapter.critic(actions)
        with torch.no_grad():
            next_sample = self._sample(next_batch)
            next_actions = self.action_adapter.critic(self.action_adapter.physical(next_sample.executed_actions(c.action_dim, c.action_steps)))
            next_q = self.target_critic(next_features, next_actions, torch.ones_like(mask))
            if c.step_reward_per_control is None:
                target = vector_td_target(next_dones, next_q, terminal, c.discount, horizons,
                                          r_done=c.r_done, r_step=c.r_step, cap=c.value_cap)
            else:
                # Explicit alternative: accumulate the control-step cost over the
                # actual chunk, discounting each cost. Completion-head rewards
                # and terminal cap retain the same reference convention.
                costs = [c.step_reward_per_control * sum(c.discount**i for i in range(r.executed_steps)) for r in rows]
                target = torch.cat([vector_td_target(next_dones[i:i+1], next_q[i:i+1], terminal[i:i+1],
                                                     c.discount, r.executed_steps, r_done=c.r_done,
                                                     r_step=costs[i], cap=c.value_cap)
                                    for i, r in enumerate(rows)])
        predicted = self.critic(features, actions, mask)
        critic_loss = vector_critic_loss(predicted, target)
        critic_grad = self._step_optimizer(self.critic_opt, self.critic, critic_loss)
        self.critic_updates += 1
        ema_update(self.target_critic, self.critic, c.critic_ema)
        metrics = dict(critic_loss=float(critic_loss.detach()), critic_grad_norm=critic_grad,
                       critic_updates=self.critic_updates, actor_updates=self.actor_updates,
                       terminated_fraction=float(terminal.float().mean()),
                       truncated_fraction=float(np.mean([r.truncated for r in rows])),
                       actual_horizon_mean=float(horizons.float().mean()), td_target_mean=float(target.mean()))
        if self.critic_updates <= c.critic_warmup_updates:
            return metrics
        repeated = {key: value.repeat_interleave(c.group_size, dim=0) for key, value in batch.items()}
        with torch.no_grad():
            sample = self._sample(repeated)
            candidate_actions = self.action_adapter.critic(self.action_adapter.physical(sample.executed_actions(c.action_dim, c.action_steps)))
            candidate_q = self.target_critic(features.repeat_interleave(c.group_size, dim=0), candidate_actions,
                                             torch.ones(c.batch_size*c.group_size, c.action_steps, device=self.device))
            advantages = conservative_group_advantages(candidate_q.reshape(c.batch_size, c.group_size, -1, c.subgoals), dones)
        # Keep the group baseline/advantages joint, but accumulate the exact
        # sample-mean objective without retaining every candidate's VLA graphs.
        # Sampling uses these same batch sizes to preserve numerical kernel layout.
        self.actor_opt.zero_grad(set_to_none=True)
        current_logs = []
        policy_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        count = len(sample.latents)
        for start in range(0, count, c.actor_microbatch_size):
            stop = min(start+c.actor_microbatch_size, count)
            selected = {key: value[start:stop] for key, value in repeated.items()}
            current_log = chain_log_prob(self.actor, selected, sample.latents[start:stop], c.chain)
            loss = clipped_chain_objective(current_log, sample.log_prob[start:stop],
                                           advantages.flatten()[start:stop], c.clip_epsilon) * (stop-start)/count
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite actor microbatch loss; no actor optimizer step performed")
            loss.backward()
            policy_loss += loss.detach()
            current_logs.append(current_log.detach())
        current_log = torch.cat(current_logs)
        ratio = full_chain_ratio(current_log, sample.log_prob)
        anchor_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        if self.bc_anchor is not None:
            anchor_sample = self._sample(batch, self.bc_anchor)
            for start in range(0, c.batch_size, c.actor_microbatch_size):
                stop = min(start+c.actor_microbatch_size, c.batch_size)
                selected = {key: value[start:stop] for key, value in batch.items()}
                score = chain_log_prob(self.actor, selected, anchor_sample.latents[start:stop], c.chain)
                loss = (anchor_sample.log_prob[start:stop]-score).sum()/c.batch_size
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite anchor microbatch loss; no actor optimizer step performed")
                (c.bc_anchor_weight * loss).backward()
                anchor_loss += loss.detach()
        actor_loss = policy_loss + c.bc_anchor_weight * anchor_loss
        actor_grad = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), c.grad_norm, error_if_nonfinite=True))
        self.actor_opt.step()
        self.actor_updates += 1
        ema_update(self.target_actor, self.actor, c.actor_ema)
        metrics.update(actor_updates=self.actor_updates, actor_loss=float(actor_loss.detach()),
                       actor_grad_norm=actor_grad, bc_anchor_loss=float(anchor_loss.detach()),
                       ratio_min=float(ratio.detach().min()), ratio_max=float(ratio.detach().max()),
                       ratio_clip_fraction=float(((ratio-1).abs() > c.clip_epsilon).float().mean()),
                       nonzero_advantage_fraction=float((advantages != 0).float().mean()),
                       ratio_ess=float((ratio.detach().sum().square() / ratio.detach().square().sum().clamp_min(1e-300))),
                       max_abs_log_ratio=float((current_log.detach()-sample.log_prob).abs().max()),
                       chain_shape=list(sample.latents.shape))
        return metrics

    def save_checkpoint(self, path: str | Path, replay: PersistentReplay):
        """Atomically save learned/target/anchor states, Adam and sampling RNG."""
        state = dict(version=1, config=asdict(self.config), provenance=self.provenance,
                     actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                     target_actor=self.target_actor.state_dict(), target_critic=self.target_critic.state_dict(),
                     bc_anchor=self.bc_anchor.state_dict() if self.bc_anchor is not None else None,
                     feature_encoder=self.feature_fn.state_dict() if hasattr(self.feature_fn, "state_dict") else None,
                     actor_optimizer=self.actor_opt.state_dict(), critic_optimizer=self.critic_opt.state_dict(),
                     generator=self.generator.get_state(), replay=replay.checkpoint_state(),
                     action_low=self.action_adapter.low, action_high=self.action_adapter.high,
                     critic_updates=self.critic_updates, actor_updates=self.actor_updates,
                     environment_steps=self.environment_steps,
                     torch_rng=torch.get_rng_state())
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def load_checkpoint(self, path: str | Path, replay: PersistentReplay):
        state = torch.load(path, map_location=self.device, weights_only=True)
        if state["version"] != 1 or state["config"] != asdict(self.config) or state["provenance"] != self.provenance:
            raise ValueError("Checkpoint configuration/provenance differs")
        if not torch.equal(state["action_low"].cpu(), self.action_adapter.low.cpu()) or not torch.equal(state["action_high"].cpu(), self.action_adapter.high.cpu()):
            raise ValueError("Checkpoint robot action limits differ")
        replay.restore_sampling(state["replay"])
        encoder_state = state.get("feature_encoder")
        if encoder_state is not None:
            if not hasattr(self.feature_fn, "load_state_dict"):
                raise ValueError("Checkpoint requires its frozen critic feature encoder")
            self.feature_fn.load_state_dict(encoder_state)
        elif hasattr(self.feature_fn, "state_dict"):
            raise ValueError("Checkpoint is missing the frozen critic feature state")
        for name in ("actor", "critic", "target_actor", "target_critic"):
            getattr(self, name).load_state_dict(state[name])
        if self.bc_anchor is not None:
            self.bc_anchor.load_state_dict(state["bc_anchor"])
        self.actor_opt.load_state_dict(state["actor_optimizer"])
        self.critic_opt.load_state_dict(state["critic_optimizer"])
        self.generator.set_state(state["generator"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        self.critic_updates, self.actor_updates = state["critic_updates"], state["actor_updates"]
        self.environment_steps = state["environment_steps"]


def collect_chunk(env, observation, info, trainer: OGPOTrainer, replay: PersistentReplay, *, max_steps=None):
    """Execute only robot commands; store the real pre-reset terminal observation.

    The collector never calls reset or writes a particle state. Caller owns
    explicit episode resets. Exogenous collection cuts are recorded as time-limit
    truncations so they bootstrap, unlike collision or task-completion terminals.
    """
    c = trainer.config
    if info.get("powder_dynamics") != "physical_grains_v1":
        raise ValueError("Online collection requires the physical granular environment")
    powder_id = int(info["powder_id"])
    if powder_id in replay.contract["heldout_ids"] or not 0 <= powder_id <= 5:
        raise ValueError("Held-out material cannot enter online training")
    if max_steps is not None and (not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError("Collection allowance must be positive")
    before = actor_observation(observation, env.instruction, c.state_dim)
    planned = trainer.sample_actions([before])[0].detach().cpu().numpy()
    actions = np.zeros_like(planned)
    limit = min(c.action_steps, max_steps) if max_steps is not None else c.action_steps
    terminated = truncated = False
    reward_sum = 0.0
    for index in range(limit):
        actions[index] = planned[index]
        observation, reward, terminated, truncated, final_info = env.step(actions[index])
        reward_sum += float(reward)
        if terminated or truncated:
            # Support explicit Gym final-observation metadata, never reset frames.
            observation = final_info.get("final_observation", observation)
            final_info = final_info.get("final_info", final_info)
            break
    executed = index + 1
    truncated = bool(truncated or (executed < c.action_steps and not terminated))
    after = actor_observation(observation, env.instruction, c.state_dim)
    transition = Transition(before, after, actions, np.asarray(info["subgoal_dones"]),
                            np.asarray(final_info["subgoal_dones"]), bool(terminated), truncated,
                            executed, powder_id)
    transition_id = replay.add(transition)
    trainer.environment_steps += executed
    return observation, final_info, dict(transition_id=transition_id, terminated=bool(terminated),
                                        truncated=truncated, executed_steps=executed, reward_sum=reward_sum)


def run_online(env, trainer: OGPOTrainer, replay: PersistentReplay, *, max_environment_steps: int,
               max_updates: int, max_seconds: float, checkpoint_path: str | Path,
               seed=0, checkpoint_every=100, on_update: Callable | None = None) -> dict:
    """Bounded serial rollout/update runner. No Modal or GPU allocation side effects.

    Time is checked between chunks/updates, so a single in-flight model call or
    simulator chunk can exceed the wall limit. Remote job limits remain external.
    Resume starts a fresh physical episode; simulator trajectory is not serialized.
    """
    if (any(not isinstance(n, int) or n < 1 for n in (max_environment_steps, max_updates, checkpoint_every))
            or not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("Runner budgets must be positive and finite")
    started = time.monotonic()
    collected = updates = episodes = 0
    observation, info = env.reset(seed=seed)
    while collected < max_environment_steps and updates < max_updates and time.monotonic()-started < max_seconds:
        observation, info, rollout = collect_chunk(env, observation, info, trainer, replay,
                                                   max_steps=max_environment_steps-collected)
        collected += rollout["executed_steps"]
        if len(replay) >= trainer.config.batch_size and time.monotonic()-started < max_seconds:
            metrics = trainer.update(replay)
            updates += 1
            if on_update is not None:
                on_update(metrics)
            if updates % checkpoint_every == 0:
                trainer.save_checkpoint(checkpoint_path, replay)
        if rollout["terminated"] or rollout["truncated"]:
            episodes += 1
            if collected < max_environment_steps and updates < max_updates and time.monotonic()-started < max_seconds:
                observation, info = env.reset(seed=seed+episodes)
    trainer.save_checkpoint(checkpoint_path, replay)
    return dict(environment_steps=collected, updates=updates, completed_episodes=episodes,
                elapsed_seconds=time.monotonic()-started, checkpoint=str(checkpoint_path),
                closed_loop_evaluated=False)
