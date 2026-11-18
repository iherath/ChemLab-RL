"""Image/sensor adapter for the separate native vial-placement experiment.

This task has a 20 g rigid vial and no granular powder. Native MuJoCo dynamics,
actuators, support-force measurement, and success geometry are retained. The
historical collector did not certify contacts against the granular task gates.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np

TASK_ID = "vial_placement_native_v1"
INSTRUCTION = "Open the top draft shield, place the vial on the balance pan, and close the draft shield."
CAMERAS = ("top1", "left_wrist", "right_wrist")
SUBGOALS = ("open_door", "grasp_vial", "place_vial", "close_shield")


def source_identity() -> str:
    """Fingerprint native simulation code/assets and this task adapter."""
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve()]
    for relative in ("chem_rl_sim/chem_rl_sim", "chem_rl_sim/third_party/urgantry_sim"):
        files.extend(p for p in (root / relative).rglob("*")
                     if p.is_file() and not any(x in {".git", "__pycache__", ".venv"} for x in p.parts)
                     and p.suffix.lower() in {".py", ".xml", ".json", ".stl", ".obj", ".png", ".jpg"})
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + digest.hexdigest()


def sequential_subgoals(previous, joint_pos, obj, *, pan_position,
                        native_success=None) -> np.ndarray:
    """Training labels, excluded from actor input; enforce native pan geometry.

    The second stage is a lift/gripper proxy inherited from the collector's
    available sensor channels, not a contact-force grasp certificate.
    """
    previous = np.asarray(previous, dtype=np.float32)
    q = np.asarray(joint_pos)
    obj = np.asarray(obj)
    pan = np.asarray(pan_position)
    if previous.shape != (4,) or q.shape != (14,) or obj.shape != (11,) or pan.shape != (3,):
        raise ValueError("Unexpected vial subgoal input shape")
    if not all(np.isfinite(x).all() for x in (previous, q, obj, pan)):
        raise ValueError("Nonfinite vial subgoal input")
    upright = 1 - 2 * (obj[4] ** 2 + obj[5] ** 2) > .95
    placed = bool(np.linalg.norm(obj[:2] - pan[:2]) < .035
                  and abs(obj[2] - pan[2]) < .005 and upright
                  and abs(obj[7] - 20.) < 1.)
    success = bool(placed and np.all(obj[8:11] < .005))
    if native_success is not None and success != bool(native_success):
        raise ValueError("Recorded/live native success disagrees with task geometry")
    instantaneous = (np.max(obj[8:11]) > .05,
                     (q[6] > .4 or q[13] > .4) and obj[2] > .815,
                     placed, success)
    result = previous.copy()
    for i, reached in enumerate(instantaneous):
        result[i] = max(result[i], float(reached and (i == 0 or result[i - 1] == 1)))
    return result


class VialVLAEnv:
    """Gym-style wrapper exposing three RGB views and measured state at 20 Hz."""

    instruction = INSTRUCTION
    task_id = TASK_ID

    def __init__(self, *, image_size=64, max_steps=1600):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        root = Path(__file__).resolve().parents[1]
        for path in (root / "chem_rl_sim", root / "chem_rl_sim/third_party/urgantry_sim"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from chem_rl_sim.env import VialWeighingEnv
        self.native = VialWeighingEnv(
            image_size=image_size, control_hz=20., max_episode_steps=max_steps,
            normalized_actions=True, prompt=self.instruction, hand="robotiq",
            layout="weigh", clutter=False, randomize=False, state_mode="proprio")
        self.action_space = self.native.action_space
        self.max_steps = max_steps
        self.subgoal_dones = np.zeros(4, dtype=np.float32)
        self.control_steps = 0
        self.max_penetration_m = 0.

    @property
    def pan_position(self):
        return self.native.data.site("balance_pan_site").xpos.copy()

    def _observation(self, native_observation):
        env = self.native
        q = np.asarray(native_observation["state"], dtype=np.float32).copy()
        vial = env.data.body("vial_a")
        obj = np.concatenate([vial.xpos, vial.xquat, [env.balance_g()], env.doors()]).astype(np.float32)
        if q.shape != (14,) or not np.isfinite(q).all() or not np.isfinite(obj).all():
            raise FloatingPointError("Nonfinite/invalid native vial sensors")
        obs = {"joint_pos": q, "object": obj, "proprio": q.copy(),
               "scale_reading_g": obj[7:8].copy()}
        frames = {"top1": native_observation["image"],
                  **env.render_cameras(["left_wrist", "right_wrist"])}
        for key, frame in frames.items():
            obs[key] = np.asarray(frame, dtype=np.uint8).copy()
            obs[key + "_image"] = obs[key]
        obs["robot0_eye_in_hand_image"] = obs["left_wrist"]
        obs["robot1_eye_in_hand_image"] = obs["right_wrist"]
        return obs

    def _info(self, native_info, observation):
        self.subgoal_dones = sequential_subgoals(
            self.subgoal_dones, observation["joint_pos"], observation["object"],
            pan_position=self.pan_position, native_success=native_info["success"])
        # Post-control diagnostic only: no claim of every-substep certification.
        data = self.native.data
        penetration = max([0.] + [-float(data.contact[i].dist) for i in range(data.ncon)])
        self.max_penetration_m = max(self.max_penetration_m, penetration)
        return {**native_info, "task_id": TASK_ID, "subgoal_dones": self.subgoal_dones.copy(),
                "fixed_scene": True, "control_steps": self.control_steps, "control_hz": 20.,
                "max_postcontrol_penetration_m": self.max_penetration_m,
                "contact_certification": "not_recorded_every_substep",
                "instruction": self.instruction}

    def reset(self, *, seed=None, options=None):
        native_obs, info = self.native.reset(seed=seed, options=options)
        self.subgoal_dones.fill(0)
        self.control_steps = 0
        self.max_penetration_m = 0.
        obs = self._observation(native_obs)
        return obs, self._info(info, obs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("Expected 14 finite normalized actuator targets")
        if np.any(np.abs(action) > 1. + 1e-6):
            raise ValueError("Vial normalized actuator targets must be in [-1,1]")
        native_obs, _, terminated, truncated, info = self.native.step(action)
        self.control_steps += 1
        obs = self._observation(native_obs)
        # All methods report the same native final-success objective. VaDar's
        # separate auxiliary stage targets are defined in its trainer.
        return obs, float(info["success"]), bool(terminated), bool(truncated), self._info(info, obs)

    def close(self):
        self.native.close()
