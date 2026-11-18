"""Additional image-based vial demos with per-substep contact auditing.

Each candidate starts from the native fixed scene, varies a continuous expert
recipe, and is saved only after native success and force/contact checks. This
does not modify the independent granular weighing collector or environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import numpy as np

from scripts.collect_grok_demos import VialWeighingEnv, Session, Recorder, _slide_door, _enter, run_all, top_down
from chemlab_rl.vial_vla_env import source_identity
import mujoco


def recipe_for(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return dict(name=f"continuous_top_{seed}", door="top", door_arm="right", pick_arm="left",
                yaw_deg=None if seed % 4 == 0 else float(rng.uniform(0, 60)),
                grasp_dz=float(rng.uniform(-.0005, .0005)),
                grasp_xy=rng.uniform(-.0005, .0005, size=2).tolist())


def weigh_recipe(session, recipe):
    """Carry via contact; derive placement height from the measured grasp."""
    s, status = session, {"recipe": recipe["name"]}
    if not _slide_door(s, status, "right", "top", .20, "open"):
        return status
    yaw = recipe["yaw_deg"]
    for phase in ("above vial", "grasp vial"):
        s.snap("left", "vial")
        pos, rotation = s.targets["left"]
        if yaw is not None:
            rotation = top_down(np.deg2rad(yaw))
        if phase == "grasp vial":
            pos = pos + np.array([*recipe["grasp_xy"], recipe["grasp_dz"]])
        s.set_target("left", pos, rotation)
        if not _enter(s, status, phase):
            return status
    s.grip("left", True)
    run_all(s)
    vial_id = s.model.body("vial_a").id
    original_speed = s.speed
    for label, clearance in (("above pan", .12), ("near pan", .015), ("place", .002)):
        holds = [h for h in s._carried() if h[0] == "left" and h[1] == vial_id]
        if not holds:
            raise ContactViolation("Vial is not held by physical gripper contact")
        _, _, _, relative_position, _ = holds[0]
        rotation = s.arms["left"].tool_pose()[1]
        desired_base = s.data.site("balance_pan_site").xpos + [0., 0., clearance]
        s.set_target("left", desired_base - rotation @ relative_position, rotation)
        s.speed = min(original_speed, .3) if label != "above pan" else original_speed
        if not _enter(s, status, label):
            return status
    s.speed = original_speed
    s.grip("left", False)
    run_all(s)
    s.snap("left", "home")
    if not _enter(s, status, "pick home"):
        return status
    if not _slide_door(s, status, "right", "top", -.20, "close"):
        return status
    status["ok"] = True
    return status


class ContactViolation(RuntimeError):
    pass


class SuccessfulTerminal(Exception):
    """Stop after the recorder has appended the first successful transition."""


class AuditedVialEnv(VialWeighingEnv):
    """Native control transition with checks after every unchanged 2 ms step."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.session = None
        self.checks = 0
        self.max_penetration_m = 0.
        self.max_unintended_penetration_m = 0.
        self.max_abs_scale_g = 0.
        self.excessive_load_seconds = 0.
        self.max_excessive_load_seconds = 0.
        self.contact_extrema = {}

    def _audit(self):
        m, d, s = self.model, self.data, self.session
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise ContactViolation("Nonfinite native dynamics")
        self.checks += 1
        reading = abs(self.balance_g())
        self.max_abs_scale_g = max(self.max_abs_scale_g, reading)
        # A brief impact can exceed static weight. Reject sustained robot load,
        # while preserving the unfiltered peak and duration in the audit.
        self.excessive_load_seconds = (self.excessive_load_seconds + m.opt.timestep) if reading > 100. else 0.
        self.max_excessive_load_seconds = max(self.max_excessive_load_seconds, self.excessive_load_seconds)
        if self.excessive_load_seconds > .1:
            raise ContactViolation(f"Balance load above 100 g for {self.excessive_load_seconds:.6f} s")
        for c in d.contact[:d.ncon]:
            penetration = max(0., -float(c.dist))
            if not penetration:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            name = "|".join(sorted([m.geom(g1).name or str(g1), m.geom(g2).name or str(g2)]))
            self.contact_extrema[name] = max(self.contact_extrema.get(name, 0.), penetration)
            self.max_penetration_m = max(self.max_penetration_m, penetration)
            owners = (s.geom_owner[g1], s.geom_owner[g2])
            unintended = ((owners[0] in ("left", "right") and owners[1] == "static")
                          or (owners[1] in ("left", "right") and owners[0] == "static")
                          or set(owners) == {"left", "right"})
            for robot_geom, other_geom in ((g1, g2), (g2, g1)):
                side = s.geom_owner[robot_geom]
                if side in ("left", "right") and s.geom_owner[other_geom] == "movable":
                    # Only the physical finger pads are intended contact surfaces.
                    # Camera mounts and gripper shells must not hit the shield.
                    if "_pad" not in (m.geom(robot_geom).name or ""):
                        unintended = True
            if unintended:
                self.max_unintended_penetration_m = max(self.max_unintended_penetration_m, penetration)
            limit = .00005 if unintended else .001
            if penetration > limit:
                raise ContactViolation(f"Contact {name}: {penetration:.9f} m exceeds {limit:.9f} m")

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.model.nu,) or not np.isfinite(action).all():
            raise ValueError("Invalid native actuator command")
        # Session supplies absolute targets including its gravity compensation.
        if self.normalized_actions:
            raise ValueError("Collection Session requires absolute actuator commands")
        self.data.ctrl[:] = np.clip(action, self._ctrl_low, self._ctrl_high)
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
            self._audit()
        self._step_count += 1
        info = self.task_status()
        if self._step_count % 200 == 0:
            print(json.dumps(dict(event="collection_progress", controls=self._step_count,
                                  balance_g=info["balance_g"], physics_checks=self.checks)), flush=True)
        return self._get_obs(), float(info["success"]), bool(info["success"]), self._step_count >= self.max_episode_steps, info


def collect_one(output: Path, seed: int, *, max_seconds=540, baseline=False) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError("Candidate artifacts already exist; choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    recipe = recipe_for(seed)
    if baseline:
        recipe.update(yaw_deg=None, grasp_dz=0., grasp_xy=[0., 0.])
    kwargs = dict(image_size=64, control_hz=20., hand="robotiq", layout="weigh",
                  clutter=False, max_episode_steps=2000, randomize=False, normalized_actions=False)
    env = AuditedVialEnv(**kwargs)
    rec = Recorder(output, {"env_name": "ChemVialWeighing-v0", "type": "chem_rl_sim",
                           "env_kwargs": {**kwargs, "normalized_actions": True},
                           "collector": "audited_continuous_vial_v1"}, 20., "robotiq")
    speed = 1. if baseline else float(np.random.default_rng(seed + 999999).uniform(.9, 1.))
    session = Session(env, rec, speed=speed)
    env.session = session
    original_tick = session.tick
    def terminal_tick():
        result = original_tick()
        if session.recording and session.info.get("success"):
            raise SuccessfulTerminal()
        return result
    session.tick = terminal_tick
    original_step = env.step

    def bounded_step(action):
        if time.monotonic() - started > max_seconds or env._step_count >= 2000:
            raise TimeoutError("Vial candidate exhausted its time/control budget")
        return original_step(action)

    env.step = bounded_step
    result = dict(seed=seed, recipe=recipe, speed=speed, accepted=False,
                  task_id="vial_placement_native_v1", fixed_scene=True,
                  source_version=source_identity(), collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  image_size=64, control_hz=20., physics_timestep_s=float(env.model.opt.timestep),
                  limits=dict(unintended_rigid_penetration_m=.00005, other_contact_penetration_m=.001,
                              sustained_scale_load_g=100., sustained_scale_duration_s=.1))
    try:
        session.begin_episode(seed=seed)
        try:
            status = weigh_recipe(session, recipe)
        except SuccessfulTerminal:
            status = {"ok": True, "stopped_at_first_native_success": True}
        if not status.get("ok") or not session.info.get("success"):
            raise RuntimeError(f"Recipe did not complete native task: {status}")
        # Keep precisely the first terminal transition, no post-terminal tail.
        first = int(np.flatnonzero(np.asarray(rec.rewards) > 0)[0]) + 1
        rec.truncate(first)
        if first < 2:
            raise ValueError("Invalid demonstration length")
        n = session.save_episode()
        rec.wait()
        if rec.last_error:
            raise RuntimeError(rec.last_error)
        result.update(accepted=True, samples=n, first_terminal_control=first,
                      final_reading_g=float(session.info["balance_g"]), episode=rec.last_saved)
        with h5py.File(output, "a") as f:
            g = f["data"][rec.last_saved]
            g.attrs["physics_validation"] = "native_vial_substep_contacts_v1"
            g.attrs["physical_valid"] = True
            g.attrs["seed"] = seed
            g.attrs["recipe"] = json.dumps(recipe)
            g.attrs["source_version"] = result["source_version"]
            g.attrs["collector_sha256"] = result["collector_sha256"]
            g.attrs["physics_checks"] = env.checks
            result["trajectory_sha256"] = hashlib.sha256(g["actions"][:].tobytes() + g["obs/joint_pos"][:].tobytes()).hexdigest()
    except Exception as error:
        session.discard_episode()
        result["accepted"] = False
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        rec.wait()
        result.update(elapsed_seconds=time.monotonic() - started, physics_checks=env.checks,
                      max_penetration_m=env.max_penetration_m,
                      max_unintended_penetration_m=env.max_unintended_penetration_m,
                      max_abs_scale_g=env.max_abs_scale_g,
                      max_excessive_load_seconds=env.max_excessive_load_seconds,
                      contact_extrema=env.contact_extrema, controls=env._step_count)
        env.close()
        output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "contact_extrema"}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-seconds", type=float, default=540)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    if not collect_one(args.output, args.seed, max_seconds=args.max_seconds, baseline=args.baseline)["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
