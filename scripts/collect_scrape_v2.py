"""Collect pre-action image demonstrations through ScrapeWeighEnv.step only.

The geometric expert uses joint planning and authoritative per-physics-step
contact validation. It cannot pin objects, attach them, or modify powder mass.
It opens the doors by their handles and gently sets the source on the housing
before regrasping. Real friction carries the source and tool; no recorded state
is restored. Legacy action replay is retained only as a failing diagnostic.
Failed attempts are logged and never written as successful demonstrations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chemlab_rl.scrape_env import ACTION_SEMANTICS, CAMERAS, SCHEMA_VERSION, ScrapeWeighEnv


class EpisodeFinished(Exception):
    pass


class RecorderBridge:
    """Adapt Session's controller API without bypassing env.step()."""

    def __init__(self, env: ScrapeWeighEnv, record: bool = True):
        self.env = env
        self.session = None
        self.rows = []
        self.record = record
        self.observation = env.observe()

    def __getattr__(self, name):
        if name == "n_substeps":
            return self.env.n_substeps
        return getattr(self.env.scene, name)

    @property
    def bad(self):
        return self.env.scene.bad

    @bad.setter
    def bad(self, value):
        self.env.scene.bad = value

    @property
    def pinned(self):
        return None

    @pinned.setter
    def pinned(self, value):
        if value is not None:
            raise RuntimeError("Expert-only object pinning is forbidden in v2")

    def task_status(self):
        return self.env.info()

    def snapshot(self, *args, **kwargs):
        # Legacy Demo's incidental frame buffer is not the dataset recorder.
        # All transitions/images are captured exactly once in this bridge.
        return {}

    def watch(self):
        self.env.watch()

    def step(self, _raw_ctrl):
        s = self.session
        action = np.concatenate([s.q_cmd["left"], [s.grip_cmd["left"]], s.q_cmd["right"], [s.grip_cmd["right"]]]).astype(np.float32)
        previous = self.observation
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.record:
            info = {**info, "qpos": self.env.data.qpos.copy(), "qvel": self.env.data.qvel.copy()}
            self.rows.append((previous, action.copy(), obs, reward, terminated, truncated, info.copy()))
        self.observation = obs
        if terminated or truncated:
            raise EpisodeFinished(info["failure"] or ("success" if info["success"] else "horizon"))
        return obs, reward, terminated, truncated, info


def _make_demo(env: ScrapeWeighEnv, bridge: RecorderBridge):
    legacy = env.legacy

    class ActionDemo(legacy.Demo):
        def latch(self, side: str, name: str):
            adr = env.scene.free[name][0]
            held = next((h for h in env.scene.carried if h.side == side and h.adr == adr), None)
            if held is None:
                env.scene.bad = f"{side} failed to acquire {name} through the action-controlled grasp proxy"
            return held

        def unlatch(self, side: str):
            if any(h.side == side for h in env.scene.carried):
                raise RuntimeError("Object still attached after opening action")

    bridge.session = legacy.Session(bridge)
    bridge.session._carried = env._carried_metadata
    bridge.session._carry_patched = True
    if env.info().get("powder_dynamics") == "physical_grains_v1":
        # The kinematic preview moves a held vial while leaving grains at their
        # current world poses. Those ghost overlaps cannot predict granular
        # dynamics. Keep every rigid-obstacle check; real grain contacts are
        # resolved and audited by env.step at every physics substep.
        rigid_contacts = bridge.session._contacts
        def contacts_without_grain_ghosts(*args, **kwargs):
            return {item for item in rigid_contacts(*args, **kwargs)
                    if not any(label.startswith("grain_") for label in item[1:])}
        bridge.session._contacts = contacts_without_grain_ghosts
    return ActionDemo(bridge, bridge.session)


def _open_shield_strict(demo):
    """Open all three doors through handles, with no glass contact.

    The top handle requires an 18 mm pinch offset. The old 16 mm offset
    clips a glass panel during approach at 10 Hz. This sequence was checked
    by ScrapeWeighEnv's watchdog at every physics step.
    """
    session = demo.session
    legacy = demo.scene.env.legacy
    handle = demo.scene.data.geom("balance_door_top_handle").xpos.copy()
    low = handle + np.array([0.0, 0.0, 0.018])
    rotation = legacy.top_down(np.pi / 2)
    print("  open top through handle", flush=True)
    for target in (low + np.array([0.0, 0.0, 0.10]), low):
        if not demo.go("right", target, rotation):
            return False
    if not demo.grip("right", True):
        return False
    if not demo.go("right", low + np.array([0.0, 0.20, 0.0]), rotation):
        return False
    if not demo.grip("right", False):
        return False
    if float(demo.scene.data.qpos[demo.scene.top_adr]) < 0.15:
        demo.scene.bad = "top door failed to open through handle"
        return False
    session.snap("right", "home")
    if not demo.go("right", *session.targets["right"]):
        return False
    for side in ("left", "right"):
        print(f"  open {side} through handle", flush=True)
        if not legacy._open_side_door(demo, side):
            return False
    return True


def _approach_strict(demo, env):
    """Physical picks and supported regrasp; every move goes through env.step."""
    legacy = env.legacy
    if not _open_shield_strict(demo):
        return None, None
    print("  pick source from above", flush=True)
    origin = env.data.body("source").xpos.copy()
    pinch = origin + np.array([0.0, 0.0, 0.034])
    face = legacy.top_down(0.0)
    here, _ = demo.session.arms["left"].tool_pose()
    stages = [np.array([here[0], here[1], 1.12])]
    stages += [np.array([pinch[0], float(y), 1.12]) for y in np.linspace(here[1], pinch[1], 4)[1:]]
    stages += [pinch + np.array([0.0, 0.0, 0.08]), pinch]
    for target in stages:
        if not demo.go("left", target, face):
            return None, None
    if not demo.grip("left", True):
        return None, None
    source = demo.latch("left", "source")
    if source is None:
        return None, None
    print("  carry source to supported regrasp", flush=True)
    # The pan's left edge is x=.055. This housing position leaves >15 mm
    # clearance between the 28 mm vial and the pan, as well as the left edge.
    # Move 10 mm forward of the earlier placement. The upward withdrawal
    # otherwise brushes the opened left glass with the gripper coupler; this
    # placement gives about 8 mm geometric clearance at that closest pose.
    destination = np.array([0.025, 0.08, 0.853])
    if not legacy._carry_through_top(demo, source, destination, legacy._rz(np.pi / 2)):
        return None, None
    print("  settle source on housing before release", flush=True)
    housing = env.model.geom("balance_housing").id
    source_root = env.model.body("source").id
    def supported():
        for c in env.data.contact[:env.data.ncon]:
            gs = (int(c.geom1), int(c.geom2))
            if housing in gs and c.dist <= 0:
                other = gs[1] if gs[0] == housing else gs[0]
                if env.model.body_rootid[env.model.geom_bodyid[other]] == source_root:
                    return True
        return False
    for _ in range(160):
        if supported():
            break
        op, ort = legacy._object_pose(env.data, "source")
        if op[2] < 0.849:
            env.scene.bad = "source placement missed housing support"
            return None, None
        target = op - np.array([0.0, 0.0, 0.0005])
        p, r = legacy._tool_for(source, target, ort)
        s = demo.session
        s._scratch.qpos[:] = env.data.qpos
        q, pe, re = s.arms["left"].ik(s._scratch, p, r, s.q_cmd["left"], iters=40)
        if pe > 0.003 or re > 0.06:
            env.scene.bad = "source setdown IK failed"
            return None, None
        s.q_cmd["left"] = q
        demo.scene.step(None)
    else:
        env.scene.bad = "source failed to reach housing support"
        return None, None
    if not demo.grip("left", False):
        return None, None
    demo.unlatch("left")
    print("  regrasp supported source from side", flush=True)
    origin, _ = legacy._object_pose(env.data, "source")
    here, here_r = demo.session.arms["left"].tool_pose()
    pinch = origin + np.array([0.0, 0.0, 0.035])
    face = legacy._from_side("left")
    # 20 mm higher than the pose that put the left wrist 0.05 mm into the bench
    # under NoSlip3 (run 053, board_neg_x). The pinch height is unchanged.
    outside = origin + np.array([-0.096, 0.0, 0.100])
    if not demo.go("left", here + np.array([0.0, 0.0, 0.10]), here_r):
        return None, None
    # Clear the empty hand farther above the vial before turning. The earlier
    # turn near z=.99 crossed an IK branch in the 200-grain carried state.
    # This extra 80 mm and slower requested trajectory passed the physical
    # regrasp/lift continuation with a 1.411 rad/s planned peak; the planner's
    # 2 rad/s limit and every runtime contact check remain unchanged.
    withdrawn, withdrawn_r = demo.session.arms["left"].tool_pose()
    stages = [(withdrawn + np.array([0.0, 0.0, 0.08]), withdrawn_r),
              (outside + np.array([0.0, 0.0, 0.14]), face),
              (outside, face), (pinch, face)]
    previous_speed = demo.session.speed
    demo.session.speed = min(previous_speed, 0.6)
    try:
        for target, rot in stages:
            if not demo.go("left", target, rot):
                return None, None
    finally:
        demo.session.speed = previous_speed
    # Close gradually so light supported glass is not struck by the jaws.
    for value in np.linspace(demo.session.grip_cmd["left"], 255.0, 31)[1:]:
        demo.session.grip_cmd["left"] = float(value)
        demo.scene.step(None)
    if not demo.idle(10):
        return None, None
    source = demo.latch("left", "source")
    if source is None:
        return None, None
    origin, rot = legacy._object_pose(env.data, "source")
    print("  lift regrasped source", flush=True)
    if not legacy._carry_held(demo, source, origin + np.array([0.0, 0.0, 0.060]), rot):
        return None, None
    print("  pick tool", flush=True)
    spatula = _pick_tool(demo, env)
    if spatula is None:
        return None, None
    # Lift clear of the table before rotating the long tool for insertion.
    here, rot = demo.session.arms["right"].tool_pose()
    if not demo.go("right", np.array([here[0], here[1], 1.37]), rot):
        return None, None
    # Rotate outside the right wall before crossing the shield boundary.
    # At a 1.10 m blade-tip height the 26 cm tool handle remains below
    # the controller's 1.375 m workspace ceiling.
    down = np.array([0.0, 0.0, -1.0])
    print("  orient tool outside shield", flush=True)
    if not legacy._go_tip(demo, spatula, np.array([0.33, 0.09, 1.10]), down):
        return None, None
    source_p = env.data.body("source").xpos.copy()
    print("  cross vertically above source", flush=True)
    if not legacy._go_tip(demo, spatula, np.array([source_p[0], source_p[1], 1.10]), down):
        return None, None
    return source, spatula


def _pick_source_from_side(demo, env):
    legacy = env.legacy
    origin = env.data.body("source").xpos.copy()
    face = legacy._from_side("left")
    # Side grasp before lifting keeps the mouth accessible throughout transport.
    pinch = origin + np.array([0.0, 0.0, 0.038])
    here, _ = demo.session.arms["left"].tool_pose()
    for pos in (np.array([here[0], here[1], 1.12]),
                pinch + np.array([-0.10, 0.0, 0.16]),
                pinch + np.array([-0.08, 0.0, 0.02]),
                pinch):
        if not demo.go("left", pos, face):
            return None
    if not demo.grip("left", True):
        return None
    return demo.latch("left", "source")


def _pick_tool(demo, env):
    legacy = env.legacy
    origin = env.data.body("spatula").xpos.copy()
    face = legacy.top_down(np.pi / 2)
    here, here_r = demo.session.arms["right"].tool_pose()
    for pos, rot in ((np.array([here[0], here[1], 1.32]), here_r),
                     (np.array([origin[0], origin[1], 1.32]), face),
                     (origin + np.array([0.0, 0.0, 0.10]), face),
                     (origin, face)):
        if not demo.go("right", pos, rot):
            return None
    # The general waypoint controller tolerates centimetres of error. A
    # cylindrical handle needs a centered pinch before closing: an offset
    # pinch can squeeze it toward the edge and eject it under scoop load.
    # Measure the resting handle, including its small tilt on the bench.
    # The Robotiq pad patch lies 8.5 mm behind its pinch site along this
    # approach axis. Center the physical pads, rather than the pinch marker,
    # on the handle's 40 mm axial span.
    pinch_offset = np.array([0., 0., .0085])
    session = demo.session
    arm = session.arms["right"]
    aligned = 0
    for _ in range(30):
        body = env.data.body("spatula")
        origin = body.xpos.copy()
        tool_rotation = body.xmat.reshape(3, 3).copy()
        face = tool_rotation @ np.diag([-1., 1., -1.])
        session._scratch.qpos[:] = env.data.qpos
        target = origin + tool_rotation @ pinch_offset
        joints, pe, re = arm.ik(session._scratch, target, face,
                                session.q_cmd["right"], iters=90, damping=.01)
        if pe > .0005 or re > .002:
            env.scene.bad = f"tool pregrasp IK miss {pe * 1000:.3f} mm/{re:.5f} rad"
            return None
        delta = joints - session.q_cmd["right"]
        session.q_cmd["right"] += np.clip(.5 * delta, -.02, .02)
        demo.scene.step(None)
        pinch, rotation = arm.tool_pose()
        body = env.data.body("spatula")
        frame = body.xmat.reshape(3, 3)
        error = np.linalg.norm(pinch - body.xpos - frame @ pinch_offset)
        angle = legacy.rot_angle(rotation, frame @ np.diag([-1., 1., -1.]))
        aligned = aligned + 1 if error <= .0005 and angle <= .002 else 0
        if aligned >= 3:
            break
    else:
        env.scene.bad = f"tool pregrasp not centered: {error * 1000:.3f} mm/{angle:.5f} rad"
        return None
    for value in np.linspace(session.grip_cmd["right"], 255., 31)[1:]:
        session.grip_cmd["right"] = float(value)
        demo.scene.step(None)
    if not demo.idle(5):
        return None
    held = demo.latch("right", "spatula")
    if held is None:
        return None
    body = env.data.body("spatula")
    local = body.xmat.reshape(3, 3).T @ (arm.tool_pose()[0] - body.xpos)
    if abs(local[0]) > .002 or abs(local[2] - pinch_offset[2]) > .004:
        env.scene.bad = f"tool handle grasp off-center: shaft={local[0] * 1000:.3f} mm, axis={local[2] * 1000:.3f} mm"
        return None
    return held


def _fine_tip(demo, env, spatula, tip, direction, speed=0.05):
    """Short Cartesian servo trajectory, recorded through the same action API."""
    legacy = env.legacy
    current = legacy._tip(env.data)
    distance = float(np.linalg.norm(tip - current))
    count = max(1, int(np.ceil(distance / (speed * env.dt))))
    for n in range(1, count + 1):
        target = current + (np.asarray(tip) - current) * n / count
        p, r = legacy._tool_for(spatula, *legacy._spatula_pose(target, direction))
        s = demo.session
        scratch = s._scratch
        scratch.qpos[:] = env.data.qpos
        q, pe, re = s.arms["right"].ik(scratch, p, r, s.q_cmd["right"], iters=40)
        if pe > 0.006 or re > 0.08:
            env.scene.bad = f"fine spatula IK miss {pe:.4f} m/{re:.3f} rad"
            return False
        s.q_cmd["right"] = q
        demo.scene.step(None)
        if env.scene.bad:
            return False
    return True


def expert_episode(env: ScrapeWeighEnv, *, seed: int, powder_id: int, target_g: float, record: bool = True,
                   approach: str = "strict"):
    env.reset(seed=seed, options={"powder_id": powder_id, "target_g": target_g})
    bridge = RecorderBridge(env, record=record)
    demo = _make_demo(env, bridge)
    legacy = env.legacy
    try:
        if approach == "strict":
            source, spatula = _approach_strict(demo, env)
            if source is None or spatula is None:
                return bridge, env.info()
        elif approach == "reference":
            print("replaying action-only approach", flush=True)
            with h5py.File(ROOT / "data/teleop/scrape_vial_bc.hdf5", "r") as f:
                actions = f["data/demo_0/actions"][:1482]
            for action in actions:
                demo.session.q_cmd = {"left": action[:6].astype(float), "right": action[7:13].astype(float)}
                demo.session.grip_cmd = {"left": float(action[6]), "right": float(action[13])}
                bridge.step(None)
            demo.session._reset_state()
            source = demo.latch("left", "source")
            if source is None:
                return bridge, env.info()
            spatula = demo.latch("right", "spatula")
            # In action replay the unpinned source settles on the balance base
            # before regrasp. Lift it back to the reachable scoop height using
            # the held object's current transform, not a state restoration.
            source_p, source_r = legacy._object_pose(env.data, "source")
            scoop_origin = source_p + np.array([0.0, 0.0, 0.055])
            print("lifting regrasped source", flush=True)
            if not legacy._carry_held(demo, source, scoop_origin, source_r):
                return bridge, env.info()
        else:
            print("opening shield", flush=True)
            if not legacy._open_shield(demo):
                return bridge, env.info()
            print("grasping source from side", flush=True)
            source = _pick_source_from_side(demo, env)
            if source is None:
                return bridge, env.info()
            origin, rot = legacy._beside_pan()
            print("carrying source", flush=True)
            if not legacy._carry_through_top(demo, source, origin, rot):
                return bridge, env.info()
            print("grasping tool", flush=True)
            spatula = _pick_tool(demo, env)
        if spatula is None:
            return bridge, env.info()
        direction = np.array([0.0, 0.0, -1.0])
        receiver = env.data.body("vial_a").xpos.copy()
        rim = receiver + np.array([0.0, 0.0, legacy.VIAL_H])
        for scoop in range(10):
            origin, rot = legacy._object_pose(env.data, "source")
            mouth = origin + rot @ np.array([0.0, 0.0, legacy.SOURCE_H])
            print(f"scoop {scoop}: mass={env.powder.mass_g:.3f}", flush=True)
            load_height = 0.002 + 0.040 * env.powder.source_g / 8.0 - 0.004
            for tip in (mouth + np.array([0.0, 0.0, 0.09]), origin + rot @ np.array([0.0, 0.0, load_height])):
                if not legacy._go_tip(demo, spatula, tip, direction):
                    return bridge, env.info()
            # Expert may use simulator state, but mass changes ONLY in step.
            for _ in range(100):
                if env.powder.tool_g >= 0.95:
                    break
                if not demo.idle(1):
                    return bridge, env.info()
            else:
                env.scene.bad = "tool did not load at source mouth"
                return bridge, env.info()
            for tip in (mouth + np.array([0.0, 0.0, 0.08]), rim + np.array([0.0, 0.0, 0.08])):
                if not legacy._go_tip(demo, spatula, tip, direction):
                    return bridge, env.info()
            # Use a low discharge rate and withdraw vertically while the tail
            # remains above the receiver. Targets never enter dynamics.
            for _ in range(250):
                m = env.material
                remaining = env.target_g - env.powder.mass_g
                if remaining <= max(0.015, env.powder.tail_g + 0.01):
                    break
                if env.powder.tool_g <= 0.015:
                    break
                desired_rate = min(0.20, max(0.025, remaining * 0.3))
                gate = min(0.6, desired_rate / (m.flow_g_per_s * (1.0 - m.cohesion)))
                engagement = m.onset + gate * (1.0 - m.onset)
                z = 0.05 - 0.04 * engagement
                if not _fine_tip(demo, env, spatula, rim + np.array([0.0, 0.0, z]), direction):
                    return bridge, env.info()
                if not demo.idle(1):
                    return bridge, env.info()
            if not _fine_tip(demo, env, spatula, rim + np.array([0.0, 0.0, 0.08]), direction, speed=0.15):
                return bridge, env.info()
            if env.target_g - env.powder.mass_g <= 0.045:
                for _ in range(60):
                    demo.idle(1)
                env.scene.bad = "withdrawn but scale failed to settle in band"
                break
        if not env.scene.bad:
            env.scene.bad = "expert exhausted scoop attempts"
    except EpisodeFinished:
        pass
    return bridge, env.info()


def write_episode(group, bridge: RecorderBridge, env: ScrapeWeighEnv):
    if env.info().get("powder_dynamics") != "physical_grains_v1":
        raise ValueError("Height-gated surrogate rollouts do not satisfy contact-driven powder transfer")
    if not env.info()["success"] or not bridge.rows:
        raise ValueError("Only successful, recorded action rollouts can be written")
    if not env.info()["physical_valid"] or env.physics_checks < env.steps * env.n_substeps:
        raise ValueError("Successful demonstrations require complete per-physics-step validation")
    rows = bridge.rows
    for prefix, idx in (("obs", 0), ("next_obs", 2)):
        for key in rows[0][idx]:
            values = np.stack([r[idx][key] for r in rows])
            group.create_dataset(f"{prefix}/{key}", data=values, compression="gzip", compression_opts=4)
    group.create_dataset("actions", data=np.stack([r[1] for r in rows]))
    group.create_dataset("rewards", data=np.array([r[3] for r in rows], np.float32))
    group.create_dataset("dones", data=np.array([r[4] for r in rows], np.bool_))
    group.create_dataset("timeouts", data=np.array([r[5] for r in rows], np.bool_))
    group.create_dataset("subgoal_dones", data=np.stack([r[6]["subgoal_dones"] for r in rows]))
    group.create_dataset("material", data=np.array(list(env.material.as_dict().values()), np.float32))
    group.create_dataset("physics/qpos", data=np.stack([r[6]["qpos"] for r in rows]), compression="gzip")
    group.create_dataset("physics/qvel", data=np.stack([r[6]["qvel"] for r in rows]), compression="gzip")
    group.create_dataset("physics/checks", data=np.array([r[6]["physics_checks"] for r in rows], np.int64))
    for field in ("mass_g", "balance_g", "source_g", "tool_g", "spilled_g", "inflight_g",
                  "raw_net_mass_g", "mass_force_error_g", "conservation_error_g",
                  "max_grain_contact_penetration_m", "elapsed_s"):
        group.create_dataset(f"physics/{field}", data=np.array([r[6][field] for r in rows], np.float64))
    group.attrs["physics_audit"] = json.dumps({"contact_minima_m": env.contact_minima,
                                               "grasp_events": env.grasp_events,
                                               "intentional_contact_tolerance_m": 0.001,
                                               "forbidden_contact_tolerance_m": 0.00005,
                                               "grain_contact_tolerance_m": 0.2 * env.material.radius_m,
                                               "mass_force_tolerance_g": env.accounting.mass_tolerance_g,
                                               "material": env.material.as_dict()})
    group.attrs.update(success=1, num_samples=len(rows), powder_id=env.powder_id,
                       seed=env.episode_seed, physical_valid=1, physics_validation="rigid_contact_v1",
                       physical_validation_scope=env.info()["physical_validation_scope"], powder_model_calibrated=0,
                       powder_model_status=env.info()["powder_model_status"],
                       powder_dynamics=env.info()["powder_dynamics"],
                       grain_count=env.grain_count, grain_radius_m=env.material.radius_m,
                       grain_mass_g=env.material.grain_mass_g,
                       physics_checks=env.physics_checks, physics_steps_per_action=env.n_substeps,
                       grasp_model="mujoco_friction_only",
                       target_g=env.target_g, instruction=env.instruction,
                       scale_model=env.info()["scale_model"],
                       settlement_definition="stable_supported_pan_force_and_containment; horizontal_grain_rolling_allowed",
                       mass_g=env.powder.mass_g, reading_g=env.powder.reading_g,
                       spilled_g=env.powder.spilled_g, control_hz=env.control_hz)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "data/teleop/scrape_v2.hdf5")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--powders", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--target", type=float, default=None)
    p.add_argument("--no-images", action="store_true", help="Local diagnostic only; not VLA data")
    p.add_argument("--diagnostic", action="store_true", help="Run one expert and write only a JSON outcome")
    p.add_argument("--approach", choices=("strict", "reference", "side_grasp"), default="strict")
    args = p.parse_args()
    if args.out.exists() and not args.diagnostic:
        raise SystemExit(f"Refusing to overwrite {args.out}")
    env = ScrapeWeighEnv(images=not args.no_images, max_steps=4200)
    if not args.diagnostic and env.info().get("powder_dynamics") != "physical_grains_v1":
        raise SystemExit("Production collection is blocked until the contact-driven grain environment passes validation; use --diagnostic for rigid-body checks only")
    rng = np.random.default_rng(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    outcomes = []
    handle = None
    try:
        if not args.diagnostic:
            handle = h5py.File(args.out, "x")
            handle.attrs.update(schema_version=SCHEMA_VERSION, observation_timing="pre_action",
                                powder_model_status="uncalibrated_hand_designed_surrogate",
                                powder_model_calibrated=0, physical_validation_scope="rigid_bodies_only",
                                action_semantics=ACTION_SEMANTICS, control_hz=env.control_hz,
                                held_out_powder_ids=json.dumps([i for i in range(6) if i not in args.powders]),
                                powder_parameter_vectors=json.dumps([p.as_dict() for p in env.legacy.POWDERS]),
                                state_semantics="proprio14: left6,left_gripper,right6,right_gripper; lowdim20 uses legacy ordering")
            handle.create_group("data")
        kept = 0
        for attempt in range(1 if args.diagnostic else args.n * 3):
            powder_id = args.powders[kept % len(args.powders)]
            target = args.target if args.target is not None else float(rng.uniform(1.3, 3.5))
            bridge, info = expert_episode(env, seed=args.seed + attempt, powder_id=powder_id, target_g=target,
                                          record=not args.diagnostic, approach=args.approach)
            outcome = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in info.items()}
            outcomes.append(outcome)
            print(json.dumps(outcome), flush=True)
            if args.diagnostic:
                # For visual inspection only, never used as a training reset.
                np.savez(args.out.with_suffix(".final.npz"), qpos=env.data.qpos.copy(),
                         qvel=env.data.qvel.copy(), reading_g=env.powder.reading_g,
                         mass_g=env.powder.mass_g, target_g=env.target_g)
            if info["success"] and handle is not None:
                write_episode(handle["data"].create_group(f"demo_{kept}"), bridge, env)
                handle.flush()
                kept += 1
                if kept == args.n:
                    break
            if args.diagnostic:
                break
        args.out.with_suffix(".outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
        if not any(r["success"] for r in outcomes):
            raise SystemExit("No successful action-driven episodes; inspect the outcome log before training.")
    finally:
        if handle is not None:
            handle.close()
        env.close()


if __name__ == "__main__":
    main()
