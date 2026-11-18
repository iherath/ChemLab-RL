"""Collect contact-driven robot scooping through 14-dimensional joint actions."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from chemlab_rl.granular_env import GranularWeighEnv
from chemlab_rl.granular_material import GRANULAR_MATERIALS, HELD_OUT_MATERIAL_IDS
from chemlab_rl.granular_geometry import SCOOP_TIP, SCOOP_FRAME, SCOOP_OUTER_RADIUS
from chemlab_rl.scrape_env import ACTION_SEMANTICS, SCHEMA_VERSION
from scripts.collect_scrape_v2 import (RecorderBridge, EpisodeFinished, _make_demo,
                                     _approach_strict, write_episode)

EXPERT_CONTROLLER = "guarded_complete_batch_v3_centered_pad_patch"
MIN_LOADING_HEIGHT_M = 0.006
MAX_LOADING_HEIGHT_M = 0.016
DEPTH_TOLERANCE_M = 0.00025
LATERAL_TOLERANCE_M = 0.001
BATCH_MARGIN_G = 0.025
INITIAL_BATCH_BUDGET_G = {"bulk": 0.40, "medium": 0.20}
OBSERVED_LOAD_HEADROOM_G = 0.025
MAX_LOAD_ATTEMPTS = 4
EMPTY_RETRY_STEP_M = 0.00025


def source_identity():
    """Hash the transition, expert, controls, and scene assets independently of Git."""
    names = ["chemlab_rl/scrape_env.py", "chemlab_rl/weigh_scene.py",
             "chemlab_rl/granular_env.py", "chemlab_rl/granular_scene.py",
             "chemlab_rl/granular_material.py", "chemlab_rl/granular_geometry.py",
             "chemlab_rl/granular_accounting.py", "scripts/collect_leo_scrape.py",
             "scripts/collect_scrape_v2.py", "scripts/collect_granular_demos.py"]
    paths = [ROOT / name for name in names]
    for directory in (ROOT / "chem_rl_sim/chem_rl_sim", ROOT / "chem_rl_sim/third_party/urgantry_sim/urgantry_sim"):
        paths.extend(p for p in directory.rglob("*") if p.is_file()
                     and p.suffix.lower() in {".py", ".xml", ".obj", ".stl", ".png", ".jpg", ".json"}
                     and "__pycache__" not in p.parts)
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(set(paths))}
    version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return "sha256:" + version, hashes


def _move_cup(demo, env, hold, target, rotation=None, *, seconds=None, speed=0.02):
    """Servo a cup-bottom pose with measured grasp transform and smooth targets."""
    mj, legacy = env.mj, env.legacy
    current = legacy._tip(env.data)
    current_r = env.data.body("spatula").xmat.reshape(3, 3).copy()
    target = np.asarray(target, float)
    rotation = current_r if rotation is None else np.asarray(rotation, float)
    q0, q1 = np.empty(4), np.empty(4)
    mj.mju_mat2Quat(q0, current_r.ravel())
    mj.mju_mat2Quat(q1, rotation.ravel())
    if np.dot(q0, q1) < 0:
        q1 = -q1
    angle = 2 * np.arccos(np.clip(np.dot(q0, q1), -1, 1))
    duration = max(float(seconds or 0), np.linalg.norm(target - current) / speed, angle / 0.5, env.dt)
    count = int(np.ceil(duration / env.dt))
    for index in range(1, count + 1):
        u = index / count
        u = u * u * (3 - 2 * u)
        tip = current + u * (target - current)
        quaternion = (1 - u) * q0 + u * q1
        quaternion /= np.linalg.norm(quaternion)
        mat = np.empty(9)
        mj.mju_quat2Mat(mat, quaternion)
        rot = mat.reshape(3, 3)
        # Contact loss/reacquisition replaces the metadata object. Resolve it
        # each tick rather than retaining the Hold returned by the first pick.
        adr = env.scene.free["spatula"][0]
        hold = next((item for item in env.scene.carried
                     if item.side == "right" and item.adr == adr), None)
        if hold is None:
            env.scene.bad = "right lost contact-supported spatula grasp"
            return False
        position, orientation = legacy._tool_for(hold, tip - rot @ SCOOP_TIP, rot)
        session = demo.session
        session._scratch.qpos[:] = env.data.qpos
        joints, position_error, rotation_error = session.arms["right"].ik(
            session._scratch, position, orientation, session.q_cmd["right"], iters=60)
        if position_error > 0.003 or rotation_error > 0.06:
            env.scene.bad = f"scoop IK miss {position_error:.4f} m/{rotation_error:.3f} rad"
            return False
        session.q_cmd["right"] = joints
        demo.scene.step(None)
        if env.scene.bad:
            return False
    # Accept only a measured tip near its requested position. Do not restore it.
    error = float(np.linalg.norm(legacy._tip(env.data) - target))
    if error > 0.004:
        env.scene.bad = f"scoop tracking error {error * 1000:.2f} mm"
        return False
    return True


def _align_cup_axis(rotation, axis):
    """Align the opening by its shortest rotation, preserving existing yaw."""
    rotation, axis = np.asarray(rotation, float), np.asarray(axis, float)
    if not np.isfinite(rotation).all() or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
        raise ValueError("non-finite or undefined cup alignment")
    normal = rotation @ SCOOP_FRAME[:, 2]
    normal = normal / np.linalg.norm(normal)
    axis = axis / np.linalg.norm(axis)
    cross, cosine = np.cross(normal, axis), float(np.clip(normal @ axis, -1., 1.))
    if np.linalg.norm(cross) < 1e-12:
        if cosine < 0:
            raise ValueError("cup alignment would require an arbitrary 180-degree flip")
        return rotation.copy()
    x, y, z = cross
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return (np.eye(3) + skew + skew @ skew / (1. + cosine)) @ rotation


def _source_pose(env):
    body = env.data.body("source")
    return body.xpos.copy(), body.xmat.reshape(3, 3).copy()


def _source_target(env, height):
    origin, rotation = _source_pose(env)
    current = env.data.body("spatula").xmat.reshape(3, 3)
    return origin + rotation @ np.array([0., 0., height]), _align_cup_axis(current, rotation[:, 2])


def _local_bed_surface(env):
    """Measure grain tops under the cup, in the current source-vial frame."""
    origin, rotation = _source_pose(env)
    local = (env.data.xpos[env.grain_bodies] - origin) @ rotation
    selected = ((env.grain_state["labels"] == "source")
                & (np.linalg.norm(local[:, :2], axis=1) <= SCOOP_OUTER_RADIUS + env.material.radius_m))
    if not np.any(selected):
        raise ValueError("no source grains under the scoop footprint")
    surface = float(np.max(local[selected, 2]) + env.material.radius_m)
    if not np.isfinite(surface):
        raise ValueError("non-finite local powder surface")
    return surface


def _candidate_depths(surface):
    """Candidate cup-bottom heights above the source base, not fixed loads."""
    if not np.isfinite(surface):
        raise ValueError("non-finite local powder surface")
    bulk = float(np.clip(surface - .011, MIN_LOADING_HEIGHT_M, MAX_LOADING_HEIGHT_M - .001))
    medium = float(np.clip(max(surface - .0105, bulk + .0005),
                           bulk + .0005, MAX_LOADING_HEIGHT_M - .0005))
    fine = float(np.clip(max(surface - .010, medium + .0005),
                         medium + .0005, MAX_LOADING_HEIGHT_M))
    return bulk, medium, fine


def _choose_load_mode(upper_margin_g, previous_loads_g):
    # Initial budgets include headroom and are identical for every material.
    # Observed loads refine each mode separately; neither rule assigns a yield.
    for mode in ("bulk", "medium"):
        budget = (previous_loads_g[mode] + OBSERVED_LOAD_HEADROOM_G
                  if mode in previous_loads_g else INITIAL_BATCH_BUDGET_G[mode])
        if upper_margin_g >= budget:
            return mode
    return "fine"


def _mass_snapshot(env):
    return {name: float(env.grain_state[name]) for name in ("receiver_g", "tool_g", "inflight_g")}


def _batch_fits(receiver_g, tool_g, inflight_g, target_g):
    quantities = np.array([receiver_g, tool_g, inflight_g, target_g], float)
    return bool(np.isfinite(quantities).all() and np.all(quantities >= 0)
                and receiver_g + tool_g + inflight_g <= target_g + BATCH_MARGIN_G + 1e-12)


def _log_expert(bridge, env, phase, **details):
    bridge.expert_phase = phase
    row = dict(phase=phase, sim_seconds=float(env.data.time), **_mass_snapshot(env), **details)
    bridge.expert_trace.append(row)
    print("expert " + json.dumps(row), flush=True)


def _check_batch(bridge, env, phase, **details):
    masses = _mass_snapshot(env)
    fits = _batch_fits(**masses, target_g=env.target_g)
    _log_expert(bridge, env, phase, batch_fits=fits,
                upper_mass_g=env.target_g + BATCH_MARGIN_G, **details)
    if not fits:
        env.scene.bad = ("unsafe retained batch before pour: "
                         f"receiver/tool/inflight={masses['receiver_g']:.4f}/"
                         f"{masses['tool_g']:.4f}/{masses['inflight_g']:.4f} g, "
                         f"upper bound={env.target_g + BATCH_MARGIN_G:.4f} g")
    return fits


def _verify_loading_depth(demo, env, scoop, bridge, depth, *, cycle, attempt):
    """Measure the bottom height in the source frame after a physical hold."""
    # Recompute the moving vial's pose and use ordinary controls for correction.
    for correction in range(3):
        target, rotation = _source_target(env, depth)
        _log_expert(bridge, env, "loading_descent", cycle=cycle, attempt=attempt,
                    correction=correction, requested_depth_m=depth)
        if not _move_cup(demo, env, scoop, target, rotation,
                         seconds=.3 if correction else None,
                         speed=.003 if correction else .015):
            return False
        if not demo.idle(4):
            return False
        origin, frame = _source_pose(env)
        measured = frame.T @ (env.legacy._tip(env.data) - origin)
        z_error = float(measured[2] - depth)
        xy_error = float(np.linalg.norm(measured[:2]))
        _log_expert(bridge, env, "depth_check", cycle=cycle, attempt=attempt,
                    correction=correction, requested_depth_m=depth,
                    measured_depth_m=float(measured[2]), z_error_m=z_error,
                    lateral_error_m=xy_error)
        if (np.isfinite(measured).all() and abs(z_error) <= DEPTH_TOLERANCE_M
                and xy_error <= LATERAL_TOLERANCE_M):
            return True
    env.scene.bad = f"loading depth miss: z={z_error * 1000:.3f} mm, xy={xy_error * 1000:.3f} mm"
    return False


def _load_batch(demo, env, scoop, bridge, cycle, previous_loads_g):
    above, rotation = _source_target(env, .078)
    _log_expert(bridge, env, "source_approach", cycle=cycle)
    if not _move_cup(demo, env, scoop, above, rotation, speed=.025):
        return None
    surface = _local_bed_surface(env)
    bulk, medium, fine = _candidate_depths(surface)
    masses = _mass_snapshot(env)
    margin = env.target_g + BATCH_MARGIN_G - masses["receiver_g"] - masses["inflight_g"]
    mode = _choose_load_mode(margin, previous_loads_g)
    depth = {"bulk": bulk, "medium": medium, "fine": fine}[mode]
    # A fine attempt remains strictly shallower than the bulk candidate, even
    # after clamping and bounded empty-scoop retries.
    minimum = MIN_LOADING_HEIGHT_M if mode == "bulk" else bulk + EMPTY_RETRY_STEP_M
    for attempt in range(MAX_LOAD_ATTEMPTS):
        _log_expert(bridge, env, "load_candidate", cycle=cycle, attempt=attempt,
                    mode=mode, local_surface_m=surface, requested_depth_m=depth,
                    bulk_depth_m=bulk, medium_depth_m=medium, fine_depth_m=fine,
                    previous_loads_g=dict(previous_loads_g))
        if not _verify_loading_depth(demo, env, scoop, bridge, depth, cycle=cycle, attempt=attempt):
            return None
        above, rotation = _source_target(env, .078)
        _log_expert(bridge, env, "load_lift", cycle=cycle, attempt=attempt)
        if not _move_cup(demo, env, scoop, above, rotation, speed=.020) or not demo.idle(4):
            return None
        if not _check_batch(bridge, env, "retained_batch", cycle=cycle, attempt=attempt, mode=mode):
            return None
        mass = float(env.grain_state["tool_g"])
        if mass >= .5 * env.material.grain_mass_g:
            return mode, mass
        deeper = max(minimum, depth - EMPTY_RETRY_STEP_M)
        if deeper >= depth - 1e-12:
            break
        depth = deeper
    env.scene.bad = "physical scoop empty after bounded depth attempts"
    return None


def _pour_complete_batch(demo, env, scoop, bridge, cycle):
    upright = _align_cup_axis(env.data.body("spatula").xmat.reshape(3, 3), [0., 0., 1.])
    receiver = env.data.body("vial_a").xpos.copy()
    pivot = receiver + [-.007, 0., .068]
    transit = np.array([pivot[0], pivot[1], max(env.legacy._tip(env.data)[2], pivot[2] + .015)])
    _log_expert(bridge, env, "receiver_transit", cycle=cycle)
    if not _move_cup(demo, env, scoop, transit, upright, speed=.030):
        return False
    if not _move_cup(demo, env, scoop, pivot, upright, speed=.020):
        return False
    # Recheck after carrying: all received, retained, and airborne mass counts.
    if not _check_batch(bridge, env, "pre_pour_guard", cycle=cycle):
        return False
    theta = np.deg2rad(110.)
    turn = np.array([[np.cos(theta), 0., np.sin(theta)], [0., 1., 0.],
                     [-np.sin(theta), 0., np.cos(theta)]])
    _log_expert(bridge, env, "complete_pour", cycle=cycle)
    if not _move_cup(demo, env, scoop, pivot, turn @ upright, seconds=4.):
        return False
    _log_expert(bridge, env, "drain_hold", cycle=cycle)
    for _ in range(30):
        if max(env.grain_state["tool_g"], env.grain_state["inflight_g"]) < .5 * env.material.grain_mass_g:
            break
        if not demo.idle(1):
            return False
    else:
        env.scene.bad = "complete batch did not drain within physical hold"
        return False
    _log_expert(bridge, env, "return_upright", cycle=cycle)
    if not _move_cup(demo, env, scoop, pivot, upright, seconds=4.):
        return False
    _log_expert(bridge, env, "receiver_retract", cycle=cycle)
    return _move_cup(demo, env, scoop, transit, upright, speed=.025)


def _expert_info(env, bridge):
    return {**env.info(), "expert_controller": EXPERT_CONTROLLER,
            "expert_terminal_phase": bridge.expert_phase,
            "expert_trace": bridge.expert_trace}


def expert_episode(env, *, seed, target_g, record=True, approach_only=False):
    env.reset(seed=seed, options={"target_g": target_g})
    bridge = RecorderBridge(env, record=record)
    bridge.expert_phase, bridge.expert_trace = "approach", []
    demo = _make_demo(env, bridge)
    try:
        source, scoop = _approach_strict(demo, env)
        if source is None or scoop is None or approach_only:
            return bridge, _expert_info(env, bridge)
        previous_loads_g = {}
        for cycle in range(20):
            loaded = _load_batch(demo, env, scoop, bridge, cycle, previous_loads_g)
            if loaded is None:
                break
            mode, retained_g = loaded
            previous_loads_g[mode] = retained_g
            if not _pour_complete_batch(demo, env, scoop, bridge, cycle):
                break
            if env.powder.mass_g >= env.target_g - 0.05:
                _log_expert(bridge, env, "settle_at_target", cycle=cycle)
                for _ in range(100):
                    if not demo.idle(1):
                        break
                if env.scene.bad:
                    break
                env.scene.bad = "physical scale did not settle at target"
                break
        if not env.scene.bad and not env.info()["success"]:
            env.scene.bad = "physical expert exhausted scoop attempts"
    except EpisodeFinished:
        pass
    except ValueError as error:
        env.scene.bad = f"expert geometry or measurement: {error}"
    # Environment success may terminate during a pour/return. Never resume its
    # finished physics or claim the tool retracted merely because it succeeded.
    return bridge, _expert_info(env, bridge)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--powder", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", type=float, default=0.5)
    parser.add_argument("--grains", type=int, default=400)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--approach-only", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.diagnostic:
        raise SystemExit(f"Refusing to overwrite {args.out}")
    if args.powder in HELD_OUT_MATERIAL_IDS and not args.diagnostic:
        raise SystemExit("Held-out materials cannot enter BC demonstration collection")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    version, hashes = source_identity()
    env = GranularWeighEnv(images=not args.diagnostic, max_steps=5000,
                          powder_id=args.powder, grain_count=args.grains)
    try:
        bridge, info = expert_episode(env, seed=args.seed, target_g=args.target,
                                      record=not args.diagnostic, approach_only=args.approach_only)
        outcome = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in info.items()}
        outcome["wall_seconds"] = time.perf_counter() - started
        outcome.update(source_version=version, source_files_sha256=hashes,
                       expert_controller=EXPERT_CONTROLLER)
        args.out.with_suffix(".outcome.json").write_text(json.dumps(outcome, indent=2) + "\n")
        np.savez(args.out.with_suffix(".final.npz"), qpos=env.data.qpos, qvel=env.data.qvel,
                 ctrl=env.data.ctrl, reading_g=env.powder.reading_g, target_g=env.target_g)
        print(json.dumps(outcome), flush=True)
        if info["success"] and not args.diagnostic:
            with h5py.File(args.out, "x") as handle:
                handle.attrs.update(schema_version=SCHEMA_VERSION, observation_timing="pre_action",
                                    powder_dynamics="physical_grains_v1", action_semantics=ACTION_SEMANTICS,
                                    control_hz=env.control_hz, held_out_powder_ids=json.dumps(HELD_OUT_MATERIAL_IDS),
                                    training_powder_ids=json.dumps([i for i in range(len(GRANULAR_MATERIALS)) if i not in HELD_OUT_MATERIAL_IDS]),
                                    source_version=version, source_files_sha256=json.dumps(hashes, sort_keys=True),
                                    expert_controller=EXPERT_CONTROLLER,
                                    expert_terminal_phase=info["expert_terminal_phase"],
                                    mujoco_version=env.mj.__version__,
                                    powder_parameter_vectors=json.dumps([m.as_dict() for m in GRANULAR_MATERIALS]),
                                    active_randomized_parameters=json.dumps(["density_kg_m3", "sliding_friction"]),
                                    grain_count=env.grain_count)
                write_episode(handle.create_group("data/demo_0"), bridge, env)
        if not info["success"] and not (args.approach_only and not info["failure"]):
            raise SystemExit("No successful physical demonstration; see outcome JSON")
    finally:
        env.close()


if __name__ == "__main__":
    main()
