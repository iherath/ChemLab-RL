"""Grok-collected weigh demos via the same Session API as the keyframe app.

Diversity Leo asked for: different draft-shield slides and different vial grasps.
Failed plans are discarded, not forced through red ghosts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "glfw")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chem_rl_sim" / "third_party" / "urgantry_sim"))
sys.path.insert(0, str(ROOT / "chem_rl_sim"))

from chem_rl_sim.env import VialWeighingEnv
from chem_rl_sim.teleop.keyframe import HANDLE_ABOVE, HANDLE_DZ, Session, top_down
from chem_rl_sim.teleop.keyframe_script import run_all
from chem_rl_sim.teleop.recorder import Recorder

L, R = "left", "right"
DOOR_TRAVEL = 0.20
HANDLES = {
    "top": "balance_door_top_handle",
    "left": "balance_door_left_handle",
    "right": "balance_door_right_handle",
}

# Grok diversity schedule. Side-door recipes approach from outside in x so the
# gripper is not dropped onto the housing (that is what failed last time).
RECIPES = [
    dict(name="top_yaw_default", door="top", door_arm=R, pick_arm=L, yaw_deg=None, grasp_dz=0.0, grasp_xy=(0.0, 0.0)),
    dict(name="top_yaw_0", door="top", door_arm=R, pick_arm=L, yaw_deg=0.0, grasp_dz=0.0, grasp_xy=(0.0, 0.0)),
    dict(name="top_yaw_35", door="top", door_arm=R, pick_arm=L, yaw_deg=35.0, grasp_dz=0.0, grasp_xy=(0.0, 0.0)),
    dict(name="top_yaw_70", door="top", door_arm=R, pick_arm=L, yaw_deg=70.0, grasp_dz=0.0, grasp_xy=(0.0, 0.0)),
    dict(name="top_yaw_90", door="top", door_arm=R, pick_arm=L, yaw_deg=90.0, grasp_dz=0.0, grasp_xy=(0.0, 0.0)),
    dict(name="top_yaw_120", door="top", door_arm=R, pick_arm=L, yaw_deg=120.0, grasp_dz=0.0, grasp_xy=(0.0, 0.0)),
    dict(name="top_yaw_160", door="top", door_arm=R, pick_arm=L, yaw_deg=160.0, grasp_dz=0.0, grasp_xy=(0.0, 0.0)),
    dict(name="top_offset_x", door="top", door_arm=R, pick_arm=L, yaw_deg=45.0, grasp_dz=0.0, grasp_xy=(0.012, 0.0)),
    dict(name="top_offset_y", door="top", door_arm=R, pick_arm=L, yaw_deg=45.0, grasp_dz=0.0, grasp_xy=(0.0, 0.012)),
    dict(name="top_mid_neck", door="top", door_arm=R, pick_arm=L, yaw_deg=20.0, grasp_dz=0.012, grasp_xy=(0.0, 0.0)),
    dict(name="left_door_outside", door="left", door_arm=L, pick_arm=L, yaw_deg=0.0, grasp_dz=0.0, grasp_xy=(0.0, 0.0)),
    dict(name="right_door_outside", door="right", door_arm=R, pick_arm=R, yaw_deg=180.0, grasp_dz=0.0, grasp_xy=(0.0, 0.0)),
]


def _enter(s: Session, st: dict, label: str) -> bool:
    st[label] = s.plan().status
    ok, msg = s.execute(force=False)
    if not ok:
        st[label] = f"refused:{msg}"
        return False
    run_all(s)
    return True


def _slide_door(s: Session, st: dict, arm: str, door: str, dy: float, tag: str) -> bool:
    handle = s.data.geom(HANDLES[door]).xpos.copy()
    if door == "top":
        low = handle + np.array([0.0, 0.0, HANDLE_DZ])
        high = low + np.array([0.0, 0.0, HANDLE_ABOVE])
        rot = top_down(np.pi / 2)
    else:
        # Stand off the housing in x, then come in to the handle.
        sx = -1.0 if door == "left" else 1.0
        high = handle + np.array([sx * 0.09, -0.02, 0.10])
        low = handle + np.array([sx * 0.028, 0.0, 0.0])
        rot = top_down(0.0 if door == "left" else np.pi)
    s.set_target(arm, high, rot)
    if not _enter(s, st, f"{tag} above {door}"):
        return False
    s.set_target(arm, low, rot)
    if not _enter(s, st, f"{tag} grasp {door}"):
        return False
    s.grip(arm, True)
    run_all(s)
    s.set_target(arm, s.targets[arm][0] + np.array([0.0, dy, 0.0]))
    if not _enter(s, st, f"{tag} slide {door}"):
        return False
    s.grip(arm, False)
    run_all(s)
    s.snap(arm, "home")
    return _enter(s, st, f"{tag} home")


def _pick_and_place(s: Session, st: dict, recipe: dict) -> bool:
    arm = recipe["pick_arm"]
    yaw = recipe["yaw_deg"]
    dz = float(recipe["grasp_dz"])
    dxy = np.array([*recipe["grasp_xy"], dz], dtype=float)
    s.snap(arm, "vial")
    if yaw is not None:
        s.set_target(arm, rot=top_down(np.deg2rad(yaw)))
    if not _enter(s, st, "above vial"):
        return False
    s.snap(arm, "vial")
    pos, rot = s.targets[arm]
    if yaw is not None:
        rot = top_down(np.deg2rad(yaw))
    s.set_target(arm, pos + dxy, rot)
    if not _enter(s, st, "grasp vial"):
        return False
    s.grip(arm, True)
    run_all(s)
    door = recipe["door"]
    if door != "top":
        if not _place_through_side(s, st, arm, door, yaw):
            return False
    else:
        s.snap(arm, "pan")
        if yaw is not None:
            s.set_target(arm, rot=top_down(np.deg2rad(yaw)))
        if not _enter(s, st, "above pan"):
            return False
        s.snap(arm, "pan")
        if not _enter(s, st, "place"):
            return False
    s.grip(arm, False)
    run_all(s)
    s.snap(arm, "home")
    return _enter(s, st, "pick home")


def _place_through_side(s: Session, st: dict, arm: str, door: str, yaw: float | None) -> bool:
    """In through the open side door, not over the closed top glass."""
    pan = s.data.site("balance_pan_site").xpos.copy()
    sx = -1.0 if door == "left" else 1.0
    rot = top_down(np.deg2rad(yaw) if yaw is not None else (0.0 if door == "left" else np.pi))
    waypoints = [
        ("outside", pan + np.array([sx * 0.30, 0.0, 0.14])),
        ("doorway", pan + np.array([sx * 0.14, 0.0, 0.14])),
        ("above pan", pan + np.array([0.0, 0.0, 0.14])),
        ("place", pan + np.array([0.0, 0.0, 0.038])),
    ]
    for label, pos in waypoints:
        s.set_target(arm, pos, rot)
        if not _enter(s, st, label):
            return False
    return True


def weigh_recipe(s: Session, recipe: dict) -> dict:
    st = {"recipe": recipe["name"]}
    if not _slide_door(s, st, recipe["door_arm"], recipe["door"], DOOR_TRAVEL, "open"):
        st["ok"] = False
        return st
    if not _pick_and_place(s, st, recipe):
        st["ok"] = False
        return st
    if not _slide_door(s, st, recipe["door_arm"], recipe["door"], -DOOR_TRAVEL, "close"):
        st["ok"] = False
        return st
    st["ok"] = True
    return st


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "data" / "teleop" / "weigh_vial_grok_v2.hdf5")
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=20)
    args = p.parse_args()

    if args.out.exists():
        args.out.unlink()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".json")

    env_kwargs = dict(
        image_size=args.image_size,
        control_hz=20.0,
        hand="robotiq",
        layout="weigh",
        clutter=False,
        max_episode_steps=10**9,
        randomize=True,
    )
    env = VialWeighingEnv(**env_kwargs)
    rec = Recorder(
        args.out,
        {
            "env_name": "ChemVialWeighing-v0",
            "type": "chem_rl_sim",
            "env_kwargs": {**env_kwargs, "normalized_actions": True},
            "collector": "grok_session_v2",
        },
        20.0,
        "robotiq",
    )
    sess = Session(env, rec)
    log = []

    print(f"Grok collecting {len(RECIPES)} recipes -> {args.out}")
    for i, recipe in enumerate(RECIPES):
        sess.begin_episode(seed=args.seed + i)
        st = weigh_recipe(sess, recipe)
        success = bool(sess.info.get("success")) and st.get("ok")
        n = 0
        if success:
            n = sess.save_episode()
            rec.wait()
            if rec.last_error:
                raise RuntimeError(rec.last_error)
            saved = rec.last_saved
        else:
            sess.discard_episode()
            saved = None
        refused = {k: v for k, v in st.items() if isinstance(v, str) and v.startswith("refused")}
        entry = {
            "recipe": recipe,
            "success": int(bool(sess.info.get("success"))),
            "operator_ok": bool(st.get("ok")),
            "saved": saved,
            "samples": n,
            "balance": sess.info.get("balance_display"),
            "refused": refused,
        }
        log.append(entry)
        print(
            f"{recipe['name']}: saved={saved} success={entry['success']} "
            f"samples={n} refused={refused or 'none'}"
        )

    rec.wait()
    env.close()
    log_path.write_text(json.dumps(log, indent=2, default=str))
    print(f"wrote {log_path}")
    kept = sum(1 for e in log if e["saved"])
    print(f"kept {kept}/{len(RECIPES)} diverse demos")


if __name__ == "__main__":
    main()
