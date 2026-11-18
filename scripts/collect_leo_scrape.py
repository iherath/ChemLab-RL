"""BC demonstrations in Leo's weigh scene, with his collisions left on.

The source is an open vial the same size as the receiving vial. There is no
stand. The left gripper opens the left draft door and holds the source through
that opening, with its mouth a few centimetres from the receiving vial. The
right gripper opens the right door. Scoop demos bring the spatula in through
that opening and scrape across the short gap. Pour demos tilt the source so
the powder drops that same short distance. Nothing writes a door joint.

Powder is a set of 0.02 g pucks. The LCD is the tared pan-force reading.
Each demo stores the powder parameter vector. Images are top1 and both wrists.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "chem_rl_sim"),
    str(ROOT / "chem_rl_sim" / "third_party" / "urgantry_sim"),
]

from chemlab_rl.powder_dispense import Material
from chem_rl_sim.assets import add_asset
from chem_rl_sim.balance import pan_reading_g
from chem_rl_sim.scene import BOARD_TOP, PAN_XY, PAN_Z, build_spec, set_initial_pose
from chem_rl_sim.teleop.arm import ARM_JOINTS
from chem_rl_sim.teleop.keyframe import Session, rot_angle, top_down

TOL_G = 0.05
PUCK_G = 0.02
N_PUCKS = 350
VIAL_H = 0.057
OD = 0.028
# Same 20 mL class as the receiving vial: 28 mm OD, 57 mm tall. Held through
# the left door, so the mouth can sit a few centimetres from the vial on the pan.
SOURCE_H = VIAL_H
SOURCE_OR = OD / 2
SOURCE_IR = 0.011
BLADE = 0.26
IMAGE = 224
PAN = np.array([*PAN_XY, BOARD_TOP + PAN_Z])
SRC_REST = np.array([-0.34, 0.06, BOARD_TOP])
SPAT_REST = np.array([0.28, 0.12, BOARD_TOP + 0.021])
DISCARD_REST = np.array([0.28, 0.02, BOARD_TOP])
HIGH_Z = 1.32
POWDERS = [
    Material(1.6, 0.15, 0.15, 0.05, 0.25, 0.008, 0.9, 0.05),
    Material(1.1, 0.20, 0.25, 0.08, 0.35, 0.010, 0.88, 0.15),
    Material(0.7, 0.18, 0.40, 0.12, 0.45, 0.012, 0.92, 0.25),
    Material(2.0, 0.12, 0.12, 0.04, 0.20, 0.006, 0.86, 0.10),
    Material(0.9, 0.28, 0.30, 0.10, 0.40, 0.009, 0.9, 0.20),
    Material(1.4, 0.22, 0.20, 0.15, 0.30, 0.011, 0.84, 0.30),
]
MATERIAL_NAMES = (
    "flow_g_per_s", "onset", "delay_s", "residual_g",
    "lag_s", "noise_g", "spill_tilt", "cohesion",
)
_DIGIT = {
    " ": "", "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
    "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg",
}
_SEG_BOX = {
    "a": (0.0, 0.006, 0.0045, 0.0007),
    "b": (0.004, 0.003, 0.0007, 0.0026),
    "c": (0.004, -0.003, 0.0007, 0.0026),
    "d": (0.0, -0.006, 0.0045, 0.0007),
    "e": (-0.004, -0.003, 0.0007, 0.0026),
    "f": (-0.004, 0.003, 0.0007, 0.0026),
    "g": (0.0, 0.0, 0.0045, 0.0007),
}


def _ry(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], float)


def _rz(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], float)


def _quat(rot: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(rot, float).reshape(-1))
    return q


def _side_grasp() -> np.ndarray:
    """Fingertips come down onto the vial's -y side so the palm stays above the bench."""
    z = np.array([0.0, 0.75, -0.55])
    z /= np.linalg.norm(z)
    y = np.array([1.0, 0.0, 0.0])
    y = y - z * np.dot(y, z)
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


def _add_lcd(spec: mujoco.MjSpec) -> None:
    tilt = np.deg2rad(20)
    rot = np.array(
        [[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]],
        float,
    )
    quat = _quat(rot)
    lcd = np.array([0.10, 0.04 - 0.110, BOARD_TOP + 0.077])
    for i, x in enumerate((-0.028, -0.012, 0.010, 0.026)):
        for name, (px, py, hx, hy) in _SEG_BOX.items():
            spec.worldbody.add_geom(
                name=f"lcd_{i}_{name}", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[hx, hy, 0.0005], pos=(lcd + rot @ np.array([x + px, py, 0.004])).tolist(),
                quat=quat.tolist(), rgba=[0.02, 0.03, 0.04, 0.0], contype=0, conaffinity=0,
            )
    spec.worldbody.add_geom(
        name="lcd_dot", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0012, 0.0012, 0.0005],
        pos=(lcd + rot @ np.array([-0.001, -0.006, 0.004])).tolist(), quat=quat.tolist(),
        rgba=[0.02, 0.03, 0.04, 1.0], contype=0, conaffinity=0,
    )


def _add_source(spec: mujoco.MjSpec) -> None:
    """Open tube. Wall boxes leave the cavity empty so the spatula can enter."""
    body = spec.worldbody.add_body(name="source", pos=SRC_REST.tolist())
    body.add_freejoint(name="source_joint")
    n = 8
    mid = (SOURCE_OR + SOURCE_IR) / 2
    thick = (SOURCE_OR - SOURCE_IR) / 2
    chord = SOURCE_OR * np.tan(np.pi / n) + 0.0008
    rgba = [0.62, 0.74, 0.78, 0.55]
    for i in range(n):
        rot = _rz(2 * np.pi * i / n)
        body.add_geom(
            name=f"source_wall_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[thick, chord, SOURCE_H / 2],
            pos=(rot @ np.array([mid, 0.0, SOURCE_H / 2])).tolist(), quat=_quat(rot).tolist(),
            rgba=rgba, mass=0.006, friction=[1.0, 0.005, 0.0001], condim=4,
        )
    body.add_geom(
        name="source_bottom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[SOURCE_OR, 0.002],
        pos=[0.0, 0.0, 0.002], rgba=[0.55, 0.66, 0.70, 0.9], mass=0.012,
    )


def build():
    spec = build_spec("weigh", "robotiq", clutter=False)
    _add_source(spec)
    add_asset(spec, spec.worldbody, "vial_20ml_open", "discard", DISCARD_REST)
    spatula = spec.worldbody.add_body(name="spatula", pos=SPAT_REST.tolist())
    spatula.add_freejoint(name="spatula_joint")
    spatula.add_geom(
        name="spatula_handle", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, 0.02],
        rgba=[0.45, 0.28, 0.14, 1], mass=0.03,
    )
    spatula.add_geom(
        name="spatula_blade", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        fromto=[0.03, 0, 0, BLADE, 0, 0], size=[0.003, 0, 0],
        rgba=[0.75, 0.76, 0.8, 1], mass=0.015,
    )
    for i in range(N_PUCKS):
        puck = spec.worldbody.add_body(name=f"puck_{i}", pos=[1.2, -1.2, 0.05])
        puck.add_freejoint(name=f"puck_{i}")
        puck.add_geom(
            name=f"puck_{i}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.0022, 0.0012],
            mass=PUCK_G / 1000.0, rgba=[0.72, 0.58, 0.32, 1], contype=2, conaffinity=0,
        )
    _add_lcd(spec)
    # Bit 2 has to be set before compile. MuJoCo freezes the collision pairs then,
    # so doing this on the compiled model leaves the pucks with nothing to land on.
    for geom in spec.geoms:
        name = geom.name or ""
        if name == "balance_pan" or name.startswith(("vial_a_collision", "discard_collision")):
            geom.conaffinity = int(geom.conaffinity) | 2
    model = spec.compile()
    data = mujoco.MjData(model)
    set_initial_pose(model, data, "weigh")
    return model, data


def _puck_local(i: int) -> np.ndarray:
    """Non-overlapping stack inside the receiving vial's cavity."""
    layer, slot = divmod(i, 19)
    if slot == 0:
        xy = np.zeros(2)
    elif slot < 7:
        ang = 2 * np.pi * (slot - 1) / 6
        xy = 0.0048 * np.array([np.cos(ang), np.sin(ang)])
    else:
        ang = 2 * np.pi * (slot - 7) / 12 + 0.26
        xy = 0.0094 * np.array([np.cos(ang), np.sin(ang)])
    return np.array([xy[0], xy[1], 0.006 + layer * 0.0028])


class Hold:
    def __init__(self, side: str, adr: int, dof: int, rel_p: np.ndarray, rel_r: np.ndarray):
        self.side, self.adr, self.dof = side, adr, dof
        self.rel_p, self.rel_r = rel_p, rel_r


class Scene:
    """Leo's weigh model plus the scrape props. Duck-types the env Session drives."""

    hand = "robotiq"

    def __init__(self):
        self.model, self.data = build()
        self.n_substeps = max(1, round((1.0 / 20.0) / self.model.opt.timestep))
        m = self.model
        self.top_adr = m.joint("balance_door_top").qposadr[0]
        self.side_adr = [m.joint(f"balance_door_{s}").qposadr[0] for s in ("left", "right")]
        self.vial_adr = m.joint("vial_a_joint").qposadr[0]
        self.vial_dof = m.joint("vial_a_joint").dofadr[0]
        self.free = {
            name: (m.joint(f"{name}_joint").qposadr[0], m.joint(f"{name}_joint").dofadr[0])
            for name in ("source", "discard", "spatula", "vial_a")
        }
        self.puck_adr = [m.joint(f"puck_{i}").qposadr[0] for i in range(N_PUCKS)]
        self.puck_dof = [m.joint(f"puck_{i}").dofadr[0] for i in range(N_PUCKS)]
        self.arm_act = {
            side: np.array([m.actuator(f"{side}_{j}").id for j in ARM_JOINTS])
            for side in ("left", "right")
        }
        self.arm_qadr = {
            side: np.array([m.joint(f"{side}_{j}_joint").qposadr[0] for j in ARM_JOINTS])
            for side in ("left", "right")
        }
        self.pinch = {side: m.site(f"{side}_grip_pinch").id for side in ("left", "right")}
        self.lcd = {
            i: {name: m.geom(f"lcd_{i}_{name}").id for name in _SEG_BOX} for i in range(4)
        }
        self.shield = set()
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name == "balance_pan" or name.startswith("vial_a_collision"):
                m.geom_conaffinity[g] |= 2
            if any(part in name for part in ("glass", "housing", "post")):
                self.shield.add(g)
        self.top_glass = m.geom("balance_door_top_glass").id
        self.handle = m.geom("balance_door_top_handle").id
        self.side_handle = [m.geom(f"balance_door_{name}_handle").id for name in ("left", "right")]
        self.arm_geoms = set()
        for g in range(m.ngeom):
            body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
            if body.startswith("left_") or body.startswith("right_"):
                self.arm_geoms.add(g)
        self._ctrl_low = m.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = m.actuator_ctrlrange[:, 1].copy()
        self.carried: list[Hold] = []
        self.pinned: tuple | None = None
        self.bad: str | None = None
        self.prev_door = 0.0
        self.prev_side = [0.0, 0.0]
        self.max_pen = 0.0
        self.recv_mass_g = 0.0
        self.removed_g = 0.0
        self.on_spatula_g = 0.0
        self.deposit_ok = False
        self.puck_role = np.full(N_PUCKS, -2)
        self._step_count = 0
        self.tare = 0.0
        self.prepare()

    def prepare(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        set_initial_pose(m, d, "weigh")
        d.qpos[self.vial_adr:self.vial_adr + 7] = [*PAN, 1, 0, 0, 0]
        self._write_free("source", SRC_REST, np.eye(3))
        self._write_free("discard", DISCARD_REST, np.eye(3))
        self._write_free("spatula", SPAT_REST, np.eye(3))
        for side, acts in self.arm_act.items():
            d.ctrl[acts] = d.qpos[self.arm_qadr[side]]
        for side in ("left", "right"):
            d.ctrl[m.actuator(f"{side}_grip_fingers_actuator").id] = 0.0
        self.carried = []
        self.pinned = None
        self.recv_mass_g = 0.0
        self.removed_g = 0.0
        self.on_spatula_g = 0.0
        self.deposit_ok = False
        self.puck_role[:] = -2
        self.bad = None
        self.prev_door = 0.0
        self.prev_side = [0.0, 0.0]
        self.max_pen = 0.0
        self._step_count = 0
        for _ in range(400):
            self._place_pucks(force=True)
            mujoco.mj_step(m, d)
        self.tare = pan_reading_g(m, d)
        self.puck_role[:] = -2
        mujoco.mj_forward(m, d)

    def reset(self, seed=None):
        self.prepare()
        return {"state": np.zeros(1, np.float32), "image": np.zeros((1, 1, 3), np.uint8)}, self.task_status()

    def task_status(self) -> dict:
        reading = self.reading()
        return {
            "success": 0, "balance_g": reading, "balance_display": f"{max(reading, 0.0):.4f} g",
            "doors": np.array([
                self.data.qpos[self.side_adr[0]], self.data.qpos[self.side_adr[1]],
                self.data.qpos[self.top_adr],
            ]),
        }

    def reading(self) -> float:
        return pan_reading_g(self.model, self.data) - self.tare

    def show(self, grams: float) -> None:
        text = f"{max(grams, 0.0):5.2f}"
        chars = [text[0], text[1], text[3], text[4]]
        for i, ch in enumerate(chars):
            on = _DIGIT.get(ch, "")
            for name, gid in self.lcd[i].items():
                self.model.geom_rgba[gid, 3] = 1.0 if name in on else 0.0

    def step(self, action):
        d = self.data
        d.ctrl[:] = np.clip(np.asarray(action, float).reshape(-1), self._ctrl_low, self._ctrl_high)
        for _ in range(self.n_substeps):
            self._hold_carried()
            if self.pinned is not None:
                self._write_free(*self.pinned)
            self._place_pucks()
            mujoco.mj_step(self.model, d)
        self._step_count += 1
        if not np.isfinite(d.qpos).all():
            self.bad = "non-finite state"
        return (
            {"state": np.zeros(1, np.float32), "image": np.zeros((1, 1, 3), np.uint8)},
            0.0, False, False, self.task_status(),
        )

    def _write_free(self, name: str, pos, rot) -> None:
        adr, dof = self.free[name]
        self.data.qpos[adr:adr + 3] = pos
        self.data.qpos[adr + 3:adr + 7] = _quat(rot)
        self.data.qvel[dof:dof + 6] = 0.0

    def _hold_carried(self) -> None:
        d = self.data
        for hold in self.carried:
            p = d.site_xpos[self.pinch[hold.side]]
            rot = d.site_xmat[self.pinch[hold.side]].reshape(3, 3)
            adr, dof = hold.adr, hold.dof
            d.qpos[adr:adr + 3] = p + rot @ hold.rel_p
            d.qpos[adr + 3:adr + 7] = _quat(rot @ hold.rel_r)
            d.qvel[dof:dof + 6] = 0.0

    def _receiver_held(self) -> bool:
        return any(hold.adr == self.vial_adr for hold in self.carried)

    def _place_pucks(self, force: bool = False) -> None:
        n_recv = int(np.clip(round(self.recv_mass_g / PUCK_G), 0, N_PUCKS))
        n_removed = int(np.clip(round(self.removed_g / PUCK_G), 0, N_PUCKS - n_recv))
        held = self._receiver_held()
        vial_p = self.data.qpos[self.vial_adr:self.vial_adr + 3]
        vial_q = self.data.qpos[self.vial_adr + 3:self.vial_adr + 7]
        vial_r = np.zeros(9)
        mujoco.mju_quat2Mat(vial_r, vial_q)
        vial_r = vial_r.reshape(3, 3)
        discard_p = self.data.qpos[self.free["discard"][0]:self.free["discard"][0] + 3]
        for i, (adr, dof) in enumerate(zip(self.puck_adr, self.puck_dof)):
            role = 0 if i < n_recv else 1 if i < n_recv + n_removed else -1
            if role == 0 and not held and self.puck_role[i] == 0 and not force:
                continue
            self.puck_role[i] = role
            if role == 0:
                pos, quat = vial_p + vial_r @ _puck_local(i), vial_q
            elif role == 1:
                local = np.array([((i % 5) - 2) * 0.0015, 0.0, 0.02])
                pos, quat = discard_p + local, np.array([1.0, 0.0, 0.0, 0.0])
            else:
                pos = np.array([1.2 + (i % 20) * 0.016, -1.2 - (i // 20) * 0.016, 0.05])
                quat = np.array([1.0, 0.0, 0.0, 0.0])
            self.data.qpos[adr:adr + 3] = pos
            self.data.qpos[adr + 3:adr + 7] = quat
            self.data.qvel[dof:dof + 6] = 0.0

    def watch(self) -> None:
        """Reject a door that moves on its own, a side door that moves, or a body in the glass."""
        if self.bad:
            return
        d, m = self.data, self.model
        door = float(d.qpos[self.top_adr])
        for i, adr in enumerate(self.side_adr):
            opened = float(d.qpos[adr])
            if opened > self.prev_side[i] + 0.002:
                handle = d.geom_xpos[self.side_handle[i]]
                near = min(
                    np.linalg.norm(d.site_xpos[self.pinch[s]] - handle) for s in ("left", "right")
                )
                if near > 0.07:
                    self.bad = "a side door moved with the gripper away from its handle"
                    return
            self.prev_side[i] = opened
        pinch = d.site_xpos[self.pinch["right"]]
        handle = d.geom_xpos[self.handle]
        if door > self.prev_door + 0.002 and np.linalg.norm(pinch - handle) > 0.07:
            self.bad = f"top door moved with the gripper {np.linalg.norm(pinch - handle)*100:.0f} cm away"
            return
        self.prev_door = door
        moving = set(self.arm_geoms)
        for hold in self.carried:
            root = int(m.dof_bodyid[hold.dof])
            for g in range(m.ngeom):
                if int(m.body_rootid[m.geom_bodyid[g]]) == root:
                    moving.add(g)
        for c in d.contact[:d.ncon]:
            if c.dist >= -0.004:
                continue
            pair = {int(c.geom1), int(c.geom2)}
            hit = pair & self.shield
            if not hit or not (pair & moving):
                continue
            geom = next(iter(hit))
            if geom == self.top_glass and c.dist > -0.008:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, geom)
            other = next(iter(pair - hit))
            other_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other) or "arm"
            self.max_pen = min(self.max_pen, float(c.dist))
            self.bad = f"penetration {c.dist*1000:.1f} mm of {other_name} into {name}"
            return

    def snapshot(self, session: Session) -> dict:
        d = self.data
        handle = d.geom_xpos[self.handle]
        pinch = d.site_xpos[self.pinch["right"]]
        return {
            "qpos": d.qpos.copy(),
            "mocap_pos": d.mocap_pos.copy(),
            "mocap_quat": d.mocap_quat.copy(),
            "reading": float(self.reading()),
            "mass": float(self.recv_mass_g),
            "removed": float(self.removed_g),
            "q_left": d.qpos[self.arm_qadr["left"]].copy(),
            "q_right": d.qpos[self.arm_qadr["right"]].copy(),
            "a_left": session.q_cmd["left"].copy(),
            "a_right": session.q_cmd["right"].copy(),
            "grips": np.array([session.grip_cmd["left"], session.grip_cmd["right"]], np.float32),
            "door": float(d.qpos[self.top_adr]),
            "side": np.array([float(d.qpos[self.side_adr[0]]), float(d.qpos[self.side_adr[1]])], np.float32),
            "handle_dist": float(np.linalg.norm(pinch - handle)),
        }


def _tool_for(hold: Hold, origin, rot):
    tool_r = rot @ hold.rel_r.T
    return origin - tool_r @ hold.rel_p, tool_r


def _slew(demo: Demo, side: str, pos, rot, steps: int = 4) -> bool:
    """Short steps. One big turn does not finish before the settle timeout."""
    cur, cur_r = demo.session.arms[side].tool_pose()
    q0, q1, q = np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0, np.asarray(cur_r, float).reshape(-1))
    mujoco.mju_mat2Quat(q1, np.asarray(rot, float).reshape(-1))
    pos = np.asarray(pos, float)
    for i in range(1, steps + 1):
        u = i / steps
        a, b = q0.copy(), q1.copy()
        dot = float(np.dot(a, b))
        if dot < 0.0:
            b, dot = -b, -dot
        if dot > 0.9995:
            q[:] = a + u * (b - a)
        else:
            theta = np.arccos(np.clip(dot, -1.0, 1.0))
            q[:] = (np.sin((1.0 - u) * theta) * a + np.sin(u * theta) * b) / np.sin(theta)
        q[:] /= np.linalg.norm(q)
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, q)
        if not demo.go(side, (1.0 - u) * cur + u * pos, mat.reshape(3, 3)):
            return False
    return True


def _object_pose(data, name):
    pos = data.body(name).xpos.copy()
    return pos, data.body(name).xmat.copy().reshape(3, 3)


def _tip(data) -> np.ndarray:
    pos, rot = _object_pose(data, "spatula")
    return pos + rot @ np.array([BLADE, 0.0, 0.0])


def _inside(point, origin, rot) -> bool:
    local = rot.T @ (point - origin)
    return abs(local[0]) < 0.008 and abs(local[1]) < 0.008 and SOURCE_H - 0.02 < local[2] < SOURCE_H + 0.004


class Demo:
    def __init__(self, scene: Scene, session: Session):
        self.scene, self.session = scene, session
        self.frames: list[dict] = []
        self.ticks = 0
        # Gravity compensation only sees a free body the gripper is touching.
        # The kinematic hold is the record of what the hand is carrying, so keep
        # it in that list for the whole demonstration.
        if not getattr(session, "_carry_patched", False):
            original = session._carried

            def carried():
                found = original()
                known = {body for _, body, *_ in found}
                for hold in scene.carried:
                    body = int(scene.model.dof_bodyid[hold.dof])
                    if body not in known:
                        found.append((hold.side, body, hold.adr, hold.rel_p, hold.rel_r))
                return found

            session._carried = carried
            session._carry_patched = True

    def drain(self) -> bool:
        while self.session.tick():
            self.ticks += 1
            self.scene.watch()
            if self.scene.bad:
                return False
            if self.ticks % 2 == 0:
                self.frames.append(self.scene.snapshot(self.session))
        return not self.scene.bad

    def go(self, side: str, pos, rot=None, _retry: bool = True) -> bool:
        other = "right" if side == "left" else "left"
        self.session.target_to_current(other)
        self.session.set_target(side, np.asarray(pos, float), rot)
        plan = self.session.plan()
        def _harmless(reason: str) -> bool:
            # The top door is already open. A fingertip may meet its rear edge;
            # a real puncture of the glass still fails the penetration watch.
            if "balance door top" in reason and float(self.scene.data.qpos[self.scene.top_adr]) > 0.15:
                return True
            if reason.startswith((
                "left arm touches balance door", "left gripper touches balance door",
                "right arm touches balance door", "right gripper touches balance door",
            )):
                return True
            # Pucks ride inside the held vial. The planner leaves them where they
            # were, so the moving vial is reported as touching its own powder.
            if " touches " not in reason:
                return False
            left_name, right_name = reason.split(" touches ", 1)
            pair = {left_name, right_name}
            return any(name.startswith("puck_") for name in pair) and bool(pair & {"vial_a", "discard"})

        harmless = plan.reasons and all(_harmless(reason) for reason in plan.reasons)
        if plan.status == "blocked" or (plan.status == "contact" and not harmless):
            where = np.array2string(np.asarray(pos, float), precision=3)
            left = np.array2string(self.session.arms["left"].tool_pose()[0], precision=3)
            right = np.array2string(self.session.arms["right"].tool_pose()[0], precision=3)
            elbows = []
            for name in ("left_upper_arm_link", "left_forearm_link", "right_upper_arm_link", "right_forearm_link"):
                try:
                    elbows.append(f"{name[-12:]} {np.array2string(self.scene.data.body(name).xpos, precision=2)}")
                except KeyError:
                    pass
            path_z = plan.path[side][:, 2]
            self.scene.bad = (
                f"{side} {where} {plan.status}: " + "; ".join(plan.reasons)
                + f" | tools L {left} R {right} | path z {path_z.min():.3f}..{path_z.max():.3f}"
                + f" | " + " ".join(elbows)
            )
            return False
        ok, msg = self.session.execute()
        if not ok:
            self.scene.bad = msg
            return False
        if not self.drain():
            return False
        got, got_r = self.session.arms[side].tool_pose()
        asked, asked_r = self.session.targets[side]
        miss = float(np.linalg.norm(got - asked))
        twist = float(rot_angle(got_r, asked_r))
        # The settle window is one second. A long carry can still be a few
        # centimetres out; one more command to the same pose closes it.
        if _retry and miss < 0.025 and np.deg2rad(8) < twist < np.deg2rad(25):
            return self.go(side, asked, asked_r, _retry=False)
        if _retry and miss < 0.08 and twist < np.deg2rad(20) and (miss > 0.05 or twist > np.deg2rad(15)):
            return self.go(side, asked, asked_r, _retry=False)
        if miss > 0.05 or twist > np.deg2rad(12):
            arm = self.session.arms[side]
            qerr = float(np.abs(self.scene.data.qpos[arm.qadr] - self.session.q_cmd[side]).max())
            self.scene.bad = (
                f"{side} stopped short of {np.array2string(asked, precision=3)} "
                f"at {np.array2string(got, precision=3)} "
                f"twist {np.degrees(twist):.0f} deg joints {qerr:.3f} rad"
            )
            return False
        return True

    def grip(self, side: str, close: bool) -> bool:
        ok, msg = self.session.grip(side, close)
        if not ok:
            self.scene.bad = msg
            return False
        return self.drain()

    def idle(self, n: int) -> bool:
        ctrl = self.scene.data.ctrl.copy()
        for _ in range(n):
            self.scene.step(ctrl)
            self.ticks += 1
            self.scene.watch()
            if self.scene.bad:
                return False
            if self.ticks % 2 == 0:
                self.frames.append(self.scene.snapshot(self.session))
        return True

    def move_held(self, hold: Hold, origin, rot, via_xy=None, via_z: float = HIGH_Z) -> bool:
        pos, mat = _tool_for(hold, origin, rot)
        cur, cur_r = self.session.arms[hold.side].tool_pose()
        mid = np.array([pos[0], pos[1], via_z]) if via_xy is None else np.array([via_xy[0], via_xy[1], via_z])
        via = [
            (np.array([cur[0], cur[1], max(cur[2], HIGH_Z)]), cur_r),
            (mid, mat),
            (pos, mat),
        ]
        for waypoint, face in via:
            here, here_r = self.session.arms[hold.side].tool_pose()
            if np.linalg.norm(here - waypoint) < 0.012 and rot_angle(here_r, face) < np.deg2rad(4):
                continue
            if not self.go(hold.side, waypoint, face):
                return False
        return True

    def latch(self, side: str, name: str) -> Hold | None:
        arm = self.session.arms[side]
        root = self.scene.model.body(name).id
        touched = False
        for c in self.scene.data.contact[:self.scene.data.ncon]:
            bodies = [int(self.scene.model.geom_bodyid[g]) for g in (c.geom1, c.geom2)]
            roots = [int(self.scene.model.body_rootid[b]) for b in bodies]
            if root in roots and any(b in arm.grip_bodies for b in bodies):
                touched = True
                break
        if not touched:
            tool_p, _ = arm.tool_pose()
            obj_p, _ = _object_pose(self.scene.data, name)
            self.scene.bad = (
                f"{side} gripper is not touching {name} "
                f"tool {np.array2string(tool_p, precision=3)} "
                f"obj {np.array2string(obj_p, precision=3)}"
            )
            return None
        tool_p, tool_r = arm.tool_pose()
        obj_p, obj_r = _object_pose(self.scene.data, name)
        adr, dof = self.scene.free[name]
        hold = Hold(side, adr, dof, tool_r.T @ (obj_p - tool_p), tool_r.T @ obj_r)
        self.scene.carried.append(hold)
        return hold

    def unlatch(self, side: str) -> None:
        self.scene.carried = [hold for hold in self.scene.carried if hold.side != side]


def _material_vector(material: Material) -> np.ndarray:
    return np.array([getattr(material, name) for name in MATERIAL_NAMES], np.float64)


def _rot_from_z(z_axis: np.ndarray) -> np.ndarray:
    z = np.asarray(z_axis, float)
    z = z / max(np.linalg.norm(z), 1e-8)
    helper = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _slerp_rot(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    q0, q1, q = np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0, np.asarray(a, float).reshape(-1))
    mujoco.mju_mat2Quat(q1, np.asarray(b, float).reshape(-1))
    aa, bb = q0.copy(), q1.copy()
    dot = float(np.dot(aa, bb))
    if dot < 0.0:
        bb, dot = -bb, -dot
    if dot > 0.9995:
        q[:] = aa + u * (bb - aa)
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        q[:] = (np.sin((1.0 - u) * theta) * aa + np.sin(u * theta) * bb) / np.sin(theta)
    q[:] /= np.linalg.norm(q)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, q)
    return mat.reshape(3, 3)


def _rim() -> float:
    return float(PAN[2] + VIAL_H)


def _pose_from_mouth(mouth: np.ndarray, toward_mouth: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray(toward_mouth, float)
    axis = axis / max(np.linalg.norm(axis), 1e-8)
    mouth = np.asarray(mouth, float)
    origin = mouth - axis * SOURCE_H
    return origin, _rot_from_z(axis), mouth


def _scoop_pose() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mouth points at the receiving vial from a few centimetres to its right."""
    mouth = np.array([PAN[0] + 0.030, PAN[1], _rim() + 0.022])
    origin, rot, mouth = _pose_from_mouth(mouth, np.array([-1.0, 0.0, 0.0]))
    return origin, rot @ _rz(np.pi / 2), mouth


def _pour_pose() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mouth over the receiving rim, tilted down from the right door."""
    mouth = np.array([PAN[0] + 0.012, PAN[1], _rim() + 0.020])
    return _pose_from_mouth(mouth, np.array([-0.82, 0.0, -0.57]))


def _spatula_pose(tip: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = np.asarray(direction, float)
    d = d / max(np.linalg.norm(d), 1e-8)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(d, up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0])
    y = np.cross(up, d)
    y /= np.linalg.norm(y)
    z = np.cross(d, y)
    rot = np.column_stack([d, y, z])
    return tip - rot @ np.array([BLADE, 0.0, 0.0]), rot


def _open_door(demo: Demo, arm: str = "right") -> bool:
    """Slide the top door. Used only after the tube has been set down."""
    session = demo.session
    for _ in range(2):
        session.snap(arm, "door")
        if not demo.go(arm, *session.targets[arm]):
            return False
    if not demo.grip(arm, True):
        return False
    session.set_target(arm, session.targets[arm][0] + np.array([0.0, 0.20, 0.0]))
    if not demo.go(arm, *session.targets[arm]):
        return False
    if not demo.grip(arm, False):
        return False
    if float(demo.scene.data.qpos[demo.scene.top_adr]) < 0.15:
        demo.scene.bad = f"top door only reached {demo.scene.data.qpos[demo.scene.top_adr]:.3f} m"
        return False
    session.snap(arm, "home")
    return demo.go(arm, *session.targets[arm])


def _from_side(which: str) -> np.ndarray:
    """Tool points in from that side of the balance. Fingers close front-to-back."""
    sign = 1.0 if which == "right" else -1.0
    z = np.array([-sign, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    return np.column_stack([np.cross(y, z), y, z])


def _open_side_door(demo: Demo, which: str) -> bool:
    """The arm on that side slides the door by its handle. The joint is never written."""
    session = demo.session
    sign = 1.0 if which == "right" else -1.0
    handle = demo.scene.data.geom(f"balance_door_{which}_handle").xpos.copy()
    rot = _from_side(which)
    outside = handle + np.array([sign * 0.07, 0.0, 0.0])
    grasp = handle + np.array([sign * 0.015, 0.0, 0.0])
    for pos in (outside + np.array([0.0, 0.0, 0.10]), outside, grasp):
        if not demo.go(which, pos, rot):
            return False
    if not demo.grip(which, True):
        return False
    if not demo.go(which, grasp + np.array([0.0, 0.18, 0.0]), rot):
        return False
    if not demo.grip(which, False):
        return False
    adr = demo.scene.side_adr[0 if which == "left" else 1]
    opened = float(demo.scene.data.qpos[adr])
    if opened < 0.16:
        demo.scene.bad = f"{which} door only reached {opened:.3f} m"
        return False
    here, here_r = session.arms[which].tool_pose()
    if not demo.go(which, here + np.array([sign * 0.06, 0.0, 0.08]), here_r):
        return False
    session.snap(which, "home")
    return demo.go(which, *session.targets[which])


def _side_demo(demo: Demo, scene: Scene, session: Session) -> list | None:
    """Open the right draft door and bring the spatula in through that opening."""
    print("  open right door", flush=True)
    if not _open_side_door(demo, "right"):
        return None
    print("  pick spatula", flush=True)
    spatula = _pick(demo, "right", "spatula", SPAT_REST, top_down(np.pi / 2), np.array([0.0, 0.0, 0.10]))
    if spatula is None:
        return None
    print("  reach in from the side", flush=True)
    direction = np.array([-1.0, 0.0, 0.0])
    mouth = np.array([PAN[0], PAN[1], PAN[2] + VIAL_H + 0.01])
    # Slide in from outside the door. The blade stays level, so it passes through
    # the opening the door just made instead of through the glass.
    for tip_x in (0.40, 0.32, 0.24, 0.16, mouth[0] + 0.02):
        tip = np.array([tip_x, mouth[1], mouth[2]])
        if not demo.go("right", *_tool_for(spatula, *_spatula_pose(tip, direction))):
            return None
    if not demo.idle(20):
        return None
    opened = float(scene.data.qpos[scene.side_adr[1]])
    tip = _tip(scene.data)
    print(
        f"  done side door={opened:.3f} tip {np.array2string(tip, precision=3)} pen={scene.max_pen*1000:.1f} mm",
        flush=True,
    )
    if scene.bad:
        return None
    if opened < 0.12:
        scene.bad = "right door did not stay open"
        return None
    if abs(tip[0] - mouth[0]) > 0.04 or abs(tip[1] - mouth[1]) > 0.03:
        scene.bad = f"spatula tip missed the vial ({tip})"
        return None
    return demo.frames


def _pick(demo: Demo, side: str, name: str, pinch, rot, approach) -> Hold | None:
    pinch = np.asarray(pinch, float)
    cur, cur_r = demo.session.arms[side].tool_pose()
    via = [
        (np.array([cur[0], cur[1], HIGH_Z]), cur_r),
        (np.array([pinch[0], pinch[1], HIGH_Z]), rot),
        (pinch + approach, rot),
        (pinch, rot),
    ]
    for pos, face in via:
        here, here_r = demo.session.arms[side].tool_pose()
        if np.linalg.norm(here - pos) < 0.012 and rot_angle(here_r, face) < np.deg2rad(4):
            continue
        if not demo.go(side, pos, face):
            return None
    obj_p, obj_r = _object_pose(demo.scene.data, name)
    demo.scene.pinned = (name, obj_p, obj_r)
    if not demo.grip(side, True):
        demo.scene.pinned = None
        return None
    hold = demo.latch(side, name)
    demo.scene.pinned = None
    if hold is not None and np.linalg.norm(hold.rel_p) > 0.13:
        demo.scene.bad = f"{side} grasp of {name} slipped ({np.linalg.norm(hold.rel_p)*100:.0f} cm)"
        return None
    return hold


def _source_mouth(data) -> np.ndarray:
    origin, rot = _object_pose(data, "source")
    return origin + rot @ np.array([0.0, 0.0, SOURCE_H])


def _mouth_close(mouth: np.ndarray) -> bool:
    """The source lip is a few centimetres from the receiving rim, not a hand-span above it."""
    gap = float(mouth[2] - _rim())
    return float(np.linalg.norm(mouth[:2] - PAN[:2])) < 0.06 and 0.004 < gap < 0.050


def _dose(scene: Scene, material: Material, goal: float) -> bool:
    rate = material.flow_g_per_s * (1.0 - material.cohesion) * 0.55
    room = (goal - 0.01) - scene.recv_mass_g
    if room <= 0.0:
        return False
    scene.recv_mass_g += min(rate, room)
    return True


def _transfer_in(scene: Scene, material: Material, goal: float) -> bool:
    origin, rot = _object_pose(scene.data, "source")
    if not _mouth_close(_source_mouth(scene.data)):
        return False
    if not _inside(_tip(scene.data), origin, rot):
        return False
    return _dose(scene, material, goal)


def _pour_in(scene: Scene, material: Material, goal: float) -> bool:
    _, rot = _object_pose(scene.data, "source")
    mouth = _source_mouth(scene.data)
    if (rot @ np.array([0.0, 0.0, 1.0]))[2] > -0.2:
        return False
    if not _mouth_close(mouth):
        return False
    return _dose(scene, material, goal)


def _arrive_held(demo: Demo, hold: Hold, origin, rot, tol: float = 0.016) -> bool:
    origin = np.asarray(origin, float)
    command = origin.copy()
    got = origin.copy()
    for _ in range(3):
        if not demo.go(hold.side, *_tool_for(hold, command, rot)):
            return False
        got, got_r = _object_pose(demo.scene.data, "source")
        err = origin - got
        if float(np.linalg.norm(err)) < tol and float(rot_angle(got_r, rot)) < np.deg2rad(12):
            return True
        command = command + err
    demo.scene.bad = (
        f"source stopped {np.linalg.norm(origin - got) * 1000:.0f} mm from "
        f"{np.array2string(origin, precision=3)} at {np.array2string(got, precision=3)}"
    )
    return False


def _carry_held(demo: Demo, hold: Hold, origin, rot, step: float = 0.07) -> bool:
    cur, cur_r = _object_pose(demo.scene.data, "source")
    origin = np.asarray(origin, float)
    dist = float(np.linalg.norm(origin - cur))
    ang = float(rot_angle(cur_r, rot))
    n = max(int(np.ceil(dist / step)), int(np.ceil(ang / np.deg2rad(25))), 1)
    for i in range(1, n + 1):
        u = i / n
        if not _arrive_held(demo, hold, (1.0 - u) * cur + u * origin, _slerp_rot(cur_r, rot, u)):
            return False
    return True


def _pick_above(demo: Demo, side: str, name: str) -> Hold | None:
    origin, _ = _object_pose(demo.scene.data, name)
    pinch = origin + np.array([0.0, 0.0, 0.034])
    face = top_down(0.0)
    cur, _ = demo.session.arms[side].tool_pose()
    stages = [np.array([cur[0], cur[1], 1.12])]
    for y in np.linspace(cur[1], pinch[1], 4)[1:]:
        stages.append(np.array([pinch[0], float(y), 1.12]))
    stages += [pinch + np.array([0.0, 0.0, 0.08]), pinch]
    for pos in stages:
        if not demo.go(side, pos, face):
            return None
    obj_p, obj_r = _object_pose(demo.scene.data, name)
    demo.scene.pinned = (name, obj_p, obj_r)
    if not demo.grip(side, True):
        demo.scene.pinned = None
        return None
    hold = demo.latch(side, name)
    demo.scene.pinned = None
    return hold


def _regrasp_side(demo: Demo) -> Hold | None:
    """Move the hand off the mouth so the spatula can enter from the side."""
    origin, rot = _object_pose(demo.scene.data, "source")
    demo.scene.pinned = ("source", origin.copy(), rot.copy())
    if not demo.grip("left", False):
        demo.scene.pinned = None
        return None
    demo.unlatch("left")
    here, here_r = demo.session.arms["left"].tool_pose()
    # Reach in through the open left door. The front pane does not move.
    z = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    face = np.column_stack([np.cross(y_axis, z), y_axis, z])
    pinch = origin + np.array([-(SOURCE_OR + 0.002), 0.0, SOURCE_H * 0.35])
    outside = pinch + np.array([-0.08, 0.0, 0.04])
    for pos in (here + np.array([0.0, 0.0, 0.08]), outside + np.array([0.0, 0.0, 0.06]), outside, pinch):
        if not demo.go("left", pos, face if not np.allclose(pos, here + np.array([0.0, 0.0, 0.08])) else here_r):
            demo.scene.pinned = None
            return None
    if not demo.grip("left", True):
        demo.scene.pinned = None
        return None
    hold = demo.latch("left", "source")
    demo.scene.pinned = None
    return hold


def _deliver_source(demo: Demo, hold: Hold, origin, rot) -> Hold | None:
    cur, cur_r = _object_pose(demo.scene.data, "source")
    sign = 1.0 if hold.side == "right" else -1.0
    clear = np.array([sign * 0.50, -0.12, 1.18])
    print("  lift source", flush=True)
    if not _carry_held(demo, hold, clear, cur_r):
        return None
    print("  turn source toward the vial", flush=True)
    if not _carry_held(demo, hold, clear, rot):
        return None
    print("  carry in through the side door", flush=True)
    low = np.array([clear[0], clear[1], float(origin[2])])
    if not _carry_held(demo, hold, low, rot):
        return None
    beside = np.array([sign * 0.22, float(origin[1]), float(origin[2])])
    if not _carry_held(demo, hold, beside, rot):
        return None
    if not _carry_held(demo, hold, origin, rot):
        return None
    mouth = _source_mouth(demo.scene.data)
    print(f"  mouth {np.array2string(mouth, precision=3)} gap {(mouth[2] - _rim()) * 1000:.0f} mm", flush=True)
    if not _mouth_close(mouth):
        demo.scene.bad = f"source mouth is not at the vial ({np.array2string(mouth, precision=3)})"
        return None
    return hold


def _scrape(demo: Demo, spatula: Hold, material: Material, goal: float) -> bool:
    inside = steps = 0
    for wiggle in range(40):
        if demo.scene.recv_mass_g >= goal - 0.02:
            break
        origin, rot = _object_pose(demo.scene.data, "source")
        axis = rot @ np.array([0.0, 0.0, 1.0])
        mouth = origin + axis * SOURCE_H
        direction = -axis
        side = rot @ np.array([0.0, 1.0, 0.0])
        tip = mouth - axis * 0.008 + side * (0.003 if wiggle % 2 == 0 else -0.003)
        if wiggle == 0:
            for dist in (0.20, 0.12, 0.05):
                staged = mouth + axis * dist
                if not demo.go(spatula.side, *_tool_for(spatula, *_spatula_pose(staged, direction))):
                    return False
        if not demo.go(spatula.side, *_tool_for(spatula, *_spatula_pose(tip, direction))):
            return False
        steps += 1
        if _inside(_tip(demo.scene.data), origin, rot):
            inside += 1
            _transfer_in(demo.scene, material, goal)
        if not demo.idle(3):
            return False
    else:
        demo.scene.bad = f"scrape stalled at {demo.scene.recv_mass_g:.2f} g"
        return False
    if steps == 0 or inside / steps < 0.5:
        demo.scene.bad = f"tip inside the source on {inside}/{steps} scrapes"
        return False
    return True


def _set_tube_down(demo: Demo, session: Session) -> bool:
    """Carry the source back out of the left door and stand it on the bench."""
    print("  set source on the bench", flush=True)
    hold = next((item for item in demo.scene.carried if item.side == "left"), None)
    if hold is None:
        demo.scene.bad = "left arm is not holding the source"
        return False
    if not _carry_held(demo, hold, SRC_REST + np.array([0.0, 0.0, 0.04]), np.eye(3)):
        return False
    if not demo.grip("left", False):
        return False
    demo.unlatch("left")
    session.snap("left", "home")
    return demo.go("left", *session.targets["left"])


def _withdraw_spatula(demo: Demo, spatula: Hold) -> bool:
    print("  withdraw spatula", flush=True)
    src_p, src_r = _object_pose(demo.scene.data, "source")
    axis = src_r @ np.array([0.0, 0.0, 1.0])
    mouth = src_p + axis * SOURCE_H
    pos, mat = _tool_for(spatula, *_spatula_pose(mouth + axis * 0.04, -axis))
    return demo.go("right", pos, mat)


def _transfer_out(scene: Scene, target: float) -> None:
    take = min(0.20, scene.recv_mass_g - (target - 0.01))
    if take > 0:
        scene.recv_mass_g -= take
        scene.removed_g += take


def _joint_move(demo: Demo, side: str, goal: np.ndarray) -> bool:
    """Move an empty arm in joint space. Cartesian IK cannot untwist this wrist."""
    session, scene = demo.session, demo.scene
    arm = session.arms[side]
    start = scene.data.qpos[arm.qadr].copy()
    scratch, m = session._scratch, scene.model
    for i in range(1, 31):
        q = start + (goal - start) * (i / 30.0)
        scratch.qpos[:] = scene.data.qpos
        scratch.qpos[arm.qadr] = q
        mujoco.mj_kinematics(m, scratch)
        mujoco.mj_collision(m, scratch)
        for con in scratch.contact[:scratch.ncon]:
            if con.dist >= -0.004:
                continue
            names = [
                mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or ""
                for g in (con.geom1, con.geom2)
            ]
            if not any(n.startswith(f"{side}_") for n in names):
                continue
            other = names[0] if names[1].startswith(f"{side}_") else names[1]
            if other.startswith(("left_", "right_")):
                continue
            scene.bad = f"joint move would put {side} through {other}"
            return False
        session.q_cmd[side] = q
        for _ in range(4):
            ff = session.gravity_ff()
            ctrl = scene.data.ctrl.copy()
            for s, a in session.arms.items():
                ctrl[a.arm_act] = session.q_cmd[s] + ff[s]
                ctrl[a.eff_act] = session.grip_cmd[s]
            scene.step(np.clip(ctrl, session.lo, session.hi))
            demo.ticks += 1
            scene.watch()
            if scene.bad:
                return False
            if demo.ticks % 2 == 0:
                demo.frames.append(scene.snapshot(session))
    session.q_cmd[side] = goal
    for _ in range(25):
        ff = session.gravity_ff()
        ctrl = scene.data.ctrl.copy()
        for s, a in session.arms.items():
            ctrl[a.arm_act] = session.q_cmd[s] + ff[s]
            ctrl[a.eff_act] = session.grip_cmd[s]
        scene.step(np.clip(ctrl, session.lo, session.hi))
        demo.ticks += 1
        scene.watch()
        if scene.bad:
            return False
    err = float(np.abs(scene.data.qpos[arm.qadr] - goal).max())
    if err > 0.12:
        scene.bad = f"{side} did not reach the joint target ({err:.2f} rad)"
        return False
    pos, rot = arm.tool_pose()
    session.set_target(side, pos, rot)
    return True


def _tip_at_vial(tip: np.ndarray) -> bool:
    gap = float(tip[2] - _rim())
    return float(np.linalg.norm(tip[:2] - PAN[:2])) < 0.03 and 0.004 < gap < 0.04


def _open_shield(demo: Demo) -> bool:
    """Slide the top and both side doors toward the back. Front and rear panes stay shut."""
    print("  open top", flush=True)
    if not _open_door(demo, "right"):
        return False
    print("  open left door", flush=True)
    if not _open_side_door(demo, "left"):
        return False
    print("  open right door", flush=True)
    return _open_side_door(demo, "right")


def _beside_pan() -> tuple[np.ndarray, np.ndarray]:
    """Upright source just left of the receiving vial, mouth a couple of centimetres above it."""
    mouth = np.array([PAN[0] - 0.055, PAN[1], _rim() + 0.018])
    # Yaw the round vial so the fingers close front-to-back, not into the receiving vial.
    return mouth - np.array([0.0, 0.0, SOURCE_H]), _rz(np.pi / 2)


def _carry_through_top(demo: Demo, hold: Hold, origin, rot) -> bool:
    """Cross above the shield, then drop through the open top. Stay off the front pane and the corner posts."""
    cur, cur_r = _object_pose(demo.scene.data, "source")
    origin = np.asarray(origin, float)
    high = np.array([cur[0], cur[1], 1.28])
    above = np.array([origin[0], origin[1], 1.28])
    print("  lift source", flush=True)
    if not _carry_held(demo, hold, high, cur_r):
        return False
    print("  cross above the open top", flush=True)
    if not _carry_held(demo, hold, above, rot):
        return False
    print("  lower through the top opening", flush=True)
    return _carry_held(demo, hold, origin, rot)


def _go_tip(demo: Demo, spatula: Hold, tip, direction) -> bool:
    """Command the blade tip. A carried spatula sags, so the next command takes up the miss."""
    tip = np.asarray(tip, float)
    command = tip.copy()
    for _ in range(3):
        if not demo.go("right", *_tool_for(spatula, *_spatula_pose(command, direction))):
            return False
        err = tip - _tip(demo.scene.data)
        if float(np.linalg.norm(err)) < 0.008:
            return True
        command = command + err
    return True


def _over_mouth(point, origin, rot) -> bool:
    local = rot.T @ (point - origin)
    return abs(local[0]) < 0.012 and abs(local[1]) < 0.012 and SOURCE_H - 0.015 < local[2] < SOURCE_H + 0.012


def _shown(scene: Scene) -> float:
    """Scale reading once powder is dynamic. Without pucks, the accounted mass."""
    if getattr(scene, "no_pucks", False):
        return scene.recv_mass_g
    return scene.reading()


def _scoop_transfer(demo: Demo, spatula: Hold, material: Material, goal: float) -> bool:
    """One scrape through the open top, then hold the blade over the vial until the scale is in band."""
    scene = demo.scene
    cur, cur_r = demo.session.arms["right"].tool_pose()
    if not demo.go("right", np.array([cur[0], cur[1], 1.20]), cur_r):
        return False
    origin, rot = _object_pose(scene.data, "source")
    mouth = origin + rot @ np.array([0.0, 0.0, SOURCE_H])
    print(f"  scrape at {np.array2string(mouth, precision=3)}", flush=True)
    into = np.array([0.0, 0.0, -1.0])
    lip_z = float(mouth[2] + 0.003)
    for height in (lip_z + 0.08, lip_z + 0.03):
        tip = np.array([mouth[0], mouth[1], height])
        if not demo.go("right", *_tool_for(spatula, *_spatula_pose(tip, into))):
            return False
    if not _go_tip(demo, spatula, np.array([mouth[0], mouth[1], lip_z]), into):
        return False
    if not _over_mouth(_tip(scene.data), origin, rot):
        scene.bad = "spatula did not meet the source mouth"
        return False
    for tip in (
        np.array([mouth[0], mouth[1], lip_z + 0.06]),
        np.array([PAN[0], PAN[1], lip_z + 0.06]),
        np.array([PAN[0], PAN[1], _rim() + 0.016]),
    ):
        if not demo.go("right", *_tool_for(spatula, *_spatula_pose(tip, into))):
            return False
    if not _tip_at_vial(_tip(scene.data)):
        scene.bad = f"spatula tip is not at the vial ({np.array2string(_tip(scene.data), precision=3)})"
        return False
    scene.deposit_ok = True
    rate = max(0.12, material.flow_g_per_s * (1.0 - material.cohesion) * 0.2)
    print(f"  dose at {rate:.2f} g/step toward {goal:.2f}", flush=True)
    stable = 0
    for step in range(80):
        if not demo.idle(4):
            return False
        shown = _shown(scene)
        if goal - 0.03 <= shown <= goal + 0.03:
            stable += 1
            if stable >= 3:
                break
            continue
        stable = 0
        if shown > goal + 0.03:
            break
        scene.recv_mass_g += min(rate, max(0.06, goal - shown))
        if step % 8 == 0:
            print(f"    scale {shown:.3f} mass {scene.recv_mass_g:.3f}", flush=True)
    else:
        if not (goal - 0.05 <= _shown(scene) <= goal + 0.05):
            scene.bad = f"scoop stalled at {_shown(scene):.2f} g"
            return False
    return True


def rollout(scene: Scene, session: Session, material: Material, target: float, mode: str) -> list | None:
    session.reset()
    demo = Demo(scene, session)
    demo.frames.append(scene.snapshot(session))
    strategy = "pour" if mode == "pour" else "scoop"
    if not _open_shield(demo):
        return None
    print("  pick source", flush=True)
    source = _pick_above(demo, "left", "source")
    if source is None:
        return None
    if strategy == "pour":
        origin, rot, _mouth = _pour_pose()
    else:
        origin, rot = _beside_pan()
    print("  hold the source vial", flush=True)
    if not _carry_through_top(demo, source, origin, rot):
        return None
    mouth = _source_mouth(scene.data)
    print(f"  source mouth {np.array2string(mouth, precision=3)}", flush=True)
    if strategy == "scoop":
        print("  regrasp beside the mouth", flush=True)
        source = _regrasp_side(demo)
        if source is None:
            return None
    if strategy == "pour":
        if not _mouth_close(mouth):
            scene.bad = f"pour mouth is not at the vial ({np.array2string(mouth, precision=3)})"
            return None
        print(f"  pour toward {target:.2f}", flush=True)
        for _ in range(28):
            if scene.recv_mass_g >= target - 0.02:
                break
            _pour_in(scene, material, target)
            if not demo.idle(4):
                return None
        else:
            scene.bad = f"pour stalled at {scene.recv_mass_g:.2f} g"
            return None
        scene.deposit_ok = _mouth_close(_source_mouth(scene.data))
    else:
        print("  pick spatula", flush=True)
        spatula = _pick(
            demo, "right", "spatula", SPAT_REST, top_down(np.pi / 2), np.array([0.0, 0.0, 0.10]),
        )
        if spatula is None:
            return None
        print(f"  scoop toward {target:.2f}", flush=True)
        if not _scoop_transfer(demo, spatula, material, target):
            return None
    if not demo.idle(20):
        return None
    reading = scene.reading()
    print(
        f"  done read={reading:.3f} mass={scene.recv_mass_g:.3f} "
        f"pen={scene.max_pen * 1000:.1f} mm",
        flush=True,
    )
    if scene.bad:
        return None
    if not getattr(scene, "no_pucks", False) and abs(reading - target) > TOL_G:
        scene.bad = f"reading {reading:.3f} vs target {target:.3f}"
        return None
    if float(scene.data.qpos[scene.top_adr]) < 0.15:
        scene.bad = "top door did not stay open"
        return None
    for name, adr in (("left", scene.side_adr[0]), ("right", scene.side_adr[1])):
        opened = float(scene.data.qpos[adr])
        if opened < 0.16:
            scene.bad = f"{name} door did not stay open ({opened:.3f})"
            return None
    if not scene.deposit_ok:
        scene.bad = "transfer did not finish at the vial"
        return None
    return demo.frames


def _dones(frame, target) -> np.ndarray:
    side = frame.get("side")
    door = frame["door"] > 0.08 or (side is not None and float(np.max(side)) > 0.08)
    poured = frame["mass"] > 0.05
    weighed = abs(frame["reading"] - target) <= TOL_G and door
    return np.array([door, poured, weighed], np.float32)


def _action(frame) -> np.ndarray:
    return np.concatenate(
        [frame["a_left"], [frame["grips"][0]], frame["a_right"], [frame["grips"][1]]]
    ).astype(np.float32)


def render_images(scene: Scene, renderer, frame) -> dict[str, np.ndarray]:
    _restore(scene, frame)
    out = {}
    for role, cam in (("top1", "top1"), ("left_wrist", "left_grip_wrist"), ("right_wrist", "right_grip_wrist")):
        renderer.update_scene(scene.data, camera=cam)
        out[role] = renderer.render().copy()
    return out


def _restore(scene: Scene, frame) -> None:
    scene.data.qpos[:] = frame["qpos"]
    scene.data.qvel[:] = 0
    if len(scene.data.mocap_pos):
        scene.data.mocap_pos[:] = frame["mocap_pos"]
        scene.data.mocap_quat[:] = frame["mocap_quat"]
    scene.show(frame["reading"])
    mujoco.mj_forward(scene.model, scene.data)


def _write_demo(group, demo_frames, images, mode, target, powder_id, material) -> None:
    for role, frames in images.items():
        group.create_dataset(f"obs/{role}", data=np.stack(frames), compression="gzip", compression_opts=4)
    low = np.stack([
        np.concatenate([
            f["q_left"], f["q_right"], f["grips"],
            [f["door"], float(f["side"][0]), float(f["side"][1]), f["reading"], target, f["reading"] - target],
        ])
        for f in demo_frames
    ]).astype(np.float32)
    group.create_dataset("obs/lowdim", data=low)
    group.create_dataset("actions", data=np.stack([_action(f) for f in demo_frames]).astype(np.float32))
    group.create_dataset("subgoal_dones", data=np.stack([_dones(f, target) for f in demo_frames]).astype(np.float32))
    group.create_dataset("qpos", data=np.stack([f["qpos"] for f in demo_frames]))
    group.create_dataset("mocap_pos", data=np.stack([f["mocap_pos"] for f in demo_frames]))
    group.create_dataset("mocap_quat", data=np.stack([f["mocap_quat"] for f in demo_frames]))
    group.attrs["success"] = 1
    group.attrs["num_samples"] = len(demo_frames)
    group.attrs["mode"] = mode
    group.attrs["target_g"] = target
    group.attrs["reading_g"] = demo_frames[-1]["reading"]
    group.attrs["mass_g"] = demo_frames[-1]["mass"]
    group.attrs["removed_g"] = demo_frames[-1]["removed"]
    group.attrs["powder_id"] = powder_id
    group.create_dataset("material", data=_material_vector(material))
    group.attrs["material_names"] = list(MATERIAL_NAMES)
    group.attrs["approach"] = (
        "left arm lowers the vial through the open top, regrasps through the left door; "
        "right arm scrapes down through the open top"
    )
    group.attrs["transfer"] = "scrape"
    group.attrs["overshoot"] = 0
    group.attrs["correction"] = "none"
    group.attrs["source_h_m"] = SOURCE_H
    group.attrs["door_opened_by"] = "grippers slide the top, left, and right doors toward the back"
    group.attrs["arm_collision"] = 1
    if mode == "pour":
        instruction = (
            "Hold the source vial through the draft door and pour into the receiving vial "
            f"until the scale reads the target mass. Target {target:.2f} g."
        )
    else:
        instruction = (
            "Hold the source vial and scrape powder across into the receiving vial "
            f"until the scale reads the target mass. Target {target:.2f} g."
        )
    group.attrs["instruction"] = instruction
    for key, value in material.as_dict().items():
        group.attrs[key] = value


def _write_metadata_table(path: Path, rows: list[dict]) -> None:
    fields = [
        "demo", "approach", "hold_arm", "tool_arm", "doors", "transfer",
        "overshoot", "correction", "target_g", "reading_g", "mass_g", "removed_g",
        "powder_id", *MATERIAL_NAMES, "material_vector", "steps", "penetration_mm", "source_h_m",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    print(f"wrote {path}", flush=True)


def _write_preview_video(scene: Scene, frames: list, name: str) -> None:
    import imageio.v2 as imageio
    from render_leo_scrape import label

    path = ROOT / "data" / "teleop" / "scrape_videos" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(scene.model, 360, 480)
    cameras = (("top1", "top1"), ("left wrist", "left_grip_wrist"), ("right wrist", "right_grip_wrist"))
    with imageio.get_writer(path, fps=10, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames:
            _restore(scene, frame)
            panels = []
            for title, cam in cameras:
                renderer.update_scene(scene.data, camera=cam)
                panels.append(label(renderer.render(), title))
            writer.append_data(np.concatenate(panels, axis=1))
    print(f"wrote {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--target", type=float, default=2.5)
    parser.add_argument("--preview", choices=("scoop", "pour", "clean"), default=None)
    parser.add_argument("--no-pucks", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "teleop" / "scrape_vial_bc.hdf5")
    args = parser.parse_args()
    scene = Scene()
    session = Session(scene)
    print(f"timestep {scene.model.opt.timestep} substeps {scene.n_substeps} tare {scene.tare:.3f} g", flush=True)
    if args.no_pucks:
        scene.no_pucks = True
        scene._place_pucks = lambda force=False: None
    if args.preview:
        frames = rollout(scene, session, POWDERS[0], args.target, "scoop" if args.preview == "clean" else args.preview)
        if frames is None:
            print("preview", args.preview, "FAIL", scene.bad)
        else:
            print("preview", args.preview, "OK", f"T={len(frames)} read={frames[-1]['reading']:.3f}")
            if args.no_pucks:
                return
            name = "pour_close.mp4" if args.preview == "pour" else "weigh_full.mp4"
            _write_preview_video(scene, frames, name)
            if args.preview != "pour":
                out = ROOT / "data" / "teleop" / "weigh_full_demo.hdf5"
                out.parent.mkdir(parents=True, exist_ok=True)
                renderer = mujoco.Renderer(scene.model, IMAGE, IMAGE)
                images = {role: [] for role in ("top1", "left_wrist", "right_wrist")}
                for frame in frames:
                    shot = render_images(scene, renderer, frame)
                    for role in images:
                        images[role].append(shot[role])
                with h5py.File(out, "w") as handle:
                    handle.attrs["collector"] = "chem_rl_sim_weigh_vial"
                    handle.attrs["scene"] = "chem_rl_sim weigh layout, robotiq, collisions on, grippers open the side doors"
                    handle.attrs["tol_g"] = TOL_G
                    handle.attrs["material_names"] = list(MATERIAL_NAMES)
                    group = handle.create_group("data/demo_0")
                    _write_demo(group, frames, images, "scoop", 2.5, 0, POWDERS[0])
                print(f"wrote {out}", flush=True)
        return
    rng = np.random.default_rng(0)
    renderer = mujoco.Renderer(scene.model, IMAGE, IMAGE)
    partial = args.out.with_suffix(".partial.hdf5")
    if partial.exists():
        partial.unlink()
    kept = []
    attempts = 0
    instruction = "Hold the source vial and transfer powder into the receiving vial until the scale reads the target mass."
    videos = set()
    with h5py.File(partial, "w") as handle:
        handle.attrs["collector"] = "chem_rl_sim_weigh_vial"
        handle.attrs["scene"] = "chem_rl_sim weigh layout, robotiq, collisions on, grippers open the side doors"
        handle.attrs["tol_g"] = TOL_G
        handle.attrs["instruction"] = instruction
        handle.attrs["material_names"] = list(MATERIAL_NAMES)
        data = handle.create_group("data")
        while len(kept) < args.n and attempts < args.n * 4:
            attempts += 1
            mode = "scoop"
            powder_id = len(kept) % len(POWDERS)
            powder = POWDERS[powder_id]
            target = float(rng.uniform(1.3, 5.5))
            print(f"attempt {attempts} {mode} powder {powder_id} target {target:.2f}", flush=True)
            frames = rollout(scene, session, powder, target, mode)
            if frames is None:
                print(f"  miss {scene.bad}", flush=True)
                continue
            images = {role: [] for role in ("top1", "left_wrist", "right_wrist")}
            for frame in frames:
                shot = render_images(scene, renderer, frame)
                for role in images:
                    images[role].append(shot[role])
            group = data.create_group(f"demo_{len(kept)}")
            _write_demo(group, frames, images, mode, target, powder_id, powder)
            handle.flush()
            vector = _material_vector(powder)
            row = {
                "demo": f"demo_{len(kept)}",
                "approach": "top then left door",
                "hold_arm": "left",
                "tool_arm": "right",
                "doors": "top, left, right",
                "transfer": "scrape",
                "overshoot": "no",
                "correction": "none",
                "target_g": round(target, 4),
                "reading_g": round(float(frames[-1]["reading"]), 4),
                "mass_g": round(float(frames[-1]["mass"]), 4),
                "removed_g": round(float(frames[-1]["removed"]), 4),
                "powder_id": powder_id,
                "material_vector": vector.tolist(),
                "steps": len(frames),
                "penetration_mm": round(float(scene.max_pen) * 1000.0, 2),
                "source_h_m": SOURCE_H,
            }
            row.update({name: float(vector[i]) for i, name in enumerate(MATERIAL_NAMES)})
            kept.append(row)
            print(f"kept {len(kept):02d}/{args.n} {mode} T={len(frames)}", flush=True)
            if len(kept) in (1, 6, 12, 18) or mode not in videos:
                _write_preview_video(scene, frames, f"demo_{len(kept) - 1:02d}_{mode}.mp4")
                videos.add(mode)
                if len(kept) == 1:
                    _write_preview_video(scene, frames, "weigh_full.mp4")
    if len(kept) < args.n:
        raise SystemExit(f"only {len(kept)} successes in {attempts} attempts ({partial})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    partial.replace(args.out)
    summary = list(kept)
    args.out.with_suffix(".json").write_text(json.dumps({
        "attempts": attempts,
        "material_names": list(MATERIAL_NAMES),
        "approach": "Left arm holds a vial-sized source. It enters through the open top and regrasps through the open left door. Right arm scrapes through the open top, a few centimetres above the receiving mouth. Front and rear panes stay shut.",
        "demos": summary,
    }, indent=2))
    _write_metadata_table(args.out.with_suffix(".csv"), kept)
    err = np.array([abs(row["reading_g"] - row["target_g"]) for row in kept])
    print(f"wrote {args.out}\nkept {len(kept)}/{attempts}  abs error mean {err.mean():.4f} max {err.max():.4f}")


if __name__ == "__main__":
    main()
