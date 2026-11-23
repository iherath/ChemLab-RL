"""Parameterized closed-loop skills + verifiers for subtask 2 (SkillWeaver-lite).

Every motion goes through ``SolutionPrepEnv.step`` with a 14-D joint/gripper
action produced by Leo's keyframe ``Session`` (IK + min-jerk plans, or its
speed-capped follow mode). Nothing is teleported or attached; the env audits
every physics substep. The expert reads privileged state (liquid volumes,
object poses) to close its loops; the recorded observations do not include it.

Skill = function(expert, params) -> None, raising ``SkillFailed(reason)`` when
its post-condition verifier fails. ``run_expert`` chains them into one episode.
``EpisodeFinished`` is raised by the env bridge when the env terminates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from chemlab_rl.solution_env import SolutionPrepEnv, plan_draws
from chemlab_rl.solution_scene import (BENCH, CAP_HALF_H, HUB_L, NEEDLE_L, STAND_COLLAR, STAND_H,
                                       THREAD_PITCH, THREAD_TURNS, VIAL_FLOOR, VIAL_H)
from chemlab_rl.solution_state import SYRINGES, TOTAL_VOLUME_ML, VOLUME_TOL_ML

TIP = HUB_L + NEEDLE_L         # barrel bottom -> needle tip


class SkillFailed(Exception):
    pass


class EpisodeFinished(Exception):
    pass


def _face(z, y):
    z = np.asarray(z, float); z /= np.linalg.norm(z)
    y = np.asarray(y, float); y = y - z * np.dot(y, z); y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


LEFT_SIDE = _face([1, 0, 0], [0, 1, 0])     # left hand from -x, jaws close along y
RIGHT_FRONT = _face([0, 1, 0], [1, 0, 0])   # right hand from the robot side, jaws along x
# Syringe roles: the RIGHT hand carries the barrel (grasped from the front in the one
# pocket it reaches well, then moved only by local servo); the LEFT hand touches the
# syringe only to pull/push the plunger (it reaches the whole left work area).
BARREL, PLUNGER = "right", "left"
BARREL_ROT = RIGHT_FRONT
TOP_DOWN = _face([0, 0, -1], [0, 1, 0])     # pointing down, jaws close along y
TOP_DOWN_X = _face([0, 0, -1], [1, 0, 0])   # pointing down, jaws along x (wrist mount off the balance)


# ------------------------------------------------------------------ parameters
PARAM_RANGES = {
    "left_standoff": (0.05, 0.09),      # m, pre-grasp distance along the approach axis
    "right_standoff": (0.05, 0.08),
    "lift": (0.06, 0.09),               # m, syringe lift out of the rack
    "hover": (0.015, 0.03),             # m, needle tip above a rim before inserting
    "res_tip_above_floor": (0.008, 0.014),
    "vial_tip_below_rim": (0.035, 0.045),   # deep (still above 3 mL of liquid): hands stay low
    "follow_speed": (0.8, 1.2),         # Session speed scale for fine moves
    "cap_carry_h": (0.05, 0.08),
    "twist_rate": (2.0, 3.0),           # rad/s of cap rotation while screwing
    "lift_clear": (0.03, 0.05),         # slow first pull out of the clamp pads
    "cam_dist": (0.075, 0.09),          # wrist camera -> vial axis at inspection
    "cam_dz": (0.006, 0.014),           # camera above the aim point (looks slightly down)
    "shake_axis_jitter": (-12.0, 12.0), # deg off the axis perpendicular to the jaws (see shake_burst)
    "burst_s": (2.5, 3.5),              # shake burst between looks
    "start_level": (0.0, 1.999),        # first vigour level (floor -> 0 or 1)
    "first_half": (0.0, 1.999),         # backdrop half looked at first (floor -> light/dark)
    "air_gap_mL": (0.06, 0.10),         # air pulled in after withdrawing: carry buffer
    "cap_grip": (210.0, 225.0),         # gentler close on the rigid cap (255 over-squeezes)
}


def open_level(gap_m: float) -> float:
    """Robotiq command that leaves a jaw gap of gap_m (0 = fully open at ~93 mm)."""
    return float(np.clip(255.0 * (1.0 - gap_m / 0.093), 0.0, 190.0))


def sample_params(rng: np.random.Generator) -> dict[str, float]:
    return {k: float(rng.uniform(*v)) for k, v in PARAM_RANGES.items()}


# ---------------------------------------------------------------- env bridge
class Bridge:
    """Session's env: turns Session commands into env.step actions and records them."""

    def __init__(self, env: SolutionPrepEnv, record: bool = True):
        self.env, self.record, self.rows = env, record, []
        self.shake_cmd = np.zeros(3)            # (amplitude m, frequency Hz, axis deg); 0 = off
        self.session = None
        self.observation = env.observe()

    def __getattr__(self, name):
        return getattr(self.env, name)

    def task_status(self):
        return self.env.info()

    def step(self, _raw_ctrl):
        s = self.session
        action = np.concatenate([s.q_cmd["left"], [s.grip_cmd["left"]],
                                 s.q_cmd["right"], [s.grip_cmd["right"]], self.shake_cmd]).astype(np.float32)
        prev = self.observation
        obs, reward, term, trunc, info = self.env.step(action)
        if self.record:
            self.rows.append((prev, action.copy(), obs, reward, term, trunc, info))
        self.observation = obs
        if term or trunc:
            raise EpisodeFinished(info["failure"] or ("success" if info["success"] else "horizon"))
        return obs, reward, term, trunc, info


# -------------------------------------------------------------------- expert
@dataclass
class Expert:
    env: SolutionPrepEnv
    params: dict[str, float]
    record: bool = True
    log: list = field(default_factory=list)

    def __post_init__(self):
        from chem_rl_sim.teleop.keyframe import Session
        self.bridge = Bridge(self.env, self.record)
        self.s = Session(self.bridge)
        self.s._carried = self.env._carried_metadata
        # Leo's planner tolerates 1 mm of preview penetration; the env audit forbids
        # 0.05 mm, so planned moves could graze. Count any penetration in previews.
        contacts = self.s._contacts
        self.s._contacts = lambda s_, carried, tol=0.0: contacts(s_, carried, tol=0.0)
        self.bridge.session = self.s
        self.syr = SYRINGES[self.env.syringe_key]
        self.g = self.env.scene.geom
        self.home_q = {side: self.s.q_cmd[side].copy() for side in ("left", "right")}
        self.plunger_rot, self.plunger_pre = TOP_DOWN_X, np.array([0, 0, 0.08])
        self.barrel_rot = BARREL_ROT          # updated by level_syringe()

    # ---------------- primitives
    def fail(self, why: str):
        raise SkillFailed(why)

    def drain(self):
        while self.s.tick():
            pass

    def tool(self, side):
        return self.s.arms[side].tool_pose()

    def cmd_tool(self, side):
        """Tool pose of the COMMANDED joints. Incremental targets must start here:
        restarting from the measured pose releases contact preload (the plunger
        hand would jump back by its servo offset)."""
        import mujoco
        sc = self.s._scratch
        sc.qpos[:] = self.env.data.qpos
        sc.qpos[self.s.arms[side].qadr] = self.s.q_cmd[side]
        mujoco.mj_kinematics(self.env.model, sc)
        return self.s.arms[side].tool_pose(sc)

    VIAS = {"right": ([0.10, -0.20, 1.05], [0.30, -0.10, 1.00], [0.20, -0.15, 1.10]),
            "left": ([-0.30, -0.15, 1.05], [-0.40, 0.00, 1.10])}

    def go(self, side: str, pos, rot, allow: tuple[str, ...] = (), tol: float = 0.01,
           _via: bool = True) -> None:
        """Collision-checked planned move of one arm (the other holds).

        If the direct plan is blocked, retry through fixed via points in front of
        the balance (same orientation), then the target.
        """
        other = "right" if side == "left" else "left"
        for attempt in range(2):
            self.s.target_to_current(other)
            self.s.set_target(side, np.asarray(pos, float), rot)
            plan = self.s.plan()
            if plan.status == "blocked" and _via:
                if self.joint_to(side, pos, rot):
                    p, _ = self.tool(side)
                    if np.linalg.norm(p - pos) < tol:
                        return
                for via in self.VIAS[side]:                    # two checked joint-space legs
                    if self.joint_to(side, via, rot) and self.joint_to(side, pos, rot):
                        p, _ = self.tool(side)
                        if np.linalg.norm(p - pos) < tol:
                            return
                for via in self.VIAS[side]:
                    self.s.target_to_current(other)
                    self.s.set_target(side, np.asarray(via, float), rot)
                    if self.s.plan().status == "ok":
                        self.go(side, via, rot, allow, tol=0.02, _via=False)
                        return self.go(side, pos, rot, allow, tol, _via=False)
            to = np.round(np.asarray(pos, float), 3).tolist()
            if plan.status == "blocked":
                self.fail(f"{side} move blocked: {'; '.join(plan.reasons)} (to {to})")
            if plan.status == "contact" and not all(any(a in r for a in allow) for r in plan.reasons):
                self.fail(f"{side} move contacts: {'; '.join(plan.reasons)} (to {to})")
            self.s.execute(force=True)
            self.drain()
            p, _ = self.tool(side)
            if np.linalg.norm(p - pos) < tol:
                return
        self.fail(f"{side} stopped {np.linalg.norm(p - pos) * 1000:.0f} mm short")

    def follow(self, targets: dict[str, tuple], max_s: float = 8.0, tol: float = 0.003,
               speed: float | None = None, check: bool = True) -> None:
        """Speed-capped IK servo of one or both arms toward targets (no plan preview)."""
        for side in ("left", "right"):
            if side in targets:
                self.s.set_target(side, *targets[side])
            else:
                self.s.target_to_current(side)
        self.s.speed = speed if speed is not None else self.params["follow_speed"]
        self.s.follow = True
        n = 0
        try:
            while self.s.tick() and n < max_s * 10:
                n += 1
        finally:
            self.s.follow = False
            self.s.speed = 1.0
        if check:
            for side, (pos, _) in targets.items():
                err = np.linalg.norm(self.tool(side)[0] - pos)
                if err > tol:
                    self.fail(f"{side} follow missed by {err * 1000:.1f} mm")

    def servo(self, targets: dict[str, tuple], ticks: int = 6) -> None:
        """Fine move below follow mode's 1 mm deadband: IK -> joint target, hold for ticks."""
        sc = self.s._scratch
        sc.qpos[:] = self.env.data.qpos
        for side, (pos, rot) in targets.items():
            arm = self.s.arms[side]
            q, ep, _ = arm.ik(sc, np.asarray(pos, float), rot, self.s.q_cmd[side], iters=60)
            if ep > 0.002:
                self.fail(f"{side} servo IK miss {ep * 1000:.1f} mm")
            self.s.q_cmd[side] = q
        for _ in range(ticks):
            self.bridge.step(None)

    def joint_home(self, side: str, max_speed: float = 1.0) -> None:
        """Min-jerk joint-space return to the episode's start configuration.
        Escapes IK branches (e.g. after a top-down grasp) that Cartesian plans cannot."""
        q0, q1 = self.s.q_cmd[side].copy(), self.home_q[side]
        n = max(5, int(np.ceil(np.abs(q1 - q0).max() / (max_speed * 0.1))))
        for i in range(1, n + 4):
            u = min(1.0, i / n); u = u * u * u * (10 - 15 * u + 6 * u * u)
            self.s.q_cmd[side] = q0 + u * (q1 - q0)
            self.bridge.step(None)
        self.s.target_to_current(side)

    def joint_to(self, side: str, pos, rot, max_speed: float = 0.8, samples: int = 24) -> bool:
        """Joint-space min-jerk move to an IK solution, collision-checked at sampled
        configurations with the planner's own checker. Returns False if unsafe."""
        import mujoco
        arm, sc = self.s.arms[side], self.s._scratch
        sc.qpos[:] = self.env.data.qpos
        q0 = self.s.q_cmd[side].copy()
        best = None
        for seed_q in (q0, self.home_q[side]):
            q1, ep, er = arm.ik(sc, np.asarray(pos, float), rot, seed_q, iters=200)
            if ep < 0.002 and er < 0.03:
                best = q1
                break
        if best is None:
            return False
        carried = self.env._carried_metadata()
        for u in np.linspace(0, 1, samples):
            sc.qpos[:] = self.env.data.qpos
            sc.qpos[arm.qadr] = q0 + u * (best - q0)
            mujoco.mj_forward(self.env.model, sc)
            if any(kind == "blocked" for kind, *_ in self.s._contacts(sc, carried)):
                return False
        n = max(5, int(np.ceil(np.abs(best - q0).max() / (max_speed * 0.1))))
        for i in range(1, n + 4):
            u = min(1.0, i / n); u = u * u * u * (10 - 15 * u + 6 * u * u)
            self.s.q_cmd[side] = q0 + u * (best - q0)
            self.bridge.step(None)
        self.s.target_to_current(side)
        return True

    RIGHT_HIGH = np.array([0.30, 0.10, 1.25])   # above the balance (top at z 1.13)

    def right_out(self, rot) -> None:
        """Leave the right arm's home (behind/right of the balance) high over the
        balance; a direct plan grazes its draft-shield glass. 8/8 seeds clean."""
        p, _ = self.tool("right")
        if p[0] > 0.2 and p[1] > 0.05:
            self.go("right", self.RIGHT_HIGH, rot)

    def right_home(self) -> None:
        """Return the right arm home by the same high route."""
        p, r = self.cmd_tool("right")
        if not (p[0] > 0.2 and p[1] > 0.05):
            self.go("right", np.array([p[0], p[1], max(p[2], 1.20)]), r)
            self.go("right", self.RIGHT_HIGH, r)
        self.s.snap("right", "home")
        self.go("right", *self.s.targets["right"])
        self.joint_home("right")      # same pose, but restore the exact start joints (IK branch)

    def loosen(self, side: str, level: float = 140.0) -> None:
        """Open partway: frees a ~28 mm part without swinging the pads out into a
        neighbouring hand (a full open on the Robotiq sweeps ~45 mm per side)."""
        self.s.queue.append({"kind": "grip", "side": side, "to": float(level)})
        self.drain()

    def swap_grips(self, closing: str, level: float, opening: str, open_to: float = 140.0,
                   ticks: int = 8) -> None:
        """Hand-over: one gripper closes while the other eases off once the closing
        one has contact, so the two never squeeze the part against each other."""
        c0, o0 = self.s.grip_cmd[closing], self.s.grip_cmd[opening]
        for i in range(1, ticks + 1):
            u = i / ticks
            self.s.grip_cmd[closing] = c0 + u * (level - c0)
            if u > 0.5:
                v = (u - 0.5) / 0.5
                self.s.grip_cmd[opening] = o0 + v * (open_to - o0)
            self.bridge.step(None)
        for _ in range(5):
            self.bridge.step(None)

    def retreat(self, side: str, dirs, dist: float, allow=("vial_a", "cap")) -> None:
        """Back off by `dist` along the first direction the planner accepts."""
        p0, r0 = self.cmd_tool(side)
        other = "right" if side == "left" else "left"
        for d in dirs:
            tgt = p0 + np.asarray(d, float) / np.linalg.norm(d) * dist
            self.s.target_to_current(other); self.s.set_target(side, tgt, r0)
            pl = self.s.plan()
            if pl.status == "ok" or (pl.status == "contact" and all(any(a in r for a in allow) for r in pl.reasons)):
                self.go(side, tgt, r0, allow=allow, _via=False)
                return
        self.fail(f"{side} has no clear retreat direction")

    def grip(self, side: str, close: bool, level: float | None = None) -> None:
        """Ramp the gripper; ``level`` (200..255) closes more gently on rigid parts.
        Grasps register at >= 200 (the env's bilateral-contact rule)."""
        if level is None or not close:
            self.s.grip(side, close)
        else:
            self.s.queue.append({"kind": "grip", "side": side, "to": float(level)})
        self.drain()

    def syringe_top_z(self) -> float:
        """Highest point of the syringe (thumb press) right now."""
        from chemlab_rl.solution_scene import ROD_EXPOSED
        return float(self.syringe_base()[2] + self.g.barrel_len + ROD_EXPOSED + self.stroke() + 0.003)

    def holding(self, side: str) -> str | None:
        return self.env.held.get(side)

    # ---------------- syringe geometry helpers
    def syringe_base(self) -> np.ndarray:
        return self.env.data.body("syringe").xpos.copy()

    def tip(self) -> np.ndarray:
        return self.env.scene.needle_tip()

    def stroke(self) -> float:
        return self.env.scene.plunger_stroke()

    def carry(self, d, speed: float = 0.08, settle: int = 3) -> None:
        """Carry the syringe by displacement d.

        Normally ONE hand (BARREL) carries it; the plunger is held by its own
        joint friction. Only while the PLUNGER hand is on the rod (short moves in
        the reservoir) does that hand follow, with an integral stroke hold.
        """
        d = np.asarray(d, float)
        both = self.holding(PLUNGER) == "syringe_plunger"
        bp0, _ = self.cmd_tool(BARREL)
        pp0 = self.cmd_tool(PLUNGER)[0] if both else None
        hold, axis = self.stroke(), self.env.data.body("syringe").xmat.reshape(3, 3)[:, 2]
        n = max(3, int(np.ceil(np.linalg.norm(d) / (speed * self.params["follow_speed"] * 0.1))))
        corr = 0.0
        for i in range(1, n + 1 + settle):
            u = min(1.0, i / n)
            u = u * u * u * (10 - 15 * u + 6 * u * u)           # min-jerk
            targets = {BARREL: (bp0 + u * d, self.barrel_rot)}
            if both:
                corr += 0.5 * (hold - self.stroke())
                targets[PLUNGER] = (pp0 + u * d + axis * corr, self.plunger_rot)
            self.servo(targets, ticks=1)
            if self.holding(BARREL) != "syringe" or (both and self.holding(PLUNGER) != "syringe_plunger"):
                self.fail(f"lost syringe grasp while carrying: {self.env.held}")

    def level_syringe(self, tol_deg: float = 0.8) -> None:
        """Rotate the barrel hand (about its pinch) so the syringe axis is vertical
        again: with one hand on the barrel, plunger strokes twist it in the pads."""
        from chem_rl_sim.teleop.keyframe import axis_rot
        if self.holding(PLUNGER) == "syringe_plunger" or self.holding(BARREL) != "syringe":
            return
        for _ in range(4):
            a = self.env.data.body("syringe").xmat.reshape(3, 3)[:, 2]
            ang = float(np.arccos(np.clip(a[2], -1, 1)))
            if np.degrees(ang) < tol_deg:
                return
            k = np.cross(a, [0, 0, 1.0]); k /= np.linalg.norm(k)
            bp, br = self.cmd_tool(BARREL)
            self.barrel_rot = axis_rot(k, ang) @ br
            self.servo({BARREL: (bp, self.barrel_rot)}, ticks=4)

    def move_syringe(self, tip_target, *, tol: float = 0.0015, max_s: float = 10.0) -> None:
        """Carry the held syringe so the needle tip reaches tip_target (with correction passes)."""
        self.level_syringe()
        for _ in range(4):
            d = np.asarray(tip_target, float) - self.tip()
            if np.linalg.norm(d) < tol:
                return
            self.carry(d)
        err = np.linalg.norm(np.asarray(tip_target) - self.tip())
        if err > 3 * tol:
            self.fail(f"needle tip {err * 1000:.1f} mm from target")

    def plunger_to(self, stroke_m: float, tol_mL: float = 0.01) -> None:
        """PLUNGER hand moves along the barrel axis (BARREL hand holds) until the
        plunger reaches stroke_m. Coarse travel in follow mode, the last
        millimetres with the fine servo on the measured stroke. Emptying aims
        0.6 mm past the stop so the seal bottoms out."""
        stroke_m = float(np.clip(stroke_m, 0.0, self.g.stroke))
        tol = tol_mL / self.syr.mL_per_m
        aim = stroke_m - (0.0006 if stroke_m == 0.0 else 0.0)
        axis = self.env.data.body("syringe").xmat.reshape(3, 3)[:, 2]
        bp, _ = self.cmd_tool(BARREL)
        err = aim - self.stroke()
        if abs(err) > 0.004:
            pp, _ = self.cmd_tool(PLUNGER)
            self.follow({PLUNGER: (pp + axis * err * 0.9, self.plunger_rot), BARREL: (bp, self.barrel_rot)},
                        max_s=6.0, check=False, speed=0.5)
        done = (lambda: self.stroke() < tol) if stroke_m == 0.0 else (lambda: abs(stroke_m - self.stroke()) < tol)
        for _ in range(8):
            if done():
                break
            pp, _ = self.cmd_tool(PLUNGER)
            self.servo({PLUNGER: (pp + axis * (aim - self.stroke()), self.plunger_rot)}, ticks=5)
            if self.holding(PLUNGER) != "syringe_plunger":
                self.fail("lost the plunger")
        if not done() and abs(stroke_m - self.stroke()) > 3 * tol:
            self.fail(f"plunger {self.stroke() * 1000:.2f} mm, wanted {stroke_m * 1000:.2f} mm")

    # ---------------- bookkeeping
    def info(self) -> dict[str, Any]:
        return self.env.info()

    def note(self, skill: str, **kw) -> None:
        self.log.append(dict(skill=skill, step=self.env.steps, **kw))


# ====================================================================== skills
def _rod_pinch(x: Expert) -> np.ndarray:
    return x.syringe_base() + [0, 0, x.g.rod_grip_z + x.stroke()]


def _plan_ok(x: Expert, side: str, pos, rot, allow=()) -> bool:
    other = "right" if side == "left" else "left"
    x.s.target_to_current(other)
    x.s.set_target(side, np.asarray(pos, float), rot)
    pl = x.s.plan()
    return pl.status == "ok" or (pl.status == "contact" and all(any(a in r for a in allow) for r in pl.reasons))


def engage_plunger(x: Expert) -> None:
    """PLUNGER (left) hand pinches the rod from the -x side (top-down as a fallback)
    if a clean plan exists. Its pads stay above the barrel hand."""
    p = x.params
    rp = _rod_pinch(x)
    # Side pinch first: from above, the pads (reaching ~23 mm above the pinch)
    # would hit the wider thumb press on the 40 mm of exposed rod.
    options = [(LEFT_SIDE, rp, [np.array([-p["left_standoff"], 0, 0.03]), np.array([-p["left_standoff"], 0, 0])]),
               (TOP_DOWN_X, rp - [0, 0, 0.004], [np.array([0, 0, 0.08])]),
               (TOP_DOWN, rp - [0, 0, 0.004], [np.array([0, 0, 0.08])])]
    for rot, pinch, pre in options:
        if all(_plan_ok(x, PLUNGER, pinch + off, rot, ("syringe",)) for off in pre):
            break
    else:
        x.fail("no clean approach to the plunger rod")
    for off in pre:
        x.go(PLUNGER, pinch + off, rot, allow=("syringe",))
    x.follow({PLUNGER: (pinch, rot)}, tol=0.003, speed=0.5)
    x.grip(PLUNGER, True)
    if x.holding(PLUNGER) != "syringe_plunger":
        x.fail(f"{PLUNGER} did not acquire the plunger: {x.env.held}")
    x.plunger_rot = rot
    x.plunger_pre = pre[0]


LEFT_PARK = np.array([-0.42, -0.10, 1.10])       # high, left-front: out of the carry path


def park_plunger_hand(x: Expert) -> None:
    """Move the plunger (left) hand out of the barrel hand's carry path."""
    p, _ = x.tool(PLUNGER)
    if np.linalg.norm(p - LEFT_PARK) > 0.02:
        here, hr = x.cmd_tool(PLUNGER)
        x.follow({PLUNGER: (np.array([here[0], here[1], max(here[2], 1.08)]), hr)}, tol=0.01, check=False)
        x.go(PLUNGER, LEFT_PARK, TOP_DOWN_X)


def release_plunger(x: Expert) -> None:
    """PLUNGER hand lets go and backs off the way it came; BARREL hand keeps the syringe."""
    x.loosen(PLUNGER)
    pp, pr = x.cmd_tool(PLUNGER)
    x.follow({PLUNGER: (pp + x.plunger_pre, pr)}, tol=0.008)
    x.grip(PLUNGER, False)
    if x.holding(BARREL) != "syringe":
        x.fail("barrel hand lost the syringe when the plunger hand let go")
    park_plunger_hand(x)


def acquire_syringe(x: Expert) -> None:
    """BARREL (right) hand grips the barrel from the front, coming out of home by
    the high route and a staging point."""
    p, g = x.params, x.g
    bp = x.syringe_base() + [0, 0, g.grip_z]
    front = bp + [0, -p["right_standoff"], 0]
    x.right_out(BARREL_ROT)
    x.go(BARREL, bp + [0.08, -0.08, 0.08], BARREL_ROT)
    # Pre-close to a barrel-sized gap: fully open, the fingers sweep ~45 mm per
    # side into the vial and reservoir standing either side of the rack.
    x.loosen(BARREL, open_level(2 * g.outer_r + 0.016))
    for pos in (front + [0, 0, 0.03], front, bp):
        x.go(BARREL, pos, BARREL_ROT, allow=("syringe",))
    x.grip(BARREL, True)
    if x.holding(BARREL) != "syringe":
        x.fail(f"{BARREL} did not acquire the barrel: {x.env.held}")
    x.barrel_rot = BARREL_ROT
    x.note("acquire_syringe", held=dict(x.env.held))


def lift_syringe(x: Expert) -> None:
    tip = x.tip()
    x.move_syringe(tip + [0, 0, x.params["lift"] + 0.02], tol=0.004)
    rack_top = BENCH + x.g.rest_z - HUB_L
    if x.tip()[2] < rack_top + 0.03:
        x.fail("needle did not clear the rack")
    x.note("lift_syringe", stroke_mm=x.stroke() * 1000)


def _over(x: Expert, vial: str, height: float) -> np.ndarray:
    mouth, _ = x.env.scene.mouth(vial)
    return mouth + [0, 0, height]


def transit_to(x: Expert, vial: str) -> None:
    """Up, across, and down to hover above the vial mouth."""
    hover = x.params["hover"]
    s0 = x.stroke()
    tallest = max(x.env.scene.mouth(v)[0][2] for v in ("vial_a", "reservoir"))
    high = max(min(x.tip()[2], tallest + 0.05), tallest + 0.025)   # carry low: raised hands meet
    x.move_syringe(np.array([*x.tip()[:2], high]), tol=0.004)
    target = _over(x, vial, hover)
    x.move_syringe(np.array([*target[:2], high]), tol=0.003)
    x.move_syringe(target, tol=0.0015)
    x.note("transit_to", vial=vial, stroke_drift_mL=(x.stroke() - s0) * x.syr.mL_per_m)


def draw(x: Expert, volume_mL: float, max_dips: int = 3) -> None:
    """Insert into the reservoir, purge air, draw volume_mL, leave an air gap, withdraw.

    Closed loop on the liquid actually in the barrel AFTER the tip is out of
    the solvent: if drips/creep left it short, dip again and top up.
    """
    p = x.params
    park_plunger_hand(x)
    transit_to(x, "reservoir")
    res_xy = x.env.data.body("reservoir").xpos[:2].copy()
    purged = False
    for dip in range(max_dips):
        res_z = x.env.data.body("reservoir").xpos[2]
        x.move_syringe(np.array([*res_xy, res_z + VIAL_FLOOR + p["res_tip_above_floor"]]), tol=0.0015)
        if not x.env._geometry()["tip_in_reservoir_liquid"]:
            x.fail("needle tip not submerged in the reservoir")
        if x.holding(PLUNGER) != "syringe_plunger":
            engage_plunger(x)
        if not purged:
            x.plunger_to(0.0)                           # purge air (and stray liquid) back out
            purged = True
        else:                                           # expel the air gap before topping up
            x.plunger_to(max(0.0, x.stroke() - x.env.liquid.syringe_air_mL / x.syr.mL_per_m))
        short = volume_mL - x.env.liquid.syringe_liquid_mL
        x.plunger_to(x.stroke() + short / x.syr.mL_per_m)
        for _ in range(3):
            err = volume_mL - x.env.liquid.syringe_liquid_mL
            if abs(err) < 0.008:
                break
            x.plunger_to(x.stroke() + err / x.syr.mL_per_m)
        # Tip out of the liquid, still inside the reservoir: air gap, then out.
        surface = res_z + VIAL_FLOOR + x.env.liquid.reservoir_mL * 1e-6 / (np.pi * 0.0128 ** 2)
        x.move_syringe(np.array([*res_xy, min(surface + 0.006, res_z + VIAL_H - 0.006)]), tol=0.002)
        gap = min(p["air_gap_mL"], 0.98 * x.syr.capacity_mL - x.env.liquid.syringe_liquid_mL)
        x.plunger_to(x.stroke() + gap / x.syr.mL_per_m, tol_mL=0.01)
        got = x.env.liquid.syringe_liquid_mL
        if abs(got - volume_mL) < 0.012:
            break
    else:
        x.fail(f"drew {got:.3f} mL after {max_dips} dips, wanted {volume_mL:.3f} mL")
    release_plunger(x)                                  # tip still inside the reservoir
    x.move_syringe(_over(x, "reservoir", p["hover"]), tol=0.003)
    got = x.env.liquid.syringe_liquid_mL
    if abs(got - volume_mL) > 0.015:
        x.fail(f"barrel holds {got:.3f} mL after release, wanted {volume_mL:.3f} mL")
    x.note("draw", want_mL=volume_mL, got_mL=got, dips=dip + 1, air_mL=x.env.liquid.syringe_air_mL)


def dispense(x: Expert) -> None:
    p = x.params
    before = x.env.liquid.vial_mL
    park_plunger_hand(x)
    transit_to(x, "vial_a")
    mouth, _ = x.env.scene.mouth("vial_a")
    x.move_syringe(mouth - [0, 0, p["vial_tip_below_rim"]], tol=0.0015)
    if not x.env._geometry()["tip_in_vial_mouth"]:
        x.fail("needle tip not inside the vial mouth")
    engage_plunger(x)
    x.plunger_to(0.0, tol_mL=0.005)
    for _ in range(3):                                  # let the tail drain
        x.env_idle()
    release_plunger(x)
    x.move_syringe(_over(x, "vial_a", p["hover"]), tol=0.003)
    added = x.env.liquid.vial_mL - before
    if x.env.liquid.syringe_liquid_mL > 0.02:
        x.fail(f"{x.env.liquid.syringe_liquid_mL:.3f} mL left in the barrel")
    x.note("dispense", added_mL=added, vial_mL=x.env.liquid.vial_mL)


def return_syringe(x: Expert) -> None:
    """BARREL hand sets the syringe back into the rack, backs off and goes home."""
    g = x.g
    rack = x.env.model.body_pos[x.env.scene.rack_body][:2]
    rest_tip = np.array([*rack, BENCH + g.rest_z - TIP])
    high = max(x.tip()[2], BENCH + g.rest_z - TIP + 0.06)
    x.move_syringe(np.array([*x.tip()[:2], high]), tol=0.004)
    def tip_for_base(xy, base_z):
        """Needle-tip target that puts the BARREL BOTTOM at (xy, base_z): one hand
        lets the syringe tilt a little, so align the barrel, not the tip."""
        return np.array([*xy, base_z]) + (x.tip() - x.syringe_base())
    x.move_syringe(np.array([*rack, high]), tol=0.0008)
    x.level_syringe(tol_deg=0.4)
    for dz in (0.025 + TIP, 0.012 + TIP, 0.004 + TIP):          # collar clearance is 1.5 mm
        for _ in range(2):
            x.move_syringe(tip_for_base(rack, BENCH + g.rest_z + dz - TIP), tol=0.0006)
    tilt = np.degrees(np.arccos(np.clip(x.env.data.body("syringe").xmat[8], -1, 1)))
    if tilt > 2.0:
        x.fail(f"syringe tilted {tilt:.1f} deg in the hand: cannot seat it in the rack")
    x.move_syringe(tip_for_base(rack, BENCH + g.rest_z - 0.001), tol=0.004)   # settle the hub
    x.loosen(BARREL)
    bp, br = x.cmd_tool(BARREL)
    x.follow({BARREL: (bp + [0, -x.params["right_standoff"], 0], br)}, tol=0.008)   # straight out the front
    x.grip(BARREL, False)
    x.go(BARREL, bp + [0.08, -0.08, 0.08], BARREL_ROT)
    x.right_home()
    base = x.syringe_base()
    tilt = np.degrees(np.arccos(np.clip(x.env.data.body("syringe").xmat[8], -1, 1)))
    if np.linalg.norm(base[:2] - rack) > 0.004 or tilt > 6:
        x.fail(f"syringe not seated in rack (offset {np.linalg.norm(base[:2] - rack) * 1000:.1f} mm, tilt {tilt:.1f} deg)")
    x.note("return_syringe", tilt_deg=tilt)


def _wrist_reindex(x: Expert, side: str, need: float) -> float:
    """wrist_3 offset (a multiple of pi/2: the cap is round) leaving `need` rad of
    tightening travel (tightening = wrist_3 positive) and >= 0.6 rad of margin the
    other way; prefers jaws along world y, then the most balanced option."""
    arm = x.s.arms[side]
    q3 = x.s.q_cmd[side][5]
    jaw = x.cmd_tool(side)[1][:, 1]                       # current jaw axis (world)
    best, score = None, -np.inf
    for k in range(-4, 5):                                # quarter turns: round cap
        t = q3 + k * np.pi / 2
        low, room = t - arm.q_lo[5], arm.q_hi[5] - t
        if low < 0.6 or room < need:
            continue
        # yaw change about world z is -k*pi/2; prefer jaws along y (the right
        # hand's fingers sit on the vial's +-x sides while it holds the body)
        a = -k * np.pi / 2
        jy = abs(np.sin(a) * jaw[0] + np.cos(a) * jaw[1])
        sc = 10.0 * (jy > 0.7) + min(low - 0.6, room - need)
        if sc > score:
            best, score = k * np.pi / 2, sc
    if best is None:
        x.fail(f"no wrist index leaves {np.degrees(need):.0f} deg of tightening travel")
    return best


def _rotate_wrist(x: Expert, side: str, delta: float, n: int = 30) -> None:
    """Joint-space wrist_3 rotation in place (used for a full-turn re-index)."""
    q0 = x.s.q_cmd[side].copy(); q1 = q0.copy(); q1[5] += delta
    for i in range(1, n + 3):
        u = min(1.0, i / n); u = u * u * u * (10 - 15 * u + 6 * u * u)
        x.s.q_cmd[side] = q0 + u * (q1 - q0)
        x.bridge.step(None)


def screw_cap(x: Expert, side: str = "left", max_turns: float = 1.25, hold: bool = True) -> None:
    """Pick the cap top-down, lower it onto the rim until the thread engages, then
    turn it (clockwise from above) until it is tight, following the helix down. The vial
    sits in the stand's spring clamp, which takes the reaction torque (one hand only).
    ``hold``: keep the cap in hand (the tight thread then carries the vial)."""
    p = x.params
    cap = x.env.data.body("cap").xpos.copy()
    pinch = np.array([cap[0], cap[1], BENCH + 2 * CAP_HALF_H + 0.0062])
    high = max(pinch[2] + 0.10, x.syringe_top_z() + 0.08)
    above = np.array([*pinch[:2], high])
    x.go(side, above, TOP_DOWN_X)
    shift = _wrist_reindex(x, side, need=2 * np.pi * THREAD_TURNS + 0.4)
    if shift:
        _rotate_wrist(x, side, shift)                         # same grasp, wrist re-indexed
    R0 = x.cmd_tool(side)[1]                                  # use the re-indexed orientation
    x.follow({side: (pinch + [0, 0, 0.03], R0)}, tol=0.004)    # straight down: no re-plan
    x.follow({side: (pinch, R0)}, tol=0.003, speed=0.5)
    x.grip(side, True, level=p["cap_grip"])
    if x.holding(side) != "cap":
        x.fail(f"{side} did not acquire the cap: {x.env.held}")
    off = x.tool(side)[0] - x.env.data.body("cap").xpos       # pinch relative to cap centre
    mouth, _ = x.env.scene.mouth("vial_a")
    seat = mouth + [0, 0, CAP_HALF_H]
    carry = seat + off + [0, 0, p["cap_carry_h"]]
    here = x.tool(side)[0]
    x.follow({side: (np.array([*here[:2], carry[2]]), R0)}, tol=0.005)   # up above the vial top
    x.follow({side: (carry, R0)}, tol=0.004, max_s=10)                    # then across
    x.follow({side: (seat + off + [0, 0, 0.004], R0)}, tol=0.003, speed=0.5)
    corr, dz = np.zeros(3), 0.004
    for k in range(60):                                       # lower onto the thread start,
        if x.env.engaged:                                     # closing the loop on the cap
            break                                             # centre vs the (held) vial mouth.
        mouth, _ = x.env.scene.mouth("vial_a")                # The cap passes through the vial
        cap_now = x.env.data.body("cap").xpos                 # until engaged, so never go below
        err = mouth[:2] - cap_now[:2]                         # the rim and only descend centred.
        if np.linalg.norm(err) < 0.0008:                      # Closed loop on x, y AND height:
            dz = max(dz - 0.00025, 0.0002)                    # the cap sags in the grip, so the
        want = mouth + [0, 0, CAP_HALF_H + dz]                # grasp-time offset overshoots.
        corr += 0.7 * (want - cap_now)
        x.servo({side: (want + off + corr, R0)}, ticks=2)
    if not x.env.engaged:
        c = x.env.data.body("cap").xpos - seat
        x.fail(f"cap did not engage the thread (offset {np.round(c * 1000, 2)} mm)")
    # Twist: rotate the tool about the vertical, descending one pitch per turn.
    start, _ = x.cmd_tool(side)
    rate, turn, sign = p["twist_rate"], 0.0, -1.0              # -z rotation = clockwise from above
    from chem_rl_sim.teleop.keyframe import axis_rot
    for n in range(int(max_turns * 2 * np.pi / (rate * 0.1)) + 20):
        if x.env.tight:
            break
        turn = min(turn + rate * 0.1, 2 * np.pi * max_turns)
        rot = axis_rot([0, 0, 1], sign * turn) @ R0
        pos = start - [0, 0, THREAD_PITCH * turn / (2 * np.pi)]
        x.servo({side: (pos, rot)}, ticks=1)
        if n == 8 and x.env.scene.thread_turn() < 0.05:
            x.fail("cap is not turning with the hand (slipping)")
    for _ in range(3):
        x.env_idle()
    if not x.env.tight:
        x.fail(f"cap not tight: thread at {np.degrees(x.env.scene.thread_turn()):.0f} deg")
    if hold:
        regrip_body(x, side)
        x.note("screw_cap", turns=x.env.scene.thread_turn() / (2 * np.pi))
        return
    # Open only partway: fully open, the pads swing into the holding hand's fingers.
    x.s.queue.append({"kind": "grip", "side": side, "to": 140.0})
    x.drain()
    tp, tr = x.tool(side)
    x.follow({side: (tp + [0, 0, 0.06], tr)}, tol=0.006)
    x.grip(side, False)
    x.follow({side: (tp + [0, 0, 0.14], tr)}, tol=0.008)
    x.joint_home(side)                                        # clear the space above the vial
    x.note("screw_cap", turns=x.env.scene.thread_turn() / (2 * np.pi))


# ------------------------------------------------- lift, inspect, shake, put back
BODY_GRIP_Z = 0.037         # pinch height above the vial base for the body grasp (20 mm below the rim;
                            # 34 mm gave 8.6 mm pad penetrations under shaking)
BODY_GRIP = (220.0, 232.0)  # Robotiq command on the 28 mm glass body


def regrip_body(x: Expert, side: str = "left") -> None:
    """With the vial still in the clamp, let go of the tight cap and grip the vial BODY
    20 mm below the rim, fingers straddling the cap. Shaking a vial held by its cap tilted
    it ~4 deg in the pads and the cap's sharp rim dug > 1 mm into a pad; the glass body
    (flat collision faces, 20 g) is the sturdy hold, as a person would shake it."""
    x.loosen(side, open_level(0.044))                       # ~44 mm: clears the 27 mm cap
    tp, tr = x.cmd_tool(side)
    x.follow({side: (tp + [0, 0, 0.004], tr)}, tol=0.003, speed=0.3)
    # Jaws along world x: the clamp rollers sit at +-y (pads on the rollers' sides tilted the
    # vial 4-5 deg going back in), and with jaws along y the shaking went unstable. Quarter
    # turn of wrist_3 if needed (round parts).
    jaw = x.cmd_tool(side)[1][:, 1]
    if abs(jaw[0]) < 0.7:
        arm, q3 = x.s.arms[side], x.s.q_cmd[side][5]
        turn = min((np.pi / 2, -np.pi / 2), key=lambda t: -min(q3 + t - arm.q_lo[5], arm.q_hi[5] - q3 - t))
        _rotate_wrist(x, side, turn, n=20)
    base = x.env.data.body("vial_a").xpos
    tp, tr = x.cmd_tool(side)
    x.follow({side: (np.array([*tp[:2], base[2] + BODY_GRIP_Z]), tr)}, tol=0.002, speed=0.3)
    x.grip(side, True, level=float(np.mean(BODY_GRIP)))
    if x.holding(side) != "vial_a":
        x.fail(f"{side} did not grip the vial body: {x.env.held}")


def lift_out(x: Expert) -> None:
    """Left (holding the tight cap) pulls the vial straight up out of the clamp, then on
    up to above the inspection height."""
    from chemlab_rl.solution_scene import INSPECT_BASE_Z
    if x.holding("left") != "vial_a" or not x.env.tight:
        x.fail(f"lift_out needs the tightened cap in the left hand (held {x.env.held}, tight {x.env.tight})")
    lp, lr = x.cmd_tool("left")
    v0 = x.env.data.body("vial_a").xpos.copy()
    # (pads on the cap also touch the neck once it is screwed down: "held" may say vial_a)
    x.follow({"left": (lp + [0, 0, x.params["lift_clear"]], lr)}, tol=0.004, speed=0.3)
    if x.env.data.body("vial_a").xpos[2] - v0[2] < 0.8 * x.params["lift_clear"]:
        x.fail("vial stayed in the clamp (cap slipped?)")
    lp, lr = x.cmd_tool("left")
    up = INSPECT_BASE_Z + 0.05 - x.env.data.body("vial_a").xpos[2]
    x.follow({"left": (lp + [0, 0, max(0.0, up)], lr)}, tol=0.006)
    x.note("lift_out", vial_z=float(x.env.data.body("vial_a").xpos[2]))


def present_vial(x: Expert, half: str) -> None:
    """Left brings the held vial to the inspection spot in front of `half` of the backdrop
    (seen by the fixed 'inspect' camera), arriving from above."""
    from chemlab_rl.solution_scene import inspect_spot
    lp, lr = x.cmd_tool("left")
    off = lp - x.env.data.body("vial_a").xpos                 # tool relative to the vial base
    target = inspect_spot(half) + off
    x.go("left", target + [0, 0, 0.04], lr, allow=("vial_a", "cap"))
    x.follow({"left": (target, lr)}, tol=0.003, speed=0.5)
    x.vial_half = half


def move_spot(x: Expert, half: str) -> None:
    """Slide the vial sideways to the spot in front of the other half (6 cm)."""
    from chemlab_rl.solution_scene import inspect_spot
    lp, lr = x.cmd_tool("left")
    off = lp - x.env.data.body("vial_a").xpos
    x.follow({"left": (inspect_spot(half) + off, lr)}, tol=0.003, speed=0.5)
    x.vial_half = half


# (Hz, wrist rotation amplitude rad), all under the UR5e wrist speed limit. The top level is
# capped at ~15 m/s^2 peak at the tool: above that (4 Hz, 0.11 rad: ~18 m/s^2) the cap pushed
# the gripper pads just over 1 mm in (pad compression audit).
SHAKE_LEVELS = [(2.0, 0.15), (2.5, 0.14), (3.0, 0.13), (3.5, 0.11), (4.0, 0.09)]


def _shake_axis_deg(x: Expert) -> float:
    """Horizontal rotation axis PERPENDICULAR to the jaw line: rotating about the line
    through the two pad contacts is resisted only by torsional friction (the vial swung
    10-18 deg in the grip and slipped out), any other axis needs the pads to slip."""
    jaw = x.cmd_tool("left")[1][:, 1]
    ax = np.cross([0.0, 0.0, 1.0], [jaw[0], jaw[1], 0.0])
    a = np.degrees(np.arctan2(ax[1], ax[0])) + x.params["shake_axis_jitter"]
    return float((a + 90.0) % 180.0 - 90.0)                  # an axis: +-180 deg equivalent


def shake_burst(x: Expert, level: int, seconds: float) -> None:
    hz, amp = SHAKE_LEVELS[level]
    x.shake_level = level
    x.bridge.shake_cmd = np.array([amp, hz, _shake_axis_deg(x)])
    try:
        _idle(x, int(round(seconds * 10)))
    finally:
        x.bridge.shake_cmd = np.zeros(3)
    _idle(x, 4)                                              # amplitude ramps down (0.3 s)
    if x.holding("left") != "vial_a":
        x.fail(f"lost the vial while shaking: {x.env.held}")


def look(x: Expert):
    """Hold still for the grains to settle, then the camera detector looks."""
    from chemlab_rl.solution_detector import SETTLE_S
    _idle(x, int(round(SETTLE_S * 10)))
    for _ in range(6):                        # camera-only stillness check (vial may still swing)
        if x.detector.still(lambda: _idle(x, 3)):
            break
    lk = x.detector.look(x.vial_half)
    x.looks.append(dict(t=lk.t, half=lk.backdrop, score=lk.score, clear=lk.clear,
                        level=getattr(x, "shake_level", None),
                        undissolved_frac=x.env.liquid.undissolved_fraction))   # truth: LOG ONLY
    return lk


def choose_backdrop(x: Expert) -> None:
    """Look at the still vial in front of both halves; keep the one where the solid
    shows the most (contrast depends on the solid's colour, seen only by the camera)."""
    first = x.vial_half
    a = look(x)
    other = "dark" if first == "light" else "light"
    move_spot(x, other)
    b = look(x)
    best = first if a.score > b.score else other
    if best != other:
        move_spot(x, best)
    x.note("choose_backdrop", scores={first: a.score, other: b.score}, chosen=best)


def dissolve_by_sight(x: Expert, max_s: float = 150.0) -> None:
    """Shake bursts until two consecutive looks are clear. Vigour adapts to progress:
    a burst that removed < 20 % of the visible solid steps the level up."""
    level = int(x.params["start_level"])
    prev = x.looks[-1]["score"] if x.looks else None
    streak, t0, bursts = 0, x.env.data.time, []
    while x.env.data.time - t0 < max_s:
        shake_burst(x, level, x.params["burst_s"])
        lk = look(x)
        bursts.append((level, lk.score))
        streak = streak + 1 if lk.clear else 0
        if streak >= 2:
            break
        if prev is not None and not lk.clear and (prev - lk.score) < 0.2 * max(prev, 1):
            level = min(level + 1, len(SHAKE_LEVELS) - 1)
        prev = lk.score
    else:
        x.fail(f"solid still visible after {max_s:.0f} s of shaking (score {lk.score})")
    x.note("dissolve_by_sight", bursts=bursts, seconds=float(x.env.data.time - t0))


def _upright_vial(x: Expert, tol_deg: float = 0.7) -> None:
    """Rotate the left hand about the vial base until the held vial hangs vertical."""
    from chem_rl_sim.teleop.keyframe import axis_rot
    for _ in range(5):
        v = x.env.data.body("vial_a")
        a = v.xmat.reshape(3, 3)[:, 2]
        ang = float(np.arccos(np.clip(a[2], -1, 1)))
        if np.degrees(ang) < tol_deg:
            return
        k = np.cross(a, [0, 0, 1.0]); k /= np.linalg.norm(k)
        Rc = axis_rot(k, ang)
        lp, lr = x.cmd_tool("left")
        base = v.xpos.copy()
        x.servo({"left": (base + Rc @ (lp - base), Rc @ lr)}, ticks=6)


def _centre_over(x: Expert, stand) -> None:
    for _ in range(3):
        err = stand - x.env.data.body("vial_a").xpos[:2]
        lp, lr = x.cmd_tool("left")
        x.servo({"left": (lp + [*err, 0], lr)}, ticks=4)


def put_back(x: Expert) -> None:
    """Left straightens the vial, lowers it into the clamp (slowly past the pads), lets go
    and leaves."""
    from chemlab_rl.solution_scene import STAND_H
    stand = x.env.model.body_pos[x.env.scene.stand_body][:2].copy()
    lp, lr = x.cmd_tool("left")
    off = lp - x.env.data.body("vial_a").xpos
    x.go("left", np.array([*(stand + off[:2]), lp[2]]), lr, allow=("vial_a", "cap"))
    _upright_vial(x)
    lp, lr = x.cmd_tool("left")
    off = lp - x.env.data.body("vial_a").xpos
    seat = np.array([*stand, BENCH + STAND_H]) + off
    x.follow({"left": (seat + [0, 0, 0.03], lr)}, tol=0.004, max_s=10)
    _centre_over(x, stand)
    lp, _ = x.cmd_tool("left")
    x.follow({"left": (np.array([*lp[:2], seat[2] + 0.02]), lr)}, tol=0.003, speed=0.3)
    _upright_vial(x, tol_deg=0.5)
    _centre_over(x, stand)
    floor = BENCH + STAND_H
    for k in range(6):                       # closed loop on the vial's height: push past the
        dz = x.env.data.body("vial_a").xpos[2] - floor          # rollers until it sits down
        if dz < 0.001:
            break
        _centre_over(x, stand)
        lp, lr = x.cmd_tool("left")
        x.follow({"left": (lp - [0, 0, dz + 0.0015 + 0.001 * k], lr)}, tol=0.004, speed=0.12, check=False)
    dz = x.env.data.body("vial_a").xpos[2] - floor
    if dz > 0.002:
        x.fail(f"vial not seated in the clamp ({dz * 1000:.1f} mm high)")
    # Release: open to ~60 mm (42 mm dragged a finger link over the 27 mm cap), lift just
    # until the pads clear the cap top, slide away from the parked syringe (a wide-open hand
    # rising past it clipped its thumb press), then open fully and go home.
    x.loosen("left", open_level(0.060))
    tp, tr = x.cmd_tool("left")
    x.follow({"left": (tp + [0, 0, 0.05], tr)}, tol=0.005, speed=0.4)
    away = x.env.data.body("vial_a").xpos[:2] - x.syringe_base()[:2]
    away /= max(np.linalg.norm(away), 1e-9)
    tp, tr = x.cmd_tool("left")
    x.follow({"left": (tp + [*(0.05 * away), 0.02], tr)}, tol=0.006, speed=0.5)
    x.grip("left", False)
    tp, tr = x.cmd_tool("left")              # planned moves out (an unchecked joint-space
    x.go("left", tp + [0, 0, 0.15], tr)       # return home swept a finger through the syringe)
    x.go("left", LEFT_PARK, TOP_DOWN_X)
    dz = x.env.data.body("vial_a").xpos[2] - floor
    x.note("put_back", seated_dz_mm=float(dz * 1000))


def _idle(x: Expert, n: int = 1) -> None:
    for side in ("left", "right"):
        x.s.target_to_current(side)
    for _ in range(n):
        x.bridge.step(None)


Expert.env_idle = lambda self, n=1: _idle(self, n)


# ===================================================================== episode
def run_expert(env: SolutionPrepEnv, *, seed: int, syringe: str, solid_id: int, solid_g: float,
               params: dict | None = None, record: bool = True) -> dict:
    """One full episode from reset. Returns outcome, rows, params and the skill log."""
    env.reset(seed=seed, options=dict(syringe=syringe, solid_id=solid_id, solid_g=solid_g))
    params = params or sample_params(np.random.default_rng(seed + 10_000))
    x = Expert(env, params, record=record)
    from chemlab_rl.solution_detector import GrainDetector
    x.detector, x.looks = GrainDetector(env), []
    stage, outcome = "start", "unfinished"
    try:
        stage = "acquire_syringe"; acquire_syringe(x)
        stage = "lift_syringe"; lift_syringe(x)
        n = len(plan_draws(SYRINGES[syringe].capacity_mL))
        for i in range(n):              # size each draw from what the vial still needs
            vol = (TOTAL_VOLUME_ML - env.liquid.vial_mL) / (n - i)
            stage = f"draw_{i}"; draw(x, vol)
            stage = f"dispense_{i}"; dispense(x)
        stage = "return_syringe"; return_syringe(x)
        stage = "screw_cap"; screw_cap(x, hold=True)
        stage = "lift_out"; lift_out(x)
        first = ("light", "dark")[int(params["first_half"])]
        stage = "present_vial"; present_vial(x, first)
        stage = "choose_backdrop"; choose_backdrop(x)
        stage = "dissolve_by_sight"; dissolve_by_sight(x)
        stage = "put_back"; put_back(x)
        stage = "settle"; _idle(x, 40)
    except SkillFailed as e:
        outcome = f"skill_failed[{stage}]: {e}"
    except EpisodeFinished as e:
        outcome = "success" if str(e) == "success" else f"env_terminated[{stage}]: {e}"
    info = env.info()
    return dict(outcome=outcome, success=bool(info["success"]), stage=stage, info=info,
                rows=x.bridge.rows, params=params, log=x.log, steps=env.steps, looks=x.looks)
