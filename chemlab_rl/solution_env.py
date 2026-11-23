"""Subtask 2 (solution prep) environment on Leo's bimanual MuJoCo bench.

Task: with a two-handed syringe, move 3.0 mL of solvent from the reservoir vial
into the receiving vial (which holds a weighed solid), screw the cap on tight, and
swirl until the solid has dissolved; leave the vial upright on the bench.

Same conventions as ``scrape_env.ScrapeWeighEnv`` (subtask 1):
* Actions: [left 6 joints (rad), left gripper 0..255, right 6 joints, right
  gripper], servo targets held for 0.1 s; gravity feed-forward added inside.
* Rigid bodies move only through MuJoCo dynamics. Grasps are detected from
  bilateral loaded pad contact and never constrain motion. Every physics
  substep is audited for forbidden contact.
* Liquid is synthetic bookkeeping (``solution_state.SolutionState``) driven by
  measured geometry: plunger slide position, needle-tip location, vial tilt,
  cap state and vial agitation. Hidden parameters (solid dissolution rate,
  syringe size label) are not observations; the syringe is visible.
* Screw cap: when the held cap is lowered onto the rim, centred and level, the
  weld ``cap_thread`` engages it to the vial's helical thread body (the ONE
  non-contact constraint). Turning the cap with a gripper advances it along the
  helix; friction rises over the last 60 deg and the hinge end stop is "tight".
  The thread seals the vial against spills only once tight.

Subgoal dones (VaDar, latched in order, never given to the actor):
  0 loaded      first draw complete: barrel liquid >= plan_draws()[0] - tol, tip out
                of the reservoir liquid
  1 dispensed   vial holds 3.0 +/- 0.05 mL, barrel < 0.05 mL, spill <= 0.05 mL
  2 capped      cap screwed on tight (thread at its end stop)
  3 homogeneous dissolved & mixed, vial upright and resting in its stand,
                no gripper holding it, for 1 s
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from chemlab_rl.solution_scene import (BENCH, CAP_HALF_H, NECK_INNER_R, NEEDLE_R, RESERVOIR_ML, STAND_H,
                                       THREAD_FRICTION, THREAD_TURNS,
                                       VIAL_FLOOR, VIAL_H, VIAL_INNER_R, SolutionScene)
from chemlab_rl.solution_grains import SOLID_RGB
from chemlab_rl.solution_scene import SOLVENT as SOLVENT_RGBA
from chemlab_rl.solution_state import (HELD_OUT_SOLID, HELD_OUT_SYRINGE, SOLIDS, SPILL_TOL_ML, SYRINGES,
                                       TOTAL_VOLUME_ML, VOLUME_TOL_ML, SolutionState)

CONTROL_HZ = 10
VIAL_LIQUID_ALPHA = 0.18
CAMERAS = {"top1": "top1", "left_wrist": "left_grip_wrist", "right_wrist": "right_grip_wrist", "inspect": "inspect"}
SCHEMA_VERSION = "solution_prep_v3"   # v3: 17-D actions (shake command), 4th camera "inspect"
SUBGOALS = ("loaded", "dispensed", "capped", "homogeneous")
TRAIN_SYRINGES = tuple(k for k in SYRINGES if k != HELD_OUT_SYRINGE)
TRAIN_SOLIDS = tuple(i for i in range(len(SOLIDS)) if i != HELD_OUT_SOLID)

ENGAGE_XY_TOL = 0.0015     # cap axis within 1.5 mm of the vial axis
ENGAGE_TILT_DEG = 5.0
ENGAGE_GAP = (-0.0005, 0.0015)   # cap bottom relative to the rim
TIGHT_MARGIN_RAD = np.deg2rad(4)  # within 4 deg of the thread's end stop = tight
FRICTION_RAMP_RAD = np.deg2rad(60)
SETTLE_S = 1.0
AGITATION_ACCEL = 1.0      # m/s^2 of vial acceleration -> agitation 1 (calibrated to a robot swirl, 2-3 cm @ 1.2-1.8 Hz)
AGITATION_OMEGA = 8.0      # rad/s of vial angular speed -> agitation 1

GRASPABLE = ("syringe", "syringe_plunger", "cap", "vial_a")

# Shake command (action[14:17]): the LEFT arm superimposes a sinusoidal ROTATION about a
# horizontal axis through its wrist flange on its commanded joints, at every physics
# substep (a 10 Hz action stream cannot express a 3-6 Hz shake). Like a wrist flick it
# is carried mostly by the light wrist joints; translating the whole arm at several Hz
# is beyond the UR5e servo bandwidth (a +-2 cm command at 3 Hz moved the tool +-5 mm).
# Command: amplitude (rad, 0 = off), frequency (Hz), axis angle in the horizontal plane
# (deg, 0 = rotation about world x).
SHAKE_LOW = np.array([0.0, 0.0, -90.0], np.float32)
SHAKE_HIGH = np.array([0.35, 6.0, 90.0], np.float32)
SHAKE_RAMP_M_S = 0.35 / 0.3          # amplitude slew (rad/s): full amplitude in 0.3 s
WRIST_SPEED_MAX = 3.0                # rad/s: UR5e wrist joint speed limit (180 deg/s), with margin
# The sim's joint servos are heavily damped (wrist kp 500, kv 100: ~0.8 Hz bandwidth), so
# the oscillation is fed forward through the position command: u = q + kv/kp qd + I/kp qdd.
ACTION_DIM = 17
GRASP_CMD_MIN = 200.0
HOLD_RADIUS = 0.06                   # m: the shake hold covers the inspection station only                # gripper command from which bilateral pad contact counts as held
SUPPORT_OK = {  # resting/support contacts (body roots), any normal
    ("syringe", "syringe_rack"), ("vial_a", "vial_stand"), ("reservoir", "reservoir_stand"),
}


def plan_draws(capacity_mL: float, total_mL: float = TOTAL_VOLUME_ML) -> list[float]:
    """Equal draws, each <= 98% of capacity, summing to total (shared with the expert)."""
    n = int(np.ceil(total_mL / (0.98 * capacity_mL) - 1e-9))
    return [total_mL / n] * n


class SolutionPrepEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(self, *, images: bool = True, image_size: int = 224, max_steps: int = 4500,
                 syringe: str = "3mL", solid_id: int = 0, solid_g: float = 0.12,
                 check_collisions: bool = True, layout_jitter_m: float = 0.02,
                 layout_jitter_yaw_deg: float = 20.0):
        super().__init__()
        import mujoco
        self.mj = mujoco
        self.images, self.image_size = bool(images), int(image_size)
        self.max_steps, self.check_collisions = int(max_steps), bool(check_collisions)
        self.layout_jitter_m, self.layout_jitter_yaw_deg = float(layout_jitter_m), float(layout_jitter_yaw_deg)
        self.dt = 1.0 / CONTROL_HZ
        self._scenes: dict[str, SolutionScene] = {}
        self.syringe_key, self.solid_id, self.solid_g = syringe, int(solid_id), float(solid_g)
        self._use_scene(syringe)
        low, high = [], []
        for side in ("left", "right"):
            arm = self.session.arms[side]
            low.extend([*arm.q_lo, 0.0]); high.extend([*arm.q_hi, 255.0])
        low.extend(SHAKE_LOW); high.extend(SHAKE_HIGH)
        self.action_space = gym.spaces.Box(np.array(low, np.float32), np.array(high, np.float32))
        spaces = {"proprio": gym.spaces.Box(-np.inf, np.inf, (14,), np.float32),
                  "lowdim": gym.spaces.Box(-np.inf, np.inf, (16,), np.float32)}
        if self.images:
            spaces.update({k: gym.spaces.Box(0, 255, (image_size, image_size, 3), np.uint8) for k in CAMERAS})
        self.observation_space = gym.spaces.Dict(spaces)
        self._grasp_force = np.zeros(6)
        self._finished = True

    # ------------------------------------------------------------- scene/session
    def _use_scene(self, key: str) -> None:
        from chem_rl_sim.teleop.keyframe import Session
        if key not in self._scenes:
            self._scenes[key] = SolutionScene(SYRINGES[key], control_dt=self.dt)
        self.scene = self._scenes[key]
        self.model, self.data = self.scene.model, self.scene.data
        self.n_substeps = self.scene.n_substeps
        self.hand = "robotiq"
        self._step_count = 0
        self.session = Session(self)
        self.session._carried = self._carried_metadata
        self._renderer = None
        self._render_option = self.mj.MjvOption()
        self._render_option.sitegroup[:] = 0
        m = self.model
        self._root = lambda g: m.body(int(m.body_rootid[m.geom_bodyid[g]])).name
        self._body = lambda g: m.body(int(m.geom_bodyid[g])).name
        self._pads = {s: {g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(f"{s}_grip_")
                          and "_pad" in (m.geom(g).name or "")} for s in ("left", "right")}
        self._arm_geoms = set().union(*(a.geoms for a in self.session.arms.values()))
        self._needle = m.geom("syringe_needle").id
        self._board = {g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("board_")}
        self._cap_geom = m.geom("cap_shell").id
        self._vial_geoms = {g for g in range(m.ngeom) if self._root(g) == "vial_a"}
        self.grains = self.scene.grains
        self._grain_mask = np.zeros(m.ngeom, bool)
        self._grain_mask[list(self.grains.geom_set | self.grains.shell_set)] = True

    def task_status(self) -> dict:
        return {"success": bool(getattr(self, "subgoal_dones", np.zeros(4))[3])}

    @property
    def instruction(self) -> str:
        return (f"Use the syringe to add {TOTAL_VOLUME_ML:.1f} mL of solvent to the vial, screw the cap on tight, "
                "and shake until the solid has dissolved. Put the vial back in its stand.")

    def hold_action(self) -> np.ndarray:
        m, d = self.model, self.data
        return np.concatenate([
            d.qpos[self.scene.arm_qadr["left"]], [d.ctrl[m.actuator("left_grip_fingers_actuator").id]],
            d.qpos[self.scene.arm_qadr["right"]], [d.ctrl[m.actuator("right_grip_fingers_actuator").id]],
            np.zeros(3),
        ]).astype(np.float32)

    # --------------------------------------------------------------------- reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        key = options.get("syringe", self.syringe_key)
        if key not in SYRINGES:
            raise ValueError(f"unknown syringe {key}")
        if key != self.syringe_key or not hasattr(self, "session"):
            self._use_scene(key)
        self.syringe_key = key
        self.solid_id = int(options.get("solid_id", self.solid_id))
        self.solid_g = float(options.get("solid_g", self.solid_g))
        if not 0.01 <= self.solid_g <= 0.5:
            raise ValueError("solid_g must be within 0.01..0.5 g")
        self.episode_seed = seed
        jitter = options.get("layout_jitter_m", self.layout_jitter_m)
        layout = self.scene.prepare(self.np_random if jitter > 0 else None, jitter,
                                    options.get("layout_jitter_yaw_deg", self.layout_jitter_yaw_deg),
                                    solid_g=self.solid_g, grain_seed=seed, solid_rgb=SOLID_RGB[self.solid_id])
        self.layout = layout
        self.session._reset_state()
        self.liquid = SolutionState(syringe=SYRINGES[key], solid=SOLIDS[self.solid_id],
                                    reservoir_mL=RESERVOIR_ML, undissolved_g=self.solid_g)
        self.first_draw_mL = plan_draws(SYRINGES[key].capacity_mL)[0]
        self.subgoal_dones = np.zeros(4, np.float32)
        self.steps, self.settled_s = 0, 0.0
        self.physics_checks, self.contact_minima, self.grasp_events = 0, {}, []
        self.held: dict[str, str] = {}
        self.engage_step = None
        self._vial_vel = np.zeros(3)
        self.agitation = 0.0
        self.slip_agitation = 0.0
        self._osc_amp, self._osc_phase, self._osc_hz = 0.0, 0.0, 0.0
        self._osc_cmd, self._osc_axis, self._osc_dq = (0.0, 0.0), np.array([1.0, 0, 0]), np.zeros(6)
        self._osc_kv_kp = self._osc_i_kp = np.zeros(6)
        self._osc_amp_cap = 0.0
        self.shake_holds = 0
        self._hold_grip = 0.0
        self._hold_anchor = np.zeros(3)
        self._left_grip_act = self.model.actuator("left_grip_fingers_actuator").id
        self._jac_data = self.mj.MjData(self.model)
        self._flange = {s: self.model.body(f"{s}_wrist_3_link").id for s in ("left", "right")}
        self._finished = False
        self._episode_t0 = float(self.data.time)
        self._refresh_visuals()
        self.last_action = self.hold_action()
        return self.observe(), self.info()

    # --------------------------------------------------------------- grasps
    def _carried_metadata(self):
        """Free bodies each hand holds (for gravity feed-forward and plan previews)."""
        out = []
        for side, name in self.held.items():
            if name == "syringe_plunger":
                continue      # a child of the syringe; the barrel hand carries the assembly
            adr, dof = self.scene.free[name]
            p, r = self.session.arms[side].tool_pose()
            b = self.model.body(name)
            out.append((side, b.id, adr, r.T @ (self.data.xpos[b.id] - p),
                        r.T @ self.data.xmat[b.id].reshape(3, 3)))
        return out

    def _pad_objects(self, side: str) -> dict[str, set]:
        """Object (body for the plunger, root otherwise) -> jaws pressing it."""
        found: dict[str, set] = {}
        for i, c in enumerate(self.data.contact[:self.data.ncon]):
            if c.efc_address < 0 or c.dist > 1e-4:
                continue
            for g, h in ((int(c.geom1), int(c.geom2)), (int(c.geom2), int(c.geom1))):
                if g not in self._pads[side]:
                    continue
                name = self._body(h) if self._body(h) == "syringe_plunger" else self._root(h)
                if name == "cap" and self.engaged:
                    name = "vial_a"       # a screwed-on cap is part of the vial
                if name not in GRASPABLE:
                    continue
                self.mj.mj_contactForce(self.model, self.data, i, self._grasp_force)
                if self._grasp_force[0] > 0.01:
                    jaw = "left" if f"{side}_grip_left_pad" in self.model.geom(g).name else "right"
                    found.setdefault(name, set()).add(jaw)
        return found

    def _update_grasps(self, action: np.ndarray) -> None:
        held = {}
        for side, grip in (("left", action[6]), ("right", action[13])):
            if grip < GRASP_CMD_MIN:
                continue
            for name, jaws in self._pad_objects(side).items():
                if len(jaws) == 2:
                    held[side] = name
                    if self.held.get(side) != name:
                        self.grasp_events.append(dict(step=self.steps, side=side, object=name))
                    break
        self.held = held

    # ------------------------------------------------------------- audit
    def _allowed_contact(self, g: int, h: int, normal) -> bool:
        ra, rb = self._root(g), self._root(h)
        if ra == rb and ra != "world":
            return True
        if self._needle in (g, h):
            # The needle touches nothing except its own rack slot (holder contact).
            return {ra, rb} == {"syringe", "syringe_rack"}
        for pad, obj, other in ((g, rb, h), (h, ra, g)):
            if pad in self._pads["left"] | self._pads["right"]:
                body = self._body(other)
                return obj in GRASPABLE or body in GRASPABLE
        if (ra, rb) in SUPPORT_OK or (rb, ra) in SUPPORT_OK:
            return True
        for sup, obj in ((g, rb), (h, ra)):
            if sup in self._board and obj in ("vial_a", "reservoir", "cap") and abs(float(normal[2])) > 0.9:
                return True
        return False

    def watch(self) -> None:
        self.physics_checks += 1
        if self.scene.bad:
            return
        d = self.data
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            self.scene.bad = "non-finite physics state"
            return
        n = d.ncon
        if n == 0:
            idx = ()
        else:
            con = d.contact
            gg = con.geom[:n]
            idx = np.flatnonzero((con.efc_address[:n] >= 0) & (con.dist[:n] <= 0)
                                 & ~self._grain_mask[gg].any(axis=1))
        for i in idx:
            c = d.contact[i]
            g, h = int(c.geom1), int(c.geom2)
            names = [self.model.geom(x).name or f"{self._body(x)}#{x}" for x in (g, h)]
            key = "|".join(sorted(names))
            self.contact_minima[key] = min(self.contact_minima.get(key, 0.0), float(c.dist))
            allowed = self._allowed_contact(g, h, c.frame[:3])
            limit = 0.001 if allowed else 0.00005
            if c.dist < -limit:
                self.scene.bad = (f"{'deep task contact' if allowed else 'forbidden contact'}: "
                                  f"{key}, {-c.dist * 1000:.3f} mm")
                return
        for name in ("vial_a", "reservoir", "syringe", "cap"):
            if d.body(name).xpos[2] < BENCH - 0.02:
                self.scene.bad = f"{name} fell below bench"
                return

    # ---------------------------------------------------------------- shake
    def _shake_prepare(self, cmd) -> None:
        """Latch the shake command; joint direction for a unit rotation about the axis
        through the flange at the commanded left-arm joints (damped pseudo-inverse of the
        flange's 6-D Jacobian, zero flange translation), recomputed every control step."""
        amp, hz, axis_deg = (float(v) for v in np.clip(cmd, SHAKE_LOW, SHAKE_HIGH))
        if hz < 0.2:
            amp = 0.0
        self._osc_cmd = (amp, hz)
        if amp <= 0.0 and self._osc_amp <= 0.0:
            return
        if amp > 0.0:
            self._osc_hz = hz
            a = np.deg2rad(axis_deg)
            self._osc_axis = np.array([np.cos(a), np.sin(a), 0.0])
        arm, sc = self.session.arms["left"], self._jac_data
        sc.qpos[:] = self.data.qpos
        sc.qpos[arm.qadr] = self.session.q_cmd["left"]
        self.mj.mj_kinematics(self.model, sc)
        self.mj.mj_comPos(self.model, sc)
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        self.mj.mj_jacBody(self.model, sc, jp, jr, self._flange["left"])
        Jw = jr[:, arm.dofs[3:]]                         # wrist joints only (light, fast)
        dq = np.zeros(6)
        dq[3:] = Jw.T @ np.linalg.solve(Jw @ Jw.T + 1e-4 * np.eye(3), self._osc_axis)
        self._osc_dq = dq
        # joint speed limit: scale the amplitude so max |qd| <= WRIST_SPEED_MAX
        w = 2 * np.pi * max(self._osc_hz, 1e-6)
        peak = np.abs(dq).max() * w
        self._osc_amp_cap = WRIST_SPEED_MAX / peak if peak > 0 else 0.0
        if self._osc_cmd[0] > self._osc_amp_cap:
            self._osc_cmd = (self._osc_amp_cap, self._osc_cmd[1])
        acts = arm.arm_act
        self._osc_kv_kp = -self.model.actuator_biasprm[acts, 2] / self.model.actuator_gainprm[acts, 0]
        self._osc_i_kp = (self.model.dof_armature[arm.dofs] + 0.02) / self.model.actuator_gainprm[acts, 0]

    def _shake_hold(self) -> None:
        """From the first shake burst on a vial already held between the left pads until the
        left hand opens: weld it to the left gripper base (rigid; a weld to the 3.5 g pad on the finger linkage resonated) and relax the gripper squeeze to touching
        (scene `left_shake_hold`). (Toggling it per burst let the full squeeze creep and tilt
        the vial between bursts.) It also releases as soon as the vial touches its stand or
        clamp: a soft weld pushed into the support drove a pad 8 mm into the vial. And it
        releases once the vial leaves the inspection station (HOLD_RADIUS from where it engaged):
        released only at the clamp, the full squeeze re-seated the vial 5 deg off as it went in. Sim stand-in for a firm grip: with MuJoCo soft contacts the
        squeeze itself walked the vibrated vial ~5 mm per burst down the pads, whatever the
        friction, impratio, noslip, impedance or grip force; weld + relaxed squeeze: 0.2 mm."""
        d, m, eq = self.data, self.model, self.scene.hold_eq
        shaking = self._osc_cmd[0] > 0.0 or self._osc_amp > 0.0
        if shaking and not d.eq_active[eq] and self.held.get("left") == "vial_a":
            gb, vb = m.body("left_grip_base").id, m.body("vial_a").id
            Rg = d.xmat[gb].reshape(3, 3)
            qg, qv, qgi, qrel = (np.zeros(4) for _ in range(4))
            self.mj.mju_mat2Quat(qg, d.xmat[gb]); self.mj.mju_mat2Quat(qv, d.xmat[vb])
            self.mj.mju_negQuat(qgi, qg); self.mj.mju_mulQuat(qrel, qgi, qv)
            m.eq_data[eq, 0:3] = 0.0
            m.eq_data[eq, 3:6] = Rg.T @ (d.xpos[vb] - d.xpos[gb])
            m.eq_data[eq, 6:10] = qrel
            m.eq_data[eq, 10] = 1.0
            d.eq_active[eq] = 1
            self.shake_holds += 1
            self._hold_anchor = d.xpos[vb].copy()
            # ... and relax the squeeze to the fingers' current position (touching): the
            # 45 N force-limited squeeze on the round vial is what walks it out of the pads
            j = m.joint("left_grip_left_driver_joint")
            self._hold_grip = float(np.clip(255.0 * d.qpos[j.qposadr[0]] / max(float(j.range[1]), 1e-6), 0, 255))
        elif d.eq_active[eq] and (self.last_action[6] < GRASP_CMD_MIN or self._vial_on_support()
                                  or np.linalg.norm(d.body("vial_a").xpos - self._hold_anchor) > HOLD_RADIUS):
            d.eq_active[eq] = 0                      # released when the hand opens or the vial
                                                     # touches its stand/clamp (contacts take over)

    def _vial_on_support(self) -> bool:
        d = self.data
        for c in d.contact[:d.ncon]:
            ra, rb = self._root(int(c.geom1)), self._root(int(c.geom2))
            if {ra, rb} == {"vial_a", "vial_stand"} and c.dist < 0.0005:
                return True
        return False

    def _shake_substep(self, dt: float) -> np.ndarray:
        amp_cmd, _ = self._osc_cmd
        step = SHAKE_RAMP_M_S * dt
        self._osc_amp = float(np.clip(amp_cmd, self._osc_amp - step, self._osc_amp + step))
        if self._osc_amp <= 0.0:
            self._osc_amp, self._osc_phase = 0.0, 0.0
            return np.zeros(6)
        w = 2 * np.pi * self._osc_hz
        self._osc_phase = (self._osc_phase + w * dt) % (2 * np.pi)
        sn, cs = np.sin(self._osc_phase), np.cos(self._osc_phase)
        q, qd, qdd = sn, w * cs, -w * w * sn            # per unit amplitude
        return self._osc_dq * self._osc_amp * (q + self._osc_kv_kp * qd + self._osc_i_kp * qdd)

    # ----------------------------------------------------------- measurements
    def _local(self, body: str, point: np.ndarray) -> np.ndarray:
        b = self.data.body(body)
        return b.xmat.reshape(3, 3).T @ (point - b.xpos)

    def _geometry(self) -> dict[str, Any]:
        sc, liq = self.scene, self.liquid
        tip = sc.needle_tip()
        res = self._local("reservoir", tip)
        surface = VIAL_FLOOR + liq.reservoir_mL * 1e-6 / (np.pi * VIAL_INNER_R ** 2)
        in_res_mouth = bool(np.linalg.norm(res[:2]) < NECK_INNER_R - NEEDLE_R and VIAL_FLOOR < res[2] < VIAL_H)
        in_res = in_res_mouth and bool(res[2] < surface)
        via = self._local("vial_a", tip)
        in_vial = bool(np.linalg.norm(via[:2]) < NECK_INNER_R - NEEDLE_R and VIAL_FLOOR < via[2] < VIAL_H + 0.002)
        vial_r = self.data.body("vial_a").xmat.reshape(3, 3)
        b = self.model.body("vial_a").id
        vel = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_BODY, b, vel, 0)
        omega, lin = vel[:3], vel[3:]
        accel = np.linalg.norm(lin - self._vial_vel) / self.dt
        self._vial_vel = lin.copy()
        agitation = float(np.clip(accel / AGITATION_ACCEL + np.linalg.norm(omega) / AGITATION_OMEGA, 0, 3))
        return dict(tip=tip, tip_in_reservoir_liquid=in_res, tip_in_reservoir_mouth=in_res_mouth,
                    tip_in_vial_mouth=in_vial,
                    vial_tilt=float((1.0 - vial_r[2, 2]) / 2.0), agitation=agitation,
                    plunger_mL=sc.plunger_stroke() * SYRINGES[self.syringe_key].mL_per_m)

    def _vial_resting(self) -> bool:
        d = self.data
        r = d.body("vial_a").xmat.reshape(3, 3)
        dof = self.scene.free["vial_a"][1]
        return bool(r[2, 2] > 0.995 and abs(d.body("vial_a").xpos[2] - (BENCH + STAND_H)) < 0.002
                    and np.linalg.norm(d.body("vial_a").xpos[:2] - self.model.body_pos[self.scene.stand_body][:2]) < 0.004
                    and np.linalg.norm(d.qvel[dof:dof + 6]) < 0.02 and "vial_a" not in self.held.values())

    @property
    def engaged(self) -> bool:
        return bool(self.data.eq_active[self.scene.cap_eq])

    @property
    def tight(self) -> bool:
        return self.engaged and self.scene.thread_turn() >= 2 * np.pi * THREAD_TURNS - TIGHT_MARGIN_RAD

    def _try_engage(self) -> None:
        """Engage the thread when the held cap sits on the rim, centred and level."""
        d, m, sc = self.data, self.model, self.scene
        if self.engaged or "cap" not in self.held.values():
            return
        mouth, vr = sc.mouth("vial_a")
        cp, cr = sc.body_pose("cap")
        local = vr.T @ (cp - mouth)
        tilt = np.degrees(np.arccos(np.clip(float(vr[:, 2] @ cr[:, 2]), -1, 1)))
        gap = local[2] - CAP_HALF_H
        if not (np.linalg.norm(local[:2]) < ENGAGE_XY_TOL and tilt < ENGAGE_TILT_DEG
                and ENGAGE_GAP[0] < gap < ENGAGE_GAP[1] and vr[2, 2] > 0.98):
            return
        # Thread at its start; weld the cap at the nominal start pose (centred,
        # bottom on the rim) keeping its yaw relative to the thread body.
        d.qpos[list(sc.thread_qadr)] = 0.0
        d.qvel[list(sc.thread_dof)] = 0.0
        self.mj.mj_kinematics(m, d)
        tr = d.xmat[sc.thread_body].reshape(3, 3)
        q_t, q_c, q_ti, q_rel = (np.zeros(4) for _ in range(4))
        self.mj.mju_mat2Quat(q_t, tr.reshape(-1)); self.mj.mju_mat2Quat(q_c, cr.reshape(-1))
        self.mj.mju_negQuat(q_ti, q_t); self.mj.mju_mulQuat(q_rel, q_ti, q_c)
        eq = sc.cap_eq
        m.eq_data[eq, 0:3] = 0.0
        m.eq_data[eq, 3:6] = [0.0, 0.0, CAP_HALF_H]
        m.eq_data[eq, 6:10] = q_rel
        m.eq_data[eq, 10] = 1.0
        d.eq_active[eq] = 1
        self.engage_step = self.steps

    def _thread_friction(self) -> None:
        """Friction rises over the last part of the turn (the cap tightening down)."""
        turn = self.scene.thread_turn()
        u = np.clip((turn - (2 * np.pi * THREAD_TURNS - FRICTION_RAMP_RAD)) / FRICTION_RAMP_RAD, 0, 1)
        lo, hi = THREAD_FRICTION
        self.model.dof_frictionloss[self.scene.thread_dof[0]] = lo + u * (hi - lo)

    def _refresh_visuals(self) -> None:
        """Vial liquid look from what a camera could physically see: cloudiness from the
        SUSPENDED solid (none once it settles), tint from DISSOLVED coloured solute. The
        hidden undissolved fraction is never painted onto the liquid."""
        liq = self.liquid
        rgb_s = np.array(SOLID_RGB[self.solid_id])
        susp = self.grains.suspended_fraction(VIAL_FLOOR) * min(1.0, liq.undissolved_g / 0.05)
        cloud = min(0.6, 1.5 * susp)
        conc = liq.dissolved_g / max(liq.vial_mL, 1e-6)
        chroma = float(rgb_s.max() - rgb_s.min())
        tint = chroma * min(0.35, conc / 0.15)
        solvent = np.array(SOLVENT_RGBA[:3])
        rgb = solvent + (tint + cloud) * (rgb_s - solvent)
        # clear solvent is nearly transparent (a camera looking down through the surface must
        # see settled grains); suspended solid and dissolved colour make it more opaque
        rgba = [*np.clip(rgb, 0, 1), min(0.9, VIAL_LIQUID_ALPHA + 0.55 * cloud + 0.3 * tint)]
        self.scene.update_visuals(reservoir_mL=liq.reservoir_mL, vial_mL=liq.vial_mL,
                                  syringe_mL=liq.syringe_liquid_mL, solid_g=liq.undissolved_g,
                                  turbidity=liq.turbidity, vial_rgba=rgba)

    # --------------------------------------------------------------------- step
    def step(self, action):
        if self._finished:
            raise RuntimeError("Episode finished; call reset() before step().")
        action = np.asarray(action, np.float32)
        if action.shape == (14,):                      # joints/grippers only: no shake
            action = np.concatenate([action, np.zeros(3, np.float32)])
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError(f"Expected {ACTION_DIM} finite commands (14 joint/gripper + 3 shake)")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action.copy()
        ses = self.session
        ses.q_cmd = {"left": action[:6].astype(float), "right": action[7:13].astype(float)}
        ses.grip_cmd = {"left": float(action[6]), "right": float(action[13])}
        per = max(1, round(0.05 / self.model.opt.timestep))
        self._shake_prepare(action[14:])
        self._shake_hold()
        left_act = ses.arms["left"].arm_act
        for i in range(self.n_substeps):
            if i % per == 0:
                self._update_grasps(action)
                self._try_engage()
                self._thread_friction()
                ff = ses.gravity_ff()
                for side, arm in ses.arms.items():
                    self.data.ctrl[arm.arm_act] = ses.q_cmd[side] + ff[side]
                    self.data.ctrl[arm.eff_act] = ses.grip_cmd[side]
                self.data.ctrl[:] = np.clip(self.data.ctrl, self.scene._ctrl_low, self.scene._ctrl_high)
                base_left = self.data.ctrl[left_act].copy()
                if self.data.eq_active[self.scene.hold_eq]:
                    self.data.ctrl[self._left_grip_act] = self._hold_grip
            if self._osc_amp > 0.0 or self._osc_cmd[0] > 0.0:
                self.data.ctrl[left_act] = np.clip(base_left + self._shake_substep(self.model.opt.timestep),
                                                   self.scene._ctrl_low[left_act], self.scene._ctrl_high[left_act])
            self.grains.apply_forces(self.model.opt.timestep, self.liquid.vial_mL, VIAL_INNER_R, VIAL_FLOOR)
            self.mj.mj_step(self.model, self.data)
            self.grains.contain(VIAL_FLOOR, VIAL_H)
            if self.check_collisions:
                self.watch()
                if self.scene.bad:
                    break
        self._update_grasps(action)
        self._try_engage()
        geo = self._geometry()
        if not self.scene.bad:
            self.liquid.step(self.dt, plunger_mL=geo["plunger_mL"],
                             tip_in_reservoir_liquid=geo["tip_in_reservoir_liquid"],
                             tip_in_vial_mouth=geo["tip_in_vial_mouth"], vial_tilt=geo["vial_tilt"],
                             tip_in_reservoir_mouth=geo["tip_in_reservoir_mouth"],
                             vial_capped=self.tight,
                             agitation=geo["agitation"], dissolve=False)
            self.slip_agitation = self.grains.dissolve(self.dt, self.liquid.solid, self.liquid.saturation_factor,
                                                       self.liquid.dissolve_mass)
            lost = self.grains.escaped(VIAL_FLOOR, VIAL_H)
            if lost:
                self.scene.bad = f"grains escaped the vial: {lost[:5]}"
        self.agitation = geo["agitation"]
        self._refresh_visuals()
        self.mj.mj_forward(self.model, self.data)
        self.steps += 1
        self._step_count = self.steps
        liq, dn = self.liquid, self.subgoal_dones
        if liq.spilled_mL > SPILL_TOL_ML and not self.scene.bad:
            self.scene.bad = f"spilled {liq.spilled_mL:.3f} mL"
        if not dn[0] and liq.syringe_liquid_mL >= self.first_draw_mL - VOLUME_TOL_ML and not geo["tip_in_reservoir_liquid"]:
            dn[0] = 1
        if dn[0] and not dn[1] and abs(liq.vial_mL - TOTAL_VOLUME_ML) <= VOLUME_TOL_ML \
                and liq.syringe_liquid_mL < 0.05 and liq.spilled_mL <= SPILL_TOL_ML:
            dn[1] = 1
        if dn[1] and not dn[2] and self.tight:
            dn[2] = 1
        ready = bool(dn[2] and liq.homogeneous and self._vial_resting() and not self.scene.bad)
        self.settled_s = self.settled_s + self.dt if ready else 0.0
        if self.settled_s >= SETTLE_S - 1e-8:
            dn[3] = 1
        success = bool(dn[3])
        terminated = success or self.scene.bad is not None
        truncated = self.steps >= self.max_steps and not terminated
        self._finished = bool(terminated or truncated)
        return self.observe(), float(success) - 1.0, bool(terminated), bool(truncated), self.info()

    # ------------------------------------------------------------- observe
    def observe(self) -> dict[str, np.ndarray]:
        d = self.data
        joints = {s: d.qpos[self.scene.arm_qadr[s]].astype(np.float32) for s in ("left", "right")}
        grips = []
        for side in ("left", "right"):
            j = self.model.joint(f"{side}_grip_right_driver_joint")
            grips.append(float(np.clip(d.qpos[j.qposadr[0]] / max(float(j.range[1]), 1e-6), 0, 1) * 255.0))
        proprio = np.concatenate([joints["left"], [grips[0]], joints["right"], [grips[1]]]).astype(np.float32)
        low = np.concatenate([proprio, [self.scene.plunger_stroke(), TOTAL_VOLUME_ML]]).astype(np.float32)
        out = {"proprio": proprio, "lowdim": low}
        if self.images:
            if self._renderer is None:
                self._renderer = self.mj.Renderer(self.model, self.image_size, self.image_size)
            for role, cam in CAMERAS.items():
                self._renderer.update_scene(d, camera=cam, scene_option=self._render_option)
                out[role] = self._renderer.render().copy()
        return out

    def info(self) -> dict:
        liq = self.liquid
        return dict(success=bool(self.subgoal_dones[3]), subgoal_dones=self.subgoal_dones.copy(),
                    syringe=self.syringe_key, solid_id=self.solid_id, solid_g=self.solid_g,
                    held_out=bool(self.syringe_key == HELD_OUT_SYRINGE or self.solid_id == HELD_OUT_SOLID),
                    liquid=liq.as_dict(), vial_mL=float(liq.vial_mL), syringe_mL=float(liq.syringe_liquid_mL),
                    syringe_air_mL=float(liq.syringe_air_mL), spilled_mL=float(liq.spilled_mL),
                    conservation_error_mL=float(liq.conservation_error_mL),
                    capped=self.tight, cap_engaged=self.engaged, engage_step=self.engage_step,
                    thread_turn_deg=float(np.degrees(self.scene.thread_turn())),
                    held=dict(self.held), agitation=self.agitation, seed=getattr(self, "episode_seed", None),
                    slip_agitation=getattr(self, "slip_agitation", 0.0), grains=self.grains.stats(),
                    grain_projections=int(self.grains.projections),
                    shake=dict(amp_m=self._osc_amp, hz=self._osc_hz, cmd_amp_m=self._osc_cmd[0],
                               hold_active=bool(self.data.eq_active[self.scene.hold_eq]), holds=self.shake_holds),
                    grain_mass_error_g=float(self.grains.undissolved_kg * 1e3 - liq.undissolved_g),
                    layout=getattr(self, "layout", None), elapsed_s=float(self.data.time - self._episode_t0),
                    settled_s=self.settled_s, failure=self.scene.bad or "", instruction=self.instruction,
                    physics_checks=self.physics_checks,
                    max_contact_penetration_m=-min(self.contact_minima.values(), default=0.0),
                    liquid_model="synthetic_volume_accounting_v1", cap_model="screw_thread_v1",
                    grasp_model="mujoco_friction_only", schema=SCHEMA_VERSION,
                    physical_valid=bool(self.check_collisions and not self.scene.bad))

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
