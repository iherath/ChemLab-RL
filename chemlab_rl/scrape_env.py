"""Action-driven synthetic powder weighing in Leo's bimanual MuJoCo scene.

This is a deliberately coarse powder model, not a granular-material simulator.
Grasping uses ordinary MuJoCo pad contact and friction; no object attachments. A spatula must
visit the source mouth to load a finite inventory; its position above the receiver
controls discharge. Withdrawing it stops the discharge after a residual tail.
Neither grasping nor mass transfer needs an expert-only callback.

The scale is a lagged/noisy observation of accounted receiver mass. It is NOT the
old puck contact-force sensor. Bounded vial-fill geometry visualizes mass.
Material parameters are synthetic and hidden from policy observations.

Actions are [left joints (radians), left gripper (0..255), right joints,
right gripper]. They are servo joint targets before gravity compensation, held
for 0.1 s. Returned observations are after that action; collectors must store the
previous observation with the action. There are no script phase/goal-id inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import gymnasium as gym
import numpy as np

from chemlab_rl.powder_dispense import Material


CONTROL_HZ = 10
CAMERAS = {"top1": "top1", "left_wrist": "left_grip_wrist", "right_wrist": "right_grip_wrist"}
ACTION_SEMANTICS = "joint_position_radians_gripper_0_255"
SCHEMA_VERSION = "scrape_v2"


@dataclass
class PowderState:
    """Finite mass accounting with target-independent dynamics.

    ``engagement`` is normalized spatula insertion over the receiver, NOT a
    policy action or a target-dependent multiplier. ``onset`` is its threshold.
    ``spill_tilt`` is the source-vial tilt threshold (0 upright, 1 inverted).
    All other Material fields keep their rate/delay/residual/sensor meanings.
    """

    material: Material
    rng: np.random.Generator
    source_g: float = 8.0
    tool_capacity_g: float = 1.0
    tool_g: float = 0.0
    mass_g: float = 0.0
    spilled_g: float = 0.0
    reading_g: float = 0.0
    filtered_g: float = 0.0
    active_s: float = 0.0
    tail_g: float = 0.0
    last_flow_g_s: float = 0.0
    dispensing: bool = False

    def step(self, dt: float, *, source_contact: bool, engagement: float,
             over_receiver: bool, source_tilt: float = 0.0) -> None:
        m = self.material
        engagement = float(np.clip(engagement, 0.0, 1.0))
        source_tilt = float(np.clip(source_tilt, 0.0, 1.0))
        if source_contact:
            loaded = min(self.source_g, self.tool_capacity_g - self.tool_g,
                         m.flow_g_per_s * (1.0 - m.cohesion) * dt)
            loaded = max(0.0, loaded)
            self.tool_g += loaded
            self.source_g -= loaded
        # A source that is inverted loses powder regardless of target or tool.
        spill_fraction = max(0.0, source_tilt - m.spill_tilt) / max(1e-6, 1.0 - m.spill_tilt)
        spilled = min(self.source_g, m.flow_g_per_s * spill_fraction * dt)
        self.source_g -= spilled
        self.spilled_g += spilled
        active = over_receiver and engagement > m.onset and self.tool_g > 1e-8
        self.active_s = self.active_s + dt if active else 0.0
        out = 0.0
        if active and self.active_s + 1e-9 >= m.delay_s:
            gate = (engagement - m.onset) / max(1e-6, 1.0 - m.onset)
            rate = m.flow_g_per_s * (1.0 - m.cohesion) * gate
            out = min(self.tool_g, rate * dt)
            self.tool_g -= out
            # Reserve a tail from inventory; do not create mass at shutdown.
            desired = min(m.residual_g * gate, self.tool_g)
            self.tail_g = max(self.tail_g, desired)
        elif not active and self.tail_g > 1e-8:
            out = min(self.tool_g, self.tail_g, max(0.005, self.tail_g * dt / max(dt, m.delay_s)))
            self.tool_g -= out
            self.tail_g -= out
        self.tail_g = min(max(0.0, self.tail_g), self.tool_g)
        if over_receiver:
            self.mass_g += out
        else:
            self.spilled_g += out
        self.last_flow_g_s = out / dt
        self.dispensing = bool(active or out > 1e-5)
        alpha = 1.0 - np.exp(-dt / max(m.lag_s, 1e-6))
        self.filtered_g += alpha * (self.mass_g - self.filtered_g)
        # Independent measurement noise, rather than a random walk in mass.
        self.reading_g = max(0.0, self.filtered_g + float(self.rng.normal(0.0, m.noise_g)))

    @property
    def conserved_mass_g(self) -> float:
        return self.source_g + self.tool_g + self.mass_g + self.spilled_g


def _load_legacy():
    root = Path(__file__).resolve().parents[1]
    for p in (root, root / "chem_rl_sim", root / "chem_rl_sim" / "third_party" / "urgantry_sim"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from scripts import collect_leo_scrape
    return collect_leo_scrape


class ScrapeWeighEnv(gym.Env):
    """Full reset: closed shield, source/tool on bench, empty receiver on pan.

    Rigid objects move only through MuJoCo dynamics. Bilateral loaded pad
    contacts identify grasps for planning and mass transfer, but impose no
    position constraint. Every physics substep is audited for forbidden contact.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": CONTROL_HZ}

    def __init__(self, *, images: bool = True, image_size: int = 224,
                 max_steps: int = 2600, control_hz: int = CONTROL_HZ,
                 target_g: float = 2.5, powder_id: int = 0,
                 check_collisions: bool = True, scene_factory=None):
        super().__init__()
        import mujoco
        self.mj = mujoco
        self.legacy = _load_legacy()
        from chemlab_rl.weigh_scene import make_scene
        self.scene = (scene_factory or make_scene)(self.legacy)
        self.model, self.data = self.scene.model, self.scene.data
        self.hand = "robotiq"
        self.control_hz = int(control_hz)
        self.dt = 1.0 / self.control_hz
        self.n_substeps = round(self.dt / self.model.opt.timestep)
        self.images = bool(images)
        self.image_size = int(image_size)
        self.max_steps = int(max_steps)
        self.check_collisions = bool(check_collisions)
        self.target_g = float(target_g)
        self.powder_id = int(powder_id)
        self.material = self.legacy.POWDERS[self.powder_id]
        self.powder = PowderState(self.material, np.random.default_rng(0))
        self.session = self.legacy.Session(self.scene)
        # Planning metadata is measured, never a kinematic attachment.
        self.session._carried = self._carried_metadata
        self.physics_checks = 0
        self.contact_minima = {}
        self.grasp_events = []
        self._grasp_force = np.zeros(6)
        self._renderer = None
        self._render_option = self.mj.MjvOption()
        self._render_option.sitegroup[:] = 0  # Debug markers are not scene objects.
        low, high = [], []
        for side in ("left", "right"):
            arm = self.session.arms[side]
            low.extend([*arm.q_lo, 0.0])
            high.extend([*arm.q_hi, 255.0])
        self.action_space = gym.spaces.Box(np.array(low, np.float32), np.array(high, np.float32))
        spaces = {
            "proprio": gym.spaces.Box(-np.inf, np.inf, (14,), np.float32),
            "lowdim": gym.spaces.Box(-np.inf, np.inf, (20,), np.float32),
        }
        if self.images:
            spaces.update({k: gym.spaces.Box(0, 255, (image_size, image_size, 3), np.uint8) for k in CAMERAS})
        self.observation_space = gym.spaces.Dict(spaces)
        self.subgoal_dones = np.zeros(3, np.float32)
        self.steps = 0
        self.settled_s = 0.0
        self._finished = False
        self.episode_seed = None
        self._episode_t0 = float(self.data.time)
        self.last_action = self.hold_action()

    @property
    def instruction(self) -> str:
        return f"Open the shield, hold the source vial, and scrape powder into the receiving vial. Stop dispensing and let the scale settle at {self.target_g:.2f} g."

    def hold_action(self) -> np.ndarray:
        return np.concatenate([
            self.data.qpos[self.scene.arm_qadr["left"]],
            [self.data.ctrl[self.model.actuator("left_grip_fingers_actuator").id]],
            self.data.qpos[self.scene.arm_qadr["right"]],
            [self.data.ctrl[self.model.actuator("right_grip_fingers_actuator").id]],
        ]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.episode_seed = seed
        options = options or {}
        self.target_g = float(options.get("target_g", self.target_g))
        if not 0.1 <= self.target_g <= 6.0:
            raise ValueError("target_g must be between 0.1 and 6.0 g")
        self.powder_id = int(options.get("powder_id", self.powder_id))
        self.material = options.get("material", self.legacy.POWDERS[self.powder_id])
        if isinstance(self.material, dict):
            self.material = Material(**self.material)
        self.scene.prepare()
        self._episode_t0 = float(self.data.time)
        self.physics_checks = 0
        self.contact_minima = {}
        self.grasp_events = []
        self.session._reset_state()
        self.powder = PowderState(self.material, self.np_random)
        self.steps = 0
        self.settled_s = 0.0
        self.subgoal_dones[:] = 0
        self._finished = False
        self.last_action = self.hold_action()
        return self.observe(), self.info()

    def _carried_metadata(self):
        return [(h.side, int(self.model.dof_bodyid[h.dof]), h.adr, h.rel_p, h.rel_r)
                for h in self.scene.carried]

    def _touching(self, side: str, name: str) -> bool:
        """Both opposing pads must exert positive force on this object."""
        root = self.model.body(name).id
        jaws = set()
        for i, c in enumerate(self.data.contact[:self.data.ncon]):
            if c.efc_address < 0 or c.dist > 0.0001:
                continue
            for g, h in ((int(c.geom1), int(c.geom2)), (int(c.geom2), int(c.geom1))):
                if int(self.model.body_rootid[self.model.geom_bodyid[h]]) != root:
                    continue
                label = self.model.geom(g).name or ""
                for jaw in ("left", "right"):
                    if label.startswith(f"{side}_grip_{jaw}_pad"):
                        self.mj.mj_contactForce(self.model, self.data, i, self._grasp_force)
                        if self._grasp_force[0] > 0.01:
                            jaws.add(jaw)
        return len(jaws) == 2

    def _update_grasps(self, action: np.ndarray) -> None:
        previous = {(h.side, h.adr): h for h in self.scene.carried}
        held = []
        occupied = set()
        for side, grip in (("left", action[6]), ("right", action[13])):
            if grip < 200:
                continue
            for name in ("source", "spatula"):
                adr, dof = self.scene.free[name]
                if adr in occupied or not self._touching(side, name):
                    continue
                p, r = self.session.arms[side].tool_pose()
                op = self.data.body(name).xpos.copy()
                ort = self.data.body(name).xmat.reshape(3, 3).copy()
                hold = previous.get((side, adr))
                if hold is None:
                    hold = self.legacy.Hold(side, adr, dof, r.T @ (op - p), r.T @ ort)
                else:
                    hold.rel_p, hold.rel_r = r.T @ (op - p), r.T @ ort
                held.append(hold)
                occupied.add(adr)
                if (side, adr) not in previous:
                    self.grasp_events.append(dict(step=self.steps, time=float(self.data.time),
                                                  side=side, object=name, bilateral_force=True))
                break
        self.scene.carried = held

    def _geometry(self) -> dict[str, Any]:
        held = {h.adr for h in self.scene.carried}
        has_tool = self.scene.free["spatula"][0] in held
        has_source = self.scene.free["source"][0] in held
        tip = self.legacy._tip(self.data)
        source = self.data.body("source")
        source_r = source.xmat.reshape(3, 3)
        local = source_r.T @ (tip - source.xpos)
        # The blade has a 3 mm radius. Keep its full cross-section inside
        # the narrow visual neck (11.05 mm inner radius), below the rim and
        # actually inside the rendered powder bed, whose level falls with use.
        powder_surface = 0.002 + 0.040 * self.powder.source_g / 8.0
        source_contact = (has_tool and has_source and np.linalg.norm(local[:2]) < 0.0075
                          and 0.008 < local[2] < min(0.051, powder_surface + 0.001))
        receiver = self.data.body("vial_a")
        receiver_r = receiver.xmat.reshape(3, 3)
        recv_local = receiver_r.T @ (tip - receiver.xpos)
        height = float(recv_local[2] - self.legacy.VIAL_H)
        # Remaining within the receiving mouth's horizontal footprint lets the
        # small residual tail land in the vial after vertically withdrawing.
        over_receiver = bool(has_tool and np.linalg.norm(recv_local[:2]) < 0.011 and 0.0 < height < 0.15)
        engagement = float(np.clip((0.05 - height) / 0.04, 0, 1)) if over_receiver else 0.0
        upright = float(receiver_r[2, 2]) > 0.95
        pan_distance = float(np.linalg.norm(receiver.xpos[:2] - self.legacy.PAN[:2]))
        return dict(source_contact=bool(source_contact), engagement=engagement,
                    over_receiver=over_receiver, source_tilt=float((1.0 - source_r[2, 2]) / 2.0),
                    receiver_on_pan=bool(upright and pan_distance < 0.02 and self._receiver_supported()
                                         and abs(float(receiver.xpos[2]) - self.legacy.PAN[2]) < 0.002
                                         and np.linalg.norm(self.data.qvel[self.scene.vial_dof:self.scene.vial_dof+6]) < 0.01),
                    tip=tip.copy())

    def _receiver_supported(self) -> bool:
        pan = self.model.geom("balance_pan").id
        root = self.model.body("vial_a").id
        for i, c in enumerate(self.data.contact[:self.data.ncon]):
            pair = (int(c.geom1), int(c.geom2))
            if pan not in pair or c.efc_address < 0:
                continue
            other = pair[1] if pair[0] == pan else pair[0]
            if self.model.body_rootid[self.model.geom_bodyid[other]] == root:
                self.mj.mj_contactForce(self.model, self.data, i, self._grasp_force)
                if self._grasp_force[0] > 0.01:
                    return True
        return False

    def _allowed_contact(self, g: int, h: int, normal) -> bool:
        """Only task contacts: finger/object, finger/handle, resting support.

        Pad compression up to 1 mm is a solver tolerance, not a permission
        for an arm, carried object, or tool to intersect the shield or table.
        """
        a, b = self.model.geom(g).name or "", self.model.geom(h).name or ""
        ra = self.model.body(int(self.model.body_rootid[self.model.geom_bodyid[g]])).name
        rb = self.model.body(int(self.model.body_rootid[self.model.geom_bodyid[h]])).name
        if ra == rb and ra in ("source", "spatula", "vial_a", "discard"):
            return True
        for pad, other, obj in ((a, b, rb), (b, a, ra)):
            if "_grip_" in pad and "_pad" in pad:
                if obj in ("source", "spatula") or (other.startswith("balance_door_") and other.endswith("_handle")):
                    return True
        for support, obj in ((a, rb), (b, ra)):
            if abs(float(normal[2])) < 0.9:
                continue  # A side impact is not a supported placement.
            if support.startswith("board_") and obj in ("source", "spatula", "discard"):
                return True
            if support == "balance_pan" and obj == "vial_a":
                return True
            if support == "balance_housing" and obj == "source":
                return True
        # Deliberate blade/source-wall scraping is allowed; all tool/shield,
        # tool/receiver and inter-arm contacts are forbidden.
        if (a == "spatula_blade" and rb == "source") or (b == "spatula_blade" and ra == "source"):
            return True
        return False

    def watch(self) -> None:
        """Audit every solver step, including contacts between free objects."""
        self.physics_checks += 1
        if self.scene.bad:
            return
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            self.scene.bad = "non-finite physics state"
            return
        for c in self.data.contact[:self.data.ncon]:
            if c.efc_address < 0 or c.dist > 0:
                continue
            g, h = int(c.geom1), int(c.geom2)
            names = [self.model.geom(x).name or f"geom_{x}" for x in (g, h)]
            key = "|".join(sorted(names))
            self.contact_minima[key] = min(self.contact_minima.get(key, 0.0), float(c.dist))
            allowed = self._allowed_contact(g, h, c.frame[:3])
            limit = 0.001 if allowed else 0.00005
            if c.dist < -limit:
                self.scene.max_pen = min(self.scene.max_pen, float(c.dist))
                self.scene.bad = (f"{'deep task contact' if allowed else 'forbidden contact'}: "
                                  f"{key}, {-c.dist * 1000:.3f} mm")
                return
        for name in ("source", "spatula", "vial_a", "discard"):
            if self.data.body(name).xpos[2] < self.legacy.BOARD_TOP - 0.02:
                self.scene.bad = f"{name} fell below bench"
                return

    def step(self, action):
        if self._finished:
            raise RuntimeError("Episode finished; call reset() before step().")
        action = np.asarray(action, np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("Expected 14 finite joint/gripper commands")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = action.copy()
        self.session.q_cmd = {"left": action[:6].astype(float), "right": action[7:13].astype(float)}
        self.session.grip_cmd = {"left": float(action[6]), "right": float(action[13])}
        # Recompute feedforward each 50 ms, including the currently carried load.
        control_substeps = max(1, round(0.05 / self.model.opt.timestep))
        for i in range(self.n_substeps):
            if i % control_substeps == 0:
                self._update_grasps(action)
                ff = self.session.gravity_ff()
                for side in ("left", "right"):
                    arm = self.session.arms[side]
                    self.data.ctrl[arm.arm_act] = self.session.q_cmd[side] + ff[side]
                    self.data.ctrl[arm.eff_act] = self.session.grip_cmd[side]
                self.data.ctrl[:] = np.clip(self.data.ctrl, self.scene._ctrl_low, self.scene._ctrl_high)
            self.mj.mj_step(self.model, self.data)
            if self.check_collisions:
                self.watch()
                if self.scene.bad:
                    break
        self._update_grasps(action)
        geometry = self._geometry()
        # A collision can stop a command partway through its 100 ms interval.
        # Reject that transition without inventing a full interval of powder flow.
        if not self.scene.bad:
            self.powder.step(self.dt, **{k: geometry[k] for k in ("source_contact", "engagement", "over_receiver", "source_tilt")})
        self.scene.recv_mass_g = self.powder.mass_g
        self.scene.removed_g = 0.0
        self.scene.on_spatula_g = self.powder.tool_g
        self.scene.update_powder_visuals(self.powder.source_g, self.powder.mass_g)
        self.mj.mj_forward(self.model, self.data)
        if self.check_collisions:
            self.watch()
        self.scene.show(self.powder.reading_g)
        self.steps += 1
        if not np.isfinite(self.data.qpos).all():
            self.scene.bad = "non-finite physics state"
        door_open = max(float(self.data.qpos[self.scene.top_adr]), *(float(self.data.qpos[x]) for x in self.scene.side_adr)) > 0.08
        if door_open:
            self.subgoal_dones[0] = 1
        if self.subgoal_dones[0] and self.powder.mass_g > 0.05:
            self.subgoal_dones[1] = 1
        within = abs(self.powder.mass_g - self.target_g) <= 0.05 and abs(self.powder.reading_g - self.target_g) <= 0.05
        stopped = not self.powder.dispensing and self.powder.tail_g < 0.001 and geometry["engagement"] <= self.material.onset
        stable = abs(self.powder.filtered_g - self.powder.mass_g) < 0.01
        ready = bool(self.subgoal_dones[1] and within and stopped and stable and geometry["receiver_on_pan"] and self.powder.spilled_g <= 0.05 and not self.scene.bad)
        self.settled_s = self.settled_s + self.dt if ready else 0.0
        if self.settled_s >= 1.0 - 1e-8:
            self.subgoal_dones[2] = 1
        success = bool(self.subgoal_dones[2])
        terminated = success or self.scene.bad is not None
        truncated = self.steps >= self.max_steps and not terminated
        self._finished = bool(terminated or truncated)
        return self.observe(), float(success) - 1.0, bool(terminated), bool(truncated), self.info()

    def observe(self) -> dict[str, np.ndarray]:
        joint = {side: self.data.qpos[self.scene.arm_qadr[side]].astype(np.float32).copy() for side in ("left", "right")}
        grips = []
        for side in ("left", "right"):
            j = self.model.joint(f"{side}_grip_right_driver_joint")
            q = float(self.data.qpos[j.qposadr[0]])
            grips.append(float(np.clip(q / max(float(j.range[1]), 1e-6), 0, 1) * 255.0))
        proprio = np.concatenate([joint["left"], [grips[0]], joint["right"], [grips[1]]]).astype(np.float32)
        doors = [float(self.data.qpos[self.scene.top_adr]), *(float(self.data.qpos[x]) for x in self.scene.side_adr)]
        low = np.concatenate([joint["left"], joint["right"], grips, doors,
                              [self.powder.reading_g, self.target_g, self.powder.reading_g - self.target_g]]).astype(np.float32)
        result = {"proprio": proprio, "lowdim": low}
        if self.images:
            if self._renderer is None:
                self._renderer = self.mj.Renderer(self.model, self.image_size, self.image_size)
            self.scene.show(self.powder.reading_g)
            for role, camera in CAMERAS.items():
                self._renderer.update_scene(self.data, camera=camera, scene_option=self._render_option)
                result[role] = self._renderer.render().copy()
        return result

    def info(self) -> dict:
        return dict(success=bool(self.subgoal_dones[2]), subgoal_dones=self.subgoal_dones.copy(),
                    mass_g=float(self.powder.mass_g), balance_g=float(self.powder.reading_g),
                    target_g=self.target_g, spilled_g=float(self.powder.spilled_g), tool_g=float(self.powder.tool_g),
                    source_g=float(self.powder.source_g), powder_id=self.powder_id, seed=self.episode_seed,
                    elapsed_s=float(self.data.time - self._episode_t0), settled_s=self.settled_s,
                    failure=self.scene.bad or "", instruction=self.instruction,
                    scale_model="synthetic_mass_accounting_with_lag_and_noise",
                    physics_validation="rigid_contact_v1" if self.check_collisions else "disabled",
                    physical_validation_scope="rigid_bodies_only",
                    powder_model_status="uncalibrated_hand_designed_surrogate",
                    powder_model_calibrated=False,
                    powder_dynamics="height_gated_surrogate",
                    physics_checks=self.physics_checks,
                    max_contact_penetration_m=-min(self.contact_minima.values(), default=0.0),
                    grasp_model="mujoco_friction_only",
                    physical_valid=bool(self.check_collisions and not self.scene.bad))

    def task_status(self):
        return self.info()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
