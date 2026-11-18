"""Bimanual weighing with dynamic grains and an independently measured scale.

Only joint and gripper controls are written during an episode. Grain positions,
velocities and masses are never assigned by the expert or the transition.
"""
from types import SimpleNamespace

import gymnasium as gym
import numpy as np

from chemlab_rl.scrape_env import ScrapeWeighEnv
from chemlab_rl.granular_material import GRANULAR_MATERIALS, ScaleInstrument
from chemlab_rl.granular_accounting import GranularAccounting
from chemlab_rl.weigh_scene import make_scene


class GranularWeighEnv(ScrapeWeighEnv):
    def __init__(self, *, grain_count=400, instrument=None, **kwargs):
        powder_id = int(kwargs.get("powder_id", 0))
        if not 0 <= powder_id < len(GRANULAR_MATERIALS):
            raise ValueError("powder_id must index the declared material bank")
        self.grain_material = GRANULAR_MATERIALS[powder_id]
        self.grain_count = int(grain_count)
        self.instrument = instrument or ScaleInstrument()
        super().__init__(scene_factory=lambda legacy: make_scene(
            legacy, granular_material=self.grain_material, grain_count=self.grain_count), **kwargs)
        self.material = self.grain_material
        self.observation_space = gym.spaces.Dict({**self.observation_space.spaces,
            "scale_reading_g": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)})
        self.grain_bodies = np.array([self.model.body(f"grain_{i}").id for i in range(self.grain_count)])
        self.grain_geoms = np.array([self.model.geom(f"grain_{i}").id for i in range(self.grain_count)])
        self._is_grain = np.zeros(self.model.ngeom, dtype=bool)
        self._is_grain[self.grain_geoms] = True
        self.grain_max_penetration_m = 0.0
        self.solver_cap_steps = self.max_solver_iterations = 0
        self._start_accounting(0)

    @property
    def instruction(self):
        return (f"Open the shield, hold the source vial, and scoop material into the receiving vial. "
                f"Stop transferring and let the scale settle at {self.target_g:.2f} g.")

    def _start_accounting(self, seed):
        from chemlab_rl.granular_geometry import scoop_membership
        self.accounting = GranularAccounting(
            self.model, self.data, self.grain_bodies, "source", "vial_a",
            tool_membership=scoop_membership(self.model, self.data),
            expected_density_kg_m3=self.material.density_kg_m3,
            expected_total_mass_g=self.grain_count * self.material.grain_mass_g,
            lag_s=self.instrument.lag_s, noise_g=self.instrument.noise_g, seed=seed,
            spill_plane_z=self.legacy.BOARD_TOP - 0.02)
        self.accounting.tare_empty_receiver()
        self.grain_state = self.accounting.inventory()
        self.grain_state.update(reading_g=0.0, filtered_mass_g=0.0, raw_net_mass_g=0.0,
                                settled=False, mass_force_agreement=False, mass_force_error_g=0.0)
        self._sync_accounting()

    def _sync_accounting(self):
        s = self.grain_state
        # Compatibility view for the observation/recorder. It has no step or
        # transfer method and does not determine any physical state.
        self.powder = SimpleNamespace(mass_g=s["receiver_g"], source_g=s["source_g"],
                                     tool_g=s["tool_g"], spilled_g=s["spill_g"],
                                     reading_g=s["reading_g"], filtered_g=s["filtered_mass_g"])
        self.scene.recv_mass_g = self.powder.mass_g
        self.scene.on_spatula_g = self.powder.tool_g

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        if int(options.get("powder_id", self.powder_id)) != self.powder_id or "material" in options:
            raise ValueError("Create a new granular environment when changing material")
        self.target_g = float(options.get("target_g", self.target_g))
        if not 0.1 <= self.target_g <= min(6.0, self.grain_count * self.material.grain_mass_g):
            raise ValueError("Target must lie within the physical initial inventory")
        self.episode_seed = seed
        self.scene.prepare(grain_seed=seed)
        self._episode_t0 = float(self.data.time)
        self.physics_checks = 0
        self.contact_minima = {}
        self.grasp_events = []
        self.grain_max_penetration_m = 0.0
        self.solver_cap_steps = self.max_solver_iterations = 0
        self.session._reset_state()
        self.steps = 0
        self.settled_s = 0.0
        self.subgoal_dones[:] = 0
        self._finished = False
        self.last_action = self.hold_action()
        self._start_accounting(0 if seed is None else seed)
        self.watch()
        if self.scene.bad:
            raise RuntimeError(f"Invalid granular reset: {self.scene.bad}")
        return self.observe(), self.info()

    def _allowed_contact(self, g, h, normal):
        names = (self.model.geom(g).name or "", self.model.geom(h).name or "")
        roots = [self.model.body(int(self.model.body_rootid[self.model.geom_bodyid[x]])).name for x in (g, h)]
        if any(name.startswith("spatula_scoop") and root == "source"
               for name, root in ((names[0], roots[1]), (names[1], roots[0]))):
            return True
        return super()._allowed_contact(g, h, normal)

    def watch(self):
        self.physics_checks += 1
        if self.scene.bad:
            return
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            self.scene.bad = "non-finite physics state"
            return
        contacts = self.data.contact
        # Hundreds of grain contacts are checked in arrays at every substep.
        grain = self._is_grain[contacts.geom1] | self._is_grain[contacts.geom2]
        active = (contacts.efc_address >= 0) & (contacts.dist <= 0)
        if np.any(grain & active):
            penetration = max(0.0, float(-contacts.dist[grain & active].min()))
            self.grain_max_penetration_m = max(self.grain_max_penetration_m, penetration)
            if penetration > 0.2 * self.material.radius_m:
                self.scene.bad = f"deep grain contact: {penetration * 1000:.3f} mm"
                return
        for index in np.flatnonzero(active & ~grain):
            contact = contacts[int(index)]
            g, h = int(contact.geom1), int(contact.geom2)
            key = "|".join(sorted(self.model.geom(x).name or f"geom_{x}" for x in (g, h)))
            self.contact_minima[key] = min(self.contact_minima.get(key, 0.0), float(contact.dist))
            allowed = self._allowed_contact(g, h, contact.frame[:3])
            if contact.dist < -(0.001 if allowed else 0.00005):
                self.scene.bad = f"{'deep task contact' if allowed else 'forbidden contact'}: {key}, {-contact.dist * 1000:.3f} mm"
                return
        for name in ("source", "spatula", "vial_a", "discard"):
            if self.data.body(name).xpos[2] < self.legacy.BOARD_TOP - 0.02:
                self.scene.bad = f"{name} fell below bench"
                return
        if any(w.number for w in self.data.warning):
            self.scene.bad = "MuJoCo numerical or buffer warning"

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
        start = float(self.data.time)
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
            iterations = int(np.max(self.data.solver_niter))
            self.max_solver_iterations = max(self.max_solver_iterations, iterations)
            self.solver_cap_steps += int(iterations >= self.model.opt.iterations)
            if self.check_collisions:
                self.watch()
                if self.scene.bad:
                    break
        self._update_grasps(action)
        self.mj.mj_forward(self.model, self.data)
        if self.check_collisions:
            self.watch()
        elapsed = float(self.data.time) - start
        self.grain_state = self.accounting.update(elapsed)
        self._sync_accounting()
        self.scene.show(self.powder.reading_g)
        self.steps += 1
        door_open = max(float(self.data.qpos[self.scene.top_adr]),
                        *(float(self.data.qpos[x]) for x in self.scene.side_adr)) > 0.08
        if door_open:
            self.subgoal_dones[0] = 1
        if self.subgoal_dones[0] and self.powder.mass_g > 0.05:
            self.subgoal_dones[1] = 1
        within = (abs(self.powder.mass_g - self.target_g) <= 0.05
                  and abs(self.powder.reading_g - self.target_g) <= 0.05)
        ready = (self.subgoal_dones[1] and within and self.grain_state["mass_force_agreement"]
                 and self.powder.spilled_g <= 0.05 and not self.scene.bad)
        self.settled_s = self.settled_s + elapsed if ready else 0.0
        if self.settled_s >= 1.0 - 1e-8:
            self.subgoal_dones[2] = 1
        success = bool(self.subgoal_dones[2])
        terminated = success or self.scene.bad is not None
        truncated = self.steps >= self.max_steps and not terminated
        self._finished = bool(terminated or truncated)
        return self.observe(), float(success) - 1.0, bool(terminated), bool(truncated), self.info()

    def info(self):
        state = super().info()
        state.update(powder_dynamics="physical_grains_v1", physical_validation_scope="rigid_bodies_and_grains",
                     powder_model_status="synthetic_coarse_grain_contact_dynamics",
                     scale_model="physical_pan_force_with_instrument_lag_and_noise",
                     grain_count=self.grain_count, grain_mass_g=self.material.grain_mass_g,
                     grain_radius_m=self.material.radius_m, density_kg_m3=self.material.density_kg_m3,
                     max_grain_contact_penetration_m=self.grain_max_penetration_m,
                     solver_cap_steps=self.solver_cap_steps, max_solver_iterations=self.max_solver_iterations,
                     timestep_s=float(self.model.opt.timestep))
        for name in ("inflight_g", "total_mass_g", "conservation_error_g", "raw_net_mass_g",
                     "mass_force_error_g", "mass_force_agreement", "settled", "bare_pan_grain_mass_g",
                     "all_grains_translationally_stationary", "max_receiver_grain_vertical_speed_m_s"):
            state[name] = self.grain_state[name]
        return state

    def observe(self):
        observation = super().observe()
        observation["scale_reading_g"] = np.array([self.powder.reading_g], dtype=np.float32)
        return observation
