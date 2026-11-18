"""Read-only inventory and force-sensor checks for physical MuJoCo grains.

Input: centered spherical grain bodies, canonical open vials, and a pan force
sensor. Output: disjoint mass inventories and a separately measured scale.
No particle positions, velocities, masses, forces, or transfer rates are written.
Tool containment is supplied by the caller's actual scoop geometry.
"""
from __future__ import annotations

from collections import deque
from typing import Callable
import math

import mujoco
import numpy as np


ToolMembership = Callable[[int, np.ndarray, float], bool]


class GranularAccounting:
    """Count physical mass; independently tare/filter the actual pan force.

    ``tool_membership(body_id, world_center, radius_m)`` must test the actual
    tool cavity. Tool containment takes precedence while a scoop is inside a
    source/receiver. Remaining grains are source, receiver, inflight, or spill.
    Outside grains count as spilled when slowly contacting a surface/other
    grain, or below an optional world-z spill plane; other outside grains are
    inflight. Spill is a current inventory, not a cumulative loss counter.
    """

    def __init__(self, model, data, grain_body_ids, source_body, receiver_body, *,
                 tool_membership: ToolMembership | None = None,
                 pan_sensor: str = "balance_pan_force", pan_geom: str = "balance_pan",
                 expected_density_kg_m3: float | None = None,
                 expected_total_mass_g: float | None = None,
                 lag_s: float = 0.25, noise_g: float = 0.0, seed: int = 0,
                 settling_s: float = 0.5, mass_tolerance_g: float = 0.01,
                 force_stability_g: float = 0.005, containment_tolerance_m: float = 0.0001,
                 spill_plane_z: float | None = None):
        self.model, self.data = model, data
        self.grain_body_ids = np.asarray(list(grain_body_ids), dtype=np.int32)
        if (self.grain_body_ids.ndim != 1 or not len(self.grain_body_ids)
                or len(set(self.grain_body_ids.tolist())) != len(self.grain_body_ids)):
            raise ValueError("Grain body IDs must be a nonempty unique one-dimensional sequence")
        self.source_body = self._body_id(source_body)
        self.receiver_body = self._body_id(receiver_body)
        if self.source_body == self.receiver_body:
            raise ValueError("Source and receiver must be different bodies")
        self.tool_membership = tool_membership
        self.pan_geom = model.geom(pan_geom).id
        sensor = model.sensor(pan_sensor)
        if int(sensor.type[0]) != int(mujoco.mjtSensor.mjSENS_FORCE) or int(sensor.dim[0]) != 3:
            raise ValueError("Pan sensor must be a three-axis MuJoCo force sensor")
        self.sensor_adr = int(sensor.adr[0])
        self.sensor_site = int(sensor.objid[0])
        self.pan_body = int(model.site_bodyid[self.sensor_site])
        gravity = np.asarray(model.opt.gravity, dtype=float)
        self.gravity_m_s2 = float(np.linalg.norm(gravity))
        if not np.isfinite(self.gravity_m_s2) or self.gravity_m_s2 <= 0:
            raise ValueError("Nonzero finite gravity is required for force-to-mass conversion")
        self.up = -gravity / self.gravity_m_s2
        numeric = (lag_s, noise_g, settling_s, mass_tolerance_g, force_stability_g, containment_tolerance_m)
        if not all(math.isfinite(x) for x in numeric) or min(numeric) < 0 or settling_s <= 0:
            raise ValueError("Scale/tolerance parameters must be finite, nonnegative; settling_s must be positive")
        if spill_plane_z is not None and not math.isfinite(spill_plane_z):
            raise ValueError("Spill plane must be finite")
        self.lag_s, self.noise_g = float(lag_s), float(noise_g)
        self.settling_s, self.mass_tolerance_g = float(settling_s), float(mass_tolerance_g)
        self.force_stability_g = float(force_stability_g)
        self.containment_tolerance_m = float(containment_tolerance_m)
        self.spill_plane_z = spill_plane_z
        self.rng = np.random.default_rng(seed)
        self.radii_m, self.grain_geom_ids, self.grain_dof_ids = [], [], []
        for body in self.grain_body_ids:
            if not 0 < body < model.nbody or int(model.body_geomnum[body]) != 1:
                raise ValueError("Each grain must have exactly one spherical geom")
            geom = int(model.body_geomadr[body])
            if (int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_SPHERE)
                    or not np.allclose(model.geom_pos[geom], 0.0, atol=1e-12)):
                raise ValueError("Each grain must be one centered sphere")
            joint = int(model.body_jntadr[body])
            if joint < 0 or int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
                raise ValueError("Each grain must have a physical free joint")
            self.radii_m.append(float(model.geom_size[geom, 0]))
            self.grain_geom_ids.append(geom)
            self.grain_dof_ids.append(int(model.jnt_dofadr[joint]))
        self.radii_m = np.asarray(self.radii_m)
        self.grain_geom_ids = np.asarray(self.grain_geom_ids, dtype=np.int32)
        self.grain_dof_ids = np.asarray(self.grain_dof_ids, dtype=np.int32)
        self.masses_g = np.asarray(model.body_mass[self.grain_body_ids], dtype=float).copy() * 1000
        self.volumes_m3 = 4 / 3 * np.pi * self.radii_m**3
        if (not np.isfinite(self.masses_g).all() or not np.isfinite(self.volumes_m3).all()
                or np.any(self.masses_g <= 0) or np.any(self.volumes_m3 <= 0)):
            raise ValueError("Grain volume and mass must be finite and positive")
        self.densities_kg_m3 = self.masses_g / 1000 / self.volumes_m3
        if expected_density_kg_m3 is not None:
            if (not math.isfinite(expected_density_kg_m3) or expected_density_kg_m3 <= 0
                    or not np.allclose(self.densities_kg_m3, expected_density_kg_m3, rtol=1e-8, atol=1e-8)):
                raise ValueError("Compiled grain mass does not equal declared density times sphere volume")
        self.initial_total_mass_g = float(self.masses_g.sum())
        if expected_total_mass_g is not None:
            if (not math.isfinite(expected_total_mass_g)
                    or not math.isclose(self.initial_total_mass_g, expected_total_mass_g, abs_tol=1e-8, rel_tol=1e-8)):
                raise ValueError("Compiled grain mass does not match declared initial inventory")
        self._grain_by_geom = {int(g): i for i, g in enumerate(self.grain_geom_ids)}
        self.tare_force_n = None
        self.filtered_g = self.reading_g = 0.0
        self.elapsed_s = 0.0
        self._last_sim_time = float(data.time)
        self._window = deque()

    def _body_id(self, value):
        body = self.model.body(value).id if isinstance(value, str) else int(value)
        if not 0 < body < self.model.nbody:
            raise ValueError("Invalid source/receiver body")
        return body

    def _inside_vial(self, points, body):
        rotation = self.data.xmat[body].reshape(3, 3)
        local = (points - self.data.xpos[body]) @ rotation
        tolerance = self.containment_tolerance_m
        # Conservatively use the narrowest cavity radius reached by any part
        # of the sphere, including the canonical 47–51 mm shoulder taper.
        shoulder = np.clip((local[:, 2] + self.radii_m - 0.047) / 0.004, 0.0, 1.0)
        cavity_radius = 0.0128 + shoulder * (0.01105 - 0.0128)
        return ((np.linalg.norm(local[:, :2], axis=1) + self.radii_m <= cavity_radius + tolerance)
                & (local[:, 2] - self.radii_m >= 0.002 - tolerance)
                & (local[:, 2] + self.radii_m <= 0.057 + tolerance))

    def raw_force_n(self) -> float:
        """Actual site-frame sensor force, projected against world gravity."""
        force = self.data.sensordata[self.sensor_adr:self.sensor_adr + 3]
        world = self.data.site_xmat[self.sensor_site].reshape(3, 3) @ force
        value = float(np.dot(world, self.up))
        if not math.isfinite(value):
            raise FloatingPointError("Pan force is nonfinite")
        return value

    def _contact_summary(self, labels):
        touch = np.zeros(len(self.grain_body_ids), dtype=bool)
        bare_pan = np.zeros(len(self.grain_body_ids), dtype=bool)
        receiver_support_n = 0.0
        foreign_pan = foreign_receiver = False
        force = np.zeros(6)
        for i, contact in enumerate(self.data.contact[:self.data.ncon]):
            if contact.efc_address < 0:
                continue
            g, h = int(contact.geom1), int(contact.geom2)
            gi, hi = self._grain_by_geom.get(g), self._grain_by_geom.get(h)
            bg, bh = int(self.model.geom_bodyid[g]), int(self.model.geom_bodyid[h])
            if gi is None and hi is None and self.pan_geom not in (g, h) and self.receiver_body not in (bg, bh):
                continue
            mujoco.mj_contactForce(self.model, self.data, i, force)
            if np.linalg.norm(force[:3]) <= 1e-9:
                continue
            if gi is not None:
                touch[gi] = True
                bare_pan[gi] |= h == self.pan_geom
            if hi is not None:
                touch[hi] = True
                bare_pan[hi] |= g == self.pan_geom
            if self.pan_geom in (g, h):
                other_body = bh if g == self.pan_geom else bg
                if other_body == self.receiver_body:
                    world_on_h = contact.frame.reshape(3, 3).T @ force[:3]
                    on_receiver = world_on_h if bh == self.receiver_body else -world_on_h
                    receiver_support_n += float(np.dot(on_receiver, self.up))
                else:
                    foreign_pan = True
            if self.receiver_body in (bg, bh):
                other_geom = h if bg == self.receiver_body else g
                grain = self._grain_by_geom.get(other_geom)
                if other_geom != self.pan_geom and not (grain is not None and labels[grain] == "receiver"):
                    foreign_receiver = True
        return touch, bare_pan, receiver_support_n > 1e-6, foreign_pan, foreign_receiver

    def inventory(self) -> dict:
        """Read a current partition; no scale-filter or simulation state changes."""
        if not np.array_equal(self.model.body_mass[self.grain_body_ids] * 1000, self.masses_g):
            raise RuntimeError("Grain masses changed after accounting initialization")
        if not np.array_equal(self.model.geom_size[self.grain_geom_ids, 0], self.radii_m):
            raise RuntimeError("Grain radii changed after accounting initialization")
        points = self.data.xpos[self.grain_body_ids].copy()
        if not np.isfinite(points).all():
            raise FloatingPointError("Nonfinite grain positions")
        source = self._inside_vial(points, self.source_body)
        receiver = self._inside_vial(points, self.receiver_body)
        tool = np.zeros(len(points), dtype=bool)
        if self.tool_membership is not None:
            tool = np.array([bool(self.tool_membership(int(b), p.copy(), float(r)))
                             for b, p, r in zip(self.grain_body_ids, points, self.radii_m)])
        if np.any(source & receiver & ~tool):
            raise RuntimeError("Source and receiver cavity memberships overlap")
        labels = np.full(len(points), "inflight", dtype="U8")
        labels[source] = "source"
        labels[receiver] = "receiver"
        labels[tool] = "tool"
        touch, bare_pan, supported, foreign_pan, foreign_receiver = self._contact_summary(labels)
        linear_velocity = np.array([self.data.qvel[a:a + 3] for a in self.grain_dof_ids])
        linear_speed = np.linalg.norm(linear_velocity, axis=1)
        vertical_speed = np.abs(linear_velocity @ self.up)
        angular_speed = np.array([np.linalg.norm(self.data.qvel[a + 3:a + 6]) for a in self.grain_dof_ids])
        outside = labels == "inflight"
        spill = outside & touch & (linear_speed < 0.02)
        if self.spill_plane_z is not None:
            spill |= outside & (points[:, 2] < self.spill_plane_z)
        labels[spill] = "spill"
        masses = {name + "_g": float(self.masses_g[labels == name].sum())
                  for name in ("source", "receiver", "tool", "inflight", "spill")}
        total = sum(masses.values())
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
                                self.receiver_body, velocity, 0)
        return {**masses, "spilled_g": masses["spill_g"], "labels": labels,
                "total_mass_g": total, "initial_total_mass_g": self.initial_total_mass_g,
                "conservation_error_g": total - self.initial_total_mass_g,
                "bare_pan_grain_mass_g": float(self.masses_g[bare_pan].sum()),
                "receiver_supported": bool(supported),
                "foreign_pan_contact": bool(foreign_pan), "foreign_receiver_contact": bool(foreign_receiver),
                "max_grain_speed_m_s": float(linear_speed.max()),
                "max_receiver_grain_vertical_speed_m_s": float(vertical_speed[labels == "receiver"].max(initial=0)),
                "all_grains_translationally_stationary": bool(np.all(linear_speed < 0.005)),
                "max_grain_angular_speed_rad_s": float(angular_speed.max()),
                "receiver_speed_m_s": float(np.linalg.norm(velocity[3:])),
                "receiver_angular_speed_rad_s": float(np.linalg.norm(velocity[:3]))}

    def tare_empty_receiver(self, tolerance_g: float = 0.02) -> float:
        """Capture measured tare only with empty, settled, physically supported vial.

        Checks against the known pan-plus-empty-vial load guard against taring
        an impact, an unsupported vial, or an external object/robot load.
        """
        if not math.isfinite(tolerance_g) or tolerance_g <= 0:
            raise ValueError("Tare tolerance must be finite and positive")
        state = self.inventory()
        if (state["receiver_g"] != 0 or not state["receiver_supported"]
                or state["foreign_pan_contact"] or state["foreign_receiver_contact"]
                or state["receiver_speed_m_s"] > 0.002 or state["receiver_angular_speed_rad_s"] > 0.02):
            raise RuntimeError("Tare requires an empty, supported, settled receiver with no external load")
        force_n = self.raw_force_n()
        expected_g = 1000 * (self.model.body_subtreemass[self.pan_body] + self.model.body_mass[self.receiver_body])
        if abs(force_n / self.gravity_m_s2 * 1000 - expected_g) > tolerance_g:
            raise RuntimeError("Measured empty load differs from known pan and receiver mass; wait for settling")
        self.tare_force_n = force_n
        self.filtered_g = self.reading_g = self.elapsed_s = 0.0
        self._last_sim_time = float(self.data.time)
        self._window.clear()
        return force_n

    def update(self, dt: float) -> dict:
        """Sample force once, then apply instrument lag/noise and stability checks."""
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Scale update dt must be finite and positive")
        if self.tare_force_n is None:
            raise RuntimeError("Capture an empty-receiver tare before reading the scale")
        actual_dt = float(self.data.time) - self._last_sim_time
        if actual_dt <= 0 or not math.isclose(actual_dt, dt, rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError("Scale dt must match elapsed MuJoCo time since the previous reading/tare")
        state = self.inventory()
        raw_force = self.raw_force_n()
        raw_g = (raw_force - self.tare_force_n) / self.gravity_m_s2 * 1000
        alpha = 1.0 if self.lag_s == 0 else -math.expm1(-dt / self.lag_s)
        self.filtered_g += alpha * (raw_g - self.filtered_g)
        self.reading_g = self.filtered_g + float(self.rng.normal(0.0, self.noise_g))
        self.elapsed_s += dt
        self._last_sim_time = float(self.data.time)
        # Settlement refers to the balance load. Ideal spheres can spin or roll
        # horizontally without changing their supported weight. Keep complete
        # grain motion observable; require quiet receiver/vertical grain motion,
        # unchanged containment, no material in flight, and stable measured force.
        # No velocity is damped or overwritten to make the indicator settle.
        quiet = (state["receiver_supported"] and not state["foreign_pan_contact"]
                 and not state["foreign_receiver_contact"] and state["receiver_speed_m_s"] < 0.002
                 and state["receiver_angular_speed_rad_s"] < 0.02
                 and state["max_receiver_grain_vertical_speed_m_s"] < 0.005
                 and state["inflight_g"] == 0)
        self._window.append((self.elapsed_s, raw_g, tuple(state["labels"]), bool(quiet)))
        cutoff = self.elapsed_s - self.settling_s
        while len(self._window) > 1 and self._window[1][0] <= cutoff:
            self._window.popleft()
        stable = (self._window[-1][0] - self._window[0][0] >= self.settling_s - 1e-9
                  and all(r[3] and r[2] == self._window[-1][2] for r in self._window)
                  and np.ptp([r[1] for r in self._window]) <= self.force_stability_g)
        mass_error = raw_g - state["receiver_g"]
        return {**state, "raw_pan_force_n": raw_force, "tare_force_n": self.tare_force_n,
                "raw_net_mass_g": raw_g, "filtered_mass_g": self.filtered_g, "reading_g": self.reading_g,
                "settled": bool(stable), "balance_settled": bool(stable), "mass_force_error_g": mass_error,
                "mass_force_agreement": bool(stable and abs(mass_error) <= self.mass_tolerance_g),
                "scale_model": "physical_pan_force_with_instrument_lag_and_noise"}
