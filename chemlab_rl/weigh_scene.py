"""Physical rigid-body scene for the corrected weighing environment.

The source uses the SAME measured 28 x 57 mm open-vial asset as the receiver.
No free body is pinned, welded to a hand, gravity-cancelled, or repositioned
after reset. Passing a granular material constructs dynamic physical grains
and a connected scoop. The default retains the rejected scalar surrogate only
for rigid-object diagnostics; the production collector does not accept it.
"""
from __future__ import annotations

import numpy as np


def make_scene(legacy, *, granular_material=None, grain_count=400):
    import mujoco as mj
    from chem_rl_sim.assets import add_asset
    from chem_rl_sim.scene import build_spec, set_initial_pose

    class PhysicalScene:
        hand = "robotiq"

        def __init__(self):
            spec = build_spec("weigh", "robotiq", clutter=False)
            source = add_asset(spec, spec.worldbody, "vial_20ml_open", "source", legacy.SRC_REST)
            add_asset(spec, spec.worldbody, "vial_20ml_open", "discard", legacy.DISCARD_REST)
            tool = spec.worldbody.add_body(name="spatula", pos=legacy.SPAT_REST.tolist())
            tool.add_freejoint(name="spatula_joint")
            tool.add_geom(name="spatula_handle", type=mj.mjtGeom.mjGEOM_CYLINDER,
                          size=[0.012, 0.020], mass=0.030, rgba=[0.45, 0.28, 0.14, 1])
            # The legacy blade began 30 mm from a 12 mm-radius handle, leaving
            # a visible gap. A physical neck connects them without enlarging it.
            tool.add_geom(name="spatula_neck", type=mj.mjtGeom.mjGEOM_CAPSULE,
                          fromto=[0.008, 0, 0, 0.033, 0, 0], size=[0.003],
                          mass=0.001, rgba=[0.75, 0.76, 0.8, 1])
            tool.add_geom(name="spatula_blade", type=mj.mjtGeom.mjGEOM_CAPSULE,
                          fromto=[0.030, 0, 0, legacy.BLADE, 0, 0], size=[0.003],
                          mass=0.015, rgba=[0.75, 0.76, 0.8, 1])
            self.granular = granular_material is not None
            if self.granular:
                from chemlab_rl.granular_geometry import replace_vial_collision, add_scoop
                from chemlab_rl.granular_scene import add_grains, configure_solver
                configure_solver(spec)
                for name in ("source", "vial_a", "discard"):
                    replace_vial_collision(spec, spec.body(name))
                # Preserve the grasp handle; replace the rod end by a connected cup.
                spec.delete(spec.geom("spatula_blade"))
                add_scoop(spec, tool)
                add_grains(spec, granular_material, grain_count, legacy.SRC_REST)
            else:
                for name, body in (("source", source), ("vial_a", spec.body("vial_a"))):
                    body.add_geom(name=f"{name}_powder_fill", type=mj.mjtGeom.mjGEOM_CYLINDER,
                                  pos=[0, 0, 0.0021], size=[0.010, 0.0001], mass=0,
                                  contype=0, conaffinity=0, rgba=[0.72, 0.58, 0.32, 1])
            legacy._add_lcd(spec)
            # A separate sign segment preserves all four mass digits. The
            # original renderer silently clamped a negative tared force to zero.
            tilt = np.deg2rad(20)
            rotation = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)],
                                 [0, np.sin(tilt), np.cos(tilt)]])
            panel = np.array([0.10, -0.07, legacy.BOARD_TOP + 0.077])
            spec.worldbody.add_geom(name="lcd_minus", type=mj.mjtGeom.mjGEOM_BOX,
                                    size=[0.003, 0.0007, 0.0005],
                                    pos=(panel + rotation @ [-0.041, 0, 0.004]).tolist(),
                                    quat=legacy._quat(rotation).tolist(),
                                    rgba=[0.02, 0.03, 0.04, 0], contype=0, conaffinity=0)
            self.model = m = spec.compile()
            self.data = mj.MjData(m)
            self.n_substeps = round(0.1 / m.opt.timestep)
            self.top_adr = m.joint("balance_door_top").qposadr[0]
            self.side_adr = [m.joint(f"balance_door_{s}").qposadr[0] for s in ("left", "right")]
            self.free = {name: (int(m.joint(f"{name}_joint").qposadr[0]),
                               int(m.joint(f"{name}_joint").dofadr[0]))
                         for name in ("source", "discard", "spatula", "vial_a")}
            self.vial_adr, self.vial_dof = self.free["vial_a"]
            self.arm_act = {s: np.array([m.actuator(f"{s}_{j}").id for j in legacy.ARM_JOINTS])
                            for s in ("left", "right")}
            self.arm_qadr = {s: np.array([m.joint(f"{s}_{j}_joint").qposadr[0] for j in legacy.ARM_JOINTS])
                             for s in ("left", "right")}
            self.pinch = {s: m.site(f"{s}_grip_pinch").id for s in ("left", "right")}
            self.lcd = {i: {n: m.geom(f"lcd_{i}_{n}").id for n in legacy._SEG_BOX} for i in range(4)}
            self.top_glass = m.geom("balance_door_top_glass").id
            self.handle = m.geom("balance_door_top_handle").id
            self.side_handle = [m.geom(f"balance_door_{s}_handle").id for s in ("left", "right")]
            self.arm_geoms = {g for g in range(m.ngeom)
                              if (m.body(int(m.geom_bodyid[g])).name or "").startswith(("left_", "right_"))}
            self.shield = {g for g in range(m.ngeom)
                           if any(x in (m.geom(g).name or "") for x in ("glass", "housing", "post"))}
            self._ctrl_low, self._ctrl_high = m.actuator_ctrlrange.T.copy()
            self.prepare()

        def prepare(self, *, grain_seed=None):
            m, d = self.model, self.data
            mj.mj_resetData(m, d)
            set_initial_pose(m, d, "weigh")
            for name, p in (("vial_a", legacy.PAN), ("source", legacy.SRC_REST),
                            ("spatula", legacy.SPAT_REST), ("discard", legacy.DISCARD_REST)):
                adr, _ = self.free[name]
                d.qpos[adr:adr + 7] = [*p, 1, 0, 0, 0]
            if self.granular and grain_seed not in (None, 0):
                # Reset-only packing variation. Rotate the separated initial
                # lattice within its circular clearance; dynamics settles it.
                angle = np.random.default_rng(grain_seed).uniform(0, 2 * np.pi)
                rotate = np.array([[np.cos(angle), -np.sin(angle)],
                                   [np.sin(angle), np.cos(angle)]])
                for index in range(grain_count):
                    adr = int(m.joint(f"grain_{index}_joint").qposadr[0])
                    relative = d.qpos[adr:adr + 2] - legacy.SRC_REST[:2]
                    d.qpos[adr:adr + 2] = legacy.SRC_REST[:2] + rotate @ relative
            for s, acts in self.arm_act.items():
                d.ctrl[acts] = d.qpos[self.arm_qadr[s]]
                d.ctrl[m.actuator(f"{s}_grip_fingers_actuator").id] = 0
            self.carried = []  # Measured contact metadata only; never imposes motion.
            self.pinned = None
            self.bad = None
            self.max_pen = 0.0
            self._step_count = 0
            self.recv_mass_g = self.removed_g = self.on_spatula_g = 0.0
            self.deposit_ok = False
            self.update_powder_visuals(8.0, 0.0)
            for _ in range(round(0.8 / m.opt.timestep) if self.granular else 400):
                mj.mj_step(m, d)
            mj.mj_forward(m, d)

        def update_powder_visuals(self, source_g, receiver_g):
            if self.granular:
                return  # Rendering uses the same dynamic grains as contact physics.
            for name, mass in (("source", source_g), ("vial_a", receiver_g)):
                g = self.model.geom(f"{name}_powder_fill").id
                height = max(0.0001, min(0.040, 0.040 * mass / 8.0))
                self.model.geom_size[g, 1] = height / 2
                self.model.geom_pos[g, 2] = 0.002 + height / 2
                self.model.geom_rgba[g, 3] = float(mass > 1e-6)

        def show(self, grams):
            rounded = round(float(grams), 2)
            overflow = not np.isfinite(rounded) or abs(rounded) >= 100
            text = "----" if overflow else f"{abs(rounded):5.2f}".replace(".", "")
            for index, char in enumerate(text):
                on = "g" if char == "-" else legacy._DIGIT.get(char, "")
                for name, geom in self.lcd[index].items():
                    self.model.geom_rgba[geom, 3] = float(name in on)
            self.model.geom_rgba[self.model.geom("lcd_minus").id, 3] = float(rounded < 0 and not overflow)
            self.model.geom_rgba[self.model.geom("lcd_dot").id, 3] = float(not overflow)

        def task_status(self):
            return {"success": False, "balance_g": self.recv_mass_g}

    return PhysicalScene()
