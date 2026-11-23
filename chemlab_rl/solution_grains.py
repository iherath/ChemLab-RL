"""Solid grains inside the receiving vial (subtask 2): rigid spheres + a synthetic liquid.

The undissolved solid is ``N_GRAINS`` free MuJoCo spheres that collide ONLY with
each other and with an invisible containment shell inside ``vial_a`` (collision
bit 2; every other geom uses bit 1). They never touch grippers, the needle or the
glass, so the env's contact audit ignores them and checks containment instead.

The liquid is not simulated as a fluid. Each physics substep the bed applies,
per grain below the free surface (``xfrc_applied``):

* buoyancy in the vial's accelerating frame: ``rho_l V (a_f - g)``;
* drag toward the local fluid velocity, Stokes + quadratic
  (``6 pi mu r + 0.5 rho_l Cd pi r^2 |v_rel|``), capped for explicit stability;
* a deterministic eddy velocity (seeded per episode) scaled by the slosh speed,
  so vigorous shaking suspends grains and a still vial lets them settle.

The bulk fluid velocity lags the vial (first order, ``SLOSH_TAU``); the lagged
acceleration tilts the free surface (it stays normal to ``g - a_f``).

Dissolution (per grain, per control step) follows the Hixson-Crowell cube-root law
with a Sherwood-type agitation factor driven by the grain's measured slip speed:

    d(m^(1/3))/dt = -(k/3) m0^(1/3) (a0 + BETA sqrt(slip / U_REF)) * wet * (1 - C/C_sat)

so stronger shaking dissolves faster with diminishing returns. ``k`` and ``a0`` are
the hidden per-solid parameters of :class:`chemlab_rl.solution_state.Solid`. Dry
grains (above the surface) do not dissolve. Mass is handed to
:meth:`SolutionState.dissolve_mass`, which keeps exact conservation; grain radii,
masses and inertias shrink to match, and a grain below ``GONE_FRACTION`` of its
initial mass is retired (invisible, no collisions, gravity-compensated).
"""
from __future__ import annotations

import math

import numpy as np

N_GRAINS = 50
SOLID_DENSITY = 1500.0          # kg/m^3, typical organic crystalline powder (as clumps)
LIQUID_DENSITY = 1000.0         # kg/m^3, water-like solvent
LIQUID_VISCOSITY = 1.0e-3       # Pa s
DRAG_CD = 1.0
SLOSH_TAU = 0.06                # s, bulk-fluid lag behind the vial
EDDY_GAIN = 0.6                 # eddy speed per unit slosh speed
EDDY_TAU = 0.05                 # s, eddy decorrelation time
U_REF = 0.05                    # m/s slip that counts as "agitation 1"
BETA = 1.0
K_GRAIN = 0.1                   # rate scale on Solid.dissolution_k for the grain model: vigorous
                                # shaking (4 Hz, +-3 cm) dissolves 0.12 g in ~13-40 s (k 0.6-0.2)
GONE_FRACTION = 1e-3
GRAIN_BIT = 2                   # contype/conaffinity bit shared by grains and the shell
CONTAIN_R = 0.0118              # shell inner radius (glass inner wall is 12.8 mm; neck 11 mm)
CONTAIN_T = 0.005               # shell half-thickness: thick so fast grains cannot tunnel
CONTAIN_SEGMENTS = 24
GRAIN_RGBA = (0.93, 0.90, 0.62, 1.0)
# Per-solid powder colour (index = Solid id; 5 is the held-out solid). Spans luminance so
# that neither backdrop half alone gives contrast for every solid.
SOLID_RGB = [(0.95, 0.95, 0.93),   # white crystalline
             (0.93, 0.86, 0.45),   # pale yellow
             (0.95, 0.52, 0.12),   # orange
             (0.15, 0.32, 0.85),   # blue
             (0.28, 0.17, 0.08),   # dark brown
             (0.35, 0.68, 0.30)]   # green (held out)
SUSPENDED_SPEED = 0.03          # m/s relative to the vial: a grain counts as suspended
# Stiff, thin-transition contacts for grains AND the shell: MuJoCo's defaults (20 ms,
# 1 mm impedance width) let shaken grains sink straight through the shell.
GRAIN_SOLREF = (0.005, 1.0)
PROJECT_TOL = 0.0006            # m into the shell before the containment projection acts
                                # (grain surface then still inside the 12.8 mm glass)
GRAIN_SOLIMP = (0.95, 0.99, 0.0001, 0.5, 2.0)


def grain_radius(mass_kg: float) -> float:
    return (3.0 * max(mass_kg, 0.0) / (4.0 * math.pi * SOLID_DENSITY)) ** (1.0 / 3.0)


def add_grains(spec, vial_floor: float, vial_h: float) -> None:
    """Containment shell on ``vial_a`` (bit 2 only) and N free grain bodies."""
    import mujoco as mj
    vial = spec.body("vial_a")
    seg = 2 * math.pi / CONTAIN_SEGMENTS
    # Staves are WIDE tangentially: each inner face is tangent to the CONTAIN_R circle, so
    # extending it never cuts into the interior, and a grain pressed a few mm into a stave
    # is still nearest to its inner face (narrow staves pushed it out sideways, outward).
    half_w = max((CONTAIN_R + 2 * CONTAIN_T) * math.tan(seg / 2) * 1.15, 0.008)
    half_h = (vial_h - vial_floor) / 2 + CONTAIN_T
    for k in range(CONTAIN_SEGMENTS):
        a = k * seg
        g = vial.add_geom(name=f"vial_a_contain_{k}", type=mj.mjtGeom.mjGEOM_BOX,
                          size=[CONTAIN_T, half_w, half_h],
                          pos=[(CONTAIN_R + CONTAIN_T) * math.cos(a), (CONTAIN_R + CONTAIN_T) * math.sin(a),
                               vial_floor + (vial_h - vial_floor) / 2],
                          quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)])
        _shell(g)
    for name, z in (("floor", vial_floor - CONTAIN_T), ("lid", vial_h - 0.003 + CONTAIN_T)):
        g = vial.add_geom(name=f"vial_a_contain_{name}", type=mj.mjtGeom.mjGEOM_BOX,
                          size=[CONTAIN_R + 2 * CONTAIN_T, CONTAIN_R + 2 * CONTAIN_T, CONTAIN_T], pos=[0, 0, z])
        _shell(g)
    r0 = grain_radius(0.1e-3 / N_GRAINS)
    for i in range(N_GRAINS):
        b = spec.worldbody.add_body(name=f"grain_{i}", pos=[0, 0, -1.0 - 0.01 * i])
        b.add_freejoint(name=f"grain_{i}_joint")
        g = b.add_geom(name=f"grain_{i}", type=mj.mjtGeom.mjGEOM_SPHERE, size=[r0, 0, 0],
                       rgba=list(GRAIN_RGBA), density=SOLID_DENSITY, friction=[0.4, 0.005, 0.0001],
                       condim=3, solref=list(GRAIN_SOLREF), solimp=list(GRAIN_SOLIMP))
        g.contype = g.conaffinity = GRAIN_BIT


def _shell(g) -> None:
    g.contype = g.conaffinity = GRAIN_BIT
    g.rgba = [0, 0, 0, 0]
    g.mass = 0
    g.group = 5
    g.solref = list(GRAIN_SOLREF)
    g.solimp = list(GRAIN_SOLIMP)


class GrainBed:
    """Runtime state of the grains in one compiled model."""

    def __init__(self, model, data):
        import mujoco as mj
        self.mj, self.model, self.data = mj, model, data
        m = model
        self.bodies = np.array([m.body(f"grain_{i}").id for i in range(N_GRAINS)])
        self.geoms = np.array([m.geom(f"grain_{i}").id for i in range(N_GRAINS)])
        j = [m.joint(f"grain_{i}_joint") for i in range(N_GRAINS)]
        self.qadr = np.array([int(x.qposadr[0]) for x in j])
        self.dadr = np.array([int(x.dofadr[0]) for x in j])
        self.lin = (self.dadr[:, None] + np.arange(3)).ravel()
        self.all_dofs = (self.dadr[:, None] + np.arange(6)).ravel()
        self.geom_set = frozenset(int(g) for g in self.geoms)
        self.shell_set = frozenset(g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("vial_a_contain_"))
        self.vial = m.body("vial_a").id
        vj = m.joint("vial_a_joint")
        self.vial_lin = int(vj.dofadr[0]) + np.arange(3)
        self.vial_qadr = int(vj.qposadr[0])
        self.m0 = np.zeros(N_GRAINS)
        self.mass = np.zeros(N_GRAINS)
        self.active = np.zeros(N_GRAINS, bool)
        self.rng = np.random.default_rng(0)
        self._scratch = None
        self._reset_flow()

    # ------------------------------------------------------------------ reset
    def _reset_flow(self) -> None:
        self.vf = np.zeros(3)                 # bulk fluid velocity (world)
        self.af = np.zeros(3)                 # lagged vial acceleration
        self.eddy = np.zeros((N_GRAINS, 3))
        self._slip_sum = np.zeros(N_GRAINS)
        self.projections = 0
        self._wet_sum = np.zeros(N_GRAINS)
        self._n = 0

    def place(self, solid_g: float, seed: int | None, vial_floor: float, rgb=GRAIN_RGBA[:3]) -> None:
        """Split ``solid_g`` into N grains (lognormal sizes) piled on the vial floor."""
        m, d = self.model, self.data
        rng = np.random.default_rng(seed)
        self.rng = np.random.default_rng(None if seed is None else seed + 7919)
        w = rng.lognormal(0.0, 0.35, N_GRAINS)
        self.m0 = solid_g * 1e-3 * w / w.sum()
        self.mass = self.m0.copy()
        self.active[:] = True
        self._reset_flow()
        r = np.array([grain_radius(x) for x in self.mass])
        # Hexagonal layers on the floor, in the vial frame, then into the world.
        pts, rmax = [], float(r.max())
        step = 2.05 * rmax
        z = vial_floor + rmax + 0.0003
        while len(pts) < N_GRAINS:
            for ix in range(-12, 13):
                for iy in range(-12, 13):
                    x = (ix + 0.5 * (iy % 2)) * step
                    y = iy * step * math.sqrt(3) / 2
                    if math.hypot(x, y) < 0.0118 - rmax - 0.0004 and len(pts) < N_GRAINS:
                        pts.append((x, y, z))
            z += step
        pts = np.array(pts)[rng.permutation(N_GRAINS)]
        vp, vr = d.xpos[self.vial].copy(), d.xmat[self.vial].reshape(3, 3).copy()
        for i in range(N_GRAINS):
            a = self.qadr[i]
            d.qpos[a:a + 3] = vp + vr @ pts[i]
            d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
            m.geom_contype[self.geoms[i]] = m.geom_conaffinity[self.geoms[i]] = GRAIN_BIT
            m.geom_rgba[self.geoms[i]] = [*rgb, 1.0]
            m.body_gravcomp[self.bodies[i]] = 0.0
        d.qvel[self.all_dofs] = 0.0
        d.xfrc_applied[self.bodies] = 0.0
        self._apply_sizes()

    def _apply_sizes(self) -> None:
        m = self.model
        for i in range(N_GRAINS):
            mi = max(self.mass[i], 1e-12)
            r = grain_radius(mi) if self.active[i] else 1e-4
            g, b = self.geoms[i], self.bodies[i]
            m.geom_size[g, 0] = r
            m.geom_rbound[g] = r
            mm = mi if self.active[i] else 1e-9
            m.body_mass[b] = mm
            m.body_inertia[b] = 0.4 * mm * r * r
        # Mass-derived constants (invweights, subtree masses, ...) must follow, or the
        # implicit integrator blows the grains up. Recompute on a scratch MjData so the
        # simulation state is untouched (~1 ms).
        if self._scratch is None:
            self._scratch = self.mj.MjData(m)
        self.mj.mj_setConst(m, self._scratch)

    # ------------------------------------------------------------- per substep
    def apply_forces(self, dt: float, vial_mL: float, vial_inner_r: float, vial_floor: float) -> None:
        d = self.data
        vv = d.qvel[self.vial_lin]
        a_new = (vv - self.vf) / SLOSH_TAU
        self.af = a_new
        self.vf = self.vf + (vv - self.vf) * min(1.0, dt / SLOSH_TAU)
        act = self.active
        if vial_mL <= 1e-6 or not act.any():
            d.xfrc_applied[self.bodies, :3] = 0.0
            return
        g_eff = np.array([0.0, 0.0, -9.81]) - self.af
        n = -g_eff / max(np.linalg.norm(g_eff), 1e-9)                     # free-surface normal
        vr = d.xmat[self.vial].reshape(3, 3)
        h = vial_mL * 1e-6 / (math.pi * vial_inner_r ** 2)
        p_s = d.xpos[self.vial] + vr[:, 2] * (vial_floor + h)
        pos = d.xpos[self.bodies]
        r = self.model.geom_size[self.geoms, 0]
        depth = -(pos - p_s) @ n
        wet = np.clip(depth / (2 * r) + 0.5, 0.0, 1.0) * act
        # Eddies: Ornstein-Uhlenbeck per grain, intensity from the slosh speed.
        slosh = float(np.linalg.norm(vv - self.vf)) + 0.002 * float(np.linalg.norm(self.af))
        sig = EDDY_GAIN * slosh
        k = min(1.0, dt / EDDY_TAU)
        self.eddy += -k * self.eddy + sig * math.sqrt(2 * k) * self.rng.standard_normal((N_GRAINS, 3))
        v = d.qvel[self.lin].reshape(N_GRAINS, 3)
        v_rel = v - (self.vf + self.eddy)
        speed = np.linalg.norm(v_rel, axis=1)
        mass = self.model.body_mass[self.bodies]
        c = 6 * math.pi * LIQUID_VISCOSITY * r + 0.5 * LIQUID_DENSITY * DRAG_CD * math.pi * r * r * speed
        c = np.minimum(c, 0.8 * mass / dt)                                 # explicit-stability cap
        vol = 4.0 / 3.0 * math.pi * r ** 3
        f = wet[:, None] * (LIQUID_DENSITY * vol[:, None] * (self.af - [0.0, 0.0, -9.81])
                            - c[:, None] * v_rel)
        d.xfrc_applied[self.bodies, :3] = f
        d.xfrc_applied[self.bodies, 3:] = 0.0
        self._slip_sum += speed * wet
        self._wet_sum += wet
        self._n += 1

    def contain(self, vial_floor: float, vial_h: float) -> int:
        """Containment projection, after every physics step.

        Soft contacts let a pile accelerated at several g push its outer grains
        through any shell (they creep out at ~0.1 m/s; a finer timestep made it
        worse). Grains interact with nothing but each other and the shell, so a
        grain deeper than PROJECT_TOL into the wall/floor/lid is put back on the
        surface and its outward velocity relative to the vial removed. Returns the
        number of projections (audited)."""
        d = self.data
        act = self.active
        if not act.any():
            return 0
        # Vial pose from qpos: right after mj_step, xpos/xmat still hold the pose from
        # the START of the step (one step stale: mm at shake speeds).
        vq = d.qpos[self.vial_qadr:self.vial_qadr + 7]
        vp = vq[:3].copy()
        vr9 = np.zeros(9)
        self.mj.mju_quat2Mat(vr9, vq[3:7])
        vr = vr9.reshape(3, 3)
        a = self.qadr
        pos = np.stack([d.qpos[a], d.qpos[a + 1], d.qpos[a + 2]], axis=1)
        loc = (pos - vp) @ vr
        r = self.model.geom_size[self.geoms, 0]
        rad = np.hypot(loc[:, 0], loc[:, 1])
        r_max = CONTAIN_R - r
        z_lo, z_hi = vial_floor + r, vial_h - 0.003 - r
        out_r = act & (rad > r_max + PROJECT_TOL)
        out_lo = act & (loc[:, 2] < z_lo - PROJECT_TOL)
        out_hi = act & (loc[:, 2] > z_hi + PROJECT_TOL)
        bad = out_r | out_lo | out_hi
        if not bad.any():
            return 0
        vv = d.qvel[self.vial_lin]
        for i in np.flatnonzero(bad):
            l = loc[i].copy()
            vrel = vr.T @ (d.qvel[self.dadr[i]:self.dadr[i] + 3] - vv)          # vial frame
            if out_r[i]:
                u = l[:2] / max(rad[i], 1e-9)
                l[:2] = u * r_max[i]
                vrel[:2] -= u * max(0.0, float(vrel[:2] @ u))
            if out_lo[i]:
                l[2] = z_lo[i]; vrel[2] = max(vrel[2], 0.0)
            if out_hi[i]:
                l[2] = z_hi[i]; vrel[2] = min(vrel[2], 0.0)
            d.qpos[a[i]:a[i] + 3] = vp + vr @ l
            d.qvel[self.dadr[i]:self.dadr[i] + 3] = vv + vr @ vrel
        n = int(bad.sum())
        self.projections += n
        return n

    # ---------------------------------------------------------- per control step
    def dissolve(self, dt: float, solid, saturation: float, transfer) -> float:
        """Shrink wet grains; ``transfer(dm_g) -> accepted_g`` moves mass into solution.

        Returns the slip-based agitation (mean over wet grains, in units of U_REF)."""
        n = max(self._n, 1)
        wet = self._wet_sum / n
        slip = np.where(wet > 1e-6, self._slip_sum / np.maximum(self._wet_sum, 1e-9), 0.0)
        self._slip_sum[:] = 0.0
        self._wet_sum[:] = 0.0
        self._n = 0
        act = self.active
        if not act.any() or saturation <= 0.0:
            return 0.0
        factor = solid.base_agitation + BETA * np.sqrt(slip / U_REF)
        cube = np.cbrt(self.mass) - K_GRAIN * solid.dissolution_k / 3.0 * np.cbrt(self.m0) * factor * wet * saturation * dt
        new = np.where(act, np.maximum(cube, 0.0) ** 3, self.mass)
        new = np.where(act & (new < GONE_FRACTION * self.m0), 0.0, new)   # last crumbs go too
        dm = float(np.sum(self.mass - new))
        if dm <= 0.0:
            return float(np.mean(np.sqrt(slip[wet > 0.5] / U_REF))) if (wet > 0.5).any() else 0.0
        got = transfer(dm * 1e3) * 1e-3          # kg; may be limited by saturation
        if got < dm:
            new = self.mass - (self.mass - new) * (got / dm)
        self.mass = new
        gone = act & (self.mass <= GONE_FRACTION * self.m0)
        if gone.any():
            self._retire(np.flatnonzero(gone))
        self._apply_sizes()
        wa = wet > 0.5
        return float(np.mean(np.sqrt(slip[wa] / U_REF))) if wa.any() else 0.0

    def _retire(self, idx) -> None:
        m, d = self.model, self.data
        for i in idx:
            self.active[i] = False
            g, b = self.geoms[i], self.bodies[i]
            m.geom_contype[g] = m.geom_conaffinity[g] = 0
            m.geom_rgba[g, 3] = 0.0
            m.body_gravcomp[b] = 1.0
            d.qvel[self.dadr[i]:self.dadr[i] + 6] = 0.0
            d.xfrc_applied[b] = 0.0

    # ----------------------------------------------------------------- queries
    @property
    def undissolved_kg(self) -> float:
        return float(np.sum(self.mass[self.active]))

    def escaped(self, vial_floor: float, vial_h: float) -> list[int]:
        """Active grains outside the containment volume (vial frame), 2 mm slack."""
        d = self.data
        vp, vr = d.xpos[self.vial], d.xmat[self.vial].reshape(3, 3)
        loc = (d.xpos[self.bodies] - vp) @ vr
        rad = np.hypot(loc[:, 0], loc[:, 1])
        bad = self.active & ((rad > CONTAIN_R + 0.002) | (loc[:, 2] < vial_floor - 0.002) | (loc[:, 2] > vial_h + 0.002))
        return [int(i) for i in np.flatnonzero(bad)]

    def suspended_fraction(self, vial_floor: float) -> float:
        """Mass fraction of the undissolved solid that is in suspension (moving relative
        to the vial, or off the floor pile). Drives the rendered cloudiness: a shaken
        suspension is cloudy, a settled one is clear liquid over a pile."""
        a = self.active
        if not a.any():
            return 0.0
        d = self.data
        vr = d.xmat[self.vial].reshape(3, 3)
        loc = (d.xpos[self.bodies] - d.xpos[self.vial]) @ vr
        r = self.model.geom_size[self.geoms, 0]
        vrel = np.linalg.norm(d.qvel[self.lin].reshape(N_GRAINS, 3) - d.qvel[self.vial_lin], axis=1)
        susp = a & ((vrel > SUSPENDED_SPEED) | (loc[:, 2] > vial_floor + 3.2 * r))
        return float(self.mass[susp].sum() / max(self.mass[a].sum(), 1e-12))

    def stats(self) -> dict:
        d = self.data
        vp, vr = d.xpos[self.vial], d.xmat[self.vial].reshape(3, 3)
        loc = (d.xpos[self.bodies] - vp) @ vr
        a = self.active
        return dict(n_active=int(a.sum()), undissolved_g=self.undissolved_kg * 1e3,
                    mean_height_mm=float(loc[a, 2].mean() * 1e3) if a.any() else 0.0,
                    max_speed=float(np.linalg.norm(d.qvel[self.lin].reshape(N_GRAINS, 3)[a], axis=1).max())
                    if a.any() else 0.0)


# ------------------------------------------------------------------ dev / test driver
def lift_vial_kinematic(env, lift: float = 0.10) -> None:
    """Raise vial_a and its grains together (tests, detector data): no arms involved."""
    d = env.data
    adr, _ = env.scene.free["vial_a"]
    d.qpos[adr + 2] += lift
    for a in env.grains.qadr:
        d.qpos[a + 2] += lift
    env.mj.mj_forward(env.model, d)
    env._kin_t0 = float(d.time)
    env._kin_p0 = d.qpos[adr:adr + 3].copy()


def shake_vial_kinematic(env, seconds: float, hz: float = 0.0, amp: float = 0.0, axis=(1.0, 0.0, 0.0)) -> None:
    """Prescribe a linear sinusoidal vial motion (after ``lift_vial_kinematic``) and run the
    same per-substep grain physics and per-step liquid/dissolution update as the env.
    Bypasses the arms and the contact audit: for grain unit tests and detector data only."""
    from chemlab_rl.solution_scene import VIAL_FLOOR, VIAL_H, VIAL_INNER_R
    m, d, gb = env.model, env.data, env.grains
    adr, dof = env.scene.free["vial_a"]
    q0 = d.qpos[adr + 3:adr + 7].copy()
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    w = 2 * np.pi * hz
    for _ in range(int(round(seconds / env.dt))):
        for _ in range(env.n_substeps):
            t = d.time - env._kin_t0
            ramp = min(1.0, t / 0.3)
            d.qpos[adr:adr + 3] = env._kin_p0 + ax * ramp * amp * np.sin(w * t)
            d.qpos[adr + 3:adr + 7] = q0
            d.qvel[dof:dof + 3] = ax * ramp * amp * w * np.cos(w * t)
            d.qvel[dof + 3:dof + 6] = 0.0
            gb.apply_forces(m.opt.timestep, env.liquid.vial_mL, VIAL_INNER_R, VIAL_FLOOR)
            env.mj.mj_step(m, d)
            gb.contain(VIAL_FLOOR, VIAL_H)
        env.liquid.step(env.dt, plunger_mL=env.liquid.syringe_liquid_mL + env.liquid.syringe_air_mL,
                        tip_in_reservoir_liquid=False, tip_in_vial_mouth=False, vial_tilt=0.0,
                        vial_capped=True, agitation=0.0, dissolve=False)
        env.slip_agitation = gb.dissolve(env.dt, env.liquid.solid, env.liquid.saturation_factor,
                                         env.liquid.dissolve_mass)
    env.mj.mj_forward(m, d)
