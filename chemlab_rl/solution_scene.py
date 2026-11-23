"""Rigid-body scene for subtask 2 (solution prep) on Leo's bench.

Adds to ``chem_rl_sim.scene.build_spec("weigh", "robotiq", clutter=False)``:

* ``vial_a``      receiving 20 mL vial (same asset as subtask 1), pre-loaded with solid.
* ``reservoir``   a second open 20 mL vial holding the solvent.
* ``syringe``     free body: barrel + finger flange + hub + needle (primitives).
  ``syringe_plunger`` is a CHILD body on the slide joint ``syringe_plunger_slide``
  (0 = pushed in / empty, +z = drawn). Damping and friction loss make it hold
  its stroke when released; it moves only through gripper contact.
* ``syringe_rack`` fixed stand: the hub rests on a slotted block, a collar keeps
  the barrel upright, needle points down through the slot.
* ``cap``         24-400 screw cap (primitive cylinder). Threads: ``vial_a`` carries a
  child body ``vial_a_thread`` at its mouth with a hinge ``cap_thread_turn`` and a
  slide ``cap_thread_travel`` coupled by a joint equality (one turn lowers it by
  the 8-TPI pitch). When the cap is lowered onto the rim, centred and level, the
  env activates the weld ``cap_thread`` (thread body -> cap): the ONE non-contact
  constraint (thread engagement). From then on the cap only turns/advances along
  the helix, driven by gripper contact torque; the end of the hinge range is
  the "tight" stop, with friction rising over the last part of the turn.
  Cap <-> vial contact is excluded (threads overlap the neck by design).

Liquid and dissolved solid are NOT simulated here; ``SolutionScene.update_visuals``
only renders fill heights/colours from the synthetic state the env keeps. The
UNDISSOLVED solid is physical: 50 free grain spheres inside an invisible
containment shell (``chemlab_rl.solution_grains``).
Liquid mass is not added to rigid bodies (documented simplification).

Geometry convention: the syringe body origin is the barrel's bottom centre, +z
along the barrel toward the flange. The needle tip is at -(HUB_L + NEEDLE_L).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---- fixed dimensions (m) -------------------------------------------------
BENCH = 0.775                      # urgantry BOARD_TOP; checked against the import below
VIAL_H = 0.057
VIAL_INNER_R = 0.0128              # straight-body inner radius (28 mm OD, 1.2 mm wall)
VIAL_FLOOR = 0.002
NECK_INNER_R = 0.011               # mouth opening a needle/tip must stay inside
WALL = 0.0011                      # syringe barrel wall
HUB_R, HUB_L = 0.004, 0.008       # Luer hub
NEEDLE_R, NEEDLE_L = 0.0006, 0.038
DEAD_SPACE = 0.010                 # barrel length beyond the stroke (seal + tip)
ROD_EXPOSED = 0.040                # plunger rod above the flange when empty: right-hand grip
BARREL_PINCH_H = 0.145             # barrel-hand pinch height above the bench, same for every
                                   # size (the rack height is set per size): the pads (14.6 mm
                                   # below the pinch) clear the 117 mm vial/reservoir tops either
                                   # side of the rack, and the parked syringe tops out at ~205 mm.
                                   # One shared 85 mm needle clearance put long barrels' tops at
                                   # 237 mm, into the left hand's cap/axial-grasp space.
RESERVOIR_ML = 18.0
CAP_R, CAP_HALF_H = 0.0135, 0.00625    # 24-400 cap: 27 mm OD x 12.5 mm
CAP_MASS = 0.0025                  # 24-400 polypropylene cap + liner, ~2.5 g (the vial is 20 g)
CAP_SOLIMP = (0.99, 0.9999, 0.0005, 0.5, 2)   # MuJoCo steady penetration ~ (1-dmax)/dmax x load/mass:
                                   # dmax 0.9999 (was 0.995) keeps the gripped 2.5 g cap stiff
                                   # (the handoff had raised its mass to 10 g instead)
STAND_H = 0.060                    # vial stand platform height: lifts the 57 mm vial so a
                                   # horizontal Robotiq grasp on its body clears the bench
RES_STAND_H = 0.060                # reservoir stand height (0 = on the bench, no stand)
STAND_COLLAR = 0.004               # collar height above the platform (keeps the vial upright)
STAND_CLEAR = 0.0015               # radial clearance vial <-> collar
THREAD_PITCH = 0.0254 / 8          # 24-400 finish: 8 threads per inch
THREAD_TURNS = 1.0                 # rim contact -> tight
THREAD_FRICTION = (0.004, 0.03)    # N*m hinge friction: free-running -> last 60 deg (tightening)
# Spring clamp in the vial stand (anti-rotation while ONE hand screws the cap): two rubber
# pads at +-x on spring-loaded slides squeeze the vial body. Holding torque ~ 2 mu F r.
CLAMP_PAD_R = 0.009                # roller-like capsule radius: the vial's bottom edge meets it at
                                   # ~50 deg (sin = 1 - reach/R), past the friction angle of mu 1.2,
                                   # so inserting the vial wedges the pads apart (3 mm pads jammed)
CLAMP_PAD_HALF = 0.006             # capsule half-length (tangential)
CLAMP_Z = 0.010                    # pad axis above the platform top (top 19 mm: under the fingers)
CLAMP_REACH = 0.0010               # pad intrudes this far into the vial radius at its stop
CLAMP_K = 400.0                    # N/m
CLAMP_PRELOAD = 0.002              # m of spring pre-compression at the stop
CLAMP_MU = 1.2


@dataclass(frozen=True)
class SyringeGeom:
    capacity_mL: float
    inner_r: float
    stroke: float

    @property
    def outer_r(self) -> float:
        return self.inner_r + WALL

    @property
    def barrel_len(self) -> float:
        return self.stroke + DEAD_SPACE

    @property
    def rest_z(self) -> float:
        """Barrel-bottom height above the bench when standing in the rack."""
        return BARREL_PINCH_H - self.grip_z

    @property
    def grip_z(self) -> float:
        """Left-hand pinch height on the barrel, body frame."""
        return self.barrel_len - 0.020

    @property
    def rod_grip_z(self) -> float:
        """Right-hand pinch on the rod above the flange, plunger-body frame (stroke 0)."""
        return self.barrel_len + 0.026

    @classmethod
    def from_syringe(cls, s) -> "SyringeGeom":
        return cls(float(s.capacity_mL), float(s.barrel_inner_radius_m), float(s.stroke_m))


# Default bench layout (bench xy). Randomized per episode by SolutionScene.prepare.
# Chosen by a reach/collision search over all four syringe sizes (both arms,
# rack grasp, reservoir and vial insertion at empty/full stroke): IK < 0.3 mm,
# no penetration. The plunger arm (right) must approach from the robot side
# (tool +y); approaching from +x drives its forearm through the balance.
LAYOUT = {
    "vial_a": (-0.14, 0.02),
    "reservoir": (-0.27, 0.02),
    "syringe_rack": (-0.205, 0.08),
    "cap": (-0.30, -0.10),          # left-front: clear of the balance and of the right hand holding the vial
}

# Inspection backdrop: a two-tone panel behind the prep area. The vial is inspected in
# front of whichever half contrasts with the solid (chosen at test time from a first look).
# Inspection station. Measured arm swept volumes leave no room for a camera mount near the
# work area, so the camera is a long-lens camera on a tripod BEHIND the bench (out of both
# arms' reach) looking down at the inspection spot, and the contrasting backdrop is a flat
# two-tone mat on the free bench surface where its line of sight lands beyond the vial
# (like holding a vial over a black/white card). Shifting the vial +-INSPECT_SHIFT in x puts
# the dark or the light half behind it.
INSPECT_X, INSPECT_Y, INSPECT_BASE_Z = -0.38, 0.12, 1.12     # spot (vial base), left arm's own area
INSPECT_SHIFT = 0.03
INSPECT_CAM_DIST = 1.20            # m from the aim point, toward +y, INSPECT_CAM_ELEV_DEG up
INSPECT_CAM_ELEV_DEG = 35.0
INSPECT_CAM_FOVY = 8.0             # long lens: the vial is ~150 px wide in an 896 px frame
INSPECT_AIM_DZ = 0.012             # aim point above the vial base (lower liquid region)
MAT_HALF = 0.12                    # mat half-size (each colour half is MAT_HALF x 2*MAT_HALF)
BACKDROP_RGB = {"dark": (0.07, 0.07, 0.08), "light": (0.93, 0.93, 0.91)}


def _inspect_geometry():
    a = np.deg2rad(INSPECT_CAM_ELEV_DEG)
    u = np.array([0.0, np.cos(a), np.sin(a)])                 # aim -> camera
    aim = np.array([INSPECT_X, INSPECT_Y, INSPECT_BASE_Z + INSPECT_AIM_DZ])
    cam = aim + INSPECT_CAM_DIST * u
    t = (aim[2] - BENCH) / np.sin(a)                           # ray beyond the aim to the bench
    mat = aim - t * u
    return cam, u, np.array([mat[0], mat[1], BENCH])

WHITE = [0.95, 0.94, 0.90, 1.0]
STEEL = [0.75, 0.76, 0.80, 1.0]
CLEAR = [0.85, 0.92, 0.96, 0.35]
RACK = [0.20, 0.35, 0.65, 1.0]
SOLVENT = [0.55, 0.75, 0.95, 0.55]
POWDER = [0.92, 0.88, 0.55, 1.0]


def _cyl(body, name, r, half, pos, rgba, collide=True, mass=None, **kw):
    import mujoco as mj
    g = body.add_geom(name=name, type=mj.mjtGeom.mjGEOM_CYLINDER, size=[r, half, 0],
                      pos=list(pos), rgba=list(rgba), **kw)
    if not collide:
        g.contype = g.conaffinity = 0
        g.mass = 0
    elif mass is not None:
        g.mass = mass
    return g


def _box(body, name, half, pos, rgba, **kw):
    import mujoco as mj
    return body.add_geom(name=name, type=mj.mjtGeom.mjGEOM_BOX, size=list(half), pos=list(pos),
                         rgba=list(rgba), **kw)


def add_syringe(spec, geom: SyringeGeom, rack_xy) -> None:
    """Rack (fixed) + syringe (free) + plunger (slide child)."""
    import mujoco as mj
    wb = spec.worldbody
    ro, L = geom.outer_r, geom.barrel_len
    top = BENCH + geom.rest_z - HUB_L          # rack plateau = hub bottom
    rack = wb.add_body(name="syringe_rack", pos=[*rack_xy, BENCH])
    gap = 0.0025                               # needle slot half-width (< HUB_R)
    w = 0.018
    for k, sx in enumerate((-1, 1)):
        _box(rack, f"syringe_rack_block_{k}", [w / 2, 0.018, (top - BENCH) / 2],
             [sx * (gap + w / 2), 0, (top - BENCH) / 2], RACK)
    collar_lo, collar_hi = top - BENCH, top - BENCH + HUB_L + 0.010
    inner = ro + 0.0015                        # collar clearance: tolerant re-insertion
    t = 0.003
    for k, (dx, dy, hx, hy) in enumerate(((inner + t, 0, t, inner + 2 * t), (-inner - t, 0, t, inner + 2 * t),
                                          (0, inner + t, inner, t), (0, -inner - t, inner, t))):
        _box(rack, f"syringe_rack_collar_{k}", [hx, hy, (collar_hi - collar_lo) / 2],
             [dx, dy, (collar_lo + collar_hi) / 2], RACK)

    body = wb.add_body(name="syringe", pos=[*rack_xy, BENCH + geom.rest_z])
    body.add_freejoint(name="syringe_joint")
    fr = dict(friction=[1.2, 0.01, 0.0001], condim=4, solref=[0.005, 1])
    barrel_mass = 0.002 + 0.9 * geom.capacity_mL * 1e-3
    _cyl(body, "syringe_barrel", ro, L / 2, [0, 0, L / 2], CLEAR, mass=barrel_mass, **fr)
    _box(body, "syringe_flange", [ro + 0.007, 0.0045, 0.001], [0, 0, L + 0.001], CLEAR, mass=0.0005, **fr)
    _cyl(body, "syringe_hub", HUB_R, HUB_L / 2, [0, 0, -HUB_L / 2], CLEAR, mass=0.0005, **fr)
    needle = body.add_geom(name="syringe_needle", type=mj.mjtGeom.mjGEOM_CAPSULE,
                           fromto=[0, 0, -HUB_L, 0, 0, -HUB_L - NEEDLE_L + NEEDLE_R],
                           size=[NEEDLE_R, 0, 0], rgba=STEEL, mass=0.0003)
    needle.solref = [0.005, 1]
    body.add_site(name="syringe_tip", pos=[0, 0, -HUB_L - NEEDLE_L], size=[0.001, 0, 0], group=4)
    # Visual liquid column inside the barrel (resized by update_visuals).
    _cyl(body, "syringe_liquid", geom.inner_r * 0.95, 0.0001, [0, 0, 0.0001], SOLVENT, collide=False)

    plunger = body.add_body(name="syringe_plunger", pos=[0, 0, 0])
    slide = plunger.add_joint(name="syringe_plunger_slide", type=mj.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                      range=[0, geom.stroke], limited=True, damping=3.0, frictionloss=0.4,
                      armature=0.002)
    slide.solref_limit = [0.004, 1]       # stiff end stops (seal at the barrel tip)
    seal_half = 0.003
    _cyl(plunger, "syringe_seal", geom.inner_r * 0.98, seal_half, [0, 0, seal_half],
         [0.15, 0.15, 0.15, 1], mass=0.0005, contype=0, conaffinity=0)
    rod_r = max(0.0022, 0.7 * geom.inner_r)
    rod_lo, rod_hi = 2 * seal_half, L + ROD_EXPOSED
    _cyl(plunger, "syringe_rod", rod_r, (rod_hi - rod_lo) / 2, [0, 0, (rod_lo + rod_hi) / 2],
         WHITE, mass=0.002, **fr)
    _cyl(plunger, "syringe_thumb", ro + 0.003, 0.001, [0, 0, rod_hi + 0.001], WHITE, mass=0.0005, **fr)


def add_vial_stand(spec, xy, name: str = "vial_stand") -> None:
    """Fixed stand: platform + 4-sided collar (open-top socket)."""
    st = spec.worldbody.add_body(name=name, pos=[*xy, BENCH])
    half = 0.022
    _box(st, f"{name}_platform", [half, half, STAND_H / 2], [0, 0, STAND_H / 2], RACK)
    inner, t = 0.014 + STAND_CLEAR, 0.003
    lo, hi = STAND_H, STAND_H + STAND_COLLAR
    for k, (dx, dy, hx, hy) in enumerate(((inner + t, 0, t, inner + 2 * t), (-inner - t, 0, t, inner + 2 * t),
                                          (0, inner + t, inner, t), (0, -inner - t, inner, t))):
        _box(st, f"{name}_collar_{k}", [hx, hy, (hi - lo) / 2], [dx, dy, (lo + hi) / 2], RACK)
    if name == "vial_stand":
        add_clamp(st)
        for k in range(2):                    # pads ride in the collar (world-fixed parent:
            spec.add_exclude(bodyname1=f"vial_clamp_{k}", bodyname2=name)   # no auto filter)


def add_clamp(st) -> None:
    """Two spring-loaded rubber pads (+-y: the body grip's pads go at +-x) on slides that
    retract outward. At the stop each
    pad reaches CLAMP_REACH inside the 14 mm vial radius, so a seated vial pushes them back
    ~2 mm against CLAMP_K with CLAMP_PRELOAD: ~3 N per pad, ~0.1 N m of holding torque."""
    import mujoco as mj
    c = 0.014 - CLAMP_REACH + CLAMP_PAD_R                 # pad axis radius at the stop
    for k, sy in enumerate((1.0, -1.0)):
        b = st.add_body(name=f"vial_clamp_{k}", pos=[0, sy * c, STAND_H + CLAMP_Z])
        b.add_joint(name=f"vial_clamp_{k}_slide", type=mj.mjtJoint.mjJNT_SLIDE, axis=[0, sy, 0],
                    range=[0.0, 0.010], limited=True, stiffness=CLAMP_K, springref=-CLAMP_PRELOAD,
                    damping=4.0, armature=0.01)
        g = b.add_geom(name=f"vial_clamp_pad_{k}", type=mj.mjtGeom.mjGEOM_CAPSULE,
                       size=[CLAMP_PAD_R, CLAMP_PAD_HALF, 0], quat=[0.7071068, 0, 0.7071068, 0],
                       rgba=[0.12, 0.12, 0.12, 1], mass=0.03, friction=[CLAMP_MU, 0.02, 0.0001],
                       condim=4, priority=3, solref=[0.004, 1],
                       solimp=[0.99, 0.9999, 0.001, 0.5, 2])   # stiff: light pads, mass-scaled contacts
        # spring housing (cosmetic, part of the stand)
        _box(st, f"vial_clamp_housing_{k}", [0.008, 0.004, 0.005],
             [0, sy * (c + CLAMP_PAD_R + 0.004), STAND_H + CLAMP_Z], [0.3, 0.3, 0.32, 1])


def add_cap(spec, xy) -> None:
    body = spec.worldbody.add_body(name="cap", pos=[*xy, BENCH + CAP_HALF_H])
    body.add_freejoint(name="cap_joint")
    _cyl(body, "cap_shell", CAP_R, CAP_HALF_H, [0, 0, 0], WHITE, mass=CAP_MASS,
         friction=[1.2, 0.01, 0.0001], condim=4, solref=[0.004, 1],
         solimp=list(CAP_SOLIMP), priority=2)   # rigid PP cap; beats pad priority 1


def add_fill_visuals(spec) -> None:
    for name in ("vial_a", "reservoir"):
        b = spec.body(name)
        _cyl(b, f"{name}_liquid", VIAL_INNER_R * 0.97, 0.0001, [0, 0, VIAL_FLOOR], SOLVENT, collide=False)


GRIP_BIT = 4                       # collision bit shared by gripper pads and the vial grip surface
GRIP_R, GRIP_Z0, GRIP_HALF = 0.0145, 0.015, 0.0175   # vial grip cylinder: r, bottom, half height


def build_spec(syringe: SyringeGeom, layout: dict | None = None):
    import mujoco as mj
    from chem_rl_sim.assets import add_asset
    from chem_rl_sim.scene import BOARD_TOP, build_spec as leo_build_spec
    assert abs(BOARD_TOP - BENCH) < 1e-9, "bench height changed upstream"
    layout = {**LAYOUT, **(layout or {})}
    spec = leo_build_spec("weigh", "robotiq", clutter=False)
    # Leo's weigh layout already contains vial_a; move it to the prep position.
    spec.body("vial_a").pos = [*layout["vial_a"], BENCH + STAND_H]
    # Gripper pads are condim 3 at priority 1, so pad-on-vial contacts had NO torsional
    # friction: a vial held by its body crept about the jaw line under shaking and fell out.
    # The vial's glass geoms take priority 2 with torsional friction 0.01 m (a ~1 cm rubber
    # patch on glass) and pad-like stiffness; the clamp pads (priority 3) still win.
    for g in spec.body("vial_a").geoms:
        if g.name.startswith("vial_a_collision"):
            g.priority, g.condim = 2, 4
            g.friction = [1.0, 0.01, 0.0001]
            g.solref, g.solimp = [0.004, 1.0], [0.95, 0.99, 0.001, 0.5, 2.0]
    # Grip surface: the glass collider is 12 thin slabs (1.6 mm); pads pressed 0.4 mm into them
    # and, under shaking, box-box normals flipped along the slab and walked the vial 5 mm per
    # burst down the pads (independent of friction). A solid cylinder 0.5 mm proud of the 28 mm
    # body, on collision bit 4 that ONLY the gripper pads see (the needle, grains, stand and
    # clamp still meet the hollow slab shell).
    vb = spec.body("vial_a")
    gc = vb.add_geom(name="vial_a_grip", type=mj.mjtGeom.mjGEOM_CYLINDER, size=[GRIP_R, GRIP_HALF, 0],
                     pos=[0, 0, GRIP_Z0 + GRIP_HALF], rgba=[0, 0, 0, 0], mass=0, priority=2, condim=4,
                     friction=[1.0, 0.01, 0.0001], solref=[0.004, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2.0])
    gc.contype, gc.conaffinity, gc.group = GRIP_BIT, 0, 5
    add_vial_stand(spec, layout["vial_a"])
    # The reservoir sits on its own stand too: drawing then happens at the same
    # height as dispensing, inside both arms' comfortable reach.
    add_asset(spec, spec.worldbody, "vial_20ml_open", "reservoir", (*layout["reservoir"], BENCH + RES_STAND_H))
    if RES_STAND_H > 0:
        add_vial_stand(spec, layout["reservoir"], "reservoir_stand")
    add_syringe(spec, syringe, layout["syringe_rack"])
    add_cap(spec, layout["cap"])
    add_fill_visuals(spec)
    add_thread(spec)
    add_backdrop(spec)
    add_inspect_camera(spec)
    # Close-up inspection (wrist camera ~8 cm from the vial): the near plane is relative to
    # the 12 m scene extent (default 0.01 -> 12.5 cm would clip the vial), and the detector
    # renders the wrist camera at its native 896 px.
    spec.visual.map.znear = 0.002
    # Friction-cone softness: Leo's parent spec keeps impratio 1 over the Robotiq model's 10
    # (an attach-conflict warning). At 1, a vial hanging from a cap pinch creeps ~2 deg/s
    # about the jaw axis and slips out in ~10 s; at 10 it holds (1.3 deg, stable).
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, 1024)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, 1024)
    from chemlab_rl.solution_grains import add_grains
    add_grains(spec, VIAL_FLOOR, VIAL_H)
    for g in _all_geoms(spec):                 # gripper pads also see the vial grip surface
        if "_grip_" in (g.name or "") and "_pad" in (g.name or ""):
            g.conaffinity = int(g.conaffinity) | GRIP_BIT
    return spec


def _all_geoms(spec):
    out, stack = [], [spec.worldbody]
    while stack:
        b = stack.pop()
        out.extend(b.geoms)
        stack.extend(b.bodies)
    return out


def add_backdrop(spec) -> None:
    """Two-tone mat on the bench (2 mm, matte), dark half at -x, light half at +x."""
    _, _, mat = _inspect_geometry()
    b = spec.worldbody.add_body(name="inspect_backdrop", pos=[mat[0], mat[1], BENCH])
    for half, sx in (("dark", -1.0), ("light", 1.0)):
        _box(b, f"backdrop_{half}", [MAT_HALF / 2, MAT_HALF, 0.001], [sx * MAT_HALF / 2, 0, 0.001],
             [*BACKDROP_RGB[half], 1.0])


def inspect_spot(half: str) -> np.ndarray:
    """World position of the vial BASE centre for inspection against `half`."""
    sx = -1.0 if half == "dark" else 1.0          # dark half is at -x
    return np.array([INSPECT_X + sx * INSPECT_SHIFT, INSPECT_Y, INSPECT_BASE_Z])


def add_inspect_camera(spec) -> None:
    """Fixed long-lens camera 'inspect' (tripod behind the bench: no geometry in the cell)."""
    cam, u, _ = _inspect_geometry()
    up = np.array([0.0, 0, 1]) - u * u[2]
    up /= np.linalg.norm(up)
    right = np.cross(-u, up)                                    # forward x up
    spec.worldbody.add_camera(name="inspect", pos=list(cam), fovy=INSPECT_CAM_FOVY,
                              xyaxes=[*right, *up])


def add_thread(spec) -> None:
    """Helical thread on vial_a's neck and the (inactive) engagement weld."""
    import mujoco as mj
    th = spec.body("vial_a").add_body(name="vial_a_thread", pos=[0, 0, VIAL_H])
    th.add_geom(name="vial_a_thread_mass", type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.002, 0, 0],
                mass=1e-4, contype=0, conaffinity=0, rgba=[0, 0, 0, 0])
    full = 2 * np.pi * THREAD_TURNS
    # Hinge ranges are read in the spec's angle unit (MJCF defaults to degrees).
    full_spec = np.degrees(full) if spec.compiler.degree else full
    th.add_joint(name="cap_thread_turn", type=mj.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                 range=[-full_spec, 0.0], limited=True, damping=0.002, frictionloss=THREAD_FRICTION[0],
                 armature=2e-4)       # constraint stiffness scales with inertia: keep it real-sized
    th.add_joint(name="cap_thread_travel", type=mj.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                 range=[-THREAD_PITCH * THREAD_TURNS, 0.0], limited=True, damping=0.5, armature=0.05)
    coupling = spec.add_equality(name="cap_thread_helix", type=mj.mjtEq.mjEQ_JOINT,
                                 name1="cap_thread_travel", name2="cap_thread_turn")
    coupling.data[0:5] = [0.0, THREAD_PITCH / (2 * np.pi), 0, 0, 0]   # travel = pitch/2pi * turn
    # Shake hold (sim limitation, see SolutionPrepEnv._shake_hold): active ONLY while a shake
    # burst runs on a vial already held between the left pads, at the current relative pose.
    hold = spec.add_equality(name="left_shake_hold", type=mj.mjtEq.mjEQ_WELD, objtype=mj.mjtObj.mjOBJ_BODY,
                             name1="left_grip_base", name2="vial_a")
    hold.active = False
    hold.solref = [0.004, 1]
    weld = spec.add_equality(name="cap_thread", type=mj.mjtEq.mjEQ_WELD, objtype=mj.mjtObj.mjOBJ_BODY,
                             name1="vial_a_thread", name2="cap")
    weld.active = False
    weld.solref = [0.004, 1]
    spec.add_exclude(bodyname1="cap", bodyname2="vial_a")


class SolutionScene:
    """Compiled model + data and the ids the env, skills and Session need.

    Duck-types what ``chem_rl_sim.teleop.keyframe.Session`` reads from its env
    (``model``, ``data``, ``hand``, ``n_substeps``) through the env bridge.
    """

    hand = "robotiq"
    FREE = ("vial_a", "reservoir", "syringe", "cap")

    def __init__(self, syringe, layout: dict | None = None, control_dt: float = 0.1):
        import mujoco as mj
        self.mj = mj
        self.geom = syringe if isinstance(syringe, SyringeGeom) else SyringeGeom.from_syringe(syringe)
        self.base_layout = {**LAYOUT, **(layout or {})}
        self.spec = build_spec(self.geom, self.base_layout)
        self.model = m = self.spec.compile()
        self.data = mj.MjData(m)
        self.n_substeps = round(control_dt / m.opt.timestep)
        self.free = {n: (int(m.joint(f"{n}_joint").qposadr[0]), int(m.joint(f"{n}_joint").dofadr[0]))
                     for n in self.FREE}
        j = m.joint("syringe_plunger_slide")
        self.plunger_qadr, self.plunger_dof = int(j.qposadr[0]), int(j.dofadr[0])
        self.cap_eq = m.equality("cap_thread").id
        self.hold_eq = m.equality("left_shake_hold").id
        jt, js = m.joint("cap_thread_turn"), m.joint("cap_thread_travel")
        self.thread_qadr = (int(jt.qposadr[0]), int(js.qposadr[0]))
        self.thread_dof = (int(jt.dofadr[0]), int(js.dofadr[0]))
        self.thread_body = m.body("vial_a_thread").id
        self.rack_body = m.body("syringe_rack").id
        self.stand_body = m.body("vial_stand").id
        self.clamp_qadr = [int(m.joint(f"vial_clamp_{k}_slide").qposadr[0]) for k in range(2)]
        self.res_stand_body = m.body("reservoir_stand").id if RES_STAND_H > 0 else None
        self.arm_qadr = {s: np.array([m.joint(f"{s}_{jn}_joint").qposadr[0] for jn in _ARM_JOINTS])
                         for s in ("left", "right")}
        self._ctrl_low, self._ctrl_high = m.actuator_ctrlrange.T.copy()
        self._vis = {n: m.geom(n).id for n in ("vial_a_liquid", "reservoir_liquid",
                                               "syringe_liquid")}
        from chemlab_rl.solution_grains import GrainBed
        self.grains = GrainBed(m, self.data)
        self.layout = dict(self.base_layout)
        self.prepare()

    # ------------------------------------------------------------------ reset
    def prepare(self, rng: np.random.Generator | None = None, jitter_xy: float = 0.0,
                jitter_yaw_deg: float = 0.0, solid_g: float = 0.1, grain_seed: int | None = None,
                solid_rgb=(0.93, 0.90, 0.62)) -> dict:
        """Reset to the start state. With rng, prop xy/yaw are jittered.

        The rack is a fixed body: its jitter moves model.body_pos (and the
        syringe resting in it) so every reset is a physically settled scene.
        """
        mj, m, d = self.mj, self.model, self.data
        from chem_rl_sim.scene import set_initial_pose
        mj.mj_resetData(m, d)
        set_initial_pose(m, d, "weigh")
        lay, yaw = {}, {}
        for name, xy in self.base_layout.items():
            off = np.zeros(2) if rng is None else rng.uniform(-jitter_xy, jitter_xy, 2)
            lay[name] = np.asarray(xy, float) + off
            yaw[name] = 0.0 if rng is None else float(np.deg2rad(rng.uniform(-jitter_yaw_deg, jitter_yaw_deg)))
        m.body_pos[self.rack_body] = [*lay["syringe_rack"], BENCH]
        m.body_pos[self.stand_body] = [*lay["vial_a"], BENCH]
        if self.res_stand_body is not None:
            m.body_pos[self.res_stand_body] = [*lay["reservoir"], BENCH]
        q = lambda a: [np.cos(a / 2), 0, 0, np.sin(a / 2)]
        poses = {
            "vial_a": ([*lay["vial_a"], BENCH + STAND_H], q(yaw["vial_a"])),
            "reservoir": ([*lay["reservoir"], BENCH + RES_STAND_H], q(yaw["reservoir"])),
            "syringe": ([*lay["syringe_rack"], BENCH + self.geom.rest_z], [1, 0, 0, 0]),
            "cap": ([*lay["cap"], BENCH + CAP_HALF_H], q(yaw["cap"])),
        }
        for name, (p, quat) in poses.items():
            adr, _ = self.free[name]
            d.qpos[adr:adr + 7] = [*p, *quat]
        mj.mj_kinematics(m, d)
        self.grains.place(solid_g, grain_seed, VIAL_FLOOR, solid_rgb)   # dry pile on the vial floor
        d.qpos[self.clamp_qadr] = CLAMP_REACH + 0.0001        # pads resting on the seated vial
        d.qpos[self.plunger_qadr] = 0.0
        d.eq_active[self.cap_eq] = 0
        d.eq_active[self.hold_eq] = 0
        d.qpos[list(self.thread_qadr)] = 0.0
        m.dof_frictionloss[self.thread_dof[0]] = THREAD_FRICTION[0]
        arm_act = {s: np.array([m.actuator(f"{s}_{jn}").id for jn in _ARM_JOINTS]) for s in ("left", "right")}
        for s, acts in arm_act.items():
            d.ctrl[acts] = d.qpos[self.arm_qadr[s]]
            d.ctrl[m.actuator(f"{s}_grip_fingers_actuator").id] = 0
        self.carried = []
        self.bad = None
        self.pinned = None
        self.max_pen = 0.0
        self._step_count = 0
        self.layout = {k: tuple(v) for k, v in lay.items()}
        self.update_visuals(reservoir_mL=RESERVOIR_ML, vial_mL=0.0, syringe_mL=0.0,
                            solid_g=0.1, turbidity=0.0)
        for _ in range(300):          # 0.6 s settle under gravity, arms holding home
            mj.mj_step(m, d)
        mj.mj_forward(m, d)
        return {"layout": self.layout, "yaw": yaw}

    # ---------------------------------------------------------------- visuals
    def _column(self, gid: int, height: float, z0: float) -> None:
        m = self.model
        h = max(1e-4, height)
        m.geom_size[gid, 1] = h / 2
        m.geom_pos[gid, 2] = z0 + h / 2
        m.geom_rgba[gid, 3] = 0.0 if height < 2e-4 else m.geom_rgba[gid, 3] or SOLVENT[3]

    def update_visuals(self, *, reservoir_mL: float, vial_mL: float, syringe_mL: float,
                       solid_g: float, turbidity: float, vial_rgba=None) -> None:
        """``vial_rgba`` (if given) sets the vial liquid colour directly (the env renders
        cloudiness from SUSPENDED grains and tint from DISSOLVED solute)."""
        area_v = np.pi * VIAL_INNER_R ** 2
        self._column(self._vis["reservoir_liquid"], reservoir_mL * 1e-6 / area_v, VIAL_FLOOR)
        self._column(self._vis["vial_a_liquid"], vial_mL * 1e-6 / area_v, VIAL_FLOOR)
        g = self._vis["vial_a_liquid"]
        tb = float(np.clip(turbidity, 0, 1))
        rgb = (1 - tb) * np.array(SOLVENT[:3]) + tb * np.array(POWDER[:3])
        self.model.geom_rgba[g, :3] = rgb
        if vial_mL > 1e-3:
            self.model.geom_rgba[g, 3] = 0.55 + 0.4 * tb
        if vial_rgba is not None:
            self.model.geom_rgba[g] = vial_rgba
            if vial_mL <= 1e-3:
                self.model.geom_rgba[g, 3] = 0.0
        area_s = np.pi * self.geom.inner_r ** 2
        self._column(self._vis["syringe_liquid"], syringe_mL * 1e-6 / area_s, 0.0)

    # --------------------------------------------------------------- geometry
    def body_pose(self, name: str):
        b = self.data.body(name)
        return b.xpos.copy(), b.xmat.reshape(3, 3).copy()

    def needle_tip(self) -> np.ndarray:
        return self.data.site("syringe_tip").xpos.copy()

    def plunger_stroke(self) -> float:
        return float(self.data.qpos[self.plunger_qadr])

    def thread_turn(self) -> float:
        """Tightening angle in rad (0 at engagement, 2*pi*THREAD_TURNS when tight)."""
        return -float(self.data.qpos[self.thread_qadr[0]])

    def mouth(self, vial: str) -> tuple[np.ndarray, np.ndarray]:
        p, r = self.body_pose(vial)
        return p + r @ np.array([0, 0, VIAL_H]), r


_ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
