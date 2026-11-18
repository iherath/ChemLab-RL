"""Profile-following vial colliders and a small physical powder scoop.

Dimensions are metres. Vial visuals are preserved; old collision hulls are
made non-colliding and massless. The new 24-sided collider never exceeds the
visual outer radii. Faceting conservatively reduces the cavity inradius by
up to 0.110mm; the neck clear diameter is 21.911mm (nominal 22.10mm).
"""
from __future__ import annotations

import math
from typing import Callable

import mujoco
import numpy as np

VIAL_BANDS = (
    ("body", .002, .047, .0128, .0128, .014, .014),
    ("shoulder", .047, .051, .0128, .01105, .014, .01225),
    ("neck", .051, .057, .01105, .01105, .01225, .01225),
)
VIAL_MASS_KG = .020
VIAL_SEGMENTS = 24
SCOOP_TIP = np.array([.26, 0., 0.])
# Columns are the cup's(+X,+Y,+Z) axes expressed in the tool frame.
SCOOP_FRAME = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
SCOOP_OUTER_RADIUS = .007
SCOOP_INNER_RADIUS = .006
SCOOP_DEPTH = .006
SCOOP_BOTTOM = .001
SCOOP_SEGMENTS = 16


def _quat(rotation):
    out = np.empty(4)
    mujoco.mju_mat2Quat(out, np.asarray(rotation, float).reshape(-1))
    return out.tolist()


def _disable(geom, *, hide=False):
    geom.contype = geom.conaffinity = 0
    geom.mass = 0
    if hide:
        rgba = np.asarray(geom.rgba).copy()
        rgba[3] = 0
        geom.rgba = rgba


def replace_vial_collision(spec, body):
    """Install 24 convex wedges per profile band plus a 2 mm bottom disk.

    Call after ``add_asset(..., 'vial_20ml_open', ...)`` and before compile.
    Returns dimensions used for numeric validation. Repeated calls are rejected.
    """
    prefix = body.name
    if any(g.name.startswith(prefix + "_contact_") for g in body.geoms):
        raise ValueError(f"Accurate vial collision already installed for {prefix}")
    previous = [g for g in body.geoms if g.name.startswith(prefix + "_collision_")]
    if not previous:
        raise ValueError(f"No original vial collision hulls found for {prefix}")
    for geom in previous:
        _disable(geom)
    segments = VIAL_SEGMENTS
    angle = 2 * math.pi / segments
    pieces = []
    for name, z0, z1, ri0, ri1, ro0, ro1 in VIAL_BANDS:
        # Integral of the quadrilateral cross-sectional area along the band.
        volume = ((z1-z0) * math.sin(angle) / 6 *
                  (ro0*ro0 + ro0*ro1 + ro1*ro1 - ri0*ri0 - ri0*ri1 - ri1*ri1))
        for k in range(segments):
            a, b = k*angle, (k+1)*angle
            vertices = np.array([[radius*math.cos(theta), radius*math.sin(theta), z]
                                 for z, ri, ro in ((z0, ri0, ro0), (z1, ri1, ro1))
                                 for radius in (ri, ro) for theta in (a, b)])
            pieces.append((f"{prefix}_contact_{name}_{k}", vertices, volume))
    bottom_volume = math.pi * .014**2 * .002
    total_volume = bottom_volume + sum(volume for _, _, volume in pieces)
    for name, vertices, volume in pieces:
        faces = np.array([[0,1,3],[0,3,2], [4,6,7],[4,7,5],
                          [0,4,5],[0,5,1], [2,3,7],[2,7,6],
                          [0,2,6],[0,6,4], [1,5,7],[1,7,3]])
        spec.add_mesh(name=name + "_mesh", uservert=vertices.ravel().tolist(),
                      userface=faces.ravel().tolist())
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=name + "_mesh", group=3, rgba=[.9, .3, .3, 0],
                      mass=VIAL_MASS_KG * volume / total_volume,
                      friction=[1., .005, .0001], condim=4, solref=[.005, 1])
    body.add_geom(name=prefix + "_contact_bottom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[.014, .001], pos=[0, 0, .001], group=3, rgba=[.9, .3, .3, 0],
                  mass=VIAL_MASS_KG * bottom_volume / total_volume,
                  friction=[1., .005, .0001], condim=4, solref=[.005, 1])
    return dict(mass_kg=VIAL_MASS_KG, height_m=.057, outer_diameter_m=.028,
                nominal_neck_inner_diameter_m=.0221,
                minimum_neck_inner_diameter_m=2*.01105*math.cos(math.pi/segments),
                maximum_radial_faceting_error_m=.0128*(1-math.cos(math.pi/segments)),
                segments=segments, collision_geoms=len(pieces)+1)


def add_scoop(spec, tool):
    """Replace the distal rod with a connected shaft and 14 mm scoop cup.

    Preserve the existing handle and short neck. Cup opening is toward the
    handle(-toolX), so a tip-down tool holds the cup open upward. Geometry
    dimensions match the contact-transfer fixture; density is 8000 kg/m3.
    """
    prefix = tool.name
    if any(g.name.startswith(prefix + "_scoop_") for g in tool.geoms):
        raise ValueError(f"Scoop already installed for {prefix}")
    for geom in tool.geoms:
        if geom.name == prefix + "_blade":
            _disable(geom, hide=True)
    props = dict(friction=[.4, .0001, .00002], condim=3,
                 solref=[.0005, 1], solimp=[.95, .99, .0001, .5, 2],
                 density=8000., rgba=[.65, .7, .75, 1])
    tool.add_geom(name=prefix + "_scoop_shaft", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=[.030, 0, 0, .220, -.0055, 0], size=[.001], **props)
    tool.add_geom(name=prefix + "_scoop_stem", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=[.220, -.0055, 0, .255, -.0055, 0], size=[.001], **props)
    tool.add_geom(name=prefix + "_scoop_bottom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[SCOOP_OUTER_RADIUS, SCOOP_BOTTOM/2],
                  pos=(SCOOP_TIP + SCOOP_FRAME @ [0, 0, SCOOP_BOTTOM/2]).tolist(),
                  quat=_quat(SCOOP_FRAME), **props)
    outer_plane = SCOOP_OUTER_RADIUS * math.cos(math.pi/SCOOP_SEGMENTS)
    radial_center = (outer_plane + SCOOP_INNER_RADIUS)/2
    for k in range(SCOOP_SEGMENTS):
        a = 2*math.pi*k/SCOOP_SEGMENTS
        c, sn = math.cos(a), math.sin(a)
        rotation = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.]])
        tool.add_geom(name=f"{prefix}_scoop_wall_{k}", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[(outer_plane-SCOOP_INNER_RADIUS)/2,
                            SCOOP_OUTER_RADIUS*math.sin(math.pi/SCOOP_SEGMENTS), SCOOP_DEPTH/2],
                      pos=(SCOOP_TIP + SCOOP_FRAME @ [radial_center*c, radial_center*sn, SCOOP_DEPTH/2]).tolist(),
                      quat=_quat(SCOOP_FRAME @ rotation), **props)
    return dict(outer_diameter_m=2*SCOOP_OUTER_RADIUS, inner_inradius_m=SCOOP_INNER_RADIUS,
                depth_m=SCOOP_DEPTH, bottom_thickness_m=SCOOP_BOTTOM,
                tip_in_tool_m=SCOOP_TIP.tolist(), opening_axis_in_tool=SCOOP_FRAME[:, 2].tolist(),
                steel_density_kg_m3=8000.)


def scoop_membership(model, data, bodyname="spatula", *, tip_in_tool=None, cup_frame=None) -> Callable:
    """Return callback(body_id, world_position, radius) for supported cup grains.

    A grain must fit the cup's interior and have a positive-force contact path to
    the tool. Contacts are cached once per physics time, not rebuilt for each
    grain. The envelope includes a one-grain heap above the rim; it never
    counts an airborne grain solely because that grain passes over the cup.
    While the cup is in the source bed this is occupied, supported cup volume;
    retention is established only after the cup is lifted free of the bed.
    Optional origin/frame arguments support an equivalent isolated cup fixture.
    """
    tool_body = model.body(bodyname).id
    origin = SCOOP_TIP if tip_in_tool is None else np.asarray(tip_in_tool, dtype=float)
    frame = SCOOP_FRAME if cup_frame is None else np.asarray(cup_frame, dtype=float)
    tool_bodies = {tool_body}
    for body in range(tool_body+1, model.nbody):
        if int(model.body_parentid[body]) in tool_bodies:
            tool_bodies.add(body)
    sphere_geoms = {}
    radii = {}
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if (model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and
                model.body_jntnum[b] == 1 and
                model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE):
            sphere_geoms[g] = b
            radii[b] = float(model.geom_size[g, 0])
    cache_key, connected = None, set()
    force = np.empty(6)
    angles = 2*np.pi*np.arange(SCOOP_SEGMENTS)/SCOOP_SEGMENTS
    wall_normals = np.column_stack([np.cos(angles), np.sin(angles)])

    def envelope(position, radius):
        tool_r = data.xmat[tool_body].reshape(3, 3)
        local_tool = tool_r.T @ (np.asarray(position)-data.xpos[tool_body])
        cup = frame.T @ (local_tool-origin)
        # Below the rim use the wall inner planes, not the outer radius.
        # Above the rim only the sphere's circular section at rim height must
        # clear the wall; extending the walls upward would reject real heaps.
        # Contact against the exterior wall/underside is not cup inventory.
        tolerance = .2*radius
        if cup[2] >= SCOOP_DEPTH+radius:
            radial = np.linalg.norm(cup[:2]) <= SCOOP_OUTER_RADIUS+radius
        else:
            over_rim = max(0., cup[2]-SCOOP_DEPTH)
            section = math.sqrt(max(0., radius*radius-over_rim*over_rim))
            radial = np.all(wall_normals @ cup[:2] + section <= SCOOP_INNER_RADIUS+tolerance)
        return bool(radial
                    and SCOOP_BOTTOM+radius-tolerance <= cup[2] <= SCOOP_DEPTH+2*radius)

    def rebuild():
        nonlocal cache_key, connected
        candidates = {b for b, r in radii.items() if envelope(data.xpos[b], r)}
        neighbors = {b: set() for b in candidates}
        supported = set()
        for i, contact in enumerate(data.contact[:data.ncon]):
            if contact.efc_address < 0:
                continue
            g, h = int(contact.geom1), int(contact.geom2)
            gb, hb = sphere_geoms.get(g), sphere_geoms.get(h)
            if gb not in candidates and hb not in candidates:
                continue
            mujoco.mj_contactForce(model, data, i, force)
            if force[0] <= 1e-7:
                continue
            if gb in candidates and hb in candidates:
                neighbors[gb].add(hb)
                neighbors[hb].add(gb)
            elif gb in candidates and int(model.geom_bodyid[h]) in tool_bodies:
                supported.add(gb)
            elif hb in candidates and int(model.geom_bodyid[g]) in tool_bodies:
                supported.add(hb)
        stack = list(supported)
        while stack:
            body = stack.pop()
            for neighbor in neighbors[body] - supported:
                supported.add(neighbor)
                stack.append(neighbor)
        connected = supported
        cache_key = (float(data.time), int(data.ncon))

    def contains(body_id, position, radius):
        if cache_key != (float(data.time), int(data.ncon)):
            rebuild()
        return int(body_id) in connected and envelope(position, float(radius))

    def invalidate():
        nonlocal cache_key
        cache_key = None
    contains.invalidate = invalidate
    return contains
