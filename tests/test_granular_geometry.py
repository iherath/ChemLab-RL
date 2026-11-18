"""Compiled collider dimensions and contact-backed scoop membership."""
from itertools import product
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "chem_rl_sim"), str(ROOT / "chem_rl_sim/third_party/urgantry_sim")]
from chem_rl_sim.assets import add_asset
from chemlab_rl.granular_geometry import (
    SCOOP_FRAME, SCOOP_TIP, add_scoop, replace_vial_collision, scoop_membership,
)


def tool_body(spec, *, static=False, pos=(.1, 0, 0), quat=(1, 0, 0, 0)):
    tool = spec.worldbody.add_body(name="spatula", pos=pos, quat=quat)
    if not static:
        tool.add_freejoint()
    tool.add_geom(name="spatula_handle", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                  size=[.012, .020], mass=.030)
    tool.add_geom(name="spatula_neck", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=[.008, 0, 0, .033, 0, 0], size=[.003], mass=.001)
    tool.add_geom(name="spatula_blade", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  fromto=[.030, 0, 0, .260, 0, 0], size=[.003], mass=.015)
    add_scoop(spec, tool)
    return tool


@pytest.fixture(scope="module")
def compiled():
    spec = mujoco.MjSpec()
    body = add_asset(spec, spec.worldbody, "vial_20ml_open", "source", [0, 0, 0])
    info = replace_vial_collision(spec, body)
    tool_body(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, info


def test_vial_preserves_mass_and_disables_legacy_hulls(compiled):
    model, _, info = compiled
    assert model.body_mass[model.body("source").id] == pytest.approx(.020, abs=1e-12)
    active = [g for g in range(model.ngeom) if model.geom(g).name.startswith("source_contact_")]
    assert len(active) == info["collision_geoms"] == 73
    for g in range(model.ngeom):
        if model.geom(g).name.startswith("source_collision_"):
            assert model.geom_contype[g] == model.geom_conaffinity[g] == 0
    assert model.geom("source_visual_0").matid[0] >= 0


def test_compiled_vial_vertices_stay_within_visual_dimensions(compiled):
    model, data, _ = compiled
    vertices = []
    for g in range(model.ngeom):
        if not model.geom(g).name.startswith("source_contact_") or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = int(model.geom_dataid[g])
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices.extend(model.mesh_vert[start:start+count] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
    vertices = np.asarray(vertices)
    assert np.linalg.norm(vertices[:, :2], axis=1).max() <= .014 + 2e-9
    assert vertices[:, 2].min() == pytest.approx(.002, abs=2e-9)
    assert vertices[:, 2].max() == pytest.approx(.057, abs=2e-9)


def test_cavity_narrows_through_shoulder_to_neck(compiled):
    model, data, info = compiled
    # mj_ray ignores alpha-zero geoms. Expose only new colliders for this query.
    rgba = model.geom_rgba.copy()
    try:
        for g in range(model.ngeom):
            if model.geom(g).name.startswith("source_contact_"):
                model.geom_rgba[g, 3] = 1
        groups = np.array([0, 0, 0, 1, 0, 0], np.uint8)
        direction = np.array([np.cos(np.pi/24), np.sin(np.pi/24), 0.])
        for height, radius in ((.020, .0128), (.049, (.0128+.01105)/2), (.054, .01105)):
            geom = np.array([-1], np.int32)
            distance = mujoco.mj_ray(model, data, np.array([0., 0., height]), direction, groups, True, -1, geom)
            assert distance == pytest.approx(radius*np.cos(np.pi/24), abs=2e-9)
            assert model.geom(int(geom[0])).name.startswith("source_contact_")
        assert info["minimum_neck_inner_diameter_m"] > .0219
    finally:
        model.geom_rgba[:] = rgba


def test_cup_is_inscribed_and_old_rod_is_hidden(compiled):
    model, data, _ = compiled
    root = model.body("spatula").id
    tool_r = data.xmat[root].reshape(3, 3)
    cup_vertices = []
    for g in range(model.ngeom):
        if model.geom(g).name.startswith("spatula_scoop_wall_"):
            corners = np.array(list(product((-1., 1.), repeat=3))) * model.geom_size[g]
            world = corners @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            local_tool = (world-data.xpos[root]) @ tool_r
            cup_vertices.extend((local_tool-SCOOP_TIP) @ SCOOP_FRAME)
    cup_vertices = np.asarray(cup_vertices)
    assert np.linalg.norm(cup_vertices[:, :2], axis=1).max() <= .007 + 1e-12
    assert cup_vertices[:, 2].min() == pytest.approx(0, abs=1e-12)
    assert cup_vertices[:, 2].max() == pytest.approx(.006, abs=1e-12)
    old = model.geom("spatula_blade").id
    assert model.geom_contype[old] == model.geom_conaffinity[old] == 0
    assert model.geom_rgba[old, 3] == 0
    assert model.geom_contype[model.geom("spatula_neck").id] != 0


def test_membership_requires_contact_support_not_only_proximity():
    spec = mujoco.MjSpec()
    spec.option.timestep = .00025
    rotation = SCOOP_FRAME.T
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, rotation.ravel())
    cup_base = np.array([0., 0., .020])
    tool_body(spec, static=True, pos=(cup_base-rotation@SCOOP_TIP).tolist(), quat=quaternion.tolist())
    positions = ([0, 0, .02249], [0, 0, .02548], [0, -.0045, .025],
                 [.00835, 0, .023], [0, 0, .01851], [.00545, 0, .02739])
    for i, position in enumerate(positions):
        body = spec.worldbody.add_body(name=f"grain_{i}", pos=position)
        body.add_freejoint()
        body.add_geom(name=f"grain_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[.0015], mass=.00002, priority=1,
                      solref=[.0005, 1], solimp=[.95, .99, .0001, .5, 2])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    member = scoop_membership(model, data)
    ids = [model.body(f"grain_{i}").id for i in range(len(positions))]
    assert member(ids[0], data.xpos[ids[0]], .0015)
    assert member(ids[1], data.xpos[ids[1]], .0015)  # supported by the first grain
    assert not member(ids[2], data.xpos[ids[2]], .0015)  # airborne inside envelope
    # These grains press against the tool but are outside its cup interior.
    for body in ids[3:5]:
        contacts = [i for i, c in enumerate(data.contact[:data.ncon])
                    if body in model.geom_bodyid[[c.geom1, c.geom2]]]
        forces = []
        for index in contacts:
            force = np.empty(6)
            mujoco.mj_contactForce(model, data, index, force)
            forces.append(force[0])
        assert any(f > 1e-7 for f in forces)
        assert not member(body, data.xpos[body], .0015)
    assert member(ids[5], data.xpos[ids[5]], .0015)  # grain supported at the finite-height rim
