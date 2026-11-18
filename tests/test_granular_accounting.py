"""Small physical-load tests; no height gate, transfer callback, or fake scale."""
import math

import mujoco
import numpy as np
import pytest

from chemlab_rl.granular_accounting import GranularAccounting


RADIUS = 0.0015
DENSITY = 20e-6 / (4 / 3 * math.pi * RADIUS**3)


@pytest.fixture
def physical_scene():
    # Two real 20mg spheres suffice to distinguish membership from actual load.
    # Test fixtures reposition a grain to arrange a drop; production helper
    # never writes its state or invents a source-to-receiver transfer.
    model = mujoco.MjModel.from_xml_string(f"""
    <mujoco>
      <option timestep=".001" integrator="implicitfast" cone="pyramidal"/>
      <default><geom friction=".5 .001 .0001" solref=".004 1"/></default>
      <worldbody>
        <geom name="bench" type="plane" size="1 1 .01"/>
        <body name="pan" pos="0 0 .0015">
          <geom name="balance_pan" type="cylinder" size=".045 .0015" mass=".15"/>
          <site name="pan_sensor"/>
        </body>
        <body name="receiver" pos="0 0 .003">
          <freejoint name="receiver_joint"/>
          <geom name="receiver_bottom" type="cylinder" size=".014 .001" pos="0 0 .001" mass=".02"/>
        </body>
        <body name="source" pos=".12 0 0">
          <freejoint name="source_joint"/>
          <geom name="source_bottom" type="cylinder" size=".014 .001" pos="0 0 .001" mass=".02"/>
        </body>
        <body name="grain_0" pos=".116 0 .006">
          <freejoint name="grain_joint_0"/>
          <geom name="grain_0" type="sphere" size="{RADIUS}" density="{DENSITY}"/>
        </body>
        <body name="grain_1" pos=".124 0 .006">
          <freejoint name="grain_joint_1"/>
          <geom name="grain_1" type="sphere" size="{RADIUS}" density="{DENSITY}"/>
        </body>
      </worldbody>
      <sensor><force name="balance_pan_force" site="pan_sensor"/></sensor>
    </mujoco>""")
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    grains = [model.body(f"grain_{i}").id for i in range(2)]
    return model, data, grains


def make_counter(scene, **kwargs):
    model, data, grains = scene
    return GranularAccounting(model, data, grains, "source", "receiver",
                              expected_density_kg_m3=DENSITY, expected_total_mass_g=0.04,
                              **kwargs)


def put_grain(scene, index, position):
    model, data, _ = scene
    joint = model.joint(f"grain_joint_{index}")
    adr, dof = int(joint.qposadr[0]), int(joint.dofadr[0])
    data.qpos[adr:adr + 7] = [*position, 1, 0, 0, 0]
    data.qvel[dof:dof + 6] = 0
    mujoco.mj_forward(model, data)


def advance(scene, seconds):
    model, data, _ = scene
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)


def test_density_volume_and_total_inventory_use_compiled_physical_masses(physical_scene):
    counter = make_counter(physical_scene)
    np.testing.assert_allclose(counter.volumes_m3, 4 / 3 * math.pi * RADIUS**3)
    np.testing.assert_allclose(counter.densities_kg_m3, DENSITY)
    np.testing.assert_allclose(counter.masses_g, [0.02, 0.02])
    state = counter.inventory()
    assert state["source_g"] == pytest.approx(0.04)
    assert state["receiver_g"] == state["tool_g"] == state["inflight_g"] == state["spill_g"] == 0
    assert state["total_mass_g"] == pytest.approx(0.04)
    assert abs(state["conservation_error_g"]) < 1e-12
    model, data, ids = physical_scene
    with pytest.raises(ValueError, match="density times sphere volume"):
        GranularAccounting(model, data, ids, "source", "receiver", expected_density_kg_m3=DENSITY * 2)
    with pytest.raises(ValueError, match="initial inventory"):
        GranularAccounting(model, data, ids, "source", "receiver", expected_total_mass_g=8)


def test_tare_is_measured_pan_plus_empty_vial_force(physical_scene):
    counter = make_counter(physical_scene, lag_s=0)
    tare = counter.tare_empty_receiver()
    assert tare == pytest.approx((0.15 + 0.02) * 9.81, abs=1e-7)
    advance(physical_scene, 0.1)
    state = counter.update(0.1)
    assert state["raw_net_mass_g"] == pytest.approx(0, abs=1e-6)
    assert state["reading_g"] == pytest.approx(0, abs=1e-6)


def test_horizontal_rolling_can_have_a_settled_accurate_balance_load(physical_scene):
    counter = make_counter(physical_scene, lag_s=0, settling_s=0.2, mass_tolerance_g=0.001)
    counter.tare_empty_receiver()
    put_grain(physical_scene, 0, [-0.005, 0, 0.01])
    advance(physical_scene, 0.5)
    counter.update(0.5)
    model, data, _ = physical_scene
    dof = int(model.joint("grain_joint_0").dofadr[0])
    # Arrange a physically rolling sphere. Its center moves horizontally while
    # its supported normal load stays mg. No transition changes this velocity.
    data.qvel[dof:dof + 6] = [0.006, 0, 0, 0, 0.006 / RADIUS, 0]
    mujoco.mj_forward(model, data)
    for _ in range(5):
        advance(physical_scene, 0.1)
        state = counter.update(0.1)
    assert state["max_grain_speed_m_s"] > 0.005
    assert not state["all_grains_translationally_stationary"]
    assert state["max_receiver_grain_vertical_speed_m_s"] < 1e-5
    assert state["receiver_g"] == pytest.approx(0.02)
    assert state["raw_net_mass_g"] == pytest.approx(0.02, abs=1e-5)
    assert state["balance_settled"] and state["mass_force_agreement"]


def test_physical_drop_gives_matching_in_vial_mass_and_force_after_settling(physical_scene):
    counter = make_counter(physical_scene, lag_s=0, settling_s=0.2, mass_tolerance_g=0.001)
    counter.tare_empty_receiver()
    put_grain(physical_scene, 0, [-0.004, 0, 0.025])
    for _ in range(10):
        advance(physical_scene, 0.1)
        state = counter.update(0.1)
    assert state["receiver_g"] == pytest.approx(0.02)
    assert state["source_g"] == pytest.approx(0.02)
    assert state["raw_net_mass_g"] == pytest.approx(0.02, abs=1e-5)
    assert state["mass_force_agreement"] and state["settled"]
    assert state["bare_pan_grain_mass_g"] == 0
    assert abs(state["conservation_error_g"]) < 1e-12


def test_grain_on_bare_pan_is_not_receiver_mass_or_success(physical_scene):
    counter = make_counter(physical_scene, lag_s=0, settling_s=0.2, mass_tolerance_g=0.1)
    counter.tare_empty_receiver()
    put_grain(physical_scene, 0, [0.03, 0, 0.015])
    for _ in range(10):
        advance(physical_scene, 0.1)
        state = counter.update(0.1)
    assert state["receiver_g"] == 0
    assert state["spill_g"] == pytest.approx(0.02)
    assert state["bare_pan_grain_mass_g"] == pytest.approx(0.02)
    assert state["raw_net_mass_g"] == pytest.approx(0.02, abs=1e-5)
    assert state["foreign_pan_contact"]
    assert not state["mass_force_agreement"]  # Even though loose tolerance could hide the mass difference.


def test_sphere_spin_does_not_prevent_stable_supported_load(physical_scene):
    counter = make_counter(physical_scene, lag_s=0, settling_s=0.2, mass_tolerance_g=0.001)
    counter.tare_empty_receiver()
    put_grain(physical_scene, 0, [-0.004, 0, 0.025])
    advance(physical_scene, 0.5)
    counter.update(0.5)
    model, data, _ = physical_scene
    joint = model.joint("grain_joint_0")
    dof = int(joint.dofadr[0])
    data.qvel[dof + 5] = 20.0  # Initial spin about gravity; no damping is invented.
    mujoco.mj_forward(model, data)
    for _ in range(5):
        advance(physical_scene, 0.1)
        state = counter.update(0.1)
    assert state["max_grain_angular_speed_rad_s"] > 19.0
    assert state["max_grain_speed_m_s"] < 1e-4
    assert state["receiver_g"] == pytest.approx(0.02)
    assert state["raw_net_mass_g"] == pytest.approx(0.02, abs=1e-5)
    assert state["settled"] and state["mass_force_agreement"]


def test_lifting_receiver_removes_its_real_load_from_sensor(physical_scene):
    counter = make_counter(physical_scene, lag_s=0)
    counter.tare_empty_receiver()
    model, data, _ = physical_scene
    adr = int(model.joint("receiver_joint").qposadr[0])
    data.qpos[adr + 2] += 0.05
    mujoco.mj_forward(model, data)
    advance(physical_scene, 0.001)
    state = counter.update(0.001)
    assert not state["receiver_supported"]
    assert state["raw_net_mass_g"] == pytest.approx(-20.0, abs=1e-5)
    assert state["reading_g"] < 0  # Real tared balances may display negative load.
    with pytest.raises(RuntimeError, match="supported"):
        counter.tare_empty_receiver()


def test_lag_filters_force_not_membership_and_reads_do_not_modify_physics(physical_scene):
    counter = make_counter(physical_scene, lag_s=0.25)
    counter.tare_empty_receiver()
    put_grain(physical_scene, 0, [0.03, 0, 0.015])  # Bare pan: receiver membership stays zero.
    advance(physical_scene, 0.5)
    model, data, _ = physical_scene
    before = data.qpos.copy(), data.qvel.copy(), model.body_mass.copy()
    state = counter.update(0.5)
    assert state["receiver_g"] == 0
    expected = state["raw_net_mass_g"] * (1 - math.exp(-0.5 / 0.25))
    assert state["filtered_mass_g"] == pytest.approx(expected)
    assert state["filtered_mass_g"] > 0
    np.testing.assert_array_equal(data.qpos, before[0])
    np.testing.assert_array_equal(data.qvel, before[1])
    np.testing.assert_array_equal(model.body_mass, before[2])
    with pytest.raises(ValueError, match="elapsed MuJoCo time"):
        counter.update(0.5)  # Re-reading frozen physics cannot fabricate a settled window.


def test_tool_callback_is_disjoint_even_inside_source(physical_scene):
    model, _, ids = physical_scene
    calls = []
    def in_tool(body, position, radius):
        calls.append((body, position, radius))
        return body == ids[0]
    counter = make_counter(physical_scene, tool_membership=in_tool)
    state = counter.inventory()
    assert state["tool_g"] == state["source_g"] == pytest.approx(0.02)
    assert state["receiver_g"] == 0
    assert state["total_mass_g"] == pytest.approx(0.04)
    assert len(calls) == 2 and calls[0][2] == RADIUS


def test_cavity_membership_accounts_for_entire_sphere_and_narrow_neck(physical_scene):
    counter = make_counter(physical_scene)
    _, data, _ = physical_scene
    origin = data.body("receiver").xpos.copy()
    put_grain(physical_scene, 0, origin + [0.010, 0, 0.030])
    assert counter.inventory()["receiver_g"] == pytest.approx(0.02)
    put_grain(physical_scene, 0, origin + [0.010, 0, 0.053])
    assert counter.inventory()["receiver_g"] == 0  # Center fits; full sphere does not fit neck.
    put_grain(physical_scene, 0, origin + [0, 0, 0.002])
    assert counter.inventory()["receiver_g"] == 0  # Center at bottom is not containment.


def test_inflight_spill_and_conservation_are_exclusive(physical_scene):
    counter = make_counter(physical_scene, spill_plane_z=-0.02)
    put_grain(physical_scene, 0, [0.07, 0, 0.2])
    state = counter.inventory()
    assert state["inflight_g"] == pytest.approx(0.02)
    assert state["spill_g"] == 0
    put_grain(physical_scene, 0, [0.07, 0, -0.03])
    state = counter.inventory()
    assert state["spill_g"] == pytest.approx(0.02)
    assert state["inflight_g"] == 0
    assert sum(state[k] for k in ["source_g", "receiver_g", "tool_g", "inflight_g", "spill_g"]) == pytest.approx(0.04)


def test_missing_tare_and_mutated_material_fail_explicitly(physical_scene):
    counter = make_counter(physical_scene)
    with pytest.raises(RuntimeError, match="tare"):
        counter.update(0.1)
    model, _, ids = physical_scene
    model.body_mass[ids[0]] *= 2
    with pytest.raises(RuntimeError, match="masses changed"):
        counter.inventory()


def test_tare_rejects_existing_deposited_powder(physical_scene):
    counter = make_counter(physical_scene)
    put_grain(physical_scene, 0, [-0.004, 0, 0.025])
    advance(physical_scene, 0.5)
    assert counter.inventory()["receiver_g"] == pytest.approx(0.02)
    with pytest.raises(RuntimeError, match="empty"):
        counter.tare_empty_receiver()


def test_declared_eight_gram_source_counts_all_400_physical_grains_without_stepping():
    # Compile/inspect the complete inventory without running an expensive
    # 400-grain solver benchmark. The smaller tests exercise actual transfers.
    xy = [(x, y) for x in np.arange(-0.0093, 0.00931, 0.0031)
          for y in np.arange(-0.0093, 0.00931, 0.0031) if np.hypot(x, y) < 0.0108]
    grains = []
    for i in range(400):
        layer, slot = divmod(i, len(xy))
        x, y = xy[slot]
        z = 0.002 + RADIUS + 0.0001 + layer * 0.0031
        grains.append(f'<body name="g{i}" pos="{0.12+x} {y} {z}"><freejoint/>'
                      f'<geom type="sphere" size="{RADIUS}" density="{DENSITY}"/></body>')
    model = mujoco.MjModel.from_xml_string(f"""<mujoco><worldbody>
      <body name="pan"><geom name="balance_pan" type="cylinder" size=".045 .0015" mass=".15"/><site name="sensor"/></body>
      <body name="source" pos=".12 0 0"><freejoint/><geom type="cylinder" size=".014 .001" pos="0 0 .001" mass=".02"/></body>
      <body name="receiver" pos="0 0 .002"><freejoint/><geom type="cylinder" size=".014 .001" pos="0 0 .001" mass=".02"/></body>
      {''.join(grains)}
    </worldbody><sensor><force name="balance_pan_force" site="sensor"/></sensor></mujoco>""")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    counter = GranularAccounting(model, data, [model.body(f"g{i}").id for i in range(400)],
                                 "source", "receiver", expected_density_kg_m3=DENSITY,
                                 expected_total_mass_g=8)
    state = counter.inventory()
    assert state["source_g"] == pytest.approx(8)
    assert state["receiver_g"] == state["tool_g"] == state["inflight_g"] == state["spill_g"] == 0
    assert abs(state["conservation_error_g"]) < 1e-12
