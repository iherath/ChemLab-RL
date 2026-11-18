"""Small full-scene integration checks for physical grains and real scale load.

Uses 24 grains, never the expensive 400-grain solver benchmark. Tests may arrange
receiver/grain starting poses as fixtures; normal actions remain 14 robot controls.
"""
import numpy as np
import pytest

from chemlab_rl.granular_env import GranularWeighEnv
from chemlab_rl.granular_material import GRANULAR_MATERIALS, HELD_OUT_MATERIAL_IDS, ScaleInstrument


@pytest.fixture(scope="module")
def env():
    instance = GranularWeighEnv(images=False, grain_count=24, target_g=0.2,
                                instrument=ScaleInstrument(lag_s=0.25, noise_g=0.0))
    yield instance
    instance.close()


def test_reset_contains_declared_source_mass_and_measured_empty_receiver(env):
    obs, info = env.reset(seed=41)
    expected = 24 * GRANULAR_MATERIALS[0].grain_mass_g
    assert env.observation_space.contains(obs)
    assert info["source_g"] == pytest.approx(expected)
    assert info["total_mass_g"] == pytest.approx(expected)
    assert info["mass_g"] == info["tool_g"] == info["spilled_g"] == info["inflight_g"] == 0
    assert abs(info["conservation_error_g"]) < 1e-12
    assert env.accounting.inventory()["receiver_supported"]
    assert env.accounting.raw_force_n() == pytest.approx(env.accounting.tare_force_n, abs=1e-8)
    assert info["raw_net_mass_g"] == info["balance_g"] == 0
    assert info["physical_valid"] and not info["success"]
    assert info["powder_dynamics"] == "physical_grains_v1"
    assert info["scale_model"] == "physical_pan_force_with_instrument_lag_and_noise"
    assert not info["powder_model_calibrated"]


def test_robot_action_moves_arm_without_inventing_transfer_or_editing_material(env, monkeypatch):
    from chemlab_rl.scrape_env import PowderState
    monkeypatch.setattr(PowderState, "step", lambda *a, **kw: pytest.fail("Physical scene called scalar dosing surrogate"))
    env.reset(seed=42)
    positions = env.data.qpos[env.scene.arm_qadr["left"]].copy()
    masses, sizes = env.model.body_mass.copy(), env.model.geom_size[env.grain_geoms].copy()
    action = env.hold_action()
    action[0] += 0.015
    start = float(env.data.time)
    _, _, terminated, truncated, info = env.step(action)
    assert not terminated and not truncated, info["failure"]
    assert np.linalg.norm(env.data.qpos[env.scene.arm_qadr["left"]] - positions) > 1e-4
    assert info["source_g"] == pytest.approx(env.accounting.initial_total_mass_g)
    assert info["mass_g"] == info["tool_g"] == info["spilled_g"] == info["inflight_g"] == 0
    assert env.data.time - start == pytest.approx(env.dt)
    assert info["elapsed_s"] == pytest.approx(env.dt)
    assert info["physics_checks"] >= env.n_substeps
    np.testing.assert_array_equal(env.model.body_mass, masses)
    np.testing.assert_array_equal(env.model.geom_size[env.grain_geoms], sizes)


def test_action_interface_cannot_supply_particle_states(env):
    env.reset(seed=43)
    qpos, qvel, controls = env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy()
    for invalid in (np.zeros(env.model.nq), np.full(14, np.nan)):
        with pytest.raises(ValueError, match="14 finite"):
            env.step(invalid)
    np.testing.assert_array_equal(env.data.qpos, qpos)
    np.testing.assert_array_equal(env.data.qvel, qvel)
    np.testing.assert_array_equal(env.data.ctrl, controls)


def test_target_changes_instruction_but_not_physical_grain_dynamics(env):
    traces = []
    for target in (0.15, 0.30):
        env.reset(seed=44, options={"target_g": target})
        trace = []
        action = env.hold_action()
        for _ in range(2):
            _, _, terminated, truncated, info = env.step(action)
            assert not terminated and not truncated, info["failure"]
            trace.append(np.concatenate([env.data.qpos.copy(),
                                         [info["source_g"], info["mass_g"], info["raw_net_mass_g"]]]))
        traces.append(trace)
    np.testing.assert_allclose(traces[0], traces[1], atol=1e-12, rtol=0)


def test_off_pan_receiver_has_truthful_force_and_filtered_reading(env):
    env.reset(seed=45)
    # Test perturbation only: production actions cannot set this free-body pose.
    env.data.qpos[env.scene.vial_adr + 2] += 0.06
    env.mj.mj_forward(env.model, env.data)
    _, _, terminated, truncated, info = env.step(env.hold_action())
    assert not terminated and not truncated, info["failure"]
    assert info["mass_g"] == 0
    assert info["raw_net_mass_g"] == pytest.approx(-20.0, abs=0.005)
    assert info["balance_g"] == pytest.approx(-20.0 * (1-np.exp(-env.dt/env.instrument.lag_s)), abs=0.005)
    assert not info["mass_force_agreement"] and not info["settled"] and not info["success"]
    assert env.model.geom_rgba[env.model.geom("lcd_minus").id, 3] == 1


def test_grains_settle_in_receiver_and_add_their_known_mass_to_pan_force(env):
    env.reset(seed=47, options={"target_g": 0.2})
    # Arrange three grains inside the empty receiver, then let physical gravity
    # and contacts determine their motion and support. No force/mass is assigned.
    receiver = env.model.body("vial_a").id
    orientation = env.data.xmat[receiver].reshape(3, 3)
    for body, local in zip(env.grain_bodies[:3], ((-.004, 0, .010), (0, 0, .010), (.004, 0, .010))):
        joint = env.model.body_jntadr[body]
        address, velocity = env.model.jnt_qposadr[joint], env.model.jnt_dofadr[joint]
        env.data.qpos[address:address + 3] = env.data.xpos[receiver] + orientation @ np.array(local)
        env.data.qvel[velocity:velocity + 6] = 0
    env.mj.mj_forward(env.model, env.data)
    expected = 3 * env.material.grain_mass_g
    for _ in range(40):
        _, _, terminated, truncated, info = env.step(env.hold_action())
        assert not terminated and not truncated, info["failure"]
        if info["mass_force_agreement"]:
            break
    assert info["mass_g"] == pytest.approx(expected)
    assert info["source_g"] == pytest.approx(21 * env.material.grain_mass_g)
    # Finite-tolerance contact solving need not equal mg exactly; the production
    # gate allows 10 mg, less than one nominal 20 mg grain.
    assert info["raw_net_mass_g"] == pytest.approx(expected, abs=env.accounting.mass_tolerance_g)
    assert info["balance_g"] == pytest.approx(expected, abs=env.accounting.mass_tolerance_g)
    assert info["settled"] and info["mass_force_agreement"], env.accounting.inventory()
    assert info["bare_pan_grain_mass_g"] == info["spilled_g"] == info["inflight_g"] == 0
    assert abs(info["conservation_error_g"]) < 1e-12


def test_material_cannot_change_without_rebuilding_physical_mass(env):
    env.reset(seed=46)
    with pytest.raises(ValueError, match="new granular environment"):
        env.reset(options={"powder_id": 1})
    with pytest.raises(ValueError, match="new granular environment"):
        env.reset(options={"material": GRANULAR_MATERIALS[1]})


def test_material_bank_declares_distinct_heldout_physical_combination():
    assert len(GRANULAR_MATERIALS) == 6
    assert HELD_OUT_MATERIAL_IDS == (5,)
    training_ids = set(range(len(GRANULAR_MATERIALS))) - set(HELD_OUT_MATERIAL_IDS)
    assert training_ids == {0, 1, 2, 3, 4}
    train_parameters = {(GRANULAR_MATERIALS[i].density_kg_m3, GRANULAR_MATERIALS[i].sliding_friction)
                        for i in training_ids}
    heldout = GRANULAR_MATERIALS[5]
    assert (heldout.density_kg_m3, heldout.sliding_friction) not in train_parameters
    for material in GRANULAR_MATERIALS:
        assert material.grain_mass_kg == pytest.approx(material.density_kg_m3 * material.grain_volume_m3)
        assert not hasattr(material, "flow_g_per_s") and not hasattr(material, "onset")


@pytest.mark.parametrize("powder_id", [-1, 6])
def test_invalid_material_ids_cannot_alias_heldout_or_random_material(powder_id):
    with pytest.raises(ValueError, match="material|powder"):
        GranularWeighEnv(images=False, grain_count=12, target_g=0.1, powder_id=powder_id)
