"""Behavioral tests for the synthetic transfer and action-driven environment."""
from dataclasses import replace

import numpy as np
import pytest

from chemlab_rl.powder_dispense import Material
from chemlab_rl.scrape_env import PowderState, ScrapeWeighEnv


MATERIAL = Material(1.6, 0.15, 0.15, 0.05, 0.25, 0.008, 0.9, 0.05)


def powder(material=MATERIAL):
    return PowderState(material, np.random.default_rng(4))


def test_no_mass_appears_without_loading_tool():
    p = powder()
    for _ in range(100):
        p.step(0.1, source_contact=False, engagement=1.0, over_receiver=True)
    assert p.mass_g == p.tool_g == p.spilled_g == 0
    assert p.conserved_mass_g == 8.0


def test_loaded_inventory_is_finite_and_conserved():
    p = powder()
    for _ in range(30):
        p.step(0.1, source_contact=True, engagement=0, over_receiver=False)
    assert p.tool_g == p.tool_capacity_g
    for _ in range(200):
        p.step(0.1, source_contact=False, engagement=0.6, over_receiver=True)
    assert p.mass_g == pytest.approx(1.0)
    assert p.conserved_mass_g == pytest.approx(8.0)
    assert p.source_g == pytest.approx(7.0)


def test_withdrawing_stops_transfer_after_finite_tail():
    p = powder()
    for _ in range(20):
        p.step(0.1, source_contact=True, engagement=0, over_receiver=False)
    for _ in range(5):
        p.step(0.1, source_contact=False, engagement=0.5, over_receiver=True)
    before = p.mass_g
    tail = p.tail_g
    for _ in range(100):
        p.step(0.1, source_contact=False, engagement=0, over_receiver=True)
    assert p.mass_g - before == pytest.approx(tail)
    assert p.tool_g > 0.1
    assert not p.dispensing
    assert p.tail_g == pytest.approx(0)
    assert p.filtered_g == pytest.approx(p.mass_g, abs=1e-5)


def test_delay_and_onset_are_effective():
    p = powder(replace(MATERIAL, delay_s=0.5))
    p.tool_g, p.source_g = 1, 7
    for _ in range(20):
        p.step(0.1, source_contact=False, engagement=0.10, over_receiver=True)
    assert p.mass_g == 0
    for _ in range(4):
        p.step(0.1, source_contact=False, engagement=0.7, over_receiver=True)
    assert p.mass_g == 0
    p.step(0.1, source_contact=False, engagement=0.7, over_receiver=True)
    assert p.mass_g > 0


def test_source_tilt_spills_without_losing_mass_accounting():
    p = powder()
    for _ in range(10):
        p.step(0.1, source_contact=False, engagement=0, over_receiver=False, source_tilt=1.0)
    assert p.spilled_g == pytest.approx(MATERIAL.flow_g_per_s)
    assert p.conserved_mass_g == pytest.approx(8.0)


def test_seeded_noise_does_not_change_true_mass():
    noisy, clean = powder(), powder(replace(MATERIAL, noise_g=0.0))
    for p in (noisy, clean):
        p.tool_g, p.source_g = 1, 7
    for _ in range(5):
        for p in (noisy, clean):
            p.step(0.1, source_contact=False, engagement=0.5, over_receiver=True)
    assert noisy.mass_g == clean.mass_g
    assert noisy.filtered_g == clean.filtered_g
    assert noisy.reading_g != clean.reading_g


@pytest.fixture(scope="module")
def env():
    e = ScrapeWeighEnv(images=False)
    yield e
    e.close()


def test_reset_and_idle_never_complete_task(env):
    obs, info = env.reset(seed=5)
    assert env.observation_space.contains(obs)
    action = env.hold_action()
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated
        assert not info["success"]
        assert info["mass_g"] == 0
        assert reward == -1
    assert not env.scene.carried


def test_bad_action_is_rejected_before_state_changes(env):
    env.reset(seed=5)
    q = env.data.qpos.copy()
    with pytest.raises(ValueError, match="14 finite"):
        env.step(np.full(14, np.nan))
    assert np.array_equal(q, env.data.qpos)


def test_dynamics_do_not_depend_on_target(env):
    traces = []
    for target in (1.5, 3.0):
        env.reset(seed=6, options={"target_g": target})
        action = env.hold_action()
        trace = []
        for _ in range(4):
            _, _, _, _, info = env.step(action)
            trace.append([info["mass_g"], info["balance_g"], info["source_g"]])
        traces.append(np.array(trace))
    np.testing.assert_array_equal(*traces)


def test_passive_props_stay_supported_and_settled(env):
    env.reset(seed=7)
    names = ("source", "vial_a", "spatula", "discard")
    before = {n: env.data.body(n).xpos.copy() for n in names}
    for _ in range(10):
        _, _, term, trunc, info = env.step(env.hold_action())
        assert not term and not trunc, info["failure"]
    for n in names:
        assert np.linalg.norm(env.data.body(n).xpos - before[n]) < 0.003
    assert env._receiver_supported()
    assert env.physics_checks >= 10 * env.n_substeps


def test_closed_grippers_do_not_attach_distant_or_falling_props(env):
    env.reset(seed=8)
    adr, dof = env.scene.free["source"]
    env.data.qpos[adr + 2] += 0.10  # A test perturbation, never an expert action.
    env.mj.mj_forward(env.model, env.data)
    start = float(env.data.qpos[adr + 2])
    action = env.hold_action()
    action[[6, 13]] = 255
    env.step(action)
    assert env.data.qpos[adr + 2] < start - 0.02
    assert env.data.qvel[dof + 2] < -0.5
    assert not env.scene.carried
    assert not hasattr(env.scene, "_hold_carried")


def test_receiver_above_pan_is_not_supported(env):
    env.reset(seed=9)
    env.data.qpos[env.scene.vial_adr + 2] += 0.10
    env.mj.mj_forward(env.model, env.data)
    assert not env._geometry()["receiver_on_pan"]


def test_source_pan_contact_is_forbidden_even_without_robot(env):
    env.reset(seed=10)
    adr, _ = env.scene.free["source"]
    env.data.qpos[adr:adr + 3] = env.legacy.PAN + np.array([0.03, 0, -0.002])
    env.mj.mj_forward(env.model, env.data)
    env.watch()
    assert env.scene.bad
    assert "forbidden contact" in env.scene.bad


def test_tool_cannot_load_from_empty_air_above_mouth(env, monkeypatch):
    env.reset(seed=11)
    env.scene.carried = [env.legacy.Hold(side, *env.scene.free[name], np.zeros(3), np.eye(3))
                         for side, name in (("left", "source"), ("right", "spatula"))]
    source = env.data.body("source")
    origin, rot = source.xpos.copy(), source.xmat.reshape(3, 3).copy()
    monkeypatch.setattr(env.legacy, "_tip", lambda d: origin + rot @ np.array([0, 0, 0.061]))
    assert not env._geometry()["source_contact"]
    monkeypatch.setattr(env.legacy, "_tip", lambda d: origin + rot @ np.array([0.010, 0, 0.040]))
    assert not env._geometry()["source_contact"]
    monkeypatch.setattr(env.legacy, "_tip", lambda d: origin + rot @ np.array([0, 0, 0.040]))
    assert env._geometry()["source_contact"]


def test_source_matches_receiver_asset_and_tool_is_connected(env):
    m = env.model
    assert m.body("source").mass[0] == pytest.approx(0.020)
    assert m.body("vial_a").mass[0] == pytest.approx(0.020)
    for i in range(13):
        assert m.geom(f"source_collision_{i}").dataid[0] == m.geom(f"vial_a_collision_{i}").dataid[0]
        assert m.geom(f"source_collision_{i}").contype[0] != 0
    assert m.geom("spatula_neck").contype[0] != 0
    assert not any((m.geom(i).name or "").startswith("puck_") for i in range(m.ngeom))
    assert np.all(m.body_gravcomp == 0)
