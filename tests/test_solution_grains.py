"""Grain bed: containment, settling, suspension, vigour-dependent dissolution, exact mass."""
import numpy as np
import pytest

from chemlab_rl.solution_env import SolutionPrepEnv
from chemlab_rl.solution_grains import N_GRAINS, lift_vial_kinematic, shake_vial_kinematic
from chemlab_rl.solution_scene import VIAL_FLOOR, VIAL_H

SOLID_G = 0.12


@pytest.fixture(scope="module")
def env():
    return SolutionPrepEnv(images=False)


def fresh(env, mL=3.0, solid_id=0, seed=0):
    env.reset(seed=seed, options=dict(syringe="3mL", solid_id=solid_id, solid_g=SOLID_G, layout_jitter_m=0.0))
    if mL:
        env.liquid.reservoir_mL -= mL
        env.liquid._add_to_vial(mL)
    lift_vial_kinematic(env)
    return env


def heights_mm(env):
    gb, d = env.grains, env.data
    loc = (d.xpos[gb.bodies] - d.xpos[gb.vial]) @ d.xmat[gb.vial].reshape(3, 3)
    return loc[gb.active, 2] * 1e3


def test_reset_puts_all_grains_dry_on_the_floor(env):
    env.reset(seed=3, options=dict(syringe="3mL", solid_id=0, solid_g=SOLID_G))
    gb = env.grains
    assert gb.active.sum() == N_GRAINS
    assert gb.escaped(VIAL_FLOOR, VIAL_H) == []
    assert np.isclose(gb.undissolved_kg * 1e3, SOLID_G) and np.isclose(env.liquid.undissolved_g, SOLID_G)
    assert heights_mm(env).max() < 8.0          # a 1-2 layer pile


def test_still_liquid_lets_grains_settle_and_barely_dissolve(env):
    fresh(env)
    shake_vial_kinematic(env, 2.0)
    assert env.grains.escaped(VIAL_FLOOR, VIAL_H) == []
    assert np.median(heights_mm(env)) < 4.0
    assert env.liquid.undissolved_g > 0.97 * SOLID_G


def test_shaking_suspends_grains_then_they_settle(env):
    fresh(env)
    shake_vial_kinematic(env, 0.5)
    rest = np.median(heights_mm(env))
    lifted = []
    for _ in range(10):
        shake_vial_kinematic(env, 0.2, hz=4.0, amp=0.03)
        lifted.append(np.mean(heights_mm(env)))
    assert np.mean(lifted) > rest + 2.0
    shake_vial_kinematic(env, 1.5)               # stop: they sink back
    assert np.median(heights_mm(env)) < rest + 1.0


def test_violent_shaking_keeps_every_grain_inside(env):
    fresh(env)
    shake_vial_kinematic(env, 3.0, hz=6.0, amp=0.04)
    assert env.grains.escaped(VIAL_FLOOR, VIAL_H) == []
    assert np.isfinite(env.data.qpos).all()


def test_more_vigorous_shaking_dissolves_faster(env):
    done = {}
    for name, hz, amp in (("still", 0.0, 0.0), ("gentle", 1.5, 0.02), ("vigorous", 4.0, 0.03)):
        fresh(env, solid_id=4)
        shake_vial_kinematic(env, 5.0, hz=hz, amp=amp)
        done[name] = SOLID_G - env.liquid.undissolved_g
    assert done["still"] < 0.3 * done["gentle"]
    assert done["vigorous"] > 1.4 * done["gentle"]


def test_dry_grains_do_not_dissolve(env):
    fresh(env, mL=0.0)
    shake_vial_kinematic(env, 1.5, hz=4.0, amp=0.03)
    assert env.liquid.dissolved_g == 0.0 and env.liquid.undissolved_g == SOLID_G


def test_grain_mass_matches_state_through_full_dissolution(env, monkeypatch):
    import chemlab_rl.solution_grains as G
    fresh(env, solid_id=4)
    shake_vial_kinematic(env, 1.0, hz=4.0, amp=0.03)
    liq, gb = env.liquid, env.grains
    assert abs(gb.undissolved_kg * 1e3 - liq.undissolved_g) < 1e-12
    monkeypatch.setattr(G, "K_GRAIN", 1e3)            # finish in a few steps
    shake_vial_kinematic(env, 1.0, hz=4.0, amp=0.03)
    assert gb.active.sum() == 0
    assert liq.undissolved_g < 1e-12 and abs(liq.mass_conservation_error_g) < 1e-12
    m = env.model
    assert (m.geom_rgba[gb.geoms, 3] == 0).all() and (m.geom_contype[gb.geoms] == 0).all()
