import math

import numpy as np
import pytest

from chemlab_rl.solution_state import (
    HELD_OUT_SOLID, HELD_OUT_SYRINGE, SOLIDS, SYRINGES, TOTAL_VOLUME_ML,
    SolutionState, spill_tilt_threshold,
)

DT = 0.1


def make(syringe="3mL", solid=0, **kw):
    return SolutionState(syringe=SYRINGES[syringe], solid=SOLIDS[solid], **kw)


def step(s, plunger, *, res=False, vial=False, tilt=0.0, capped=False, agit=0.0, dt=DT):
    s.step(dt, plunger_mL=plunger, tip_in_reservoir_liquid=res, tip_in_vial_mouth=vial,
           vial_tilt=tilt, vial_capped=capped, agitation=agit)


def barrel_ok(s, plunger):
    assert abs(s.syringe_liquid_mL + s.syringe_air_mL - plunger) < 1e-9


def test_syringe_geometry_consistent():
    assert HELD_OUT_SYRINGE in SYRINGES
    for syr in SYRINGES.values():
        vol = math.pi * syr.barrel_inner_radius_m ** 2 * syr.stroke_m * 1e6
        assert vol == pytest.approx(syr.capacity_mL, rel=1e-9)
        assert syr.mL_per_m * syr.stroke_m == pytest.approx(syr.capacity_mL)
        assert 0.02 < syr.stroke_m < 0.07


def test_draw_exact_volume_submerged():
    s = make()
    for p in np.linspace(0.1, 1.5, 15):
        step(s, p, res=True)
        barrel_ok(s, p)
    assert s.syringe_liquid_mL == pytest.approx(1.5)
    assert s.syringe_air_mL == pytest.approx(0.0, abs=1e-12)
    assert s.reservoir_mL == pytest.approx(3.5)


def test_draw_air_when_tip_out():
    s = make()
    step(s, 0.5, res=False)
    assert s.syringe_air_mL == pytest.approx(0.5)
    assert s.syringe_liquid_mL == 0.0
    assert s.reservoir_mL == 5.0


def test_draw_limited_by_reservoir():
    s = make(reservoir_mL=0.4)
    step(s, 1.0, res=True)
    assert s.syringe_liquid_mL == pytest.approx(0.4)
    assert s.syringe_air_mL == pytest.approx(0.6)


def test_air_expelled_before_liquid():
    s = make()
    step(s, 0.3)              # air
    step(s, 1.3, res=True)    # 1.0 liquid
    step(s, 1.1, vial=True)   # expel 0.2 -> all air
    assert s.syringe_air_mL == pytest.approx(0.1)
    assert s.vial_mL == 0.0
    step(s, 0.5, vial=True)   # 0.1 air + 0.5 liquid
    assert s.syringe_air_mL == pytest.approx(0.0, abs=1e-12)
    assert s.vial_mL == pytest.approx(0.5)
    assert abs(s.conservation_error_mL) < 1e-12


def test_dispense_vial_vs_spill():
    s = make()
    step(s, 1.0, res=True)
    step(s, 0.6, vial=True)
    step(s, 0.2, vial=False)
    assert s.vial_mL == pytest.approx(0.4)
    assert s.spilled_mL == pytest.approx(0.4)
    step(s, 0.0, vial=True, capped=True)  # capped vial: liquid goes to spill
    assert s.vial_mL == pytest.approx(0.4)
    assert s.spilled_mL == pytest.approx(0.6)


def test_three_draws_with_1ml_syringe():
    s = make("1mL")
    cap = s.syringe.capacity_mL
    for _ in range(3):
        for p in np.linspace(0.1, cap, 10):
            step(s, p, res=True)
        for p in np.linspace(cap - 0.1, 0.0, 10):
            step(s, p, vial=True)
    assert s.vial_mL == pytest.approx(TOTAL_VOLUME_ML, abs=1e-9)
    assert s.spilled_mL == 0.0
    assert abs(s.conservation_error_mL) < 1e-9


def test_volume_conservation_random_sequence():
    rng = np.random.default_rng(0)
    s = make("2mL", solid=2, reservoir_mL=4.0, undissolved_g=0.2)
    for _ in range(500):
        p = float(rng.uniform(-0.2, 2.2))
        step(s, p, res=bool(rng.random() < 0.5), vial=bool(rng.random() < 0.5),
             tilt=float(rng.uniform(0, 1)), capped=bool(rng.random() < 0.3),
             agit=float(rng.uniform(0, 2)))
        barrel_ok(s, min(max(p, 0.0), 2.0))
        assert abs(s.conservation_error_mL) < 1e-9
        assert abs(s.mass_conservation_error_g) < 1e-12
        for v in (s.reservoir_mL, s.syringe_liquid_mL, s.syringe_air_mL, s.vial_mL,
                  s.spilled_mL, s.undissolved_g, s.dissolved_g):
            assert v >= -1e-12
        assert 0.0 <= s.homogeneity <= 1.0 and 0.0 <= s.turbidity <= 1.0


def test_no_dissolution_without_solvent():
    s = make(undissolved_g=0.2)
    for _ in range(100):
        step(s, 0.0, agit=1.0)
    assert s.undissolved_g == 0.2 and s.dissolved_g == 0.0


def fill_vial(s, mL=TOTAL_VOLUME_ML):
    step(s, mL, res=True)
    step(s, 0.0, vial=True)


def time_to_dissolve(solid, agit, g=0.15, limit_s=1000.0):
    s = make(solid=solid, undissolved_g=g)
    fill_vial(s)
    t = 0.0
    while s.undissolved_g >= 0.01 * g and t < limit_s:
        step(s, 0.0, agit=agit)
        t += DT
    return t


def test_dissolution_faster_with_agitation_and_in_range():
    for i in range(len(SOLIDS)):
        for g in (0.05, 0.3):
            fast = time_to_dissolve(i, 1.0, g)
            assert 5.0 <= fast <= 30.0, (i, g, fast)
            assert time_to_dissolve(i, 0.0, g) > 300.0


def test_capped_never_spills_uncapped_inverted_spills():
    s = make()
    fill_vial(s)
    for _ in range(50):
        step(s, 0.0, tilt=1.0, capped=True)
    assert s.spilled_mL == 0.0 and s.vial_mL == pytest.approx(3.0)
    step(s, 0.0, tilt=0.1)  # gentle tilt below lip angle
    assert s.spilled_mL == 0.0
    assert spill_tilt_threshold(3.0) > 0.1
    for _ in range(50):
        step(s, 0.0, tilt=1.0)
    assert s.spilled_mL > 2.9
    assert abs(s.conservation_error_mL) < 1e-12


def test_held_out_solid_interpolates():
    ks = [SOLIDS[i].dissolution_k for i in range(len(SOLIDS)) if i != HELD_OUT_SOLID]
    assert min(ks) < SOLIDS[HELD_OUT_SOLID].dissolution_k < max(ks)


def test_homogeneous_only_after_mixing():
    s = make(solid=0, undissolved_g=0.2)
    assert not s.homogeneous
    fill_vial(s)
    assert s.homogeneity < 1e-3  # fresh solvent: fully stratified
    assert not s.homogeneous and s.turbidity > 0.5
    for _ in range(20):
        step(s, 0.0, capped=True, agit=1.0)
    assert not s.homogeneous
    t = 0
    while not s.homogeneous and t < 1000:
        step(s, 0.0, capped=True, agit=1.0)
        t += 1
    assert s.homogeneous and s.mixed_fraction > 0.97 and s.turbidity < 0.02
    assert s.fill_height_m() == pytest.approx(3e-6 / (math.pi * 0.011 ** 2))
    assert s.as_dict()["homogeneous"] is True
