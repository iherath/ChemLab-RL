"""SolutionPrepEnv: reset stability, snap-cap weld, determinism, audit."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("chem_rl_sim")

from chemlab_rl.solution_env import SolutionPrepEnv, TRAIN_SYRINGES  # noqa: E402
from chemlab_rl.solution_scene import BENCH, CAP_HALF_H, VIAL_H  # noqa: E402


@pytest.fixture(scope="module")
def env():
    e = SolutionPrepEnv(images=False)
    yield e
    e.close()


@pytest.mark.parametrize("syringe", ["1mL", "2mL", "3mL", "5mL"])
def test_reset_and_hold_is_quiet(env, syringe):
    obs, info = env.reset(seed=3, options={"syringe": syringe})
    assert obs["proprio"].shape == (14,) and info["syringe"] == syringe
    for _ in range(15):
        obs, r, term, trunc, info = env.step(env.hold_action())
        assert not term, info["failure"]
    assert info["failure"] == ""
    assert abs(info["conservation_error_mL"]) < 1e-9
    assert info["vial_mL"] == 0 and info["syringe_mL"] == 0
    assert not info["subgoal_dones"].any()
    assert env.scene.plunger_stroke() < 5e-4


def test_layout_jitter_changes_props_but_is_seeded(env):
    _, a = env.reset(seed=1)
    _, b = env.reset(seed=1)
    _, c = env.reset(seed=2)
    assert a["layout"] == b["layout"] and a["layout"] != c["layout"]


def test_same_seed_same_actions_replay_exactly(env):
    rng = np.random.default_rng(0)
    env.reset(seed=7)
    base = env.hold_action()
    actions = [base + np.r_[rng.normal(0, 0.01, 6), 0, rng.normal(0, 0.01, 6), 0].astype(np.float32)
               for _ in range(10)]
    traj = []
    for a in actions:
        env.step(a); traj.append(env.data.qpos.copy())
    env.reset(seed=7)
    for a, q in zip(actions, traj):
        env.step(a)
        np.testing.assert_array_equal(env.data.qpos, q)


def _hold_cap_on_rim(env, dx=0.0):
    """Place the cap on the rim and mark it held (stands in for a gripper)."""
    sc, d = env.scene, env.data
    mouth, _ = sc.mouth("vial_a")
    adr, dof = sc.free["cap"]
    d.qpos[adr:adr + 7] = [*(mouth + [dx, 0, CAP_HALF_H + 0.0005]), 1, 0, 0, 0]
    d.qvel[dof:dof + 6] = 0
    mujoco.mj_forward(env.model, d)
    env.held = {"right": "cap"}


def test_thread_engages_only_when_centred(env):
    env.reset(seed=0, options={"layout_jitter_m": 0.0})
    _hold_cap_on_rim(env, dx=0.004)
    env._try_engage()
    assert not env.engaged, "4 mm off-axis must not engage"
    _hold_cap_on_rim(env)
    env._try_engage()
    assert env.engaged and not env.tight


def test_turning_cap_advances_along_helix_to_tight(env):
    from chemlab_rl.solution_scene import THREAD_PITCH
    env.reset(seed=0, options={"layout_jitter_m": 0.0})
    _hold_cap_on_rim(env)
    env._try_engage()
    m, d = env.model, env.data
    cap, vial = m.body("cap").id, m.body("vial_a").id
    z0 = CAP_HALF_H                          # engaged start: cap bottom on the rim
    for _ in range(60):                      # clockwise from above = tightening;
        d.xfrc_applied[cap, 3:6] = [0, 0, -0.08]   # the vial is held: equal and
        d.xfrc_applied[vial, 3:6] = [0, 0, 0.08]   # opposite torque on its body
        _, _, term, _, info = env.step(env.hold_action())
        assert not term, info["failure"]
    d.xfrc_applied[:] = 0
    info = env.info()
    assert info["failure"] == "", info["failure"]
    assert env.tight and info["capped"], info["thread_turn_deg"]
    mouth, vr = env.scene.mouth("vial_a")
    drop = z0 - (vr.T @ (d.body("cap").xpos - mouth))[2]
    assert abs(drop - THREAD_PITCH) < 5e-4, drop      # one turn lowers the cap by the pitch


def test_screwed_cap_follows_tilted_vial(env):
    env.reset(seed=0, options={"layout_jitter_m": 0.0})
    _hold_cap_on_rim(env)
    env._try_engage()
    m, d = env.model, env.data
    s = mujoco.MjData(m)
    s.qpos[:], s.eq_active[:] = d.qpos, d.eq_active
    for name in ("vial_a", "cap"):
        s.qpos[env.scene.free[name][0] + 2] += 0.15
    mujoco.mj_forward(m, s)
    seat0 = s.body("vial_a").xmat.reshape(3, 3).T @ (s.body("cap").xpos - s.body("vial_a").xpos)
    s.qvel[env.scene.free["vial_a"][1] + 3] = 45.0   # the welded 10 g cap ~6 cm up dominates the inertia
    m.opt.gravity[:] = 0
    try:
        for _ in range(500):
            mujoco.mj_step(m, s)
    finally:
        m.opt.gravity[:] = [0, 0, -9.81]
    vr = s.body("vial_a").xmat.reshape(3, 3)
    assert vr[2, 2] < 0.7, "vial should be tilted past 45 deg"
    seat = vr.T @ (s.body("cap").xpos - s.body("vial_a").xpos)
    assert np.linalg.norm(seat - seat0) < 1e-3, (seat, seat0)
