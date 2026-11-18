"""Safety boundaries for the expert; physical transfer is validated separately."""
from types import SimpleNamespace
import json

import h5py
import mujoco
import numpy as np
import pytest

from scripts import collect_granular_demos as expert
from chemlab_rl.granular_geometry import SCOOP_FRAME


def rotation_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def fake_env(target=.3, received=0., tool=0., inflight=0.):
    bodies = {
        'source': SimpleNamespace(xpos=np.array([.02, .08, 1.]), xmat=np.eye(3).ravel()),
        'spatula': SimpleNamespace(xpos=np.zeros(3), xmat=SCOOP_FRAME.T.ravel().copy()),
        'vial_a': SimpleNamespace(xpos=np.array([0., 0., .9]), xmat=np.eye(3).ravel()),
    }
    data = SimpleNamespace(body=lambda name: bodies[name], time=0.,
                           tip=np.array([.02, .08, 1.078]),
                           qpos=np.zeros(10), qvel=np.zeros(9), ctrl=np.zeros(14))
    for value in (data.qpos, data.qvel, data.ctrl):
        value.flags.writeable = False
    return SimpleNamespace(data=data, legacy=SimpleNamespace(_tip=lambda d: d.tip.copy()),
                           material=SimpleNamespace(radius_m=.0015, grain_mass_g=.02),
                           target_g=target, scene=SimpleNamespace(bad=None),
                           grain_state=dict(receiver_g=received, tool_g=tool, inflight_g=inflight),
                           powder=SimpleNamespace(mass_g=received, reading_g=received))


def bridge():
    return SimpleNamespace(expert_trace=[], expert_phase='start')


def cup_servo_fixture():
    env = fake_env()
    env.mj, env.dt = mujoco, .1
    old = SimpleNamespace(side='right', adr=24, rel_p=np.array([0., 0., .10]), rel_r=np.eye(3))
    env.scene.free, env.scene.carried = {'spatula': (24, 22)}, [old]
    used, controls = [], []
    def tool_for(hold, origin, rotation):
        used.append(hold)
        return origin - rotation @ hold.rel_p, rotation
    env.legacy._tool_for = tool_for
    arm = SimpleNamespace(ik=lambda *args, **kwargs: (np.zeros(6), 0., 0.))
    session = SimpleNamespace(arms={'right': arm}, q_cmd={'right': np.zeros(6)},
                              _scratch=SimpleNamespace(qpos=np.zeros_like(env.data.qpos)))
    demo = SimpleNamespace(session=session, scene=SimpleNamespace(step=lambda _: controls.append(True)))
    return env, demo, old, used, controls


def test_cup_servo_uses_reacquired_live_grasp_instead_of_stale_hold():
    env, demo, old, used, controls = cup_servo_fixture()
    reacquired = SimpleNamespace(side='right', adr=24, rel_p=np.array([0., 0., .102]), rel_r=np.eye(3))
    wrong_hand = SimpleNamespace(side='left', adr=24)
    wrong_object = SimpleNamespace(side='right', adr=31)
    def step(_):
        controls.append(True)
        env.scene.carried = [wrong_hand, wrong_object, reacquired]
    demo.scene.step = step
    assert expert._move_cup(demo, env, old, env.data.tip.copy(), seconds=.3)
    assert len(controls) == 3
    assert used[0] is old and all(item is reacquired for item in used[1:])
    np.testing.assert_array_equal(env.data.qpos, np.zeros(10))


@pytest.mark.parametrize('lost_before_first_tick', [True, False])
def test_cup_servo_stops_before_next_ik_when_physical_grasp_is_absent(lost_before_first_tick):
    env, demo, old, used, controls = cup_servo_fixture()
    if lost_before_first_tick:
        env.scene.carried = []
    def step(_):
        controls.append(True)
        env.scene.carried = []
    demo.scene.step = step
    assert not expert._move_cup(demo, env, old, env.data.tip.copy(), seconds=.3)
    assert len(controls) == len(used) == (0 if lost_before_first_tick else 1)
    assert env.scene.bad == 'right lost contact-supported spatula grasp'
    np.testing.assert_array_equal(env.data.qpos, np.zeros(10))


def test_batch_guard_counts_airborne_grains_and_rejects_invalid_measurements():
    assert expert._batch_fits(.27, .04, 0., .30)
    assert not expert._batch_fits(.27, .04, .02, .30)
    # The observed partial-pour avalanche state must never start a new pour.
    assert not expert._batch_fits(.22, .14, .14, .30)
    for bad in [np.nan, np.inf, -.001]:
        assert not expert._batch_fits(.1, bad, 0., .30)
        assert not expert._batch_fits(.1, .1, bad, .30)


def test_source_targets_and_surface_use_vial_frame_without_source_yaw_flip():
    env = fake_env()
    current = env.data.body('spatula').xmat.reshape(3, 3).copy()
    env.data.body('source').xmat = rotation_z(2.4).ravel()
    _, aligned = expert._source_target(env, .007)
    np.testing.assert_allclose(aligned, current, atol=1e-14)
    frame = rotation_z(.7) @ rotation_y(.2)
    env.data.body('source').xmat = frame.ravel()
    position, aligned = expert._source_target(env, .007)
    np.testing.assert_allclose(frame.T @ (position - env.data.body('source').xpos), [0., 0., .007], atol=1e-14)
    np.testing.assert_allclose(aligned @ SCOOP_FRAME[:, 2], frame[:, 2], atol=1e-14)
    np.testing.assert_allclose(aligned.T @ aligned, np.eye(3), atol=1e-14)
    local = np.array([[0., 0., .0155], [.02, 0., .04], [.001, 0., .03]])
    env.data.xpos = local @ frame.T + env.data.body('source').xpos
    env.grain_bodies = np.arange(3)
    env.grain_state['labels'] = np.array(['source', 'source', 'tool'])
    assert expert._local_bed_surface(env) == pytest.approx(.017)


def test_depth_clamps_never_collapse_fine_into_bulk():
    for surface in [.002, .010, .017, .028, .060]:
        bulk, medium, fine = expert._candidate_depths(surface)
        assert .006 <= bulk < medium < fine <= .016
        assert medium - bulk >= .0005 - 1e-12
        assert fine - medium >= .0005 - 1e-12
    assert expert._candidate_depths(.017) == pytest.approx((.006, .0065, .007))
    assert expert._candidate_depths(.0146723) == pytest.approx((.006, .0065, .007))
    with pytest.raises(ValueError, match='non-finite'):
        expert._candidate_depths(np.nan)


def test_initial_modes_use_shared_budgets_and_observations_refine_each_mode():
    assert expert._choose_load_mode(.325, {}) == 'medium'
    assert expert._choose_load_mode(.525, {}) == 'bulk'
    assert expert._choose_load_mode(.725, {}) == 'bulk'
    # Initial budgets already include headroom: this remaining margin permits
    # a first medium candidate instead of seven single-grain correction trips.
    assert expert._choose_load_mode(.205, {'bulk': .32}) == 'medium'
    assert expert._choose_load_mode(.125, {'bulk': .32, 'medium': .08}) == 'medium'
    assert expert._choose_load_mode(.085, {'bulk': .32, 'medium': .08}) == 'fine'
    assert expert._choose_load_mode(.117, {'medium': .068}) == 'medium'
    assert expert._choose_load_mode(.205, {'medium': .19}) == 'fine'


def test_depth_correction_tracks_moving_source_through_controls(monkeypatch):
    env, record = fake_env(), bridge()
    frame = rotation_y(.1)
    env.data.body('source').xmat = frame.ravel()
    targets = []
    def move(demo, env, hold, target, rotation, **kwargs):
        targets.append(target.copy())
        env.data.tip = target + frame[:, 2] * (.0006 if len(targets) == 1 else .0001)
        return True
    def idle(_):
        if len(targets) == 1:
            env.data.body('source').xpos += frame[:, 2] * .001
        return True
    monkeypatch.setattr(expert, '_move_cup', move)
    assert expert._verify_loading_depth(SimpleNamespace(idle=idle), env, object(), record,
                                         .007, cycle=0, attempt=0)
    assert len(targets) == 2
    np.testing.assert_allclose(targets[1] - targets[0], frame[:, 2] * .001)
    assert record.expert_trace[-1]['z_error_m'] == pytest.approx(.0001)
    assert not env.scene.bad
    np.testing.assert_array_equal(env.data.qpos, np.zeros(10))


def test_depth_failure_is_bounded_and_lateral_clearance_is_separate(monkeypatch):
    env, record = fake_env(), bridge()
    moves = []
    def move(demo, env, hold, target, rotation, **kwargs):
        moves.append(target.copy())
        env.data.tip = target + [.0011, 0., 0.]
        return True
    monkeypatch.setattr(expert, '_move_cup', move)
    assert not expert._verify_loading_depth(SimpleNamespace(idle=lambda _: True), env, object(), record,
                                             .007, cycle=0, attempt=0)
    assert len(moves) == 3
    assert 'xy=1.100 mm' in env.scene.bad


@pytest.mark.parametrize('eventual_load', [0., .08, .20])
def test_empty_retry_is_bounded_deeper_and_uses_actual_retained_mass(monkeypatch, eventual_load):
    env, record = fake_env(target=.1), bridge()
    env.grain_state['labels'] = np.array(['source'])
    env.grain_bodies = np.array([0])
    env.data.xpos = np.array([[.02, .08, 1.0155]])
    depths, moves = [], []
    def verify(demo, env, scoop, bridge, depth, **kwargs):
        depths.append(depth)
        return True
    def move(demo, env, hold, target, rotation, **kwargs):
        moves.append(target.copy())
        if len(depths) == 2:
            env.grain_state['tool_g'] = eventual_load
        return True
    monkeypatch.setattr(expert, '_verify_loading_depth', verify)
    monkeypatch.setattr(expert, '_move_cup', move)
    previous_loads = {}
    result = expert._load_batch(SimpleNamespace(idle=lambda _: True), env, object(), record, 0, previous_loads)
    previous_loads['medium'] = .08
    candidates = [row for row in record.expert_trace if row['phase'] == 'load_candidate']
    assert all(row['previous_loads_g'] == {} for row in candidates)
    assert all(a > b > .006 for a, b in zip(depths, depths[1:]))
    assert len(depths) <= expert.MAX_LOAD_ATTEMPTS
    if eventual_load == .20:
        assert result is None
        assert len(depths) == 2
        assert 'unsafe retained batch' in env.scene.bad
    elif eventual_load:
        assert result == ('fine', .08)
        assert depths == pytest.approx([.007, .00675])
    else:
        assert result is None
        assert len(depths) == expert.MAX_LOAD_ATTEMPTS
        assert 'empty after bounded' in env.scene.bad
    np.testing.assert_array_equal(env.data.qpos, np.zeros(10))


def test_new_inflight_mass_during_carry_blocks_every_tilt(monkeypatch):
    env, record = fake_env(received=.20, tool=.10), bridge()
    assert expert._check_batch(record, env, 'lifted')
    rotations = []
    def move(demo, env, hold, target, rotation, **kwargs):
        rotations.append(rotation.copy())
        env.grain_state['inflight_g'] = .04
        return True
    monkeypatch.setattr(expert, '_move_cup', move)
    assert not expert._pour_complete_batch(SimpleNamespace(idle=lambda _: True), env, object(), record, 0)
    assert len(rotations) == 2
    for rotation in rotations:
        np.testing.assert_allclose(rotation @ SCOOP_FRAME[:, 2], [0., 0., 1.])
    assert 'unsafe retained batch before pour' in env.scene.bad
    assert record.expert_trace[-1]['phase'] == 'pre_pour_guard'


def test_environment_terminal_success_is_preserved_mid_pour(monkeypatch):
    env = fake_env()
    env.reset = lambda **kwargs: None
    finished = {'value': False}
    env.info = lambda: dict(success=finished['value'], failure=env.scene.bad)
    record = SimpleNamespace(rows=['existing terminal transition'])
    monkeypatch.setattr(expert, 'RecorderBridge', lambda *args, **kwargs: record)
    monkeypatch.setattr(expert, '_make_demo', lambda *args: object())
    monkeypatch.setattr(expert, '_approach_strict', lambda *args: (object(), object()))
    monkeypatch.setattr(expert, '_load_batch', lambda *args: ('bulk', .26))
    def pour(*args):
        record.expert_phase = 'complete_pour'
        finished['value'] = True
        raise expert.EpisodeFinished('success')
    monkeypatch.setattr(expert, '_pour_complete_batch', pour)
    returned, info = expert.expert_episode(env, seed=0, target_g=.3)
    assert returned is record and record.rows == ['existing terminal transition']
    assert info['success'] and not env.scene.bad
    assert info['expert_terminal_phase'] == 'complete_pour'
    assert info['expert_controller'] == expert.EXPERT_CONTROLLER


def test_inventory_in_target_band_cannot_override_environment_success(monkeypatch):
    env = fake_env()
    env.reset = lambda **kwargs: None
    env.info = lambda: dict(success=False, failure=env.scene.bad)
    record = SimpleNamespace(rows=[])
    holds = []
    demo = SimpleNamespace(idle=lambda count: holds.append(count) or True)
    monkeypatch.setattr(expert, 'RecorderBridge', lambda *args, **kwargs: record)
    monkeypatch.setattr(expert, '_make_demo', lambda *args: demo)
    monkeypatch.setattr(expert, '_approach_strict', lambda *args: (object(), object()))
    monkeypatch.setattr(expert, '_load_batch', lambda *args: ('bulk', .26))
    def pour(*args):
        env.powder.mass_g = .26
        env.grain_state['receiver_g'] = .26
        return True
    monkeypatch.setattr(expert, '_pour_complete_batch', pour)
    _, info = expert.expert_episode(env, seed=0, target_g=.3)
    assert holds == [1] * 100
    assert not info['success']
    assert info['failure'] == 'physical scale did not settle at target'


def test_outcome_and_hdf_root_share_source_and_controller_identity(monkeypatch, tmp_path):
    env = fake_env()
    env.control_hz, env.grain_count = 10, 200
    env.mj = SimpleNamespace(__version__='test-mujoco')
    env.close = lambda: None
    out = tmp_path / 'reference.hdf5'
    hashes = {'scripts/collect_granular_demos.py': 'a' * 64}
    monkeypatch.setattr(expert, 'source_identity', lambda: ('sha256:fixture', hashes))
    monkeypatch.setattr(expert, 'GranularWeighEnv', lambda **kwargs: env)
    info = dict(success=True, failure=None, expert_controller=expert.EXPERT_CONTROLLER,
                expert_terminal_phase='return_upright', expert_trace=[])
    monkeypatch.setattr(expert, 'expert_episode', lambda *args, **kwargs: (object(), info))
    monkeypatch.setattr(expert, 'write_episode', lambda group, *args: group.attrs.update(preserved=True))
    monkeypatch.setattr('sys.argv', ['collect_granular_demos', '--out', str(out), '--grains', '200'])
    expert.main()
    outcome = json.loads(out.with_suffix('.outcome.json').read_text())
    with h5py.File(out, 'r') as handle:
        assert handle.attrs['source_version'] == outcome['source_version'] == 'sha256:fixture'
        assert json.loads(handle.attrs['source_files_sha256']) == outcome['source_files_sha256'] == hashes
        assert handle.attrs['expert_controller'] == outcome['expert_controller'] == expert.EXPERT_CONTROLLER
        assert handle.attrs['expert_terminal_phase'] == 'return_upright'
        assert handle['data/demo_0'].attrs['preserved']
