"""Shake command (action[14:17]) and the camera detector's contrast choice."""
import numpy as np
import pytest

from chemlab_rl.solution_detector import choose_backdrop
from chemlab_rl.solution_env import ACTION_DIM, SolutionPrepEnv
from chemlab_rl.solution_grains import SOLID_RGB


@pytest.fixture(scope="module")
def env():
    return SolutionPrepEnv(images=False)


def _tool_x(env):
    return env.session.arms["left"].tool_pose()[0].copy()


def test_action_space_has_shake_dims(env):
    assert env.action_space.shape == (ACTION_DIM,) == (17,)
    env.reset(seed=0)
    assert env.hold_action().shape == (17,)
    env.step(env.hold_action()[:14])                       # 14-D still accepted (no shake)


def test_shake_command_rotates_the_left_hand_about_a_horizontal_axis(env):
    env.reset(seed=0, options=dict(layout_jitter_m=0.0))
    a = env.hold_action()
    a[14:] = [0.15, 4.0, 0.0]                              # ~8.6 deg, 4 Hz, about world x
    p0 = _tool_x(env)
    xs = []
    for _ in range(20):                                    # 2 s
        env.step(a)
        xs.append(_tool_x(env) - p0)
    xs = np.array(xs[5:])                                  # after the 0.3 s ramp
    # rotation about x swings the tool point in y/z (lever ~0.17 m), not along x
    assert np.ptp(xs[:, 1]) + np.ptp(xs[:, 2]) > 0.02
    assert np.ptp(xs[:, 0]) < 0.006
    a[14] = 0.0
    for _ in range(6):
        env.step(a)
    assert env.info()["shake"]["amp_m"] == 0.0             # ramps down to off
    assert np.linalg.norm(_tool_x(env) - p0) < 0.004       # and comes back


def test_backdrop_choice_contrasts_with_the_solid():
    assert choose_backdrop(np.array(SOLID_RGB[0]) * 255) == "dark"    # white powder
    assert choose_backdrop(np.array(SOLID_RGB[4]) * 255) == "light"   # dark brown powder
