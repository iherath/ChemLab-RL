"""Low-dim wrapper around Leo's ChemVialWeighing env, plus HDF5 helpers.

The MuJoCo task is vial weighing, not cyclic voltammetry. VaDar's ``G`` is the
sequential completion vector of *this* MDP:

1. ``open_door`` — a draft-shield door has been opened
2. ``grasp_vial`` — a gripper has lifted ``vial_a``
3. ``place_vial`` — the vial is upright on the pan with a ~20 g reading
4. ``close_shield`` — doors closed after the vial is on the pan (env success)

Bits latch: once a subgoal is true it stays true, so releasing the vial after
placing it does not reopen the grasp head. Instantaneous flags are computed from
``joint_pos`` (14) and ``object`` (11), the same keys Leo's recorder writes.

Observation for the policy is that 25-vector. It has no subgoal index.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SUBGOALS: tuple[str, ...] = (
    "open_door",
    "grasp_vial",
    "place_vial",
    "close_shield",
)
N_SUBGOALS = len(SUBGOALS)
OBS_DIM = 25  # 14 proprio + 11 object
ACTION_DIM = 14

BOARD_TOP = 0.775
DOOR_OPEN_M = 0.05
DOOR_CLOSED_TOL = 0.005
GRIP_CLOSED = 0.40  # rad; Robotiq driver is ~0 open, ~0.8 closed
VIAL_MASS_G = 20.0
READING_TOL_G = 1.0
LIFT_Z = BOARD_TOP + 0.04


def lowdim_from_parts(joint_pos: np.ndarray, obj: np.ndarray) -> np.ndarray:
    return np.concatenate([joint_pos, obj]).astype(np.float32)


def object_vector(env) -> np.ndarray:
    vial = env.data.body("vial_a")
    return np.concatenate(
        [vial.xpos, vial.xquat, [env.balance_g()], env.doors()]
    ).astype(np.float32)


def lowdim_obs(env) -> np.ndarray:
    return lowdim_from_parts(env._proprio(), object_vector(env))


def _quat_upright(quat_wxyz: np.ndarray) -> bool:
    """Body +z after the wxyz quaternion; same 0.95 threshold as the env."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    zz = 1.0 - 2.0 * (x * x + y * y)
    return zz > 0.95


def instantaneous_done(joint_pos: np.ndarray, obj: np.ndarray) -> np.ndarray:
    """Unlatched subgoal bits from one observation. Shape ``(4,)``."""
    vial_z = float(obj[0:3][2])
    balance_g = float(obj[7])
    doors = obj[8:11]
    grip_l = float(joint_pos[6])
    grip_r = float(joint_pos[13])
    opened = bool(np.max(doors) > DOOR_OPEN_M)
    grasped = bool(
        (grip_l > GRIP_CLOSED or grip_r > GRIP_CLOSED) and vial_z > LIFT_Z
    )
    placed = bool(
        abs(balance_g - VIAL_MASS_G) < READING_TOL_G and _quat_upright(obj[3:7])
    )
    closed = bool(placed and np.all(doors < DOOR_CLOSED_TOL))
    return np.array([opened, grasped, placed, closed], dtype=np.float32)


def latch(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    return np.maximum(previous, current).astype(np.float32)


class VialWeighingAdapter:
    """CVLab-shaped interface over ``ChemVialWeighing-v0``.

    ``step`` returns ``(obs, info)`` with ``info['done']`` the latched length-4
    vector. Images are not rendered; the policy is low-dim only.
    """

    obs_dim = OBS_DIM
    action_dim = ACTION_DIM
    n_subgoals = N_SUBGOALS

    def __init__(
        self,
        seed: int = 0,
        horizon: int = 200,
        layout: str = "weigh",
        clutter: bool = False,
    ) -> None:
        from chem_rl_sim.env import VialWeighingEnv

        self.horizon = int(horizon)
        self._env = VialWeighingEnv(
            image_size=32,
            layout=layout,
            clutter=clutter,
            normalized_actions=True,
            max_episode_steps=self.horizon,
            randomize=True,
        )
        # Training does not use pixels; skip the EGL/GLFW render on every step.
        self._env._get_obs = lambda: {
            "state": self._env._state(),
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
        }
        self._latched = np.zeros(N_SUBGOALS, dtype=np.float32)
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> np.ndarray:
        self._env.reset(seed=seed)
        self._latched = instantaneous_done(self._env._proprio(), object_vector(self._env))
        self.t = 0
        return lowdim_obs(self._env)

    def observation(self) -> np.ndarray:
        return lowdim_obs(self._env)

    def done_vector(self) -> np.ndarray:
        return self._latched.copy()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, dict]:
        _obs, _reward, terminated, truncated, info = self._env.step(action)
        self.t += 1
        inst = instantaneous_done(self._env._proprio(), object_vector(self._env))
        self._latched = latch(self._latched, inst)
        success = bool(self._latched[-1] > 0.5) or bool(info.get("success"))
        if success:
            self._latched[-1] = 1.0
        done = self._latched.copy()
        out = {
            "done": done,
            "success": success,
            "terminated": bool(terminated) or success,
            "truncated": bool(truncated) and not success,
            "t": self.t,
            "vial_on_pan": info.get("vial_on_pan"),
            "doors_closed": info.get("doors_closed"),
            "balance_g": info.get("balance_g"),
        }
        return lowdim_obs(self._env), out

    def mj_state(self) -> np.ndarray:
        """``[time, qpos, qvel]`` for GIF replay, same layout as Leo's recorder."""
        d = self._env.data
        return np.concatenate([[d.time], d.qpos, d.qvel]).astype(np.float64)

    def close(self) -> None:
        self._env.close()


def load_hdf5_demos(path: str | Path) -> dict[str, np.ndarray]:
    """Read Leo's robomimic HDF5 into flat arrays for BC and the replay buffer."""
    import h5py

    path = Path(path)
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    next_list: list[np.ndarray] = []
    done_s_list: list[np.ndarray] = []
    reward_list: list[np.ndarray] = []
    final_list: list[np.ndarray] = []
    demo_success: list[int] = []
    demo_len: list[int] = []

    from chemlab_rl.vadar import sparse_reward

    with h5py.File(path, "r") as f:
        names = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for name in names:
            g = f[f"data/{name}"]
            joint = np.asarray(g["obs/joint_pos"], dtype=np.float32)
            obj = np.asarray(g["obs/object"], dtype=np.float32)
            next_joint = np.asarray(g["next_obs/joint_pos"], dtype=np.float32)
            next_obj = np.asarray(g["next_obs/object"], dtype=np.float32)
            actions = np.asarray(g["actions"], dtype=np.float32)
            success = int(g.attrs.get("success", 0))
            t_steps = actions.shape[0]
            latched = instantaneous_done(joint[0], obj[0])
            for t in range(t_steps):
                done_s = latched.copy()
                latched = latch(latched, instantaneous_done(next_joint[t], next_obj[t]))
                obs_list.append(lowdim_from_parts(joint[t], obj[t]))
                next_list.append(lowdim_from_parts(next_joint[t], next_obj[t]))
                act_list.append(actions[t])
                done_s_list.append(done_s)
                reward_list.append(sparse_reward(latched))
                final_list.append(np.float32(success if t == t_steps - 1 else 0.0))
            demo_success.append(success)
            demo_len.append(t_steps)

    return {
        "obs": np.stack(obs_list),
        "actions": np.stack(act_list),
        "next_obs": np.stack(next_list),
        "dones": np.stack(done_s_list),
        "rewards": np.stack(reward_list),
        "finals": np.asarray(final_list, dtype=np.float32),
        "demo_success": np.asarray(demo_success, dtype=np.int32),
        "demo_len": np.asarray(demo_len, dtype=np.int32),
    }
