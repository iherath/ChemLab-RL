"""ChemVialWeighing as a vadar-public env: flat low-dim obs + info['subgoal_dones'].

Copied into vadar-public/envs/ by patch_vadar.py. Policy observation is Leo's
TASK_CONFIG concat (eef poses + object = 27). Subgoal checks still use
joint_pos + object, same as chemlab_rl.vial_env.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# mujoco binds a GL backend at import. RunPod EGL is often missing; this
# task is low-dim so OSMesa is enough. Leave MUJOCO_GL=glfw alone (local Mac).
if os.environ.get("MUJOCO_GL", "").lower() in ("", "egl"):
    os.environ["MUJOCO_GL"] = "osmesa"

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium.spaces import Box

from chemlab_rl.vial_env import instantaneous_done, object_vector
from envs import subgoal_utils as SG
from vadar.utils.datasets import Dataset

ENV_NAME = "chem_vial-low_dim"
LOW_DIM_KEYS = (
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot1_eef_pos",
    "robot1_eef_quat",
    "robot1_gripper_qpos",
    "object",
)
OBS_DIM = 27  # 3+4+1 + 3+4+1 + 11
ACTION_DIM = 14
MAX_EPISODE_STEPS = 1600


def _bootstrap_chem_rl_sim() -> None:
    chem = os.environ.get("CHEM_RL_SIM")
    roots = []
    if chem:
        roots.append(Path(chem))
    lab = os.environ.get("CHEMLAB_ROOT")
    if lab:
        roots.append(Path(lab) / "chem_rl_sim")
    here = Path(__file__).resolve()
    roots.extend(
        [
            here.parents[1] / "chem_rl_sim",
            here.parents[2] / "chem_rl_sim",
            here.parents[2] / "chemlab-rl-project" / "chem_rl_sim",
        ]
    )
    for root in roots:
        if (root / "chem_rl_sim" / "env.py").is_file() or (root / "env.py").is_file():
            pkg = root if (root / "chem_rl_sim" / "env.py").is_file() else root.parent
            ur = pkg / "third_party" / "urgantry_sim"
            if str(ur) not in sys.path:
                sys.path.insert(0, str(ur))
            if str(pkg) not in sys.path:
                sys.path.insert(0, str(pkg))
            return
    raise ImportError(
        "chem_rl_sim not found. Set CHEM_RL_SIM to the clone that contains env.py "
        "(with third_party/urgantry_sim)."
    )


def _skip_gl_renderer() -> None:
    """Leo's env always constructs mujoco.Renderer; training never uses pixels."""
    if getattr(mujoco, "_chemlab_dummy_renderer", False):
        return

    class DummyRenderer:
        def __init__(self, model, height=32, width=32, **_):
            self.height, self.width = int(height), int(width)

        def update_scene(self, *a, **k):
            pass

        def render(self):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        def close(self):
            pass

    mujoco.Renderer = DummyRenderer
    mujoco._chemlab_dummy_renderer = True


def _dataset_path(env_name: str | None = None) -> str:
    path = os.environ.get("CHEMVIAL_DATASET")
    if not path:
        raise ValueError(
            "Set CHEMVIAL_DATASET to the robomimic HDF5 (e.g. data/teleop/weigh_vial_grok_v2.hdf5)"
        )
    if not Path(path).is_file():
        raise FileNotFoundError(f"CHEMVIAL_DATASET={path} does not exist")
    return path


def _eef_parts(env) -> list[np.ndarray]:
    joint = env._proprio()
    parts: list[np.ndarray] = []
    for i, side in enumerate(("left", "right")):
        site = env.data.site(f"{side}_grip_pinch")
        pos = np.asarray(site.xpos, dtype=np.float32)
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, np.asarray(site.xmat, dtype=np.float64))
        grip = np.asarray(joint[6 if i == 0 else 13], dtype=np.float32).reshape(1)
        parts.extend([pos, quat.astype(np.float32), grip])
    parts.append(object_vector(env))
    return parts


def lowdim_obs(env) -> np.ndarray:
    return np.concatenate(_eef_parts(env)).astype(np.float32)


def _raw_goals(env) -> np.ndarray:
    return instantaneous_done(env._proprio(), object_vector(env))


class ChemVialLowDimEnv(gym.Env):
    """Gymnasium wrapper: Box obs, sparse {-1, 0} reward, sticky subgoal_dones."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, seed: int = 0, subgoal_cfg=None, max_episode_steps: int = MAX_EPISODE_STEPS):
        super().__init__()
        _bootstrap_chem_rl_sim()
        _skip_gl_renderer()
        from chem_rl_sim.env import VialWeighingEnv

        self._env = VialWeighingEnv(
            image_size=32,
            layout="weigh",
            clutter=False,
            normalized_actions=True,
            max_episode_steps=max_episode_steps,
            randomize=True,
        )
        self._env._get_obs = lambda: {
            "state": self._env._state(),
            "image": np.zeros((32, 32, 3), dtype=np.uint8),
        }
        self.max_episode_steps = int(max_episode_steps)
        self.subgoal_cfg = SG.parse_subgoal_cfg(ENV_NAME, subgoal_cfg) if subgoal_cfg else None
        self.num_subgoals = self.subgoal_cfg["num_subgoals"] if self.subgoal_cfg else 0
        self.reward_mode = self.subgoal_cfg["reward_mode"] if self.subgoal_cfg else "sparse"
        self.goal_obs = self.subgoal_cfg["goal_obs"] if self.subgoal_cfg else None
        self._shaped_reward = SG.get_shaped_reward_fn() if self.reward_mode == "shaped" else None
        dag = self.subgoal_cfg["dag"] if self.subgoal_cfg else None
        self._prereqs = SG.prerequisites(self.num_subgoals, dag=dag, mode="sequential")
        self.goal_done = [False] * self.num_subgoals
        obs_dim = OBS_DIM + (self.num_subgoals if self.goal_obs in ("cumulative", "singular") else 0)
        self.action_space = Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self._t = 0
        self._env.reset(seed=seed)

    def _obs(self) -> np.ndarray:
        obs = lowdim_obs(self._env)
        if self.goal_obs in ("cumulative", "singular"):
            suffix = SG.encode_goal_vector(self.goal_done, self.goal_obs).astype(np.float32)
            obs = np.concatenate([obs, suffix])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._env.reset(seed=seed)
        self._t = 0
        self.goal_done = [False] * self.num_subgoals
        if self.num_subgoals:
            self.goal_done = SG.ratchet(self.goal_done, _raw_goals(self._env).astype(bool), self._prereqs)
        return self._obs(), {}

    def step(self, action):
        _obs, _env_r, terminated, truncated, info = self._env.step(np.asarray(action, dtype=np.float32))
        self._t += 1
        success = bool(info.get("success"))
        prev = list(self.goal_done)
        if self.num_subgoals:
            self.goal_done = SG.ratchet(self.goal_done, _raw_goals(self._env).astype(bool), self._prereqs)
            if success:
                self.goal_done[-1] = True
            info["subgoal_dones"] = np.asarray(self.goal_done, dtype=np.float32)
        env_r = 1.0 if success else 0.0
        if self.reward_mode == "hgc":
            reward = SG.hgc_reward(env_r - 1.0, prev, self.goal_done)
        elif self.reward_mode == "shaped":
            reward = float(self._shaped_reward(np.asarray(self.goal_done, dtype=np.float32)))
        else:
            reward = env_r - 1.0
        if success:
            terminated = True
        info["success"] = success
        return self._obs(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        return self._env.render() if hasattr(self._env, "render") else None

    def close(self):
        self._env.close()


def make_env(env_name, seed=0, subgoal_cfg=None, **_kwargs):
    env = ChemVialLowDimEnv(seed=seed, subgoal_cfg=subgoal_cfg)
    env.reset(seed=seed)
    return env


def _as_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    return arr[:, None] if arr.ndim == 1 else arr


def _concat_keys(group, keys) -> np.ndarray:
    return np.concatenate([_as_2d(group[k]) for k in keys], axis=-1)


def get_dataset(env, env_name):
    """Leo robomimic HDF5 -> vadar Dataset, with post-action subgoal_dones."""
    import h5py

    dataset_path = _dataset_path(env_name)
    wrapper = env.unwrapped if hasattr(env, "unwrapped") else env
    subgoal_cfg = getattr(wrapper, "subgoal_cfg", None)
    G = subgoal_cfg["num_subgoals"] if subgoal_cfg is not None else 0
    dag = subgoal_cfg["dag"] if subgoal_cfg is not None else None
    prereqs = SG.prerequisites(G, dag=dag, mode="sequential") if G else None
    reward_mode = subgoal_cfg["reward_mode"] if subgoal_cfg is not None else "sparse"
    goal_obs = subgoal_cfg["goal_obs"] if subgoal_cfg is not None else None

    observations, actions, next_observations = [], [], []
    terminals, rewards, masks = [], [], []
    subgoal_rows, episode_starts = [], []

    with h5py.File(dataset_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        print(f"[chem_vial] loading {len(demos)} demos from {dataset_path}")
        for name in demos:
            g = f[f"data/{name}"]
            obs_g, next_g = g["obs"], g["next_obs"]
            a = np.asarray(g["actions"], dtype=np.float32)
            r = np.asarray(g["rewards"], dtype=np.float32)
            dones = np.asarray(g["dones"], dtype=np.float32)
            obs = _concat_keys(obs_g, LOW_DIM_KEYS)
            next_obs = _concat_keys(next_g, LOW_DIM_KEYS)
            # robomimic convention: {0,1} success -> {-1, 0}
            r = (r > 0.5).astype(np.float32) - 1.0
            term = dones.astype(np.float32)
            mask = 1.0 - dones.astype(np.float32)
            if term[-1] == 0:
                term[-1] = 1.0
            starts = np.zeros(len(a), dtype=bool)
            starts[0] = True

            if G:
                joint_n = np.asarray(next_g["joint_pos"], dtype=np.float32)
                obj_n = np.asarray(next_g["object"], dtype=np.float32)
                latched = [False] * G
                labels = np.zeros((len(a), G), dtype=np.float32)
                for t in range(len(a)):
                    latched = SG.ratchet(latched, instantaneous_done(joint_n[t], obj_n[t]).astype(bool), prereqs)
                    labels[t] = np.asarray(latched, dtype=np.float32)
                subgoal_rows.append(labels)

            observations.append(obs)
            next_observations.append(next_obs)
            actions.append(a)
            rewards.append(r)
            terminals.append(term)
            masks.append(mask)
            episode_starts.append(starts)

    fields = dict(
        observations=np.concatenate(observations),
        actions=np.concatenate(actions),
        next_observations=np.concatenate(next_observations),
        rewards=np.concatenate(rewards).astype(np.float32),
        terminals=np.concatenate(terminals).astype(np.float32),
        masks=np.concatenate(masks).astype(np.float32),
    )
    print(f"[chem_vial] dataset size {fields['actions'].shape[0]}  obs {fields['observations'].shape}")
    if G:
        post = np.concatenate(subgoal_rows)
        starts = np.concatenate(episode_starts)
        fields["subgoal_dones"] = post
        if reward_mode in ("hgc", "shaped"):
            fields["rewards"] = SG.relabel_rewards(reward_mode, fields["rewards"], post, starts)
        if goal_obs in ("cumulative", "singular"):
            obs_g, next_g = SG.goal_suffixes(post, starts, goal_obs)
            fields["observations"] = np.concatenate([fields["observations"], obs_g], axis=-1)
            fields["next_observations"] = np.concatenate([fields["next_observations"], next_g], axis=-1)
    return Dataset.create(**fields)
