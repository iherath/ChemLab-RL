"""Behavior-cloning warm start, then VaDar's online loop with a one-step OGPO actor.

Per environment step (Algorithm 2):

1. Act with the target policy. The action is a function of the state only.
2. Store ``(s, a, s', r, d(s), final)``. ``r`` is the vector of per-subgoal
   sparse rewards at ``s'``.
3. Fit every critic head to the clipped TD target, then Polyak-update the
   target ensemble.
4. Re-sample a group of actions from the target policy at buffer states
   (the cheap inner step of OGPO) and take one clipped PPO step on
   ``Q_agg`` with next-future weights. Polyak-update the actor.

Appendix H.6 uses a wider network and a larger ensemble than this file.
Those paper values are noted on ``TrainConfig``; the defaults are small so
a CPU smoke run finishes quickly. OGPO's 20k-step critic warmup is omitted.
Once the buffer can fill a batch, the update-to-data ratio is one critic
update and one actor update per environment step (Table 8, UTD (1, 1)).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch

from chemlab_rl.env import SUBGOALS, CVLab
from chemlab_rl.ogpo import SquashedGaussianPolicy, bc_nll, group_advantages, ppo_clip_loss
from chemlab_rl.vadar import (
    CriticEnsemble,
    ReplayBuffer,
    aggregate_q,
    get_weighting,
    polyak_update,
    sparse_reward,
    td_target,
)


@dataclass
class TrainConfig:
    """Toy hyperparameters.

    Appendix H.6, for comparison: actor/critic lr ``3e-4``, batch 256,
    replay ``2e6``, MLP widths ``(512, 512, 512, 512)``, ensemble ``M = 10``,
    entrywise mean, EMA ``tau = 0.005`` in Appendix B.4 (Table 8 lists 0.05).
    OGPO additionally uses action chunks of length 8 and 10 flow steps; this
    loop is the ``K = 1`` Gaussian reduction and does not chunk.
    """

    seed: int = 0
    n_demos: int = 8
    bc_epochs: int = 5
    online_episodes: int = 8
    horizon: int = 80
    batch_size: int = 64
    group_size: int = 8  # N_group in OGPO Eq. 3.3; not a value fixed by the paper
    n_critics: int = 2
    hidden: tuple[int, ...] = (128, 128)
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    clip_eps: float = 0.2
    replay_size: int = 50_000
    weighting: str = "next_future"
    quiet: bool = False


def run(cfg: TrainConfig | None = None) -> dict[str, object]:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    env = CVLab(seed=cfg.seed, horizon=cfg.horizon)
    replay = ReplayBuffer(cfg.replay_size, env.obs_dim, env.action_dim, env.n_subgoals)

    demo_obs, demo_act = _collect_demos(env, replay, cfg.n_demos)
    policy = SquashedGaussianPolicy(env.obs_dim, env.action_dim, cfg.hidden)
    bc_loss, _bc_hist = _behavior_clone(policy, demo_obs, demo_act, cfg, rng)
    target_policy = copy.deepcopy(policy)

    critic = CriticEnsemble(
        env.obs_dim, env.action_dim, env.n_subgoals, cfg.n_critics, cfg.hidden
    )
    target_critic = copy.deepcopy(critic)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
    actor_opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    if not cfg.quiet:
        print(
            f"VaDar {cfg.weighting} + one-step OGPO | "
            f"demos={cfg.n_demos} bc_loss={bc_loss:.3f} "
            f"online_episodes={cfg.online_episodes}"
        )

    successes = 0
    env_steps = 0
    updates = 0
    last_done = np.zeros(env.n_subgoals, dtype=np.float32)
    for episode in range(cfg.online_episodes):
        obs = env.reset()
        info = {"success": False, "done": last_done, "terminated": False, "truncated": False}
        for _ in range(env.horizon):
            done_s = env.done_vector()
            action = _act(target_policy, obs)
            next_obs, info = env.step(action)
            _store(replay, obs, action, next_obs, done_s, info)
            obs = next_obs
            env_steps += 1
            if len(replay) >= cfg.batch_size:
                _update_critic(replay, rng, policy, critic, target_critic, critic_opt, cfg)
                _update_actor(
                    replay, rng, policy, target_policy, target_critic, actor_opt, cfg
                )
                updates += 1
            if info["terminated"] or info["truncated"]:
                break
        last_done = np.asarray(info["done"], dtype=np.float32)
        successes += int(info["success"])
        if not cfg.quiet:
            bits = " ".join(
                f"{name}={int(flag)}" for name, flag in zip(SUBGOALS, last_done)
            )
            print(f"episode {episode:03d}  success={int(info['success'])}  {bits}")

    if not cfg.quiet:
        print(f"online success {successes}/{cfg.online_episodes}")
    return {
        "episodes": cfg.online_episodes,
        "successes": successes,
        "env_steps": env_steps,
        "updates": updates,
        "bc_loss": bc_loss,
        "last_done": last_done,
    }


def _collect_demos(
    env: CVLab, replay: ReplayBuffer, n_demos: int
) -> tuple[np.ndarray, np.ndarray]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for _ in range(n_demos):
        obs = env.reset()
        for _ in range(env.horizon):
            done_s = env.done_vector()
            action = env.scripted_action()
            next_obs, info = env.step(action)
            _store(replay, obs, action, next_obs, done_s, info)
            observations.append(obs)
            actions.append(np.asarray(action, dtype=np.float32))
            obs = next_obs
            if info["terminated"] or info["truncated"]:
                break
    return np.stack(observations), np.stack(actions)


def _behavior_clone(
    policy: SquashedGaussianPolicy,
    obs: np.ndarray,
    actions: np.ndarray,
    cfg: TrainConfig,
    rng: np.random.Generator,
) -> tuple[float, list[float]]:
    obs_t = torch.from_numpy(obs)
    act_t = torch.from_numpy(actions)
    # A free std fits the pre-tanh residual by widening, and the tanh
    # change-of-variables term does not depend on the network, so the NLL
    # can look good while the mean action is still wrong. Hold std fixed
    # for the clone. The online PPO optimizer is built afterward and trains it.
    with torch.no_grad():
        policy.log_std_head.weight.zero_()
        policy.log_std_head.bias.fill_(-1.0)
    for param in policy.log_std_head.parameters():
        param.requires_grad_(False)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg.lr)
    last = 0.0
    n = obs_t.shape[0]
    epoch_losses: list[float] = []
    for _ in range(cfg.bc_epochs):
        order = rng.permutation(n)
        batch_losses: list[float] = []
        for start in range(0, n, cfg.batch_size):
            index = order[start : start + cfg.batch_size]
            if index.size == 0:
                continue
            loss = bc_nll(policy, obs_t[index], act_t[index])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()
            last = float(loss.detach())
            batch_losses.append(last)
        if batch_losses:
            epoch_losses.append(float(np.mean(batch_losses)))
    for param in policy.log_std_head.parameters():
        param.requires_grad_(True)
    return last, epoch_losses


def _act(policy: SquashedGaussianPolicy, obs: np.ndarray) -> np.ndarray:
    """Target-policy action. No subgoal argument (VaDar line 5, ``pi_bar``)."""
    with torch.no_grad():
        action, _ = policy.sample(torch.from_numpy(obs).unsqueeze(0))
    return action.squeeze(0).numpy().astype(np.float32)


def _store(
    replay: ReplayBuffer,
    obs: np.ndarray,
    action: np.ndarray,
    next_obs: np.ndarray,
    done_s: np.ndarray,
    info: dict,
) -> None:
    replay.add(
        obs,
        np.asarray(action, dtype=np.float32),
        next_obs,
        sparse_reward(info["done"]),
        np.asarray(done_s, dtype=np.float32),
        float(info["terminated"]),
    )


def _update_critic(
    replay: ReplayBuffer,
    rng: np.random.Generator,
    policy: SquashedGaussianPolicy,
    critic: CriticEnsemble,
    target_critic: CriticEnsemble,
    opt: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> None:
    batch = replay.sample(cfg.batch_size, rng)
    # Bootstrap action from the online policy (Algorithm 2, line 9). The
    # target ensemble supplies the bootstrapped Q, entrywise mean (B.2).
    with torch.no_grad():
        next_action, _ = policy.sample(batch["next_obs"])
        next_q = target_critic.mean(batch["next_obs"], next_action)
        next_q = next_q * (1.0 - batch["finals"])
        target = td_target(batch["rewards"], next_q, cfg.gamma)
    loss = critic.loss(batch["obs"], batch["actions"], target)
    opt.zero_grad()
    loss.backward()
    opt.step()
    polyak_update(target_critic, critic, cfg.tau)
    return float(loss.detach())


def _update_actor(
    replay: ReplayBuffer,
    rng: np.random.Generator,
    policy: SquashedGaussianPolicy,
    target_policy: SquashedGaussianPolicy,
    target_critic: CriticEnsemble,
    opt: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> None:
    """PPO on a fresh group of actions, not on the actions stored in the buffer.

    Drawing the group is the inner denoising step. For this Gaussian it is a
    single sample; a flow policy would unroll ``K`` steps inside ``sample``.
    Weights depend only on ``d(s)`` and are shared by the group (Appendix C.18).
    """
    batch = replay.sample(cfg.batch_size, rng)
    obs = batch["obs"]
    batch_size = obs.shape[0]
    weights = torch.from_numpy(get_weighting(batch["dones"].numpy(), cfg.weighting))
    obs_rep = obs.repeat_interleave(cfg.group_size, dim=0)
    weight_rep = weights.repeat_interleave(cfg.group_size, dim=0)
    with torch.no_grad():
        actions, log_prob_old = target_policy.sample(obs_rep)
        q_values = target_critic.mean(obs_rep, actions)
        q_agg = aggregate_q(q_values, weight_rep).view(batch_size, cfg.group_size)
        advantage = group_advantages(q_agg).reshape(-1)
    log_prob = policy.log_prob(obs_rep, actions)
    loss = ppo_clip_loss(log_prob, log_prob_old, advantage, cfg.clip_eps)
    opt.zero_grad()
    loss.backward()
    opt.step()
    polyak_update(target_policy, policy, cfg.tau)
    return float(loss.detach())


if __name__ == "__main__":
    run()
