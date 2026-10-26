"""Value disaggregation (VaDar) pieces used by the lab trainer.

VaDar (Saxena et al., 2026, Algorithm 2 and Appendix A) keeps a single
subgoal-agnostic policy and puts subgoal structure in the critic:

* one sparse reward per subgoal (Eq. 3.2),
* a vector critic ``Q(s, a) -> R^{|G|}`` with a shared trunk (Appendix B.1),
* an entrywise ensemble mean and an EMA target (Appendix B.2–B.4),
* a clipped TD target so a finished subgoal cannot integrate to
  ``r_done / (1 - gamma)`` (Eq. 5.2),
* policy extraction on ``Q_agg = sum_g w[g] Q(s, a, g)`` (Eq. 5.1).

Next-future is the recommended weighting for manipulation (Appendix A).
The other three rows of Table 1 are included for ablations. Indicators are
not normalized: the paper sums the selected heads.

The replay stores the reward vector and the completion vector at ``s``,
which is what ``GetWeighting`` reads. Completion at ``s'`` is what the
reward is computed from (Algorithm 2, lines 6–7).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# Eq. 3.2. The paper's running example uses r_done = 5.
R_STEP = -1.0
R_DONE = 5.0
# Eq. 5.2. c_clip = 1 caps a completed head at r_done.
C_CLIP = 1.0

# Appendix B.4 writes tau = 0.005. Table 8 also lists 0.05; the trainer
# uses the equation value unless overridden.
TAU = 0.005

WEIGHTINGS = ("current_myopic", "current_future", "next_myopic", "next_future")


def sparse_reward(done: np.ndarray) -> np.ndarray:
    """Entrywise Eq. 3.2. ``done`` is 0/1 with shape ``(G,)`` or ``(B, G)``."""
    done_arr = np.asarray(done, dtype=np.float32)
    return np.where(done_arr > 0.5, R_DONE, R_STEP).astype(np.float32)


def get_weighting(done: np.ndarray, mode: str = "next_future") -> np.ndarray:
    """Indicator weights for policy extraction (Table 1).

    ``g_curr`` is the first incomplete subgoal. ``g_next`` is the one after
    it, or the last subgoal when the current one is already last. If every
    subgoal is done, every mode puts weight 1 on the final head only.

    Accepts ``(G,)`` or ``(B, G)``. A non-prefix completion (a later subgoal
    done while an earlier one is not) still treats the earliest zero as current,
    which is the sequential rule in Section 5.
    """
    if mode not in WEIGHTINGS:
        raise ValueError(f"unknown weighting {mode!r}; expected one of {WEIGHTINGS}")
    flags = np.asarray(done, dtype=np.float32)
    single = flags.ndim == 1
    if single:
        flags = flags[None, :]
    if flags.ndim != 2:
        raise ValueError("done must have shape (G,) or (B, G)")
    batch, n_goals = flags.shape
    incomplete = flags < 0.5
    has_open = incomplete.any(axis=1)
    # argmax is the first True. Rows with no open subgoal are overwritten below.
    first = incomplete.argmax(axis=1)
    g_curr = np.where(has_open, first, n_goals - 1)
    g_next = np.minimum(g_curr + 1, n_goals - 1)
    index = np.arange(n_goals)[None, :]
    if mode == "current_myopic":
        weights = index == g_curr[:, None]
    elif mode == "current_future":
        weights = index >= g_curr[:, None]
    elif mode == "next_myopic":
        weights = index == g_next[:, None]
    else:
        weights = index >= g_next[:, None]
    weights = weights.astype(np.float32)
    return weights[0] if single else weights


def aggregate_q(q: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``sum_g w[g] Q(s, a, g)``. ``q`` and ``weights`` broadcast on the goal axis."""
    return (q * weights).sum(dim=-1)


def td_target(
    reward: torch.Tensor,
    next_q: torch.Tensor,
    gamma: float,
    c_clip: float = C_CLIP,
    r_done: float = R_DONE,
) -> torch.Tensor:
    """Clipped vector backup, Eq. 5.2.

    ``reward`` and ``next_q`` have shape ``(..., G)``. The caller folds the
    episode-termination mask into ``next_q`` (zero it on true terminals).
    Horizon truncation should still bootstrap; that is not this function's job.
    """
    cap = c_clip * r_done
    raw = reward + gamma * next_q
    return torch.minimum(raw, torch.full_like(raw, cap))


def polyak_update(target: nn.Module, online: nn.Module, tau: float = TAU) -> None:
    """``target <- (1 - tau) * target + tau * online`` (Appendix B.4)."""
    with torch.no_grad():
        for dest, src in zip(target.parameters(), online.parameters()):
            dest.mul_(1.0 - tau).add_(src, alpha=tau)


class MultiHeadCritic(nn.Module):
    """One trunk, ``|G|`` linear outputs. Goals share features (Appendix B.1)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_subgoals: int,
        hidden: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim + action_dim
        for width in hidden:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, n_subgoals)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(torch.cat([obs, action], dim=-1)))


class CriticEnsemble(nn.Module):
    """``M`` independent multi-head critics. Aggregation is an entrywise mean.

    The paper uses ``M = 10`` (Appendix B.2). The toy default is smaller and
    is chosen by the caller.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_subgoals: int,
        n_critics: int = 2,
        hidden: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        self.critics = nn.ModuleList(
            [
                MultiHeadCritic(obs_dim, action_dim, n_subgoals, hidden)
                for _ in range(n_critics)
            ]
        )

    def mean(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack([critic(obs, action) for critic in self.critics], dim=0)
        return stacked.mean(dim=0)

    def loss(self, obs: torch.Tensor, action: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean over the ensemble of the per-head sum of squared residuals (B.5).

        Summing over heads matches the paper. Averaging over the ensemble keeps
        the step size from growing with ``M``.
        """
        per_member = [
            (critic(obs, action) - target).pow(2).sum(dim=-1)
            for critic in self.critics
        ]
        return torch.stack(per_member, dim=0).mean()


class ReplayBuffer:
    """Off-policy buffer of ``(s, a, s', r, d, final)``.

    ``r`` and ``d`` are vectors of length ``|G|``. ``d`` is completion at ``s``
    (policy weights). ``r`` is the sparse reward of completion at ``s'``.
    ``final`` is 1 when ``s'`` ended the episode by task success, so the
    backup does not bootstrap past it.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        n_subgoals: int,
    ) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_subgoals), dtype=np.float32)
        self.dones = np.zeros((capacity, n_subgoals), dtype=np.float32)
        self.finals = np.zeros((capacity, 1), dtype=np.float32)
        self.idx = 0
        self.full = False

    def __len__(self) -> int:
        return self.capacity if self.full else self.idx

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: np.ndarray,
        done_s: np.ndarray,
        final: float,
    ) -> None:
        i = self.idx
        self.obs[i] = obs
        self.actions[i] = action
        self.next_obs[i] = next_obs
        self.rewards[i] = reward
        self.dones[i] = done_s
        self.finals[i, 0] = final
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        n = len(self)
        if n < batch_size:
            raise ValueError(f"need at least {batch_size} transitions, have {n}")
        index = rng.integers(0, n, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[index]),
            "actions": torch.from_numpy(self.actions[index]),
            "next_obs": torch.from_numpy(self.next_obs[index]),
            "rewards": torch.from_numpy(self.rewards[index]),
            "dones": torch.from_numpy(self.dones[index]),
            "finals": torch.from_numpy(self.finals[index]),
        }
