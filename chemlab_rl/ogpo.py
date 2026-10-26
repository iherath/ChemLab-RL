"""One-step stand-in for Off-Policy Generative Policy Optimization.

OGPO (Patil et al., 2026) learns the critic off-policy, then improves a
generative policy with PPO on denoising trajectories that are sampled in
computation, not in the environment (Eq. 3.2–3.3). The importance ratio of
a flow policy is a product over ``K`` denoising steps (VaDar Appendix C.3,
Eq. C.15):

    ratio = prod_k pi_theta(a_{k-1} | s, a_k) / pi_bar(a_{k-1} | s, a_k)

A squashed Gaussian is the ``K = 1`` case: the product has one factor, and
``log_prob`` is that factor's log density. A later flow policy should return
the sum of per-step log densities from ``log_prob`` and draw the executed
action from ``sample``. The rest of the trainer does not need to change.

Advantages subtract the mean Q inside a group of actions drawn at the same
state (Eq. C.16). There is no learned value head. The paper drops GRPO's
extra division by the group standard deviation.

Not implemented, on purpose (OGPO+ / flow-only machinery):

* success-buffer regularization, two-sided conservative advantages, Q-variance reduction
* flow-matching or diffusion, and the Gaussian noise injected so a flow step has a density
* action chunks (the paper uses length 8) and best-of-N at test time
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def atanh(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-1.0 + 1e-4, 1.0 - 1e-4)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class SquashedGaussianPolicy(nn.Module):
    """Diagonal Gaussian pushed through ``tanh``, so actions lie in ``(-1, 1)^d``.

    ``sample`` draws ``a ~ pi(· | s)`` and returns ``log pi(a | s)``.
    ``log_prob`` scores an action that was already drawn (the PPO ratio's
    numerator and denominator). For ``K > 1`` this method is where the
    denoising-step log-densities would be summed.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for width in hidden:
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width
        self.body = nn.Sequential(*layers)
        self.mu_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)
        nn.init.constant_(self.log_std_head.bias, -0.5)

    def _mu_std(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(obs)
        mu = self.mu_head(hidden)
        std = self.log_std_head(hidden).clamp(-5.0, 2.0).exp()
        return mu, std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, std = self._mu_std(obs)
        u = mu + std * torch.randn_like(mu)
        action = torch.tanh(u)
        return action, self._log_prob_u(mu, std, u, action)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Demonstrator components sit on ±1, which is outside the open range of tanh.
        action = action.clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        mu, std = self._mu_std(obs)
        return self._log_prob_u(mu, std, atanh(action), action)

    @staticmethod
    def _log_prob_u(
        mu: torch.Tensor,
        std: torch.Tensor,
        u: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        log_norm = -0.5 * (
            ((u - mu) / std).square() + 2.0 * std.log() + math.log(2.0 * math.pi)
        ).sum(dim=-1)
        log_det = torch.log(1.0 - action.square() + 1e-6).sum(dim=-1)
        return log_norm - log_det


def group_advantages(q_agg: torch.Tensor) -> torch.Tensor:
    """``q_agg`` has shape ``(batch, group)``. Baseline is the group mean."""
    return q_agg - q_agg.mean(dim=-1, keepdim=True)


def ppo_clip_loss(
    log_prob: torch.Tensor,
    log_prob_old: torch.Tensor,
    advantage: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Clipped PPO surrogate to *minimize* (Eq. C.17 is the quantity we maximize).

    ``log_prob`` is the online policy, ``log_prob_old`` is the target policy
    that drew the actions. ``clip_eps`` defaults to the usual PPO value; a
    flow policy with a long product of ratios often wants a tighter clip.
    """
    log_ratio = (log_prob - log_prob_old).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantage
    return -torch.min(unclipped, clipped).mean()


def bc_nll(policy: SquashedGaussianPolicy, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Behavior-cloning loss: negative log-likelihood of demonstrator actions."""
    return -policy.log_prob(obs, action).mean()
