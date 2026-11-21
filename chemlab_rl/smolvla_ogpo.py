"""Experimental PyTorch OGPO/VaDar mathematical primitives for SmolVLA.

This is an explicit port, NOT the official JAX implementation or a complete RL
trainer. See docs/SMOLVLA_OGPO.md for equations, source pins, and deviations.
Likelihoods below are densities of the entire augmented latent chain, never
tractable marginal densities of the final robot actions. No chain dimension or
denoising step is averaged out of the likelihood ratio.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
import math
from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn


Batch = Mapping[str, Any]
Velocity = Callable[[Batch, Tensor, Tensor], Tensor]


def _batch_time(time: Tensor | float, latent: Tensor) -> Tensor:
    result = torch.as_tensor(time, device=latent.device, dtype=latent.dtype)
    if result.ndim == 0:
        result = result.expand(latent.shape[0])
    if result.shape != (latent.shape[0],):
        raise ValueError("Time must be scalar or [batch]")
    if not torch.isfinite(result).all():
        raise ValueError("Time must be finite")
    return result


class SmolVLAVelocity(nn.Module):
    """Expose differentiable noise->data velocity of a pretrained SmolVLA.

    Input ``batch`` is already tokenized/normalized by the BC preprocessor.
    ``latent`` retains ALL max_action_dim channels, including unused channels.
    ``tau`` runs from noise at 0 to data at 1; native SmolVLA time runs backwards.
    We recompute the prefix rather than share its mutable KV cache. ``eval``
    behavior disables dropout but does not disable gradients or freeze weights.

    The optional mask builder only supports a lightweight contract test without
    importing LeRobot. Production uses its pinned attention-mask implementation.
    """

    def __init__(self, policy: nn.Module, *, mask_builder: Callable | None = None, cache_prefix: bool = False):
        super().__init__()
        self.policy = policy
        self._mask_builder = mask_builder
        self.cache_prefix = cache_prefix
        self._prefix_cache = None
        if getattr(policy.config, "adapt_to_pi_aloha", False):
            raise ValueError("UR actions must not use the ALOHA-specific transform")
        self.policy.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # All stochasticity must be represented by the chain Gaussian draws;
        # unrecorded dropout would invalidate a recomputed chain likelihood.
        self.policy.eval()
        return self

    @contextmanager
    def prefix_context(self, batch):
        """Reuse prefix computation inside one chain, retaining its autograd graph."""
        if not self.cache_prefix:
            yield
            return
        if self._prefix_cache is not None:
            raise RuntimeError("Prefix contexts cannot overlap")
        from lerobot.policies.common.vla_utils import make_att_2d_masks
        model = self.policy.model
        images, masks = self.policy.prepare_images(batch)
        prefix, padding, attention = model.embed_prefix(
            images, masks, batch["observation.language.tokens"],
            batch["observation.language.attention_mask"], state=self.policy.prepare_state(batch))
        _, kv = model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(padding, attention),
            position_ids=padding.long().cumsum(dim=1)-1,
            past_key_values=None, inputs_embeds=[prefix, None], use_cache=True)
        self._prefix_cache = (padding, kv)
        try:
            yield
        finally:
            self._prefix_cache = None

    def forward(self, batch: Batch, latent: Tensor, tau: Tensor | float) -> Tensor:
        config = self.policy.config
        if latent.ndim != 3 or latent.shape[1:] != (config.chunk_size, config.max_action_dim):
            raise ValueError("Latent shape must be [B, chunk_size, max_action_dim]")
        tau = _batch_time(tau, latent)
        if ((tau < 0) | (tau > 1)).any():
            raise ValueError("Forward flow time tau must be in [0,1]")
        if self._prefix_cache is not None:
            padding, kv = self._prefix_cache
            saved = [(layer.keys, layer.values) for layer in kv.layers]
            try:
                # Native backward-time velocity; restore the unchanged prefix after each step.
                native = self.policy.model.denoise_step(padding, kv, latent, 1.0-tau)
            finally:
                for layer, (keys, values) in zip(kv.layers, saved):
                    layer.keys, layer.values = keys, values
            return -native
        if self._mask_builder is None:
            from lerobot.policies.common.vla_utils import make_att_2d_masks
            mask_builder = make_att_2d_masks
        else:
            mask_builder = self._mask_builder
        model = self.policy.model
        images, image_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        prefix, prefix_pad, prefix_att = model.embed_prefix(
            images, image_masks, batch["observation.language.tokens"],
            batch["observation.language.attention_mask"], state=state,
        )
        suffix, suffix_pad, suffix_att = model.embed_suffix(latent, 1.0 - tau)
        padding = torch.cat((prefix_pad, suffix_pad), dim=1)
        attention = torch.cat((prefix_att, suffix_att), dim=1)
        (_, suffix_output), _ = model.vlm_with_expert.forward(
            attention_mask=mask_builder(padding, attention),
            position_ids=padding.long().cumsum(dim=1) - 1,
            past_key_values=None,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
        )
        suffix_output = suffix_output[:, -config.chunk_size:].float()
        native_velocity = model.action_out_proj(suffix_output)
        # x_native(t)=t*z+(1-t)*a; x_forward(tau)=tau*a+(1-tau)*z.
        return -native_velocity


@dataclass(frozen=True)
class ChainConfig:
    steps: int = 10
    diffusion_scale: float = 0.05
    score_correction: bool = True

    def __post_init__(self):
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(self.diffusion_scale) or self.diffusion_scale <= 0:
            raise ValueError("diffusion_scale must be finite and strictly positive")

    @property
    def dt(self) -> float:
        return 1.0 / self.steps


@dataclass(frozen=True)
class ChainSample:
    latents: Tensor  # [B, K+1, H, D], detached target-policy draw
    log_prob: Tensor  # [B], detached complete joint density incl initial N(0,I)

    def executed_actions(self, action_dim: int, action_steps: int | None = None) -> Tensor:
        """Deterministic projection only; caller unnormalizes and steps the env.

        The likelihood is still on the FULL stored chain, including coordinates
        and future actions discarded by this projection. Do not score this slice
        as though it were the marginal density of executed actions.
        """
        if not 1 <= action_dim <= self.latents.shape[-1]:
            raise ValueError("Invalid executed action dimension")
        n = self.latents.shape[-2] if action_steps is None else action_steps
        if not 1 <= n <= self.latents.shape[-2]:
            raise ValueError("Invalid executed action horizon")
        return self.latents[:, -1, :n, :action_dim]


def transition_parameters(latent: Tensor, velocity: Tensor, tau: Tensor | float, config: ChainConfig) -> tuple[Tensor, Tensor]:
    """Euler–Maruyama mean and variance of one *nonsingular* transition.

    sigma(tau)=c*sqrt(1-tau), score=(tau*v-x)/(1-tau).
    b=v+0.5*c^2*(tau*v-x), mu=x+dt*b, var=c^2*(1-tau)*dt.
    Taper and score denominator cancel analytically; never evaluate at tau=1.
    The correction is marginal-preserving only for exact flow/continuous time.
    """
    if latent.ndim != 3 or velocity.shape != latent.shape:
        raise ValueError("latent and velocity must have equal [B,H,D] shape")
    tau = _batch_time(tau, latent)
    if ((tau < 0) | (tau >= 1)).any():
        raise ValueError("Transition start times must be in [0,1), including the last step")
    t = tau[:, None, None]
    drift = velocity
    if config.score_correction:
        drift = velocity + 0.5 * config.diffusion_scale**2 * (t * velocity - latent)
    mean = latent + config.dt * drift
    variance = config.diffusion_scale**2 * (1.0 - t) * config.dt
    return mean, variance


def _normal_log_prob(value: Tensor, mean: Tensor, variance: Tensor) -> Tensor:
    """Sum all H,D coordinates in float64 to avoid large-log-density cancellation."""
    value, mean, variance = value.double(), mean.double(), variance.double()
    return (-0.5 * ((value - mean).square() / variance + variance.log() + math.log(2 * math.pi))).sum(dim=(-2, -1))


def _initial_log_prob(latent: Tensor) -> Tensor:
    return -0.5 * (latent.double().square() + math.log(2 * math.pi)).sum(dim=(-2, -1))


@torch.no_grad()
def sample_chain(velocity: Velocity, batch: Batch, shape: tuple[int, int, int], config: ChainConfig,
                 *, device: str | torch.device = "cpu", generator: torch.Generator | None = None) -> ChainSample:
    """Sample a target-policy chain; neither draws nor old densities retain a graph.

    Every one of K transitions has fresh untruncated Gaussian noise. There is
    no noise clipping, latent clipping, action clipping, or deterministic final
    transition. Keep this target policy fixed while scoring a PPO update group.
    """
    if len(shape) != 3 or any(n < 1 for n in shape):
        raise ValueError("shape must be positive [B,H,D]")
    latent = torch.randn(shape, device=device, dtype=torch.float32, generator=generator)
    states = [latent]
    log_prob = _initial_log_prob(latent)
    context = velocity.prefix_context(batch) if hasattr(velocity, "prefix_context") else nullcontext()
    with context:
        for k in range(config.steps):
            tau = torch.full((shape[0],), k / config.steps, device=latent.device, dtype=latent.dtype)
            mean, variance = transition_parameters(latent, velocity(batch, latent, tau), tau, config)
            noise = torch.randn(shape, device=latent.device, dtype=latent.dtype, generator=generator)
            latent = mean + variance.sqrt() * noise
            log_prob = log_prob + _normal_log_prob(latent, mean, variance)
            states.append(latent)
    return ChainSample(torch.stack(states, dim=1).detach(), log_prob.detach())


def chain_log_prob(velocity: Velocity, batch: Batch, latents: Tensor, config: ChainConfig) -> Tensor:
    """Score the fixed augmented chain, differentiating current policy means only.

    ``latents.detach()`` is essential: this is a score-function update, not
    backpropagation through the target sampler. The initial prior is constant
    in parameters and cancels from the new/old ratio, but is included here.
    """
    if latents.ndim != 4 or latents.shape[1] != config.steps + 1:
        raise ValueError("Expected latents [B,K+1,H,D]")
    fixed = latents.detach()
    log_prob = _initial_log_prob(fixed[:, 0])
    context = velocity.prefix_context(batch) if hasattr(velocity, "prefix_context") else nullcontext()
    with context:
        for k in range(config.steps):
            # Match the contiguous sampler layout during density rescoring.
            x = fixed[:, k].contiguous()
            tau = torch.full((x.shape[0],), k / config.steps, device=x.device, dtype=x.dtype)
            mean, variance = transition_parameters(x, velocity(batch, x, tau), tau, config)
            log_prob = log_prob + _normal_log_prob(fixed[:, k + 1], mean, variance)
    return log_prob


def full_chain_ratio(new_log_prob: Tensor, old_log_prob: Tensor) -> Tensor:
    """Exact exp(sum log-density differences), without averaging or log clipping."""
    if new_log_prob.shape != old_log_prob.shape:
        raise ValueError("New and old log densities must have equal shape")
    ratio = torch.exp(new_log_prob.double() - old_log_prob.detach().double())
    if not torch.isfinite(ratio).all():
        raise FloatingPointError("Full-chain ratio overflow/nonfinite; reduce the actor update rather than silently clamp the log ratio")
    return ratio


def clipped_chain_objective(new_log_prob: Tensor, old_log_prob: Tensor, advantages: Tensor, clip_epsilon: float = 0.01) -> Tensor:
    """Negative PPO surrogate. Old density and Q advantages are stop-gradient."""
    if not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must be in (0,1)")
    if advantages.shape != new_log_prob.shape:
        raise ValueError("Advantages and chain log densities must have equal shape")
    advantage = advantages.detach().double()
    if not torch.isfinite(advantage).all():
        raise ValueError("Advantages must be finite")
    ratio = full_chain_ratio(new_log_prob, old_log_prob)
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantage).mean()


def _binary_flags(dones: Tensor) -> Tensor:
    if dones.ndim < 1 or dones.shape[-1] < 1 or not torch.isfinite(dones).all():
        raise ValueError("Subgoal dones need a nonempty finite goal dimension")
    if not ((dones == 0) | (dones == 1)).all():
        raise ValueError("Subgoal dones must be binary")
    return dones.detach().float()


def next_future_weights(dones: Tensor) -> Tensor:
    """Exact sequential VaDar NF, with final-head fallback even when all done.

    000->011; 100->001; 110->001; 111->001. The current unfinished
    head is excluded. A non-prefix completion vector is rejected explicitly.
    """
    d = _binary_flags(dones)
    if d.shape[-1] > 1 and (d[..., 1:] > d[..., :-1]).any():
        raise ValueError("Sequential subgoal dones must be an ordered completed prefix")
    parents = torch.cat((torch.ones_like(d[..., :1]), d[..., :-1]), dim=-1)
    weights = 1.0 - parents
    weights[..., -1] = 1.0
    return weights


@torch.no_grad()
def conservative_group_advantages(q_ensemble: Tensor, dones: Tensor) -> Tensor:
    """OGPO Eq4.5 applied AFTER VaDar scalarization, without group-std division.

    q_ensemble: [B,N,M,G], N candidate chunks, M target-critic members.
    dones: [B,G], one fixed replay completion vector shared by all N candidates.
    Positive only when every member is positive; negative only when all agree.
    """
    if q_ensemble.ndim != 4 or dones.shape != (q_ensemble.shape[0], q_ensemble.shape[-1]):
        raise ValueError("Expected q[B,N,M,G], dones[B,G]")
    if q_ensemble.shape[1] < 2 or q_ensemble.shape[2] < 1:
        raise ValueError("Need at least two candidates and one ensemble member")
    if not torch.isfinite(q_ensemble).all():
        raise ValueError("Critic values must be finite")
    weights = next_future_weights(dones)[:, None, None, :]
    q = (q_ensemble.detach() * weights).sum(dim=-1)
    centered = q - q.mean(dim=1, keepdim=True)
    lower, upper = centered.amin(dim=2), centered.amax(dim=2)
    return torch.where(lower > 0, lower, torch.where(upper < 0, upper, torch.zeros_like(lower))).detach()


@torch.no_grad()
def vector_td_target(dones: Tensor, next_q: Tensor, terminated: Tensor, discount: float | Tensor,
                     executed_steps: int | Tensor = 1, *, r_done: float = 5.0, r_step: float = -1.0,
                     cap: float = 15.0) -> Tensor:
    """Port of pinned vadar-public compute_vadar_td_target (task heads only).

    dones[B,G] are latched flags AFTER the executed chunk, next_q is either
    ensemble-mean[B,G] or ensemble[B,M,G]. ``terminated`` means a true terminal;
    a time-limit truncation must pass False and bootstrap from its final obs.
    Gamma is scalar or [G]; executed_steps is the ACTUAL h, scalar or [B].
    Completed heads bootstrap from cap; terminal-completed heads target cap.
    This source-code convention is stronger than paper Eq5.2 alone.
    """
    d = _binary_flags(dones)
    if d.ndim != 2:
        raise ValueError("Expected dones[B,G]")
    q = next_q.detach().mean(dim=1) if next_q.ndim == 3 else next_q.detach()
    if q.shape != d.shape or not torch.isfinite(q).all():
        raise ValueError("Expected finite next_q[B,G] or [B,M,G]")
    terminal = torch.as_tensor(terminated, device=d.device)
    if terminal.shape != (d.shape[0],) or not ((terminal == 0) | (terminal == 1)).all():
        raise ValueError("terminated must be binary [B]")
    gamma = torch.as_tensor(discount, device=d.device, dtype=d.dtype)
    if (gamma.ndim > 1 or (gamma.ndim == 1 and gamma.shape != (d.shape[1],))
            or not torch.isfinite(gamma).all() or ((gamma < 0) | (gamma > 1)).any()):
        raise ValueError("discount must be scalar or [G] in [0,1]")
    horizon = torch.as_tensor(executed_steps, device=d.device, dtype=d.dtype)
    if horizon.ndim == 0:
        horizon = horizon.expand(d.shape[0])
    if horizon.shape != (d.shape[0],) or (horizon < 1).any() or (horizon != horizon.round()).any():
        raise ValueError("executed_steps must be positive integer scalar or [B]")
    if not math.isfinite(r_done) or not math.isfinite(r_step):
        raise ValueError("r_done and r_step must be finite")
    if not math.isfinite(cap) or cap < r_done:
        raise ValueError("cap must be finite and at least r_done")
    rewards = torch.where(d.bool(), r_done, r_step)
    bootstrap = d * cap + (1.0 - d) * q
    target = rewards + gamma.pow(horizon[:, None]) * (~terminal.bool())[:, None] * bootstrap
    target = target.clamp_max(cap)
    return torch.where(terminal.bool()[:, None] & d.bool(), cap, target).detach()


def vector_critic_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """Sum over subgoal heads, mean over replay batch and ensemble members."""
    if predicted.ndim != 3 or target.shape != (predicted.shape[0], predicted.shape[-1]):
        raise ValueError("Expected predicted[B,M,G], target[B,G]")
    return (predicted - target.detach()[:, None, :]).square().sum(dim=-1).mean()
