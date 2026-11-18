"""CPU mathematical contracts, independent of GPU weights or rollout success.

Closed-form densities/derivatives and hand-computed targets test the estimator;
the tiny SmolVLA stand-in checks wrapper conventions, not pretrained integration.
"""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from chemlab_rl.smolvla_ogpo import (
    ChainConfig, ChainSample, SmolVLAVelocity, chain_log_prob,
    clipped_chain_objective, conservative_group_advantages, full_chain_ratio,
    next_future_weights, sample_chain, transition_parameters,
    vector_critic_loss, vector_td_target,
)


class ConstantVelocity(nn.Module):
    def __init__(self, value=0.2, dtype=torch.float64):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=dtype))

    def forward(self, batch, x, tau):
        return self.value * torch.ones_like(x)


def manual_log_density(latents, value, config):
    """Independent distribution evaluation, including every H,D coordinate."""
    result = torch.distributions.Normal(0.0, 1.0).log_prob(latents[:, 0]).sum((-2, -1))
    for k in range(config.steps):
        x = latents[:, k]
        tau = k / config.steps
        drift = value + (0.5 * config.diffusion_scale**2 * (tau * value - x)
                         if config.score_correction else 0.0)
        mean = x + drift / config.steps
        std = config.diffusion_scale * ((1 - tau) / config.steps) ** 0.5
        result += torch.distributions.Normal(mean, std).log_prob(latents[:, k + 1]).sum((-2, -1))
    return result


def test_sde_drift_and_brownian_variance_have_correct_time_scaling():
    x = torch.full((2, 3, 4), 0.7, dtype=torch.float64)
    v = torch.full_like(x, -0.4)
    config = ChainConfig(steps=4, diffusion_scale=0.4)
    mean, variance = transition_parameters(x, v, torch.tensor([0.25, 0.75]), config)
    expected_mean = torch.tensor([0.584, 0.58], dtype=torch.float64)[:, None, None].expand_as(x)
    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(variance[:, 0, 0], torch.tensor([0.03, 0.01], dtype=torch.float64))
    assert variance[-1].item() > 0  # Last transition remains a nonsingular Gaussian.
    with pytest.raises(ValueError, match="start times"):
        transition_parameters(x, v, 1.0, config)


@pytest.mark.parametrize("steps,noise", [(0, 0.1), (True, 0.1), (2, 0), (2, float("nan"))])
def test_degenerate_chain_configuration_is_rejected(steps, noise):
    with pytest.raises(ValueError):
        ChainConfig(steps=steps, diffusion_scale=noise)


def test_complete_chain_density_matches_independent_normal_distributions():
    config = ChainConfig(steps=3, diffusion_scale=0.4)
    latents = torch.linspace(-0.7, 1.1, 2 * 4 * 2 * 3, dtype=torch.float64).reshape(2, 4, 2, 3)
    actor = ConstantVelocity()
    expected = manual_log_density(latents, actor.value.detach(), config)
    actual = chain_log_prob(actor, {}, latents, config)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def test_sampled_density_recomputes_and_stored_target_chain_is_fixed():
    actor = ConstantVelocity(dtype=torch.float32)
    config = ChainConfig(steps=4, diffusion_scale=0.3)
    sample = sample_chain(actor, {}, (4, 3, 5), config,
                          generator=torch.Generator().manual_seed(40))
    assert not sample.latents.requires_grad and not sample.log_prob.requires_grad
    stored = sample.latents.clone(), sample.log_prob.clone()
    actual = chain_log_prob(actor, {}, sample.latents, config)
    torch.testing.assert_close(actual, sample.log_prob, rtol=0, atol=0)
    torch.testing.assert_close(full_chain_ratio(actual, sample.log_prob), torch.ones(4, dtype=torch.float64))
    with torch.no_grad():
        actor.value.add_(0.01)
    changed = chain_log_prob(actor, {}, sample.latents, config)
    assert not torch.equal(changed, sample.log_prob)
    torch.testing.assert_close(sample.latents, stored[0])
    torch.testing.assert_close(sample.log_prob, stored[1])
    # A target update belongs AFTER a complete optimization group. There is no
    # EMA updater implemented in this module; this verifies fixed-draw semantics.


def test_float32_mlp_chain_scores_exactly_with_sampler_memory_layout():
    class MLPVelocity(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(32, 256), nn.SiLU(), nn.Linear(256, 32))
        def forward(self, batch, x, tau):
            return self.net(x)
    # A scalar velocity cannot expose fused-Linear rounding differences between
    # contiguous sampling inputs and a strided path[:, k] scoring input.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        actor = MLPVelocity().eval()
        config = ChainConfig(steps=2, diffusion_scale=.05)
        sample = sample_chain(actor, {}, (2, 50, 32), config)
        assert not sample.latents[:, 0].is_contiguous()
        score = chain_log_prob(actor, {}, sample.latents, config)
        with torch.no_grad():
            frozen_score = chain_log_prob(actor, {}, sample.latents, config)
        torch.testing.assert_close(score.detach(), sample.log_prob, rtol=0, atol=0)
        torch.testing.assert_close(frozen_score, sample.log_prob, rtol=0, atol=0)
        score.sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())


def test_auxiliary_latent_coordinates_still_contribute_to_density():
    config = ChainConfig(steps=2, diffusion_scale=0.5)
    actor = ConstantVelocity(0.0)
    latents = torch.zeros((1, 3, 4, 6), dtype=torch.float64)
    changed = latents.clone()
    changed[:, -1, -1, -1] = 0.3
    first = ChainSample(latents, chain_log_prob(actor, {}, latents, config))
    second = ChainSample(changed, chain_log_prob(actor, {}, changed, config))
    torch.testing.assert_close(first.executed_actions(2, 1), second.executed_actions(2, 1))
    assert not torch.equal(first.log_prob, second.log_prob)
    # One changed final coordinate has variance c^2/K^2 = .0625.
    torch.testing.assert_close(second.log_prob - first.log_prob, torch.tensor([-0.72], dtype=torch.float64))


def test_score_gradient_uses_fixed_chain_not_backprop_through_sampler():
    config = ChainConfig(steps=3, diffusion_scale=0.4, score_correction=False)
    actor = ConstantVelocity(0.1)
    latent = torch.linspace(-0.3, 0.8, 4 * 2 * 3, dtype=torch.float64).reshape(1, 4, 2, 3).requires_grad_()
    log_prob = chain_log_prob(actor, {}, latent, config)
    expected_grad = 0.0
    for k in range(config.steps):
        mean = latent[:, k].detach() + config.dt * actor.value.detach()
        variance = config.diffusion_scale**2 * (1 - k / config.steps) * config.dt
        expected_grad += ((latent[:, k + 1].detach() - mean) / variance * config.dt).sum()
    log_prob.sum().backward()
    assert latent.grad is None
    torch.testing.assert_close(actor.value.grad, expected_grad)


def test_ppo_clipping_signs_and_stop_gradients():
    ratios = torch.tensor([1.5, 0.5, 1.5, 0.5], dtype=torch.float64)
    new = ratios.log().requires_grad_()
    old = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    advantage = torch.tensor([2.0, 2.0, -2.0, -2.0], dtype=torch.float64, requires_grad=True)
    loss = clipped_chain_objective(new, old, advantage, clip_epsilon=0.1)
    torch.testing.assert_close(loss, torch.tensor(0.4, dtype=torch.float64))
    loss.backward()
    torch.testing.assert_close(new.grad, torch.tensor([0.0, -0.25, 0.75, 0.0], dtype=torch.float64))
    assert old.grad is None and advantage.grad is None


def test_full_chain_ratio_is_not_averaged_or_silently_clamped():
    ratio = full_chain_ratio(torch.tensor([4.0], dtype=torch.float64), torch.tensor([1.0], dtype=torch.float64))
    torch.testing.assert_close(ratio, torch.tensor([3.0], dtype=torch.float64).exp())
    with pytest.raises(FloatingPointError, match="reduce the actor update"):
        full_chain_ratio(torch.tensor([1000.0]), torch.tensor([0.0]))


def test_next_future_excludes_current_goal_with_last_goal_fallback():
    done = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
    expected = torch.tensor([[0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=torch.float32)
    torch.testing.assert_close(next_future_weights(done), expected)
    torch.testing.assert_close(next_future_weights(torch.tensor([[0], [1]])), torch.ones(2, 1))
    with pytest.raises(ValueError, match="completed prefix"):
        next_future_weights(torch.tensor([[0, 1, 0]]))


def test_advantages_scalarize_before_conservative_ensemble_reduction():
    scalars = torch.tensor([[4, 2], [2, -1], [-3, -4], [-3, 3]], dtype=torch.float32)
    q = torch.zeros(1, 4, 2, 3)
    q[0, :, :, 0] = torch.tensor([1000, -9000, 6000, -1000])[:, None]  # Excluded current head.
    q[0, :, :, 1] = torch.tensor([[10, -10], [-10, 10], [3, -3], [-3, 3]])
    q[0, :, :, 2] = scalars - q[0, :, :, 1]
    q.requires_grad_()
    advantage = conservative_group_advantages(q, torch.zeros(1, 3))
    torch.testing.assert_close(advantage, torch.tensor([[2.0, 0.0, -3.0, 0.0]]))
    assert not advantage.requires_grad
    # No standard-deviation normalization: scaling critic values scales A.
    torch.testing.assert_close(conservative_group_advantages(q * 3, torch.zeros(1, 3)), advantage * 3)


def test_td_targets_use_true_terminal_actual_horizon_and_all_heads():
    dones = torch.tensor([[1, 0, 0], [1, 1, 0]], dtype=torch.float32)
    next_q = torch.tensor([[[1, 2, 3], [3, 4, 5]], [[1, 2, 3], [3, 4, 5]]], dtype=torch.float32, requires_grad=True)
    target = vector_td_target(dones, next_q, torch.tensor([True, False]),
                              torch.tensor([0.9, 0.8, 0.7]), executed_steps=torch.tensor([1, 2]))
    torch.testing.assert_close(target, torch.tensor([[15, -1, -1], [15, 14.6, 0.96]]))
    assert not target.requires_grad
    # Time-limit truncation must pass terminated=False and retain bootstrap.
    truncated_target = vector_td_target(dones[:1], next_q[:1], torch.tensor([False]), 0.9)
    torch.testing.assert_close(truncated_target, torch.tensor([[15, 1.7, 2.6]]))


@pytest.mark.parametrize("field,value", [("discount", float("nan")), ("discount", float("inf")),
                                         ("r_done", float("nan")), ("r_step", float("inf"))])
def test_nonfinite_td_configuration_is_rejected(field, value):
    kwargs = {"discount": 0.9, field: value}
    with pytest.raises(ValueError):
        vector_td_target(torch.zeros(1, 3), torch.zeros(1, 3), torch.tensor([False]), **kwargs)


def test_critic_loss_supervises_every_head_and_detaches_target():
    predicted = torch.tensor([[[1.0, 2, 3], [2, 4, 6]]], requires_grad=True)
    target = torch.zeros(1, 3, requires_grad=True)
    loss = vector_critic_loss(predicted, target)
    torch.testing.assert_close(loss, torch.tensor(35.0))
    loss.backward()
    torch.testing.assert_close(predicted.grad, predicted.detach())
    assert target.grad is None


class PassThroughVLM(nn.Module):
    def forward(self, *, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache):
        assert past_key_values is None and not use_cache
        assert attention_mask.shape[-1] == position_ids.shape[-1] == 3
        return inputs_embeds, None


class TinySmolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm_with_expert = PassThroughVLM()
        self.action_out_proj = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.action_out_proj.weight.copy_(torch.eye(3))
        self.dropout = nn.Dropout(0.9)
        self.native_time = None

    def embed_prefix(self, images, image_masks, tokens, attention, *, state):
        b = len(state)
        return torch.zeros(b, 1, 3), torch.ones(b, 1, dtype=torch.bool), torch.zeros(b, 1, dtype=torch.bool)

    def embed_suffix(self, latent, native_time):
        self.native_time = native_time.clone()
        b, h, _ = latent.shape
        return self.dropout(latent + native_time[:, None, None]), torch.ones(b, h, dtype=torch.bool), torch.ones(b, h, dtype=torch.bool)


class TinySmolPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(chunk_size=2, max_action_dim=3, adapt_to_pi_aloha=False)
        self.model = TinySmolModel()

    def prepare_images(self, batch):
        return [], []

    def prepare_state(self, batch):
        return batch["state"]


def test_smol_wrapper_reverses_time_and_velocity_with_gradients_and_no_dropout():
    policy = TinySmolPolicy()
    wrapper = SmolVLAVelocity(policy, mask_builder=lambda pad, att: pad[:, :, None] & pad[:, None, :])
    wrapper.train()
    assert wrapper.training and not policy.training and not policy.model.dropout.training
    batch = {"state": torch.zeros(2, 14), "observation.language.tokens": torch.ones(2, 1),
             "observation.language.attention_mask": torch.ones(2, 1)}
    latent = torch.full((2, 2, 3), 0.2)
    tau = torch.tensor([0.25, 0.75])
    output = wrapper(batch, latent, tau)
    torch.testing.assert_close(policy.model.native_time, torch.tensor([0.75, 0.25]))
    torch.testing.assert_close(output, -(latent + (1 - tau)[:, None, None]))
    output.sum().backward()
    assert policy.model.action_out_proj.weight.grad is not None
    assert policy.model.action_out_proj.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="Latent shape"):
        wrapper(batch, torch.zeros(2, 2, 2), tau)
