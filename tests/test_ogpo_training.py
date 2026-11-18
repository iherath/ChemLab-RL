"""CPU integration contracts; these do not establish real-VLA or robot success."""
from dataclasses import replace
from types import SimpleNamespace
import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

import chemlab_rl.ogpo_training as training
from chemlab_rl.ogpo_training import (
    FrozenSmolVLAFeatures, OGPOConfig, OGPOTrainer, PersistentReplay,
    PhysicalActionAdapter, Transition, VectorQEnsemble, actor_observation,
    collect_chunk, ema_update, prepare_policy_batch, run_online,
)
from chemlab_rl.smolvla_ogpo import ChainConfig
from chemlab_rl.vla_bc import CAMERAS, STATE_KEY


PROVENANCE = "test-only-BC-fingerprint-and-physical-material-bank-v1"


def observation(value=0, state_dim=15):
    raw = {"proprio": np.full(14, value, np.float32),
           "scale_reading_g": np.array([value + .25], np.float32),
           "lowdim": np.full(20, 999), "mass_g": 999, "powder_id": 5,
           "subgoal_dones": np.ones(3), "density_kg_m3": 999}
    raw.update({key: np.full((4, 4, 3), value % 255, np.uint8) for key in CAMERAS})
    return actor_observation(raw, f"Weigh {value + 1} grams", state_dim)


def transition(value=0, **kwargs):
    base = Transition(observation(value), observation(value+1), np.full((2, 14), .1, np.float32),
                      np.zeros(3, np.float32), np.zeros(3, np.float32), False, False, 2, 0)
    return replace(base, **kwargs)


def preprocessor(batch):
    result = dict(batch)
    size = len(result[STATE_KEY])
    result["observation.language.tokens"] = torch.ones(size, 3, dtype=torch.long)
    result["observation.language.attention_mask"] = torch.ones(size, 3, dtype=torch.bool)
    result["hidden_material"] = torch.ones(size, 6)  # Must be filtered after preprocessing too.
    return result


class TinyVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(32))

    def forward(self, batch, latent, tau):
        assert set(batch) == set(training.POLICY_KEYS)
        return self.bias[None, None] - .2 * latent + .01 * batch[STATE_KEY][:, :1, None]


def features(batch):
    return batch[STATE_KEY][:, :2].float()


def make_trainer(**kwargs):
    config = OGPOConfig(batch_size=2, group_size=3, critic_warmup_updates=0,
                        actor_lr=1e-7, critic_lr=1e-4,
                        chain=ChainConfig(steps=2, diffusion_scale=.2))
    config = replace(config, **kwargs)
    critic = VectorQEnsemble(2, hidden=12)
    # Identical members make this test's group consensus deterministic. Actual
    # production critics initialize independently and can disagree.
    critic.heads[1].load_state_dict(critic.heads[0].state_dict())
    return OGPOTrainer(TinyVelocity(), critic, preprocessor=preprocessor, feature_fn=features,
                       action_adapter=PhysicalActionAdapter(lambda x: x * 2, [-1]*14, [1]*14),
                       config=config, provenance=PROVENANCE, seed=123)


def make_replay(path, **kwargs):
    return PersistentReplay(path, provenance=PROVENANCE, **kwargs)


def test_actor_allowlist_contains_only_measured_sensors_images_and_language():
    obs = observation(2)
    assert set(obs) == {"proprio", "scale_reading_g", "task", *CAMERAS}
    result = prepare_policy_batch([obs], preprocessor, "cpu")
    assert set(result) == set(training.POLICY_KEYS)
    assert result[STATE_KEY].shape == (1, 15)
    assert result[STATE_KEY][0, -1] == 2.25
    images_only_scale = observation(2, state_dim=14)
    assert "scale_reading_g" not in images_only_scale
    assert prepare_policy_batch([images_only_scale], preprocessor, "cpu")[STATE_KEY].shape == (1, 14)


def test_replay_survives_reopen_evicts_oldest_and_snapshots_inputs(tmp_path):
    path = tmp_path / "replay.sqlite"
    replay = make_replay(path, capacity=3)
    for value in range(4):
        row = transition(value)
        replay.add(row)
        row.observation["proprio"][:] = 999  # Stored payload must remain immutable.
    replay.close()
    reopened = make_replay(path, capacity=3)
    assert len(reopened) == 3
    rows = reopened.sample(3)
    assert sorted(r.observation["proprio"][0] for r in rows) == [1, 2, 3]
    assert sorted(r.next_observation["proprio"][0] for r in rows) == [2, 3, 4]
    assert all(r.observation["scale_reading_g"].shape == (1,) for r in rows)
    reopened.close()
    with pytest.raises(ValueError, match="contract"):
        make_replay(path, capacity=4)


@pytest.mark.parametrize("changes", [
    {"powder_id": 5}, {"powder_id": -1}, {"executed_steps": 1},
    {"dones": np.array([0, 1, 0])},
    {"dones": np.array([1, 0, 0]), "next_dones": np.zeros(3)},
])
def test_replay_rejects_split_leakage_and_invalid_chunk_semantics(tmp_path, changes):
    replay = make_replay(tmp_path / "replay.sqlite")
    with pytest.raises(ValueError):
        replay.add(transition(**changes))
    assert len(replay) == 0
    replay.close()


def test_ema_updates_after_online_change_and_freezes_float_and_integer_state():
    online = nn.Linear(2, 1)
    online.register_buffer("counter", torch.tensor(1))
    target = copy.deepcopy(online)
    old = {k: v.clone() for k, v in target.state_dict().items()}
    with torch.no_grad():
        online.weight.add_(4)
        online.bias.sub_(8)
        online.counter.fill_(3)
    ema_update(target, online, .25)
    torch.testing.assert_close(target.weight, old["weight"] + 1)
    torch.testing.assert_close(target.bias, old["bias"] - 2)
    assert target.counter == 3 and not target.training
    assert all(not p.requires_grad for p in target.parameters())


def test_action_mapping_clips_only_physical_projection_and_uses_robot_ranges():
    # Deployment processors may return CPU/different-dtype tensors. The
    # candidate-action interface must match actor/critic input placement.
    adapter = PhysicalActionAdapter(lambda x: (x * 2).to("cpu", dtype=torch.float64), [-1, 0], [1, 255])
    latent_actions = torch.tensor([[[5., -2.], [.25, 10.]]])
    original = latent_actions.clone()
    physical = adapter.physical(latent_actions)
    assert physical.dtype == latent_actions.dtype and physical.device == latent_actions.device
    torch.testing.assert_close(physical, torch.tensor([[[1., 0.], [.5, 20.]]]))
    encoded = adapter.critic(physical)
    torch.testing.assert_close(encoded[0, 0], torch.tensor([1., -1.]))
    torch.testing.assert_close(latent_actions, original)


def test_vector_td_integration_bootstraps_time_limit_final_observation(tmp_path, monkeypatch):
    trainer = make_trainer(critic_warmup_updates=10, discount=.9)
    for critic in (trainer.critic, trainer.target_critic):
        with torch.no_grad():
            for parameter in critic.parameters():
                parameter.zero_()
            for head in critic.heads:
                head[-1].bias.fill_(2.)
    replay = make_replay(tmp_path / "replay.sqlite")
    replay.add(transition(1, terminated=True, executed_steps=1, next_dones=np.array([1, 0, 0])))
    replay.add(transition(2, truncated=True, executed_steps=1, next_dones=np.array([1, 0, 0])))
    captured = {}
    original = training.vector_td_target
    def capture(dones, next_q, terminated, discount, horizons, **kwargs):
        result = original(dones, next_q, terminated, discount, horizons, **kwargs)
        captured.update(target=result.clone(), terminal=terminated.clone(), horizons=horizons.clone())
        return result
    monkeypatch.setattr(training, "vector_td_target", capture)
    old_actor = trainer.target_actor.bias.clone()
    metrics = trainer.update(replay)
    assert metrics["actual_horizon_mean"] == 1 and metrics["truncated_fraction"] == .5
    assert metrics["actor_updates"] == 0
    for target, terminal in zip(captured["target"], captured["terminal"]):
        torch.testing.assert_close(target, torch.tensor([15., -1., -1.] if terminal else [15., .8, .8]))
    torch.testing.assert_close(trainer.target_actor.bias, old_actor)
    replay.close()


def test_opt_in_control_cost_uses_actual_discounted_horizon(tmp_path, monkeypatch):
    trainer = make_trainer(critic_warmup_updates=10, discount=.5,
                           step_reward_per_control=-.01)
    replay = make_replay(tmp_path / "replay.sqlite")
    replay.add(transition(0, terminated=True, executed_steps=1))
    replay.add(transition(1, terminated=True, executed_steps=2))
    captured = []
    original = training.vector_td_target
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result.clone())
        return result
    monkeypatch.setattr(training, "vector_td_target", capture)
    trainer.update(replay)
    assert sorted(float(row[0, 0]) for row in captured) == pytest.approx([-.015, -.01])
    replay.close()


def test_full_chain_actor_update_uses_fresh_target_draws_and_ema(tmp_path, monkeypatch):
    trainer = make_trainer(actor_ema=.25, critic_ema=.5)
    replay = make_replay(tmp_path / "replay.sqlite")
    replay.add(transition(0))
    replay.add(transition(1))
    old_actor = trainer.actor.bias.detach().clone()
    old_target_critic = {k: v.clone() for k, v in trainer.target_critic.state_dict().items()}
    original_sample = training.sample_chain
    draws = []
    def record(actor, batch, shape, config, **kwargs):
        result = original_sample(actor, batch, shape, config, **kwargs)
        draws.append((actor, shape, result.latents.clone(), result.log_prob.clone()))
        return result
    monkeypatch.setattr(training, "sample_chain", record)
    metrics = trainer.update(replay)
    # Two TD next actions plus six fresh candidates, microbatched by two;
    # behavior action chunks are never substituted for target chains.
    assert len(draws) == 4
    assert all(actor is trainer.target_actor for actor, *_ in draws)
    assert all(shape == (2, 50, 32) for _, shape, *_ in draws)
    assert metrics["chain_shape"] == [6, 3, 50, 32]
    assert metrics["ratio_min"] == metrics["ratio_max"] == 1
    assert not torch.equal(trainer.actor.bias.detach(), old_actor)
    torch.testing.assert_close(trainer.target_actor.bias, .75*old_actor + .25*trainer.actor.bias)
    for key, value in trainer.target_critic.state_dict().items():
        torch.testing.assert_close(value, .5*old_target_critic[key] + .5*trainer.critic.state_dict()[key])
    assert all(p.grad is None and not p.requires_grad for p in trainer.target_actor.parameters())
    assert all(p.grad is None and not p.requires_grad for p in trainer.target_critic.parameters())
    replay.close()


def test_actor_microbatch_accumulation_matches_joint_candidate_objective(tmp_path, monkeypatch):
    large = make_trainer(actor_microbatch_size=6)
    small = make_trainer(actor_microbatch_size=1)
    small.critic.load_state_dict(large.critic.state_dict())
    small.target_critic.load_state_dict(large.target_critic.state_dict())
    replay = make_replay(tmp_path / "replay.sqlite")
    replay.add(transition(0))
    replay.add(transition(1))
    replay_state = replay.checkpoint_state()
    rows = replay.sample(2)
    batch = large._inputs([r.observation for r in rows])
    next_batch = large._inputs([r.next_observation for r in rows])
    fixed_next = large._sample(next_batch)
    fixed_group = large._sample({k:v.repeat_interleave(3, dim=0) for k,v in batch.items()})
    for trainer in (large, small):
        draws = iter([fixed_next, fixed_group])
        monkeypatch.setattr(trainer, "_sample", lambda batch, actor=None, draws=draws: next(draws))
        replay.restore_sampling(replay_state)
        trainer.update(replay)
        assert trainer.actor_opt.state[trainer.actor.bias]["step"] == 1
        assert trainer.actor_updates == 1
    torch.testing.assert_close(small.actor.bias.grad, large.actor.bias.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(small.actor.bias, large.actor.bias, rtol=1e-5, atol=1e-12)
    torch.testing.assert_close(small.target_actor.bias, large.target_actor.bias, rtol=1e-5, atol=1e-12)
    replay.close()


def test_checkpoint_restores_exact_next_update_optimizer_and_sampling(tmp_path):
    torch.manual_seed(25)
    replay = make_replay(tmp_path / "replay.sqlite")
    for i in range(4):
        replay.add(transition(i))
    trainer = make_trainer()
    trainer.update(replay)
    checkpoint = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(checkpoint, replay)
    expected_metrics = trainer.update(replay)
    expected_actor = {k: v.clone() for k, v in trainer.actor.state_dict().items()}
    expected_critic = {k: v.clone() for k, v in trainer.critic.state_dict().items()}
    restored = make_trainer()
    restored.load_checkpoint(checkpoint, replay)
    actual_metrics = restored.update(replay)
    assert actual_metrics == expected_metrics
    for key, value in restored.actor.state_dict().items():
        torch.testing.assert_close(value, expected_actor[key], rtol=0, atol=0)
    for key, value in restored.critic.state_dict().items():
        torch.testing.assert_close(value, expected_critic[key], rtol=0, atol=0)
    replay.add(transition(8))
    with pytest.raises(ValueError, match="revision"):
        restored.load_checkpoint(checkpoint, replay)
    replay.close()


def test_optional_bc_anchor_is_frozen_and_disabled_by_default(tmp_path):
    assert make_trainer().bc_anchor is None
    trainer = make_trainer(bc_anchor_weight=.01)
    replay = make_replay(tmp_path / "replay.sqlite")
    replay.add(transition(0))
    replay.add(transition(1))
    snapshot = trainer.bc_anchor.bias.clone()
    metrics = trainer.update(replay)
    assert metrics["bc_anchor_loss"] == pytest.approx(0, abs=1e-10)
    torch.testing.assert_close(trainer.bc_anchor.bias, snapshot)
    assert trainer.bc_anchor.bias.grad is None and not trainer.bc_anchor.bias.requires_grad
    replay.close()


class TinyEnvironment:
    instruction = "Weigh one gram"
    def __init__(self, terminal_after=None, autoreset=False):
        self.terminal_after, self.autoreset = terminal_after, autoreset
        self.reset_count = 0
        self.commands = []

    def info(self):
        return {"powder_dynamics": "physical_grains_v1", "powder_id": 0,
                "subgoal_dones": np.array([1, 0, 0]) if self.steps else np.zeros(3)}

    def reset(self, seed=None):
        self.reset_count += 1
        self.steps = 0
        return observation(0), self.info()

    def step(self, command):
        assert command.shape == (14,)
        self.commands.append(command.copy())
        self.steps += 1
        truncated = self.steps == self.terminal_after
        info = self.info()
        obs = observation(self.steps)
        if truncated and self.autoreset:
            info = {"final_observation": obs, "final_info": info}
            obs = observation(99)  # Must not enter a bootstrapped next state.
        return obs, -1., False, truncated, info


def test_chunk_collection_keeps_real_final_frame_and_actual_horizon(tmp_path):
    env = TinyEnvironment(terminal_after=1, autoreset=True)
    trainer = make_trainer()
    replay = make_replay(tmp_path / "replay.sqlite")
    obs, info = env.reset()
    final, _, result = collect_chunk(env, obs, info, trainer, replay)
    row = replay.sample(1)[0]
    assert result["executed_steps"] == 1 and result["truncated"] and not result["terminated"]
    assert row.next_observation["proprio"][0] == final["proprio"][0] == 1
    assert row.observation["proprio"][0] == 0
    assert trainer.environment_steps == 1 and len(env.commands) == 1 and env.reset_count == 1
    assert np.all(np.abs(row.actions[0]) <= 1) and np.all(row.actions[1] == 0)
    replay.close()


def test_bounded_runner_stops_at_exact_step_budget_and_saves(tmp_path):
    env = TinyEnvironment()
    trainer = make_trainer(critic_warmup_updates=1)
    replay = make_replay(tmp_path / "replay.sqlite")
    checkpoint = tmp_path / "online.pt"
    result = run_online(env, trainer, replay, max_environment_steps=3, max_updates=2,
                        max_seconds=60, checkpoint_path=checkpoint)
    assert result["environment_steps"] == len(env.commands) == 3
    assert result["updates"] == 1 and checkpoint.exists()
    assert result["closed_loop_evaluated"] is False
    assert env.reset_count == 1
    rows = replay.sample(2)
    assert sorted(r.executed_steps for r in rows) == [1, 2]
    assert sum(r.truncated for r in rows) == 1
    replay.close()


def test_frozen_vlm_feature_pooling_masks_padding_and_keeps_sensor_vector():
    class PrefixModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.state_proj = nn.Linear(15, 2)
            with torch.no_grad():
                self.state_proj.weight.zero_()
                self.state_proj.bias.copy_(torch.tensor([2., 4.]))
            self.vlm_with_expert = SimpleNamespace(
                vlm=nn.Linear(2, 2).requires_grad_(False),
                config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=2)),
                forward=self.context)

        def embed_prefix(self, images, image_masks, tokens, masks, state):
            prefix = torch.cat([self.state_proj(state)[:, None], torch.tensor([[[4., 8.], [100., 100.]]])], dim=1)
            return prefix, torch.tensor([[1, 1, 0]]).bool(), torch.tensor([[1, 0, 0]]).bool()

        def context(self, **kwargs):
            # The actual cross-attention implementation needs the fresh-prefill
            # path when suffix=None; an uncached cross layer tries to use suffix.
            assert kwargs["use_cache"] is True and kwargs["past_key_values"] is None
            return (kwargs["inputs_embeds"][0], None), object()

    class PrefixPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = PrefixModel()

        def prepare_images(self, batch):
            return [], []

        def prepare_state(self, batch):
            return batch[STATE_KEY]

    policy = PrefixPolicy()
    encoder = FrozenSmolVLAFeatures(policy, mask_builder=lambda p, a: p)
    assert encoder.prefix_model.state_proj is not policy.model.state_proj
    assert encoder.prefix_model.vlm_with_expert is policy.model.vlm_with_expert
    assert all(p.requires_grad for p in policy.model.state_proj.parameters())
    assert all(not p.requires_grad for p in encoder.prefix_model.state_proj.parameters())
    batch = prepare_policy_batch([observation(1)], preprocessor, "cpu")
    result = encoder(batch)
    assert result.shape == (1, 17) and encoder.feature_dim == 17
    torch.testing.assert_close(result[:, :2], torch.tensor([[3., 6.]]))
    torch.testing.assert_close(result[:, 2:], batch[STATE_KEY])
    assert not result.requires_grad
    saved_features = copy.deepcopy(encoder.state_dict())
    with torch.no_grad():
        policy.model.state_proj.bias.add_(99)
    torch.testing.assert_close(encoder(batch), result)  # Actor learning cannot drift critic features.
    rebuilt = FrozenSmolVLAFeatures(policy, mask_builder=lambda p, a: p)
    assert not torch.equal(rebuilt(batch), result)
    rebuilt.load_state_dict(saved_features)
    torch.testing.assert_close(rebuilt(batch), result)
    policy.model.vlm_with_expert.vlm.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        FrozenSmolVLAFeatures(policy)


def test_checkpoint_restores_private_frozen_feature_state(tmp_path):
    class FrozenFeatures(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("gain", torch.tensor(2.))
        def forward(self, batch):
            return features(batch) * self.gain
    replay = make_replay(tmp_path / "replay.sqlite")
    trainer = make_trainer()
    trainer.feature_fn = FrozenFeatures()
    path = tmp_path / "state.pt"
    trainer.save_checkpoint(path, replay)
    trainer.feature_fn.gain.fill_(99)
    trainer.load_checkpoint(path, replay)
    assert trainer.feature_fn.gain == 2
    replay.close()


def test_cli_rejects_legacy_diagnostic_checkpoint_before_loading_model(tmp_path, monkeypatch):
    from scripts.train_ogpo_vadar import main
    from chemlab_rl import vla_bc
    checkpoint = tmp_path / "legacy"
    checkpoint.mkdir()
    channel = {"mean": [0.]*14, "std": [1.]*14, "min": [-1.]*14, "max": [1.]*14}
    manifest = {"manifest_version": 1, "legacy": True, "schema_version": "legacy",
                "statistics": {STATE_KEY: channel, "action": channel}}
    (checkpoint / "data_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(vla_bc, "load_smolvla", lambda *a, **k: pytest.fail("Loaded a diagnostic model for online training"))
    with pytest.raises(ValueError, match="corrected 15-D"):
        main(["--checkpoint", str(checkpoint), "--output", str(tmp_path / "out"), "--device", "cpu"])


def test_smoke_density_diagnostic_quantifies_mean_and_ratio_drift():
    from scripts.train_ogpo_vadar import density_diagnostics
    from chemlab_rl.smolvla_ogpo import ChainSample
    latents = torch.zeros(1, 2, 1, 1, dtype=torch.float64)
    old_mean, new_mean = torch.zeros(1, 1, 1, dtype=torch.float64), torch.full((1, 1, 1), .001, dtype=torch.float64)
    variance = torch.full((1, 1, 1), .01, dtype=torch.float64)
    # x=0, sigma=.1, mu shift=.001 -> log ratio = -.5*(.001/.1)^2.
    sample = ChainSample(latents, torch.zeros(1, dtype=torch.float64))
    diagnostics = density_diagnostics(sample, torch.tensor([-.00005], dtype=torch.float64),
                                      [(old_mean, variance)], [(new_mean, variance)])
    assert diagnostics["steps"][0]["mean_error_in_std_max"] == pytest.approx(.01)
    assert diagnostics["steps"][0]["implied_log_ratio_per_sample"] == pytest.approx([-.00005])
    assert diagnostics["max_implied_ratio_deviation"] == pytest.approx(abs(np.expm1(-.00005)))
