"""Bounded SmolVLA comparisons on the native rigid-body vial task.

All methods report the same native final-success reward. ``bon`` keeps the BC
actor frozen and learns a scalar chunk critic, including greedy BoN Bellman
targets. ``ogpo`` optimizes that scalar return through fresh augmented chains.
``vadar`` uses the reference auxiliary subgoal targets (+5/-1, cap 15) and
next-future weights, not a sum of subgoal bonuses as the environment reward.
This is a task-specific experimental PyTorch port, not an upstream reproduction.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import time
from typing import Mapping

import numpy as np
import torch

from chemlab_rl.ogpo_training import (
    OGPOConfig, OGPOTrainer, PersistentReplay, actor_observation,
    ema_update, prepare_policy_batch,
)
from chemlab_rl.smolvla_ogpo import (
    chain_log_prob, clipped_chain_objective, conservative_group_advantages,
    full_chain_ratio, next_future_weights, vector_critic_loss, vector_td_target,
)
from chemlab_rl.vla_bc import CAMERAS, STATE_KEY, file_sha256

TASK_ID = "vial_placement_native_v1"


@dataclass(frozen=True)
class VialTransition:
    observation: Mapping
    next_observation: Mapping
    actions: np.ndarray
    dones: np.ndarray
    next_dones: np.ndarray
    rewards: np.ndarray  # Per-control native success rewards; unused suffix zero.
    terminated: bool
    truncated: bool
    executed_steps: int


class VialReplay(PersistentReplay):
    """Reuse the tested SQLite transport with a vial-specific payload contract."""

    def __init__(self, path, *, provenance, capacity=30_000, seed=0):
        super().__init__(path, capacity=capacity, action_steps=2, action_dim=14,
                         state_dim=15, subgoals=4, heldout_ids=(),
                         provenance=TASK_ID + ":" + provenance, seed=seed)

    def _encode_transition(self, t):
        h = t.executed_steps
        if (isinstance(h, bool) or not isinstance(h, (int, np.integer)) or not 1 <= h <= 2
                or not isinstance(t.terminated, (bool, np.bool_))
                or not isinstance(t.truncated, (bool, np.bool_))
                or (h < 2 and not (t.terminated or t.truncated))):
            raise ValueError("Invalid vial terminal flags or executed horizon")
        arrays = {}
        for label, obs in (("obs", t.observation), ("next", t.next_observation)):
            clean = actor_observation(obs, obs["task"], 15)
            arrays.update({f"{label}/{key}": value for key, value in clean.items()})
        a, r = np.asarray(t.actions, np.float32), np.asarray(t.rewards, np.float32)
        if a.shape != (2, 14) or not np.isfinite(a).all() or np.any(np.abs(a[:h]) > 1.000001):
            raise ValueError("Vial actions must be finite normalized 2x14 commands")
        if r.shape != (2,) or not np.isfinite(r).all() or not np.isin(r, [0, 1]).all():
            raise ValueError("Vial rewards must be per-control native success flags")
        before, after = np.asarray(t.dones, np.float32), np.asarray(t.next_dones, np.float32)
        if before.shape != (4,) or after.shape != (4,) or np.any(after < before):
            raise ValueError("Vial subgoal labels must be four monotone flags")
        next_future_weights(torch.from_numpy(np.stack([before, after])))
        if before[-1]:
            raise ValueError("No vial transition may start after native success")
        if bool(after[-1]) != bool(t.terminated) or bool(r[:h].sum()) != bool(t.terminated):
            raise ValueError("First native success must terminate the vial transition")
        if r[:h].sum() > 1 or (r[:h].any() and r[h-1] != 1) or r[h:].any():
            raise ValueError("No controls or rewards may follow first native success")
        arrays.update(actions=a, rewards=r, dones=before, next_dones=after,
                      terminated=np.bool_(t.terminated), truncated=np.bool_(t.truncated),
                      executed_steps=np.int64(h))
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        return stream.getvalue()

    @staticmethod
    def _decode(payload):
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            def observation(prefix):
                return {key: str(arrays[f"{prefix}/{key}"].item()) if key == "task"
                        else arrays[f"{prefix}/{key}"].copy()
                        for key in ("proprio", "scale_reading_g", "task", *CAMERAS)}
            return VialTransition(observation("obs"), observation("next"), arrays["actions"].copy(),
                                  arrays["dones"].copy(), arrays["next_dones"].copy(),
                                  arrays["rewards"].copy(), bool(arrays["terminated"]),
                                  bool(arrays["truncated"]), int(arrays["executed_steps"]))


@dataclass(frozen=True)
class VialOGPOConfig(OGPOConfig):
    method: str = "ogpo"
    best_of_n: int = 8
    discount: float = .999
    subgoals: int = 1  # Number of critic heads; replay always has four labels.
    bf16: bool = False  # Opt-in CUDA compute; weights/Adam and densities stay FP32/FP64.

    def __post_init__(self):
        if self.method not in ("bon", "ogpo", "vadar"):
            raise ValueError("method must be bon, ogpo or vadar")
        object.__setattr__(self, "subgoals", 4 if self.method == "vadar" else 1)
        super().__post_init__()
        if self.action_steps != 2 or self.action_dim != 14 or self.state_dim != 15:
            raise ValueError("Native vial contract is H=2, action14, state15")
        if isinstance(self.best_of_n, bool) or not isinstance(self.best_of_n, int) or self.best_of_n < 1:
            raise ValueError("best_of_n must be positive")
        if self.bc_anchor_weight or self.step_reward_per_control is not None:
            raise ValueError("Vial comparison does not add BC penalties or shaped scalar rewards")


class VialTrainer(OGPOTrainer):
    def __init__(self, *args, config=VialOGPOConfig(), **kwargs):
        super().__init__(*args, config=config, **kwargs)
        if config.method == "bon":
            self.actor.requires_grad_(False)

    def _amp(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                              enabled=self.device.type == "cuda" and self.config.bf16)

    def _sample(self, batch, actor=None):
        with self._amp():
            return super()._sample(batch, actor)

    def _features(self, batch):
        with self._amp():
            return super()._features(batch)

    @torch.no_grad()
    def _candidate_selection(self, batch, features, count):
        """Return selected environment actions and their target-Q values."""
        c = self.config
        repeated = {key: value.repeat_interleave(count, dim=0) for key, value in batch.items()}
        sample = self._sample(repeated)
        actions = self.action_adapter.physical(sample.executed_actions(c.action_dim, c.action_steps))
        q = self.target_critic(features.repeat_interleave(count, dim=0),
                               self.action_adapter.critic(actions),
                               torch.ones(len(actions), c.action_steps, device=self.device))
        q = q.reshape(len(features), count, -1, c.subgoals)
        # No subgoal labels at inference. Scalar BoN; sum would be reference's
        # fixed weighting for vector BoN, but that variant is not enabled here.
        score = q.mean(dim=2).sum(dim=-1)
        choice = score.argmax(dim=1)
        index = torch.arange(len(features), device=self.device)
        return actions.reshape(len(features), count, c.action_steps, c.action_dim)[index, choice], q[index, choice]

    @torch.no_grad()
    def sample_actions(self, observations):
        batch = self._inputs(observations)
        if self.config.method == "bon":
            return self._candidate_selection(batch, self._features(batch), self.config.best_of_n)[0]
        sample = self._sample(batch)
        return self.action_adapter.physical(sample.executed_actions(self.config.action_dim, self.config.action_steps))

    def update(self, replay):
        c = self.config
        if replay.contract["provenance"] != TASK_ID + ":" + self.provenance:
            raise ValueError("Vial replay and BC/trainer provenance differ")
        rows = replay.sample(c.batch_size)
        batch = self._inputs([r.observation for r in rows])
        next_batch = self._inputs([r.next_observation for r in rows])
        features, next_features = self._features(batch), self._features(next_batch)
        before = torch.as_tensor(np.stack([r.dones for r in rows]), device=self.device)
        after = torch.as_tensor(np.stack([r.next_dones for r in rows]), device=self.device)
        terminal = torch.tensor([r.terminated for r in rows], device=self.device)
        horizons = torch.tensor([r.executed_steps for r in rows], device=self.device)
        valid = (torch.arange(c.action_steps, device=self.device)[None] < horizons[:, None]).float()
        actions = torch.as_tensor(np.stack([r.actions for r in rows]), device=self.device)
        encoded_actions = self.action_adapter.critic(actions)
        with torch.no_grad():
            if c.method == "bon":
                _, next_q = self._candidate_selection(next_batch, next_features, c.best_of_n)
            else:
                sample = self._sample(next_batch)
                next_actions = self.action_adapter.physical(sample.executed_actions(c.action_dim, c.action_steps))
                next_q = self.target_critic(next_features, self.action_adapter.critic(next_actions), torch.ones_like(valid))
            if c.method == "vadar":
                target = vector_td_target(after, next_q, terminal, c.discount, horizons,
                                          r_done=c.r_done, r_step=c.r_step, cap=c.value_cap)
            else:
                rewards = torch.as_tensor(np.stack([r.rewards for r in rows]), device=self.device)
                powers = c.discount ** torch.arange(c.action_steps, device=self.device)
                discounted_reward = (rewards * valid * powers).sum(dim=1)
                target = (discounted_reward[:, None] + c.discount**horizons[:, None]
                          * (~terminal)[:, None] * next_q.mean(dim=1))
        loss = vector_critic_loss(self.critic(features, encoded_actions, valid), target)
        norm = self._step_optimizer(self.critic_opt, self.critic, loss)
        self.critic_updates += 1
        ema_update(self.target_critic, self.critic, c.critic_ema)
        metrics = dict(method=c.method, critic_updates=self.critic_updates, actor_updates=self.actor_updates,
                       critic_loss=float(loss.detach()), critic_grad_norm=norm,
                       td_target_mean=float(target.mean()), actual_horizon_mean=float(horizons.float().mean()),
                       terminated_fraction=float(terminal.float().mean()),
                       sampled_terminal_transitions=int(terminal.sum()),
                       truncated_fraction=float(np.mean([r.truncated for r in rows])))
        if c.method == "bon" or self.critic_updates <= c.critic_warmup_updates:
            return metrics
        repeated = {key: value.repeat_interleave(c.group_size, dim=0) for key, value in batch.items()}
        with torch.no_grad():
            sample = self._sample(repeated)
            candidates = self.action_adapter.physical(sample.executed_actions(c.action_dim, c.action_steps))
            q = self.target_critic(features.repeat_interleave(c.group_size, dim=0),
                                   self.action_adapter.critic(candidates),
                                   torch.ones(len(candidates), c.action_steps, device=self.device))
            labels = before if c.method == "vadar" else torch.zeros(c.batch_size, 1, device=self.device)
            advantages = conservative_group_advantages(q.reshape(c.batch_size, c.group_size, -1, c.subgoals), labels)
        self.actor_opt.zero_grad(set_to_none=True)
        logs, total_loss = [], torch.zeros((), device=self.device, dtype=torch.float64)
        count = len(sample.latents)
        for start in range(0, count, c.actor_microbatch_size):
            stop = min(start+c.actor_microbatch_size, count)
            selected = {key: value[start:stop] for key, value in repeated.items()}
            # Match autocast and microbatch shape used when recording the chain.
            with self._amp():
                score = chain_log_prob(self.actor, selected, sample.latents[start:stop], c.chain)
            objective = clipped_chain_objective(score, sample.log_prob[start:stop],
                                                 advantages.flatten()[start:stop], c.clip_epsilon)
            objective = objective * (stop-start)/count
            if not torch.isfinite(objective):
                raise FloatingPointError("Nonfinite actor loss; update aborted")
            objective.backward()
            logs.append(score.detach())
            total_loss += objective.detach()
        norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), c.grad_norm, error_if_nonfinite=True)
        self.actor_opt.step()
        self.actor_updates += 1
        ema_update(self.target_actor, self.actor, c.actor_ema)
        logs = torch.cat(logs)
        ratio = full_chain_ratio(logs, sample.log_prob)
        metrics.update(actor_updates=self.actor_updates, actor_loss=float(total_loss), actor_grad_norm=float(norm),
                       ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
                       max_abs_log_ratio=float((logs-sample.log_prob).abs().max()),
                       ratio_clip_fraction=float(((ratio-1).abs() > c.clip_epsilon).float().mean()),
                       ratio_ess=float(ratio.sum().square()/ratio.square().sum().clamp_min(1e-300)),
                       nonzero_advantage_fraction=float((advantages != 0).float().mean()))
        return metrics


def canonical_hdf_observation(group, index, instruction):
    camera_keys = dict(top1="top1_image", left_wrist="robot0_eye_in_hand_image",
                       right_wrist="robot1_eye_in_hand_image")
    return actor_observation({"proprio": np.asarray(group["joint_pos"][index], np.float32),
                              "scale_reading_g": np.asarray(group["object"][index, 7:8], np.float32),
                              **{name: group[key][index] for name, key in camera_keys.items()}}, instruction)


@torch.no_grad()
def check_sampling_density(trainer, observations, *, tolerance=1e-7):
    """Gate the chosen GPU precision against sampled full-chain densities.

    Uses the same actor, microbatch shape, and autocast mode on both sides.
    Passing this check is numerical consistency, not optimizer/task validation.
    """
    batch = trainer._inputs(observations)
    sample = trainer._sample(batch)
    scores = []
    c = trainer.config
    for start in range(0, len(observations), c.actor_microbatch_size):
        stop = min(start+c.actor_microbatch_size, len(observations))
        selected = {key: value[start:stop] for key, value in batch.items()}
        with trainer._amp():
            scores.append(chain_log_prob(trainer.target_actor, selected, sample.latents[start:stop], c.chain))
    score = torch.cat(scores)
    error = float((score-sample.log_prob).abs().max())
    ratio = full_chain_ratio(score, sample.log_prob)
    result = dict(passed=bool(np.isfinite(error) and error <= tolerance),
                  max_abs_log_density_error=error, tolerance=tolerance,
                  ratio_min=float(ratio.min()), ratio_max=float(ratio.max()),
                  flow_steps=c.chain.steps, chain_shape=list(sample.latents.shape),
                  bf16=bool(c.bf16 and trainer.device.type == "cuda"),
                  parameter_dtype=str(next(trainer.actor.parameters()).dtype),
                  log_density_dtype=str(sample.log_prob.dtype))
    return result


def import_vial_demonstrations(replay, path, *, manifest, pan_position, instruction, on_progress=None):
    """Import only manifest training episodes, cut at their first native success."""
    import h5py
    from chemlab_rl.vial_vla_env import sequential_subgoals
    if len(replay):
        raise ValueError("Fresh demonstration import requires an empty replay")
    if manifest.get("task_id") != TASK_ID or file_sha256(path) != manifest["dataset_sha256"]:
        raise ValueError("Vial dataset task/hash differs from the BC manifest")
    names = manifest["train_episodes"]
    count, controls = 0, 0
    with h5py.File(path, "r") as handle:
        for name in names:
            demo = handle["data"][name]
            terminal_indices = np.flatnonzero(np.asarray(demo["dones"], bool))
            if not len(terminal_indices):
                raise ValueError(f"{name}: no native successful terminal")
            length = int(terminal_indices[0]) + 1
            if length != manifest["episode_lengths"][name]:
                raise ValueError(f"{name}: first-terminal length differs from BC manifest")
            labels = np.zeros(4, np.float32)
            for start in range(0, length, 2):
                stop = min(start+2, length)
                h = stop-start
                before = labels.copy()
                for i in range(start, stop):
                    labels = sequential_subgoals(labels, demo["next_obs/joint_pos"][i],
                                                  demo["next_obs/object"][i], pan_position=pan_position)
                terminal = stop == length
                if bool(labels[-1]) != terminal:
                    raise ValueError(f"{name}: native task labels disagree with first recorded terminal")
                actions = np.zeros((2, 14), np.float32)
                actions[:h] = demo["actions"][start:stop]
                rewards = np.zeros(2, np.float32)
                rewards[h-1] = float(terminal)
                replay.add(VialTransition(canonical_hdf_observation(demo["obs"], start, instruction),
                                          canonical_hdf_observation(demo["next_obs"], stop-1, instruction),
                                          actions, before, labels.copy(), rewards, terminal, False, h))
                count += 1
            controls += length
            if on_progress:
                on_progress(dict(event="demo_import", episode=name, effective_controls=length,
                                 replay_transitions=count))
    return dict(episodes=names, transitions=count, effective_controls=controls,
                terminal_rule="first_native_success_inclusive", heldout_imported=False)


def collect_vial_chunk(env, observation, info, trainer, replay, *, max_steps=2):
    if info.get("task_id") != TASK_ID or info.get("success"):
        raise ValueError("Vial collection requires a live nonterminal native task state")
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ValueError("Positive collection allowance required")
    before = actor_observation(observation, env.instruction)
    planned = trainer.sample_actions([before])[0].detach().cpu().numpy()
    actions, rewards = np.zeros((2, 14), np.float32), np.zeros(2, np.float32)
    terminated = truncated = False
    for i in range(min(2, max_steps)):
        actions[i] = planned[i]
        observation, reward, terminated, truncated, final_info = env.step(actions[i])
        rewards[i] = reward
        if terminated or truncated:
            observation = final_info.get("final_observation", observation)
            final_info = final_info.get("final_info", final_info)
            break
    h = i+1
    truncated = bool(truncated or (h < 2 and not terminated))
    replay.add(VialTransition(before, actor_observation(observation, env.instruction), actions,
                              np.asarray(info["subgoal_dones"], np.float32),
                              np.asarray(final_info["subgoal_dones"], np.float32), rewards,
                              bool(terminated), truncated, h))
    trainer.environment_steps += h
    return observation, final_info, dict(executed_steps=h, terminated=bool(terminated),
                                         truncated=truncated, reward_sum=float(rewards.sum()))


def run_vial_online(env, trainer, replay, *, max_environment_steps, max_updates,
                    max_seconds, checkpoint_path, seed=0, checkpoint_every=25, on_update=None):
    """Warm critics from imported demos, then alternate real chunks and updates."""
    if min(max_environment_steps, max_updates, checkpoint_every) < 1 or not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Positive finite run budgets required")
    if len(replay) < trainer.config.batch_size:
        raise ValueError("Import enough demonstration transitions before warmup")
    started, controls, updates, episodes, successes = time.monotonic(), 0, 0, 0, 0
    warmup_updates = warmup_terminals = 0
    def within_budget():
        return updates < max_updates and time.monotonic()-started < max_seconds
    def update(warmup=False):
        nonlocal updates, warmup_updates, warmup_terminals
        update_started = time.monotonic()
        metrics = trainer.update(replay)
        updates += 1
        if warmup:
            warmup_updates += 1
            warmup_terminals += metrics["sampled_terminal_transitions"]
        metrics.update(environment_steps=trainer.environment_steps, elapsed_seconds=time.monotonic()-started,
                       update_seconds=time.monotonic()-update_started,
                       phase="critic_warmup" if warmup else "online",
                       warmup_sampled_terminal_transitions=warmup_terminals)
        if on_update:
            on_update(metrics)
        if updates % checkpoint_every == 0:
            trainer.save_checkpoint(checkpoint_path, replay)
    while trainer.critic_updates < trainer.config.critic_warmup_updates and within_budget():
        update(warmup=True)
    warmup_seconds = time.monotonic()-started
    observation, info = env.reset(seed=seed)
    episode_controls, episode_stages = 0, np.zeros(4, np.float32)
    records = []
    while controls < max_environment_steps and within_budget():
        observation, info, result = collect_vial_chunk(env, observation, info, trainer, replay,
                                                      max_steps=max_environment_steps-controls)
        controls += result["executed_steps"]
        episode_controls += result["executed_steps"]
        episode_stages = np.maximum(episode_stages, info["subgoal_dones"])
        if within_budget():
            update()
        if result["terminated"] or result["truncated"]:
            successes += int(bool(info["success"]))
            episodes += 1
            record = dict(episode=episodes, controls=episode_controls, success=bool(info["success"]),
                          subgoal_dones=episode_stages.tolist(), terminated=result["terminated"],
                          truncated=result["truncated"])
            records.append(record)
            if on_update:
                on_update(dict(event="online_episode", **record))
            if controls < max_environment_steps and within_budget():
                observation, info = env.reset(seed=seed+episodes)
                episode_controls, episode_stages = 0, np.zeros(4, np.float32)
    trainer.save_checkpoint(checkpoint_path, replay)
    return dict(method=trainer.config.method, environment_steps=controls, updates=updates,
                completed_episodes=episodes, online_successes=successes, episodes=records,
                elapsed_seconds=time.monotonic()-started, actor_updates=trainer.actor_updates,
                critic_updates=trainer.critic_updates, closed_loop_evaluated=False,
                warmup_updates=warmup_updates, warmup_seconds=warmup_seconds,
                warmup_sampled_terminal_transitions=warmup_terminals,
                warmup_completed=trainer.critic_updates >= trainer.config.critic_warmup_updates)


@torch.no_grad()
def evaluate_vial_trainer(env, trainer, *, seeds=(1000,), max_steps=1600):
    """Same EMA stochastic sampler/BoN selection used online; no privileged labels."""
    if max_steps < 1 or not seeds:
        raise ValueError("Evaluation requires seeds and a positive horizon")
    saved_rng = trainer.generator.get_state()
    episodes = []
    try:
        for seed in seeds:
            trainer.generator.manual_seed(int(seed))
            obs, info = env.reset(seed=int(seed))
            stages, count, terminated, truncated = np.zeros(4), 0, False, False
            while count < max_steps and not (terminated or truncated):
                actions = trainer.sample_actions([actor_observation(obs, env.instruction)])[0].cpu().numpy()
                for action in actions[:min(2, max_steps-count)]:
                    obs, _, terminated, truncated, info = env.step(action)
                    count += 1
                    stages = np.maximum(stages, info["subgoal_dones"])
                    if terminated or truncated:
                        break
            episodes.append(dict(seed=int(seed), success=bool(info["success"]), controls=count,
                                 subgoal_dones=stages.tolist(), balance_g=float(info["balance_g"]),
                                 terminated=bool(terminated), truncated=bool(truncated or count >= max_steps),
                                 fixed_scene=True))
    finally:
        trainer.generator.set_state(saved_rng)
    return dict(method=trainer.config.method, episodes=episodes,
                success_rate=float(np.mean([e["success"] for e in episodes])),
                sampler="frozen_bc_stochastic_chain_bon" if trainer.config.method == "bon" else "ema_actor_stochastic_chain",
                best_of_n=trainer.config.best_of_n if trainer.config.method == "bon" else 1,
                flow_steps=trainer.config.chain.steps, action_steps=2,
                fixed_scene=True, independent_initial_conditions=False)
