"""Opt-in, bounded physical-grain OGPO–VaDar run from a genuine BC checkpoint.

Use --gradient-smoke with a manifest-matching HDF5 dataset to test the real VLA
likelihood/backward interface without online rollouts or an optimizer step.
This script allocates no remote resources; a job launcher must supply a device.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, required=True, help="BC save_pretrained directory")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--resume", type=Path, help="Exact trainer checkpoint; requires its unchanged replay DB")
    result.add_argument("--device", default="cuda")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--grain-count", type=int, default=400)
    result.add_argument("--powder-ids", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    result.add_argument("--target-g", type=float, default=2.5)
    result.add_argument("--max-episode-steps", type=int, default=2600)
    result.add_argument("--max-environment-steps", type=int, default=10000)
    result.add_argument("--max-updates", type=int, default=5000)
    result.add_argument("--max-seconds", type=float, default=600)
    result.add_argument("--checkpoint-every", type=int, default=100)
    result.add_argument("--replay-capacity", type=int, default=100000)
    result.add_argument("--batch-size", type=int, default=8)
    result.add_argument("--group-size", type=int, default=4)
    result.add_argument("--actor-microbatch-size", type=int, default=2)
    result.add_argument("--critic-members", type=int, default=2)
    result.add_argument("--critic-hidden", type=int, default=256)
    result.add_argument("--critic-warmup-updates", type=int, default=1000)
    result.add_argument("--actor-lr", type=float, default=1e-6)
    result.add_argument("--critic-lr", type=float, default=3e-4)
    result.add_argument("--actor-ema", type=float, default=.005)
    result.add_argument("--critic-ema", type=float, default=.005)
    result.add_argument("--discount", type=float, default=.99, help="Discount per 0.1-s environment control")
    result.add_argument("--r-done", type=float, default=5.)
    result.add_argument("--r-step", type=float, default=-1., help="Reference step cost per executed chunk")
    result.add_argument("--value-cap", type=float, default=15.)
    result.add_argument("--step-reward-per-control", type=float,
                        help="Explicit deviation: discounted control-step cost, replacing --r-step")
    result.add_argument("--flow-steps", type=int, default=10)
    result.add_argument("--diffusion-scale", type=float, default=.05)
    result.add_argument("--bc-anchor-weight", type=float, default=0,
                        help="Optional fixed-BC chain KL deviation; default off")
    result.add_argument("--gradient-smoke", action="store_true")
    result.add_argument("--dataset", type=Path, help="Required for gradient smoke; must match BC manifest")
    result.add_argument("--demo-replay", type=Path,
                        help="Opt-in deviation: seed critic replay from the exact accepted physical BC dataset")
    return result


def density_diagnostics(sample, score, sampled_parameters, scored_parameters):
    """Quantify transition-mean and implied ratio drift without changing scoring."""
    import torch
    steps = []
    for index, ((old_mean, old_var), (new_mean, new_var)) in enumerate(zip(sampled_parameters, scored_parameters, strict=True)):
        if not torch.equal(old_var, new_var):
            raise AssertionError("Sampler/scorer transition variances differ")
        mean_difference = new_mean.double() - old_mean.double()
        standardized = mean_difference / old_var.double().sqrt()
        following = sample.latents[:, index+1].double()
        implied = (.5 * ((following-old_mean.double()).square() - (following-new_mean.double()).square())
                   / old_var.double()).sum((-2, -1))
        steps.append(dict(step=index, mean_max_abs=float(mean_difference.abs().max()),
                          mean_error_in_std_max=float(standardized.abs().max()),
                          mean_error_in_std_rms=float(standardized.square().mean().sqrt()),
                          implied_log_ratio_per_sample=implied.cpu().tolist()))
    delta = score.detach()-sample.log_prob
    return dict(log_density_error_per_sample=delta.cpu().tolist(),
                max_abs_log_density_error=float(delta.abs().max()),
                max_implied_ratio_deviation=float(torch.expm1(delta).abs().max()), steps=steps)


def gradient_smoke(policy, preprocessor, postprocessor, manifest, dataset_path, *, flow_steps, diffusion_scale, device, seed):
    import numpy as np
    import torch
    from chemlab_rl.ogpo_training import FrozenSmolVLAFeatures, actor_observation, prepare_policy_batch
    from chemlab_rl.smolvla_ogpo import (ChainConfig, SmolVLAVelocity, chain_log_prob,
                                       clipped_chain_objective, sample_chain, transition_parameters)
    from chemlab_rl.vla_bc import CAMERAS, STATE_KEY, ScrapeHDFDataset, file_sha256, manifest_state_contract
    include_scale, state_dim = manifest_state_contract(manifest)
    if file_sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("Gradient smoke dataset differs from the checkpoint manifest")
    dataset = ScrapeHDFDataset(dataset_path, manifest["train_episodes"][:1], 50,
                              legacy=manifest["legacy"], include_measured_scale=include_scale)
    try:
        item = dataset[0]
    finally:
        dataset.close()
    observation = {"proprio": item[STATE_KEY][:14]}
    if include_scale:
        observation["scale_reading_g"] = item[STATE_KEY][14:15]
    observation.update({raw: np.rint(item[key].transpose(1, 2, 0)*255).astype(np.uint8) for raw, key in CAMERAS.items()})
    observation = actor_observation(observation, item["task"], state_dim)
    batch = prepare_policy_batch([observation, observation], preprocessor, device)
    actor = SmolVLAVelocity(policy)
    trainability = dict(
        vlm_trainable_parameters=sum(p.numel() for p in policy.model.vlm_with_expert.vlm.parameters() if p.requires_grad),
        state_projection_trainable_parameters=sum(p.numel() for p in policy.model.state_proj.parameters() if p.requires_grad),
        actor_trainable_parameters=sum(p.numel() for p in actor.parameters() if p.requires_grad))
    print(json.dumps({"event": "real_vla_trainability", **trainability}), flush=True)
    encoder = FrozenSmolVLAFeatures(policy, state_dim)
    feature = encoder(batch)
    chain = ChainConfig(steps=flow_steps, diffusion_scale=diffusion_scale)
    sampled_parameters, scored_parameters = [], []
    def recording_velocity(parameters):
        def velocity(batch, latent, tau):
            result = actor(batch, latent, tau)
            # Record numerical means without retaining an additional graph or
            # performing another VLA forward. The sampler/scorer formulas stay unchanged.
            with torch.no_grad():
                mean, variance = transition_parameters(latent, result.detach(), tau, chain)
                parameters.append((mean.detach().clone(), variance.detach().clone()))
            return result
        return velocity
    sample = sample_chain(recording_velocity(sampled_parameters), batch, (2, 50, 32), chain, device=device,
                          generator=torch.Generator(device=device).manual_seed(seed))
    actor.zero_grad(set_to_none=True)
    score = chain_log_prob(recording_velocity(scored_parameters), batch, sample.latents, chain)
    numerical = density_diagnostics(sample, score, sampled_parameters, scored_parameters)
    print(json.dumps({"event": "real_vla_density_diagnostics", **numerical}), flush=True)
    difference = float((score.detach()-sample.log_prob).abs().max())
    if not torch.allclose(score.detach(), sample.log_prob, rtol=0, atol=1e-7):
        with torch.no_grad():
            no_grad_score = chain_log_prob(actor, batch, sample.latents, chain)
        print(json.dumps({"event": "real_vla_density_failure_modes",
                          "no_grad_vs_sampler_max_error": float((no_grad_score-sample.log_prob).abs().max()),
                          "grad_vs_no_grad_max_error": float((score.detach()-no_grad_score).abs().max())}), flush=True)
        raise AssertionError(f"Real actor recomputed path density changed: {difference}")
    loss = clipped_chain_objective(score, sample.log_prob, torch.tensor([-1., 1.], device=device))
    loss.backward()
    gradients = [parameter.grad for parameter in actor.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("Real VLA gradients are absent or nonfinite")
    norm = float(torch.sqrt(sum(gradient.double().square().sum() for gradient in gradients)))
    if norm <= 0:
        raise AssertionError("Real VLA chain objective has zero gradient")
    actions = postprocessor(sample.executed_actions(14, 2))
    if actions.shape != (2, 2, 14) or not torch.isfinite(actions).all():
        raise AssertionError("Real BC unnormalizer returned invalid action prefix")
    actor.zero_grad(set_to_none=True)
    return dict(real_pretrained_gradient_passed=True, state_dim=state_dim,
                feature_shape=list(feature.shape), chain_shape=list(sample.latents.shape),
                action_shape=list(actions.shape), recomputed_log_density_max_error=difference,
                gradient_norm=norm, gradient_tensors=len(gradients),
                optimizer_step_performed=False, online_training_performed=False,
                closed_loop_evaluated=False, legacy_diagnostic=bool(manifest["legacy"]),
                critic_state_projection="private_frozen_bc_snapshot", density_diagnostics=numerical,
                **trainability)


def main(argv=None):
    pipeline_started = time.monotonic()
    args = parser().parse_args(argv)
    if sys.platform != "darwin":
        os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import torch
    from chemlab_rl.ogpo_training import (FrozenSmolVLAFeatures, OGPOConfig, OGPOTrainer, PersistentReplay,
                                         PhysicalActionAdapter, VectorQEnsemble, import_demonstrations, run_online)
    from chemlab_rl.smolvla_ogpo import ChainConfig, SmolVLAVelocity
    from chemlab_rl.vla_bc import load_smolvla, manifest_state_contract, file_sha256
    args.output.mkdir(parents=True, exist_ok=True)
    phase_seconds = {}
    def phase_start(name):
        print(json.dumps(dict(event="pipeline_phase_started", phase=name,
                              pipeline_elapsed_seconds=time.monotonic()-pipeline_started)), flush=True)
        return time.monotonic()

    def phase_finish(name, started):
        phase_seconds[name] = time.monotonic()-started
        timing = dict(phase_seconds=dict(phase_seconds),
                      pipeline_elapsed_seconds=time.monotonic()-pipeline_started,
                      runner_max_seconds=args.max_seconds,
                      limit_scope="run_online only; external timeout must include setup/import/export")
        (args.output / "pipeline_timings.json").write_text(json.dumps(timing, indent=2)+"\n")
        print(json.dumps(dict(event="pipeline_phase_completed", phase=name, **timing)), flush=True)

    manifest_path = args.checkpoint / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    include_scale, state_dim = manifest_state_contract(manifest)
    if args.gradient_smoke and args.dataset is None:
        raise ValueError("--gradient-smoke requires --dataset")
    if args.gradient_smoke and args.demo_replay is not None:
        raise ValueError("--demo-replay is only available in explicit online training mode")
    demo_provenance = {"mode": "online_only", "dataset_sha256": None}
    if not args.gradient_smoke:
        if manifest.get("legacy") or manifest.get("schema_version") != "scrape_v2" or not include_scale:
            raise ValueError("Online physical training requires the corrected 15-D measured-scale BC checkpoint")
        if not args.powder_ids or any(i not in range(5) for i in args.powder_ids):
            raise ValueError("Only training materials 0..4 may be used online; material 5 is held out")
        if (not 1 <= args.grain_count <= 400 or min(args.max_episode_steps, args.max_environment_steps,
                                                    args.max_updates, args.checkpoint_every,
                                                    args.critic_members, args.critic_hidden, args.replay_capacity) < 1
                or not np.isfinite(args.max_seconds) or args.max_seconds <= 0):
            raise ValueError("Physical grain count and all run budgets must be positive and valid")
        if (args.output / "trainer.pt").exists() and args.resume is None:
            raise ValueError("Existing training output requires --resume or a new directory")
        if args.demo_replay is not None:
            import mujoco
            from scripts.collect_granular_demos import source_identity
            demo_sha = file_sha256(args.demo_replay)
            if demo_sha != manifest.get("dataset_sha256"):
                raise ValueError("--demo-replay must be the exact dataset in the BC manifest")
            source_version, source_hashes = source_identity()
            if manifest.get("source_version") != source_version or manifest.get("grain_count") != args.grain_count:
                raise ValueError("--demo-replay BC sources/grain count differ from the online simulation")
            demo_provenance = dict(mode="demonstration_seeded_off_policy", dataset_sha256=demo_sha,
                                   source_version=source_version, source_files_sha256=source_hashes,
                                   mujoco_version=mujoco.__version__)
    phase_finish("preflight", pipeline_started)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config = OGPOConfig(state_dim=state_dim, batch_size=args.batch_size, group_size=args.group_size,
                        actor_microbatch_size=args.actor_microbatch_size,
                        actor_lr=args.actor_lr, critic_lr=args.critic_lr, actor_ema=args.actor_ema,
                        critic_ema=args.critic_ema, critic_warmup_updates=args.critic_warmup_updates,
                        discount=args.discount, r_done=args.r_done, r_step=args.r_step, value_cap=args.value_cap,
                        step_reward_per_control=args.step_reward_per_control,
                        bc_anchor_weight=args.bc_anchor_weight,
                        chain=ChainConfig(steps=args.flow_steps, diffusion_scale=args.diffusion_scale))
    (args.output / "ogpo_config.json").write_text(json.dumps(asdict(config), indent=2)+"\n")
    phase = phase_start("model_load")
    policy, pre, post = load_smolvla(manifest, args.device, pretrained=str(args.checkpoint),
                                   chunk_size=50, n_action_steps=2)
    phase_finish("model_load", phase)
    if args.gradient_smoke:
        result = gradient_smoke(policy, pre, post, manifest, args.dataset, flow_steps=args.flow_steps,
                                diffusion_scale=args.diffusion_scale, device=args.device, seed=args.seed)
        (args.output / "real_vla_gradient_smoke.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
        return result

    from chemlab_rl.granular_env import GranularWeighEnv
    class CyclingMaterials:
        def __init__(self):
            self.current_id = args.powder_ids[args.seed % len(args.powder_ids)]
            self.env = self.build(self.current_id)
            self.action_space = self.env.action_space

        def build(self, powder_id):
            return GranularWeighEnv(powder_id=powder_id, target_g=args.target_g, grain_count=args.grain_count,
                                    max_steps=args.max_episode_steps, images=True, check_collisions=True)

        @property
        def instruction(self):
            return self.env.instruction

        def reset(self, seed=None):
            selected = args.powder_ids[(seed or 0) % len(args.powder_ids)]
            if selected != self.current_id:
                self.env.close()
                self.env, self.current_id = self.build(selected), selected
            return self.env.reset(seed=seed)

        def step(self, action):
            return self.env.step(action)

        def close(self):
            self.env.close()

    weights = sorted(args.checkpoint.glob("*.safetensors"))
    if not weights:
        raise ValueError("BC checkpoint must include its saved safetensors weights")
    provenance_data = dict(manifest_sha256=file_sha256(manifest_path),
                           weights={path.name: file_sha256(path) for path in weights},
                           physical_code={name: file_sha256(ROOT / "chemlab_rl" / name)
                                          for name in ("granular_env.py", "granular_scene.py", "granular_material.py",
                                                       "granular_geometry.py", "granular_accounting.py", "weigh_scene.py")},
                           grain_count=args.grain_count, powder_ids=args.powder_ids,
                           target_g=args.target_g, max_episode_steps=args.max_episode_steps,
                           replay_initialization=demo_provenance)
    provenance = hashlib.sha256(json.dumps(provenance_data, sort_keys=True).encode()).hexdigest()
    (args.output / "provenance.json").write_text(json.dumps(provenance_data, indent=2)+"\n")
    env, replay = None, None
    try:
        phase = phase_start("environment_construction")
        env = CyclingMaterials()
        phase_finish("environment_construction", phase)
        phase = phase_start("trainer_initialization")
        actor = SmolVLAVelocity(policy)
        encoder = FrozenSmolVLAFeatures(policy, state_dim)
        critic = VectorQEnsemble(encoder.feature_dim, members=args.critic_members, hidden=args.critic_hidden)
        trainer = OGPOTrainer(actor, critic, preprocessor=pre, feature_fn=encoder,
                              action_adapter=PhysicalActionAdapter(post, env.action_space.low, env.action_space.high),
                              config=config, device=args.device, seed=args.seed, provenance=provenance)
        replay = PersistentReplay(args.output / "replay.sqlite", capacity=args.replay_capacity,
                                  state_dim=state_dim, provenance=provenance, seed=args.seed)
        phase_finish("trainer_initialization", phase)
        demonstration_report = None
        if args.demo_replay is not None:
            phase = phase_start("demonstration_import")
            demonstration_report = import_demonstrations(
                replay, args.demo_replay, bc_manifest=manifest,
                expected_source_version=demo_provenance["source_version"],
                expected_source_hashes=demo_provenance["source_files_sha256"],
                grain_count=args.grain_count, max_episode_steps=args.max_episode_steps,
                mujoco_version=demo_provenance["mujoco_version"],
                action_low=env.action_space.low, action_high=env.action_space.high,
                on_progress=lambda record: print(json.dumps(record), flush=True))
            phase_finish("demonstration_import", phase)
            (args.output / "demonstration_import.json").write_text(json.dumps(demonstration_report, indent=2)+"\n")
            print(json.dumps(dict(event="demonstration_replay_import", mode=demonstration_report["mode"],
                                  transitions=demonstration_report["transition_count"],
                                  already_imported=demonstration_report["already_imported"])), flush=True)
        if args.resume:
            trainer.load_checkpoint(args.resume, replay)
        with (args.output / "updates.jsonl").open("a") as stream:
            def log(metrics):
                line = json.dumps(metrics)
                stream.write(line+"\n")
                stream.flush()
                print(line, flush=True)
            phase = phase_start("online_runner")
            result = run_online(env, trainer, replay, max_environment_steps=args.max_environment_steps,
                                max_updates=args.max_updates, max_seconds=args.max_seconds,
                                checkpoint_path=args.output / "trainer.pt", seed=args.seed,
                                checkpoint_every=args.checkpoint_every, on_update=log)
            phase_finish("online_runner", phase)
        # Export the current actor in the existing BC evaluator's save format.
        phase = phase_start("actor_export")
        exported = args.output / "policy"
        trainer.actor.policy.save_pretrained(exported)
        pre.save_pretrained(exported, config_filename="policy_preprocessor.json")
        post.save_pretrained(exported, config_filename="policy_postprocessor.json")
        (exported / "data_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
        phase_finish("actor_export", phase)
        result.update(actor_updates=trainer.actor_updates, critic_updates=trainer.critic_updates,
                      policy=str(exported), method="experimental_smolvla_ogpo_vadar_port",
                      replay_initialization=demo_provenance["mode"],
                      demonstration_transitions=(demonstration_report["transition_count"] if demonstration_report else 0),
                      actor_trainable_scope="pretrained_expert_and_projections_per_bc_config",
                      train_state_proj=bool(policy.config.train_state_proj), bc_anchor_weight=args.bc_anchor_weight,
                      pipeline_phase_seconds=dict(phase_seconds),
                      pipeline_elapsed_seconds=time.monotonic()-pipeline_started)
        (args.output / "run_result.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
        return result
    finally:
        if replay is not None:
            replay.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
