"""Train/evaluate a bounded SmolVLA vial comparison from one shared BC checkpoint."""
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--method", choices=("bon", "ogpo", "vadar"), required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--action-steps", type=int, default=2)
    p.add_argument("--diffusion-scale", type=float, default=.05)
    p.add_argument("--best-of-n", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--group-size", type=int, default=4)
    p.add_argument("--actor-microbatch-size", type=int, default=1)
    p.add_argument("--actor-lr", type=float, default=1e-6)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--critic-members", type=int, default=2)
    p.add_argument("--critic-hidden", type=int, default=256)
    p.add_argument("--critic-warmup-updates", type=int, default=100)
    p.add_argument("--max-environment-steps", type=int, default=1600)
    p.add_argument("--max-updates", type=int, default=900)
    p.add_argument("--max-seconds", type=float, default=3600)
    p.add_argument("--max-episode-steps", type=int, default=1600)
    p.add_argument("--eval-episodes", type=int, default=1)
    p.add_argument("--eval-max-steps", type=int, default=1600)
    p.add_argument("--replay-capacity", type=int, default=30000)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--update-every-chunks", type=int, default=1)
    p.add_argument("--cache-prefix", action="store_true")
    p.add_argument("--seed-replay-on-resume", action="store_true",
                   help="Rebuild an identical demonstration-only replay before restoring a warmup checkpoint")
    p.add_argument("--respect-absolute-deadline", action="store_true")
    p.add_argument("--pending-replay-updates", type=int, choices=(0,1), default=0,
                   help="Complete a collected chunk's update skipped at the previous time cap")
    p.add_argument("--resume", type=Path)
    p.add_argument("--eval-only", action="store_true", help="Requires a full trainer checkpoint via --resume")
    p.add_argument("--baseline-only", action="store_true", help="Evaluate shared BC with matched stochastic chain; no updates")
    precision = p.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true", help="Opt into matched CUDA BF16 sampling/scoring")
    precision.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.set_defaults(bf16=False)
    p.add_argument("--density-check-only", action="store_true", help="Check real-model density consistency without updates or robot rollout")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform != "darwin":
        os.environ.setdefault("MUJOCO_GL", "egl")
    import numpy as np
    import torch
    from chemlab_rl.ogpo_training import FrozenSmolVLAFeatures, PhysicalActionAdapter, VectorQEnsemble
    from chemlab_rl.smolvla_ogpo import ChainConfig, SmolVLAVelocity
    from chemlab_rl.vial_ogpo_training import (VialOGPOConfig, VialReplay, VialTrainer,
        canonical_hdf_observation, check_sampling_density, evaluate_vial_trainer,
        import_vial_demonstrations, run_vial_online)
    from chemlab_rl.vial_vla_data import load_smolvla, manifest_state_contract
    from chemlab_rl.vial_vla_env import VialVLAEnv, source_identity
    from chemlab_rl.vla_bc import file_sha256

    if args.eval_episodes < 1 or args.eval_max_steps < 1 or args.max_episode_steps < 1:
        raise ValueError("Positive evaluation/native horizons required")
    if args.eval_only and args.resume is None:
        raise ValueError("--eval-only requires --resume for critic/actor state")
    if args.baseline_only and (args.method != "ogpo" or args.resume or args.eval_only):
        raise ValueError("--baseline-only uses --method ogpo and a fresh BC checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "trainer.pt").exists() and args.resume is None:
        raise ValueError("Existing output requires --resume or a new directory")
    started = time.monotonic()
    def log(record):
        line = json.dumps(record)
        with (args.output / "updates.jsonl").open("a") as stream:
            stream.write(line + "\n")
        print(line, flush=True)
    def phase(name):
        log(dict(event="phase", name=name, elapsed_seconds=time.monotonic()-started))
    manifest_path = args.checkpoint / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_state_contract(manifest)
    if file_sha256(args.dataset) != manifest["dataset_sha256"]:
        raise ValueError("RL dataset must be the shared BC checkpoint's exact source")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    weights = sorted(args.checkpoint.glob("*.safetensors"))
    if not weights:
        raise ValueError("Saved shared BC checkpoint is missing safetensors")
    provenance_data = dict(bc_manifest_sha256=file_sha256(manifest_path),
                           bc_weights={p.name: file_sha256(p) for p in weights},
                           simulation_source=source_identity(), task_id=manifest["task_id"],
                           dataset_sha256=manifest["dataset_sha256"],
                           train_episodes=manifest["train_episodes"],
                           heldout_episodes=manifest["heldout_episodes"],
                           excluded_episodes=manifest["excluded_episodes"])
    provenance_data["algorithm_sources"] = {
        name: file_sha256(ROOT / "chemlab_rl" / name)
        for name in ("vial_ogpo_training.py", "ogpo_training.py", "smolvla_ogpo.py", "vial_vla_data.py")}
    provenance = hashlib.sha256(json.dumps(provenance_data, sort_keys=True).encode()).hexdigest()
    (args.output / "provenance.json").write_text(json.dumps(provenance_data, indent=2)+"\n")
    config = VialOGPOConfig(method=args.method, best_of_n=args.best_of_n, action_steps=args.action_steps,
                           batch_size=args.batch_size, group_size=args.group_size,
                           actor_microbatch_size=args.actor_microbatch_size,
                           actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                           critic_warmup_updates=args.critic_warmup_updates,
                           bf16=args.bf16, cache_prefix=args.cache_prefix,
                           chain=ChainConfig(steps=args.flow_steps, diffusion_scale=args.diffusion_scale))
    (args.output / "ogpo_config.json").write_text(json.dumps(asdict(config), indent=2)+"\n")
    phase("load_shared_bc")
    policy, pre, post = load_smolvla(manifest, args.device, pretrained=str(args.checkpoint),
                                    chunk_size=50, n_action_steps=config.action_steps)
    encoder = FrozenSmolVLAFeatures(policy, 15)
    critic = VectorQEnsemble(encoder.feature_dim, action_steps=config.action_steps, subgoals=config.subgoals,
                            members=args.critic_members, hidden=args.critic_hidden)
    trainer = VialTrainer(SmolVLAVelocity(policy, cache_prefix=args.cache_prefix), critic, preprocessor=pre, feature_fn=encoder,
                          action_adapter=PhysicalActionAdapter(post, [-1]*14, [1]*14),
                          config=config, device=args.device, seed=args.seed, provenance=provenance)
    phase("sampling_density_check")
    import h5py
    with h5py.File(args.dataset, "r") as handle:
        group = handle["data"][manifest["train_episodes"][0]]["obs"]
        inputs = [canonical_hdf_observation(group, i, manifest["instruction"])
                  for i in range(min(2, manifest["episode_lengths"][manifest["train_episodes"][0]]))]
    density = check_sampling_density(trainer, inputs)
    (args.output / "density_check.json").write_text(json.dumps(density, indent=2)+"\n")
    if not density["passed"]:
        raise FloatingPointError("Sample/recomputed chain densities disagree; refusing training")
    if args.density_check_only:
        result = dict(method=args.method, density_check=density, optimizer_updates=0,
                      environment_steps=0, closed_loop_evaluated=False)
        (args.output / "run_result.json").write_text(json.dumps(result, indent=2)+"\n")
        log(dict(event="result", **result))
        return result
    env = replay = None
    try:
        phase("native_environment")
        env = VialVLAEnv(max_steps=args.max_episode_steps)
        env.reset(seed=args.seed)
        replay = VialReplay(args.output / "replay.sqlite", provenance=provenance,
                            capacity=args.replay_capacity, seed=args.seed, action_steps=config.action_steps)
        if args.baseline_only:
            imported = dict(episodes=[], transitions=0, effective_controls=0, heldout_imported=False)
        elif args.resume:
            if args.seed_replay_on_resume:
                phase("restore_demonstration_replay")
                imported = import_vial_demonstrations(replay, args.dataset, manifest=manifest,
                                                      pan_position=env.pan_position,
                                                      instruction=env.instruction, on_progress=log)
                (args.output / "demonstration_import.json").write_text(json.dumps(imported, indent=2)+"\n")
            trainer.load_checkpoint(args.resume, replay)
            imported = json.loads((args.output / "demonstration_import.json").read_text())
            for _ in range(args.pending_replay_updates):
                log(dict(phase="resumed_pending_update", **trainer.update(replay)))
        else:
            phase("training_demo_import")
            imported = import_vial_demonstrations(replay, args.dataset, manifest=manifest,
                                                  pan_position=env.pan_position,
                                                  instruction=env.instruction, on_progress=log)
            (args.output / "demonstration_import.json").write_text(json.dumps(imported, indent=2)+"\n")
        result = dict(method=args.method, eval_only=args.eval_only)
        if not (args.eval_only or args.baseline_only):
            phase("critic_warmup_and_online_training")
            if args.respect_absolute_deadline:
                deadline = float(os.environ["CHEMLAB_DEADLINE_UTC"])
                args.max_seconds = min(args.max_seconds, max(1., deadline-time.time()-60))
            result = run_vial_online(env, trainer, replay, max_environment_steps=args.max_environment_steps,
                                     max_updates=args.max_updates, max_seconds=args.max_seconds,
                                     checkpoint_path=args.output / "trainer.pt", seed=args.seed,
                                     checkpoint_every=args.checkpoint_every, on_update=log,
                                     update_every_chunks=args.update_every_chunks)
        # Snapshot evaluation restores the trainer RNG and uses exactly its
        # online EMA stochastic chain (or frozen BC BoN), never a different ODE.
        phase("closed_loop_evaluation")
        evaluation = evaluate_vial_trainer(env, trainer,
                                           seeds=tuple(args.seed+1000+i for i in range(args.eval_episodes)),
                                           max_steps=args.eval_max_steps, on_progress=log)
        if args.baseline_only:
            evaluation["method"] = "bc_matched_stochastic_sampler"
        (args.output / "evaluation.json").write_text(json.dumps(evaluation, indent=2)+"\n")
        result.update(closed_loop_evaluated=True, evaluation=evaluation, demonstration_import=imported,
                      density_check=density,
                      actor_updates=trainer.actor_updates, critic_updates=trainer.critic_updates,
                      native_reward="0 before success,1 on first native success",
                      critic_objective="reference_auxiliary_subgoal_heads_5_minus1_cap15_NF" if args.method == "vadar"
                                       else "discounted_native_final_success",
                      actor_frozen=args.method == "bon", fixed_scene=True,
                      total_pipeline_seconds=time.monotonic()-started,
                      cumulative_environment_steps=trainer.environment_steps,
                      training_time_limit_scope="runner only; import and final evaluation additional",
                      checkpoint=str(args.output / "trainer.pt"))
        if args.baseline_only:
            result.update(method="bc_matched_stochastic_sampler", checkpoint=None, actor_frozen=True,
                          critic_objective="none; shared BC evaluated without training")
        (args.output / "run_result.json").write_text(json.dumps(result, indent=2)+"\n")
        log(dict(event="result", **result))
        return result
    finally:
        if replay is not None:
            replay.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
