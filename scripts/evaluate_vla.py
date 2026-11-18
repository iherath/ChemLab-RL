"""Frozen SmolVLA evaluation from a full closed-shield reset.

Held-out powder results are final measurements, never a checkpoint-selection
signal. Seed, target, action history and termination reason are recorded. A
time-limited episode is not counted as a success or silently dropped.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np


def evaluation_grain_count(manifest, *, source_version, material_vectors):
    """Reject checkpoints whose physical setup cannot be reproduced exactly."""
    if (manifest.get("legacy") is not False or manifest.get("schema_version") != "scrape_v2"
            or manifest.get("use") != "synthetic_coarse_grain_control_data"):
        raise ValueError("Evaluation requires a physical granular BC/RL checkpoint")
    count = manifest.get("grain_count")
    if type(count) is not int or not 1 <= count <= 400:
        raise ValueError("Checkpoint manifest needs an explicit integer grain_count in 1..400")
    if manifest.get("source_version") != source_version:
        raise ValueError("Checkpoint physical source version differs from the evaluation environment")
    vectors = manifest.get("powder_parameter_vectors")
    if isinstance(vectors, str):
        vectors = json.loads(vectors)
    if vectors != material_vectors:
        raise ValueError("Checkpoint material vectors differ from the evaluation environment")
    return count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("seen", "heldout", "both"), default="seen")
    parser.add_argument("--episodes-per-powder", type=int, default=1)
    parser.add_argument("--powders", type=int, nargs="+")
    parser.add_argument("--targets", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--max-seconds", type=float, default=7200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args(argv)
    if args.episodes_per_powder < 1 or args.max_steps < 1 or args.max_seconds <= 0 or args.seed < 0:
        parser.error("episode counts and limits must be positive; seed must be nonnegative")
    if not all(np.isfinite(target) and target > 0 for target in args.targets):
        parser.error("targets must be finite and positive")
    import torch
    from chemlab_rl.granular_env import GranularWeighEnv
    from chemlab_rl.granular_material import GRANULAR_MATERIALS
    from chemlab_rl.vla_bc import VLABCPolicy
    from scripts.collect_granular_demos import source_identity

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.checkpoint / "data_manifest.json").read_text())
    source_version, _ = source_identity()
    grain_count = evaluation_grain_count(
        manifest, source_version=source_version,
        material_vectors=[material.as_dict() for material in GRANULAR_MATERIALS],
    )
    heldout = set(manifest["heldout_powder_ids"])
    seen = {row["powder_id"] for row in manifest["episodes"] if row["name"] in manifest["train_episodes"]}
    eligible = seen if args.split == "seen" else heldout if args.split == "heldout" else seen | heldout
    powders = sorted(eligible) if args.powders is None else args.powders
    if not powders or not set(powders) <= eligible:
        raise ValueError("Requested powders do not belong to the declared evaluation split")
    started = time.monotonic()
    deadline = float(os.environ.get("CHEMLAB_DEADLINE_UTC", "inf"))
    policy = VLABCPolicy(args.checkpoint, device=args.device)
    env = None
    results = []
    try:
        for powder in powders:
            if env is not None:
                env.close()
            env = GranularWeighEnv(images=True, max_steps=args.max_steps,
                                   powder_id=powder, grain_count=grain_count)
            for episode in range(args.episodes_per_powder):
                if time.monotonic() - started >= args.max_seconds or time.time() >= deadline - 60:
                    break
                seed = args.seed + 100 * powder + episode
                target = args.targets[episode % len(args.targets)]
                stem = f"powder_{powder}_seed_{seed}_target_{target:.2f}"
                obs, info = env.reset(seed=seed, options={"powder_id": powder, "target_g": target})
                # Pair stochastic policy draws as well as physical resets when
                # comparing frozen BC/RL checkpoints on the same episode.
                policy_noise_seed = (seed + 2000000) % (2**63 - 1)
                random.seed(policy_noise_seed)
                np.random.seed(policy_noise_seed % 2**32)
                torch.manual_seed(policy_noise_seed)
                policy.reset()
                trace, actions = [], []
                video = None
                if args.video:
                    import imageio.v2 as imageio
                    video = imageio.get_writer(args.output / f"{stem}.mp4", fps=2, codec="libx264")
                reason = "step_limit"
                clipped = 0
                peak_mass = 0.0
                episode_start = time.monotonic()
                try:
                    for step in range(args.max_steps):
                        if time.monotonic() - started >= args.max_seconds or time.time() >= deadline - 60:
                            reason = "wall_time_limit"
                            break
                        if video is not None and step % 5 == 0:
                            video.append_data(np.concatenate([obs[key] for key in ("top1", "left_wrist", "right_wrist")], axis=1))
                        action = policy.act(obs, env.instruction)
                        if action.shape != (14,) or not np.isfinite(action).all():
                            reason = "invalid_policy_action"
                            break
                        clipped += int(np.any((action < env.action_space.low) | (action > env.action_space.high)))
                        actions.append(action.copy())
                        obs, reward, terminated, truncated, info = env.step(action)
                        peak_mass = max(peak_mass, info["mass_g"])
                        trace.append({"step": step + 1, "mass_g": info["mass_g"], "reading_g": info["balance_g"],
                                      "spilled_g": info["spilled_g"], "subgoal_dones": info["subgoal_dones"].tolist()})
                        if step % 100 == 0:
                            print(json.dumps({"event": "evaluation_progress", "episode": stem, **trace[-1]}), flush=True)
                        if terminated or truncated:
                            reason = "success" if info["success"] else info["failure"] or "step_limit"
                            break
                finally:
                    if video is not None:
                        video.close()
                row = {
                    "powder_id": powder, "split": "heldout" if powder in heldout else "seen",
                    "seed": seed, "target_g": target, "success": bool(info["success"]),
                    "policy_noise_seed": policy_noise_seed,
                    "mass_g": info["mass_g"], "reading_g": info["balance_g"],
                    "absolute_mass_error_g": abs(info["mass_g"] - target),
                    "absolute_reading_error_g": abs(info["balance_g"] - target),
                    "peak_overshoot_g": max(0.0, peak_mass - target),
                    "spilled_g": info["spilled_g"], "subgoal_dones": info["subgoal_dones"].tolist(),
                    "simulated_seconds": info["elapsed_s"], "wall_seconds": time.monotonic() - episode_start,
                    "termination_reason": reason, "steps": len(actions), "clipped_action_steps": clipped,
                    "physics_validation": info["physics_validation"],
                    "powder_dynamics": info["powder_dynamics"],
                    "mass_force_error_g": info["mass_force_error_g"],
                    "max_grain_contact_penetration_m": info["max_grain_contact_penetration_m"],
                    "physical_valid": info["physical_valid"], "physics_checks": info["physics_checks"],
                    "max_contact_penetration_m": info["max_contact_penetration_m"],
                    "inference_precision": "bf16_autocast" if policy.bf16 else "fp32",
                    "completed_evaluation": reason != "wall_time_limit",
                }
                results.append(row)
                np.savez_compressed(args.output / f"{stem}_actions.npz", actions=np.asarray(actions))
                (args.output / f"{stem}_trace.json").write_text(json.dumps(trace) + "\n")
                with (args.output / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps({"event": "evaluation_episode", **row}), flush=True)
    finally:
        if env is not None:
            env.close()
    summary = {"checkpoint": str(args.checkpoint), "training_data_scope": manifest["use"],
               "environment": "scrape_v2", "scale_model": "physical_pan_force_with_instrument_lag_and_noise",
               "powder_dynamics": "physical_grains_v1", "grain_count": grain_count,
               "source_version": source_version, "powder_parameter_vectors": manifest["powder_parameter_vectors"],
               "powder_model_calibrated": False, "physical_validation_scope": "rigid_bodies_and_grains",
               "requested_split": args.split, "completed_episodes": sum(r["completed_evaluation"] for r in results),
               "episodes": results, "elapsed_seconds": time.monotonic() - started}
    for split in ("seen", "heldout"):
        rows = [r for r in results if r["split"] == split and r["completed_evaluation"]]
        summary[split] = {"n": len(rows), "success_rate": float(np.mean([r["success"] for r in rows])) if rows else None,
                          "mean_absolute_mass_error_g": float(np.mean([r["absolute_mass_error_g"] for r in rows])) if rows else None}
    (args.output / "evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
