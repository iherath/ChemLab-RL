"""Frozen native-ODE SmolVLA evaluation for fixed-scene vial placement."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def evaluate(checkpoint, output, *, max_steps=1600, seed=1000, max_seconds=1080, device="cuda", video=True,
             n_action_steps=None):
    import torch
    from chemlab_rl.vial_vla_data import VialVLABCPolicy, file_sha256
    from chemlab_rl.vial_vla_env import VialVLAEnv, source_identity
    if not 1 <= max_steps <= 1600 or not 0 < max_seconds <= 1200:
        raise ValueError("Native vial evaluation is bounded to1600controls and20minutes")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "evaluation.json").exists():
        raise FileExistsError("Evaluation evidence already exists")
    started = time.monotonic()
    deadline = float(os.environ["CHEMLAB_DEADLINE_UTC"]) if "CHEMLAB_DEADLINE_UTC" in os.environ else None
    # Identical seed is used by the stochastic baseline's evaluation generator.
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = VialVLABCPolicy(checkpoint, device=device, bf16=False)
    if n_action_steps is not None:
        if type(n_action_steps) is not int or not 1 <= n_action_steps <= policy.policy.config.chunk_size:
            raise ValueError("Executed action count must be within the predicted chunk")
        policy.policy.config.n_action_steps = n_action_steps
    env = VialVLAEnv(max_steps=1600)
    writer = None
    rows = []
    clipped, action_values = 0, 0
    termination = "evaluation_budget"
    terminated = truncated = False
    max_door_opening = 0.
    try:
        observation, info = env.reset(seed=seed)
        policy.reset()
        # Model construction may consume random numbers before weights load.
        # Seed actual rollout noise after construction, like the stochastic evaluator.
        torch.manual_seed(seed)
        if video:
            import imageio.v2 as imageio
            writer = imageio.get_writer(output / "rollout.mp4", fps=20, codec="libx264", quality=7, macro_block_size=1)
        for step in range(max_steps):
            if time.monotonic() - started >= max_seconds or (deadline is not None and time.time() >= deadline - 15):
                termination = "time_limit"
                break
            if writer is not None:
                writer.append_data(np.concatenate([observation[k] for k in ("top1", "left_wrist", "right_wrist")], axis=1))
            action = np.asarray(policy.act(observation), dtype=np.float32)
            if action.shape != (14,) or not np.isfinite(action).all():
                raise FloatingPointError("Policy produced invalid normalized actuator targets")
            clipped += int(np.count_nonzero(np.abs(action) > 1))
            action_values += action.size
            action = np.clip(action, -1, 1)
            observation, reward, terminated, truncated, info = env.step(action)
            doors = np.asarray(observation['object'][8:11], dtype=float)
            max_door_opening = max(max_door_opening, float(doors.max()))
            row = dict(control=step + 1, reward=reward, success=bool(info["success"]),
                       scale_reading_g=float(observation["object"][7]), subgoals=info["subgoal_dones"].tolist(),
                       door_openings_m=doors.tolist(), max_door_opening_m=max_door_opening,
                       max_postcontrol_penetration_m=info["max_postcontrol_penetration_m"])
            rows.append(row)
            if step == 0 or (step + 1) % 100 == 0:
                print(json.dumps(dict(event="evaluation_progress", **row)), flush=True)
            if terminated or truncated:
                termination = "native_success" if info["success"] else "native_horizon" if truncated else "native_terminal"
                break
        if writer is not None:
            writer.append_data(np.concatenate([observation[k] for k in ("top1", "left_wrist", "right_wrist")], axis=1))
        complete = bool(terminated or truncated)
        result = dict(method="bc_native_ode", sampler="SmolVLA_native_ODE", precision="float32", fixed_scene=True,
                      checkpoint=str(checkpoint), checkpoint_manifest_sha256=file_sha256(Path(checkpoint) / "data_manifest.json"),
                      simulation_source=source_identity(), seed=seed,
                      chunk_size=policy.policy.config.chunk_size, n_action_steps=policy.policy.config.n_action_steps,
                      requested_controls=max_steps, executed_controls=len(rows),
                      completed_evaluation=complete, success=bool(info["success"]) if complete else False,
                      termination_reason=termination, elapsed_seconds=time.monotonic() - started,
                      scale_reading_g=float(observation["object"][7]), subgoal_dones=info["subgoal_dones"].tolist(),
                      max_door_opening_m=max_door_opening,
                      clipped_action_elements=clipped, action_elements=action_values,
                      clipped_action_fraction=clipped / max(action_values, 1),
                      contact_certification=info["contact_certification"],
                      max_postcontrol_penetration_m=info["max_postcontrol_penetration_m"],
                      generalization_claim=False, video="rollout.mp4" if video else None)
        (output / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
        (output / "trajectory.json").write_text(json.dumps(rows) + "\n")
        print(json.dumps(dict(event="result", **result)), flush=True)
        return result
    finally:
        if writer is not None:
            writer.close()
        env.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-steps", type=int, default=1600)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max-seconds", type=float, default=1080)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--n-action-steps", type=int, default=None)
    a = p.parse_args(argv)
    evaluate(a.checkpoint, a.output, max_steps=a.max_steps, seed=a.seed,
             max_seconds=a.max_seconds, device=a.device, video=not a.no_video,
             n_action_steps=a.n_action_steps)


if __name__ == "__main__":
    main()
