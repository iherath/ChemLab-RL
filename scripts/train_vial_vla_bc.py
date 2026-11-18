"""Bounded SmolVLA behavior cloning for the separate fixed-scene vial task."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from chemlab_rl.vial_vla_data import (
    ACTION_KEY, LEROBOT_VERSION, PRETRAINED_MODEL, PRETRAINED_REVISION, RESULT_SCOPE,
    VialHDFDataset, load_smolvla, prepare_manifest, manifest_state_contract, file_sha256,
)
from chemlab_rl.vla_bc import _save_checkpoint


def heldout_loss(policy, preprocessor, dataset, *, batch_size=8, samples=128, seed=314159,
                 device="cuda", bf16=False, deadline=None):
    """Fixed observations/noise seed; preserve all training RNG and mode state."""
    import torch
    from torch.utils.data import DataLoader, Subset
    indices = np.linspace(0, len(dataset) - 1, min(samples, len(dataset)), dtype=int).tolist()
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
           torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    training = policy.training
    total, count, valid_elements = 0.0, 0, 0
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        policy.eval()
        loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch in loader:
                if deadline is not None and time.time() >= deadline - 120:
                    return dict(completed=False, evaluated_samples=count, requested_samples=len(indices),
                                reason="absolute_deadline", seed=seed, indices=indices)
                size = batch[ACTION_KEY].shape[0]
                elements = int((~batch["action_is_pad"]).sum()) * batch[ACTION_KEY].shape[-1]
                with torch.autocast("cuda" if device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=bf16):
                    loss, _ = policy(preprocessor(batch))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite held-out flow loss")
                # Native SmolVLA mean loss divides by valid time/action entries.
                total += float(loss) * elements
                valid_elements += elements
                count += size
        return dict(completed=True, loss=total / valid_elements, evaluated_samples=count, valid_action_elements=valid_elements,
                    requested_samples=len(indices),
                    seed=seed, indices=indices, metric="supervised_flow_matching_loss",
                    scope="held-out demonstration observations in the same fixed scene; not task success")
    finally:
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if rng[3] is not None:
            torch.cuda.set_rng_state_all(rng[3])
        policy.train(training)


def verify_checkpoint(checkpoint, dataset_path, device="cuda"):
    """Roundtrip with identical observations and noise, independent of task success."""
    import tempfile
    import torch
    checkpoint = Path(checkpoint)
    manifest = json.loads((checkpoint / "data_manifest.json").read_text())
    manifest_state_contract(manifest)
    if file_sha256(dataset_path) != manifest["dataset_sha256"]:
        raise ValueError("Verification dataset hash differs")
    policy, pre, post = load_smolvla(manifest, device, str(checkpoint))
    policy.eval()
    dataset = VialHDFDataset(dataset_path, manifest, manifest["train_episodes"][:1], policy.config.chunk_size)
    item = dataset[0]
    dataset.close()

    def batch():
        return {key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
                for key, value in item.items() if key not in (ACTION_KEY, "action_is_pad")}

    generator = torch.Generator(device=device).manual_seed(1729)
    noise = torch.randn((1, policy.config.chunk_size, policy.config.max_action_dim), generator=generator, device=device)
    with torch.inference_mode():
        expected = post(policy.predict_action_chunk(pre(batch()), noise=noise.clone())).detach().cpu()
    with tempfile.TemporaryDirectory(prefix="vial-vla-roundtrip-") as temporary:
        policy.save_pretrained(temporary)
        other, pre2, post2 = load_smolvla(manifest, device, temporary)
        other.eval()
        with torch.inference_mode():
            actual = post2(other.predict_action_chunk(pre2(batch()), noise=noise.clone())).detach().cpu()
    if actual.shape != expected.shape or actual.shape[-1] != 14 or not torch.isfinite(actual).all():
        raise AssertionError("Invalid reloaded vial action shape/values")
    error = float((expected - actual).abs().max())
    if not torch.allclose(expected, actual, atol=1e-4, rtol=1e-5):
        raise AssertionError(f"Roundtrip changed vial actions by {error}")
    result = dict(roundtrip_passed=True, max_action_difference=error, action_shape=list(actual.shape), closed_loop_evaluated=False)
    (checkpoint / "roundtrip_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--verify-checkpoint", type=Path)
    parser.add_argument("--pretrained", default=PRETRAINED_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--max-seconds", type=float, default=1740)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--n-action-steps", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--heldout-samples", type=int, default=128)
    parser.add_argument("--split-json", type=Path, help="Explicit versioned train/heldout/excluded episode lists; no implicit data merge")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_checkpoint:
        print(json.dumps(verify_checkpoint(args.verify_checkpoint, args.dataset, args.device)), flush=True)
        return
    if args.output is None:
        parser.error("--output is required")
    if min(args.steps, args.max_seconds, args.batch_size, args.save_every, args.heldout_samples) <= 0 or args.workers < 0:
        parser.error("Training limits must be positive and workers nonnegative")
    if args.chunk_size != 50 or args.n_action_steps != 2:
        parser.error("This vial pilot uses a fixed50-step prediction chunk and2 executed controls")
    started = time.monotonic()
    deadline = float(os.environ["CHEMLAB_DEADLINE_UTC"]) if "CHEMLAB_DEADLINE_UTC" in os.environ else None
    if deadline is not None and deadline - time.time() <= 180:
        raise TimeoutError("Insufficient time for model load and checkpoint")
    split = json.loads(args.split_json.read_text()) if args.split_json else {}
    if split and set(split) != {"train_episodes", "heldout_episodes", "excluded_episodes"}:
        raise ValueError("Split JSON must contain exactly train_episodes, heldout_episodes, excluded_episodes")
    manifest = prepare_manifest(args.dataset, args.output, **split)
    print(json.dumps(dict(event="data_ready", use=manifest["use"], state_dim=15,
                          train_samples=sum(manifest["episode_lengths"][n] for n in manifest["train_episodes"]),
                          heldout_samples=sum(manifest["episode_lengths"][n] for n in manifest["heldout_episodes"]),
                          excluded=manifest["excluded_episodes"])), flush=True)
    if args.prepare_only:
        return
    if (args.output / "run_config.json").exists() and not args.resume:
        raise ValueError("Run output already exists; use a new run or explicit resume")
    import torch
    from torch.utils.data import DataLoader
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.bf16 and (not args.device.startswith("cuda") or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 requires a capable CUDA device")
    if args.resume and json.loads((args.resume / "data_manifest.json").read_text()) != manifest:
        raise ValueError("Resume data contract differs")
    policy, pre, post = load_smolvla(manifest, args.device, str(args.resume) if args.resume else args.pretrained,
                                    args.revision if not args.resume else None, args.chunk_size, args.n_action_steps)
    policy.train()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable VLA parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-2)
    training = VialHDFDataset(args.dataset, manifest, manifest["train_episodes"], args.chunk_size)
    validation = VialHDFDataset(args.dataset, manifest, manifest["heldout_episodes"], args.chunk_size)
    loader = DataLoader(training, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=args.device.startswith("cuda"), persistent_workers=args.workers > 0)
    step = 0
    if args.resume:
        state = torch.load(args.resume / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        if state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(lerobot_version=LEROBOT_VERSION, pretrained_revision=args.revision or PRETRAINED_REVISION,
                  parameters=sum(p.numel() for p in policy.parameters()), trainable_parameters=sum(p.numel() for p in trainable),
                  result_scope=RESULT_SCOPE, state_dim=15, loss="supervised_flow_matching", closed_loop_evaluated=False)
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(dict(event="model_ready", **config)), flush=True)
    evaluations = {}

    def checkpoint():
        if step not in evaluations:
            evaluation = heldout_loss(policy, pre, validation, batch_size=args.batch_size, samples=args.heldout_samples,
                                      device=args.device, bf16=args.bf16, deadline=deadline)
            evaluations[step] = evaluation
            print(json.dumps(dict(event="heldout_loss", step=step, **evaluation)), flush=True)
        target = _save_checkpoint(policy, pre, post, optimizer, manifest, args.output, step, time.monotonic() - started, config)
        (target / "heldout_evaluation.json").write_text(json.dumps(evaluations[step], indent=2) + "\n")
        with (args.output / "heldout_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(step=step, **evaluations[step])) + "\n")
        print(json.dumps(dict(event="checkpoint", path=str(target), step=step)), flush=True)
        return target

    # Save an initial model and its deterministic held-out metric for comparison.
    last_checkpoint = checkpoint()
    iterator = iter(loader)
    losses = []
    reason = "steps"
    try:
        while step < args.steps:
            if deadline is not None and time.time() >= deadline - 180:
                reason = "absolute_deadline"
                break
            if time.monotonic() - started >= args.max_seconds:
                reason = "time_limit"
                break
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            tick = time.monotonic()
            batch = pre(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=args.bf16):
                loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step{step}")
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(trainable, 10.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            row = dict(step=step, loss=float(loss.detach()), grad_norm=float(grad),
                       elapsed_seconds=time.monotonic() - started, step_seconds=time.monotonic() - tick)
            losses.append(row["loss"])
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if step == 1 or step % 20 == 0:
                print(json.dumps(row), flush=True)
            if step % args.save_every == 0:
                last_checkpoint = checkpoint()
        if step not in evaluations:
            last_checkpoint = checkpoint()
    finally:
        training.close()
        validation.close()
    result = dict(steps=step, requested_steps=args.steps, stop_reason=reason, checkpoint=str(last_checkpoint),
                  elapsed_seconds=time.monotonic() - started, result_scope=RESULT_SCOPE,
                  first_loss=losses[0] if losses else None, last_loss=losses[-1] if losses else None,
                  initial_heldout=evaluations.get(0), final_heldout=evaluations.get(step), closed_loop_evaluated=False)
    (args.output / "training_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
