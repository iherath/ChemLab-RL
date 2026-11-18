# Vial placement: colleague training handoff

This experiment moves a rigid 20 g vial onto the balance and closes the draft shield. It is separate from the contact-driven powder-transfer task. The policy receives three 64×64 RGB cameras, 14 measured joints including gripper-driver angles, the measured scale reading, and a language instruction. Actions are 14 normalized absolute actuator targets at 20 Hz. SmolVLA predicts 50 controls and executes 2 before replanning.

The complete release is **98 quality episodes: 97 training and 1 held out**. There were 9 historical recordings; original `demo_8` is excluded for 22 instrument readings above 100 g, peaking at 3603.84 g during placement. The package retains historical `demo_0` through `demo_7` and adds 90 individually accepted native-physics recordings. Original `demo_7` remains held out. This is 98, not 100. A package may be assembled only after all 98 validated shards are present.

All episodes use the same initial scene. Held-out trajectory loss does not establish generalization to new resets, objects or powders. Historical episodes have no every-substep contact certificate. New accepted recordings carry their own native contact-audit metadata. The hybrid loader verifies every historical dataset byte against the audited original episode hashes and requires distinct provenance and physical acceptance for all90 new episodes. The standalone powder gates remain unchanged.

## Install and assemble

Use Python 3.12 and a CUDA-capable PyTorch installation for training. The tested cloud image uses LeRobot 0.7.0 and MuJoCo 3.16.0. The bootstrap script installs `requirements-vial.txt` and the pinned native simulation repositories into `.venv-vial`. The simulator also needs the system EGL/OpenGL libraries supplied by the Modal image. See [dataset and dependency setup](VIAL_DATASET.md) for the native commit identities and package format.

```sh
bash scripts/bootstrap_vial_training.sh
```

Activate that environment before the remaining commands:

```sh
source .venv-vial/bin/activate
```

The GitHub-friendly package stores one episode per HDF5 shard and includes checksums. Assemble locally; this preserves episode payloads but may produce a different whole-file HDF5 SHA. Training records the actual assembled file SHA.

```sh
python scripts/package_vial_dataset.py assemble --package-dir data/teleop/vial_placement_v1 --output data/teleop/weigh_vial_hybrid98.hdf5
```

Audit the package and fit normalization using retained training controls only:

```sh
python -m scripts.train_vial_vla_bc --dataset data/teleop/weigh_vial_hybrid98.hdf5 --split-json configs/vial_hybrid98_split.json --output runs/vial98-audit --prepare-only
```

Each episode stops at its first terminal inclusively. Subsequent historical success-labelled frames are excluded. Training batches cannot cross episode boundaries; padded controls do not contribute to the native SmolVLA loss. The loader never supplies object pose, stage labels, simulator state or operator goals to the actor.

## Behavior cloning

Start a new run directory. The following is a bounded 5000-update BC run on 97 training episodes; a time cap can produce fewer updates and is reported explicitly.

```sh
python -m scripts.train_vial_vla_bc --dataset data/teleop/weigh_vial_hybrid98.hdf5 --split-json configs/vial_hybrid98_split.json --output runs/smolvla-vial-bc97-001 --steps 5000 --batch-size 8 --max-seconds 3300 --save-every 500 --device cuda --bf16
```

The pinned base model is`lerobot/smolvla_base` revision`5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`. VLM weights are frozen; the action expert and configured projections are trained. Complete checkpoints include model weights, processors, optimizer/RNG state, the immutable data manifest and training configuration. Each checkpoint reports loss on the same128 evenly spaced held-out observations with seed314159. Evaluation restores training RNG state and weights losses by valid action elements. This is offline flow-matching loss, not rollout success.

Read`training_result.json` and`latest_checkpoint.txt`; use the actual completed checkpoint in subsequent commands. The examples below assume5000 updates completed. Validate save/reload deterministically:

```sh
python -m scripts.train_vial_vla_bc --dataset data/teleop/weigh_vial_hybrid98.hdf5 --verify-checkpoint runs/smolvla-vial-bc97-001/checkpoint_005000 --device cuda
```

For the historical baseline, the original file defaults to7 training episodes, held-out`demo_7`, and excluded`demo_8`, yielding7428 training and1011 held-out controls. Do not use the98-episode split with that original file.

## Compare policies from one BC checkpoint

Keep the same BC checkpoint, data SHA, native scene,20Hz controls,1600-control horizon and evaluation seed across comparisons. FP32 is the default for the stochastic actor likelihood path; a real sample/rescore check must pass at`1e-7` tolerance before training. No OGPO+ variant is included.

The native SmolVLA ODE baseline is reported separately from the matched stochasticK10 baseline. The native baseline writes a three-view video and reports the fraction of action elements clipped to[-1,1]. Contact metrics from evaluation are post-control diagnostics, not every-substep certification.

```sh
MUJOCO_GL=egl python -m scripts.evaluate_vial_vla --checkpoint runs/smolvla-vial-bc97-001/checkpoint_005000 --output runs/vial-bc-ode-eval-001 --max-steps 1600 --max-seconds 1080 --seed 1000
```

```sh
MUJOCO_GL=egl python -m scripts.train_vial_ogpo --checkpoint runs/smolvla-vial-bc97-001/checkpoint_005000 --dataset data/teleop/weigh_vial_hybrid98.hdf5 --output runs/vial-bc-stochastic-eval-001 --method ogpo --baseline-only --flow-steps 10 --eval-max-steps 1600 --seed 0
```

Before running learning comparisons, perform a real optimizer smoke using the sameK10 flow chain. This checks integration; four controls are not a complete task evaluation.

```sh
MUJOCO_GL=egl python -m scripts.train_vial_ogpo --checkpoint runs/smolvla-vial-bc97-001/checkpoint_005000 --dataset data/teleop/weigh_vial_hybrid98.hdf5 --output runs/vial-optimizer-smoke-001 --method ogpo --flow-steps 10 --batch-size 2 --group-size 2 --actor-microbatch-size 1 --critic-warmup-updates 1 --max-updates 3 --max-environment-steps 4 --eval-max-steps 4 --max-seconds 600 --checkpoint-every 1 --no-bf16
```

The methods are`bon`(frozen BC actor, learned scalar critic selecting among8 sampled chunks),`ogpo`(scalar success critic plus actor updates), and`vadar`(VaDar auxiliary vector critics plus OGPO actor updates). Each imports only the BC manifest's training episodes. They share the native sparse final-success reporting metric; VaDar's auxiliary targets are documented in the trainer and do not become actor observations. Run one method at a time on one GPU, or assign distinct GPUs explicitly.

```sh
MUJOCO_GL=egl python -m scripts.train_vial_ogpo --checkpoint runs/smolvla-vial-bc97-001/checkpoint_005000 --dataset data/teleop/weigh_vial_hybrid98.hdf5 --output runs/vial-bon-001 --method bon --flow-steps 10 --best-of-n 8 --batch-size 4 --group-size 4 --critic-warmup-updates 100 --max-updates 900 --max-environment-steps 1600 --max-seconds 3600 --eval-max-steps 1600 --seed 0
```

```sh
MUJOCO_GL=egl python -m scripts.train_vial_ogpo --checkpoint runs/smolvla-vial-bc97-001/checkpoint_005000 --dataset data/teleop/weigh_vial_hybrid98.hdf5 --output runs/vial-ogpo-001 --method ogpo --flow-steps 10 --best-of-n 8 --batch-size 4 --group-size 4 --critic-warmup-updates 100 --max-updates 900 --max-environment-steps 1600 --max-seconds 3600 --eval-max-steps 1600 --seed 0
```

```sh
MUJOCO_GL=egl python -m scripts.train_vial_ogpo --checkpoint runs/smolvla-vial-bc97-001/checkpoint_005000 --dataset data/teleop/weigh_vial_hybrid98.hdf5 --output runs/vial-vadar-001 --method vadar --flow-steps 10 --best-of-n 8 --batch-size 4 --group-size 4 --critic-warmup-updates 100 --max-updates 900 --max-environment-steps 1600 --max-seconds 3600 --eval-max-steps 1600 --seed 0
```

The direct RL CLI's `--max-seconds` covers its training runner. Model loading, demonstration import and final evaluation are additional. For an enforced total cloud budget, `scripts/modal_vial_rl.py` provides 15-minute optimizer smoke, 20-minute baseline, and 75-minute comparison functions. All use A100 80GB, 8 CPU cores and 32 GiB RAM, zero retries and an absolute caller deadline. Comparison launches at most 3 GPUs and requires a successful smoke artifact for identical checkpoint/data/algorithm/simulator hashes. It preserves every method's outcome independently. The smoke entrypoint accepts `--method ogpo`, `vadar`, or `bon`; the four-control smoke is explicitly an incomplete task rollout.

`scripts/modal_vial_train.py` supports`--split-json configs/vial_hybrid98_split.json` after the assembled file is uploaded. It verifies workspaceish70, dataset bytes and schema before training, uses one A100, and accepts at most60 minutes. Authentication is never included in the repository. Do not copy another person's credentials.

## Evidence and limits

The historical 7-episode BC run, `smolvla-vial-bc7-001`, completed all 2,000 updates in 512.25 seconds. Its fixed 128-observation held-out flow loss fell from 3.89086 to 0.068586. The saved checkpoint is `/runs/smolvla-vial-bc7-001/checkpoint_002000` in Modal volume `chemlab-rl-runs`. The immutable data audit is under `docs/pilot/vial_dataset_audit_001/`; the measured training result is `docs/pilot/smolvla-vial-bc7-001/training_result.json`. No learned-policy rollout was performed by this BC job.

Native replay of historical `demo_0` reproduced success after 1,002 controls in 30.34 wall seconds, with maximum replay joint error 2.38e-7 and final 20 g. See `docs/pilot/vial_replay_001/result.json`. This validates action/state replay, not the learned policy.

Actual completed update counts, held-out losses and checkpoint paths are in the run's`training_result.json`/Modal wrapper result. No learned-policy success rate or RL improvement should be inferred from offline BC loss. The powder experiment and its pending full physical demonstration acceptance remain separate.

The scalar OGPO optimizer smoke `smolvla-vial-optimizer-smoke-001` completed 3 critic and 2 actor updates on the real pretrained model. Its K10 sample/rescore density check passed with absolute error 0 and ratios exactly 1 in FP32. Evidence is in `docs/pilot/smolvla-vial-optimizer-smoke-001/wrapper_result.json`. This verifies the optimizer integration; its 4-control training/evaluation budgets do not measure full-task performance. The longer three-method comparison was not run during this handoff.
