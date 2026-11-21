# BC400k checkpoint and training handoff

Get the `checkpoint_400000` folder from the team's Google Drive. It is the
original 400,000-update BC model, not the subsequently RL-fine-tuned policy.
Large checkpoint and dataset files are shared separately from Git.

## Setup

```sh
git fetch origin
git switch --track origin/codex/vial-bc400k-handoff
bash scripts/bootstrap_vial_training.sh
source .venv-vial/bin/activate
```

Place the complete downloaded folder at `runs/shared_bc/checkpoint_400000`.
Keep `model.safetensors`, `config.json`, `data_manifest.json`, both policy
processor JSON files and their normalizer safetensors, `run_config.json`,
`heldout_evaluation.json`, and the `tokenizer/` directory together.
Keep `training_state.pt` too if continuing BC with its original Adam/RNG state.
Model weights occupy about 1.8 GB and BC optimizer/RNG state another 0.8 GB.
The original dataset is about 2.9 GB and should be placed at
`data/teleop/weigh_vial_hybrid98.hdf5`.
Python 3.12 and CUDA are recommended for training; bootstrap installs the pinned
native simulator, LeRobot 0.7.0 and MuJoCo 3.16.0. See [setup details](VIAL_VLA_RUNBOOK.md).

Use the **original shared dataset for RL**. Reassembling repository shards preserves
episode payloads but can change the HDF5 byte hash. RL requires the original
checkpoint's exact source bytes:
`eb3a946882e872f8ebdccbb3e6d4049413ef03b746af5a523797bfd7e1b1d853`.
The split remains 97 training episodes / one held-out episode (`demo_7`).

## Evaluate on CUDA

```sh
MUJOCO_GL=egl python -m scripts.evaluate_vial_vla \
  --checkpoint runs/shared_bc/checkpoint_400000 \
  --output runs/bc400k-seed1000 --seed 1000 --max-steps 1600 \
  --n-action-steps 25
```

**Execute 25 actions before replanning.** The policy predicts 50 controls at
20 Hz; 25-action execution replans every 1.25 seconds. The saved BC config
retains its original two-action setting, so the explicit flag is essential.
At seed 1000, two-action execution stalled; ten opened the shield without
completing placement; 25 completed the native task. On sampling seeds 1001–1010,
native-ODE BC achieved 7/10 successes with 25 actions and 0/10 with ten actions.
The scene is fixed; these results do not establish scene generalization.
Best fixed held-out flow loss: 0.0091718404 at 400k updates.

## Fine-tune from BC

Start with a bounded pilot before increasing the update budget:

```sh
MUJOCO_GL=egl OMP_NUM_THREADS=1 python -m scripts.train_vial_ogpo \
  --checkpoint runs/shared_bc/checkpoint_400000 \
  --dataset data/teleop/weigh_vial_hybrid98.hdf5 \
  --output runs/vadar25-check --method vadar --action-steps 25 \
  --flow-steps 10 --actor-lr 1e-9 --critic-lr 3e-4 \
  --batch-size 2 --group-size 2 --actor-microbatch-size 2 \
  --critic-warmup-updates 100 --max-updates 228 \
  --max-environment-steps 3200 --max-seconds 900 --checkpoint-every 50 \
  --eval-episodes 10 --eval-max-steps 1600 --seed 1 --cache-prefix --no-bf16
```

The critic/replay dimensions, imported demonstration chunks, collection and
evaluation all follow `--action-steps`. Partial terminal chunks retain their
actual lengths and masked suffixes. Discounting uses `gamma ** actual_controls`.
VaDar's auxiliary unfinished-subgoal cost remains -1 per chunk, following the
reference convention. Native reward remains zero before success and one on
first success. Compare against BC with the same stochastic sampler as well as
the native ODE sampler; changing samplers alone changes outcomes.

Continue BC with its original optimizer and normalization:

```sh
python -m scripts.train_vial_vla_bc \
  --dataset data/teleop/weigh_vial_hybrid98.hdf5 \
  --resume runs/shared_bc/checkpoint_400000 --output runs/bc400k-continued \
  --steps 425000 --batch-size 8 --save-every 5000 --max-seconds 7200 --device cuda --bf16
```

## Modal

Use your own Modal account. Set `MODAL_PROFILE` and `CHEMLAB_MODAL_WORKSPACE` to
your profile/workspace when using cloud launchers. Upload the checkpoint to
`chemlab-rl-runs` and the original HDF5 to `chemlab-rl-data` in that workspace.
Historical experiment launchers contain source run IDs from the original
workspace; adjust those paths before using them. Direct commands above accept
arbitrary checkpoint paths and are the portable entry points. No credentials
are included.

Relevant code: `scripts/train_vial_bc_distributed.py`,
`scripts/evaluate_vial_vla.py`, `scripts/train_vial_ogpo.py`,
`chemlab_rl/vial_ogpo_training.py`, `chemlab_rl/smolvla_ogpo.py`, and the
`modal_vial_vadar25_pilot.py` / `modal_vial_vadar25_full.py` experiment launchers.
The full launcher automatically recovers complete model/optimizer/replay
bundles at worker boundaries and targets 2,000 additional actor updates.
