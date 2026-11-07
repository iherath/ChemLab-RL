# Paper OGPO + VaDaR on ChemVialWeighing

This is the CUDA path: [vadar-public](https://github.com/simchowitzlabpublic/vadar-public) (`--algo=ogpo_vadar`) on Leo’s vial env, not the K=1 Gaussian in `chemlab_rl/`.

We do not clone [OGPO_public](https://github.com/simchowitzlabpublic/OGPO_public). vadar-public **is** upstream OGPO (`simchowitzlabpublic/OGPO@c230a0e`, package renamed `ogpo` → `vadar`) plus VaDaR. `--algo=ogpo` is that OGPO agent; `--algo=ogpo_vadar` is the same flow/PPO code with the vector critic and next-future weights. OGPO_public has no `subgoal_dones` contract.

## VRAM

The OGPO authors launch low-dim ToolHang with **48 GB** (`#SBATCH --constraint=VRAM_48GB`, `jax_frac=0.45` → ~22 GB JAX heap, 32 eval envs, 10 Qs, 512⁴ MLPs, flow K=10, batch 256, action chunks of 8).

| GPU | VRAM | What to set | Notes |
| --- | --- | --- | --- |
| RTX 4090 / A5000 / L4 | **24 GB** | `VRAM_GB=24` (script default): `jax_frac=0.65`, 8 eval envs, 1e6 replay | Minimum that should fit **low-dim**. If JAX OOM, drop `n_eval_envs` to 4 or `jax_frac` to 0.55. |
| A6000 / L40 / A100 40GB | **40–48 GB** | `VRAM_GB=48`: paper defaults, 32 eval envs | Comfortable. This is the OGPO ToolHang size. |
| A100 80GB / H100 | 80 GB | not needed | Only for **image** OGPO or a VLM actor. Skip for this task. |

System RAM: **≥ 48 GB** (32 eval MuJoCo scenes of two UR7es). A 4090 pod with ~62 GB RAM is fine at 8 eval envs.

Rent a **24 GB 4090** first. If it OOMs during JIT of the PPO step, move to an **A6000 48 GB**.

## RunPod

1. Pod: Ubuntu + CUDA 12, **≥ 80 GB disk**, 24 or 48 GB GPU.
2. Clone this repo. Low-dim demos are in `data/teleop/weigh_vial_grok_v2_lowdim.hdf5` (the 267 MB image HDF5 stays gitignored). `runpod_setup.sh` clones vadar-public and chem_rl_sim.
3. Setup and train:

```bash
bash paper_ogpo/runpod_setup.sh
source paper_ogpo/.env.sh
# compile check (~minutes)
FAST=1 bash paper_ogpo/run_chem_vial.sh
# full paper run (BC 300k + critic warmup 20k + online 3M)
VRAM_GB=24 bash paper_ogpo/run_chem_vial.sh          # or VRAM_GB=48
```

`patch_vadar.py` registers `chem_vial-low_dim` in a local vadar-public checkout: `info['subgoal_dones']` of length 4, HDF5 loaded from `CHEMVIAL_DATASET`, Q bounds `[-1/(1-γ), 0]`.

Flags match the paper scripts: `ogpo_vadar`, `weighting=next_future`, `vadar_critic=vector`, flow K=10, horizon 8, fork pins (scalar time, interleaved Q warmup, 10 Qs, BON=8). The grok HDF5 is 9 successful trajectories; the paper’s robomimic PH sets are much larger, so BC will sit closer to memorization than ToolHang. The algorithm is the same.
