# Weighing VLA pilot — 2026-11-18

## Objective

Fine-tune a pretrained VLA on bimanual powder weighing demonstrations, then
compare frozen BC and OGPO–VaDar policies on previously unseen synthetic powder
parameters. The initial compute allocation is $50–100 of the user's stated $530
Modal credits, in workspace `ish70`. SkillWeaver and WAM comparisons are deferred.

## Original-data audit

`data/teleop/scrape_vial_bc.hdf5` is preserved unchanged. It has 30 episodes,
five for each of six powder vectors, 46,442 frames, three 224×224 RGB views,
and 14-dimensional arm/gripper commands. Every episode has an identical action
sequence up to its termination. The final command change is at index 1,481.

The legacy collector directly changes mass and pins/latches objects outside
`step(action)`. It never demonstrates a target-dependent robot withdrawal to
stop dosing. The scale and accounted mass differ by as much as 1.964 g; only
14/30 final accounted masses are within 0.05 g of target. Eleven episodes have
reversing subgoal bits. Legacy observations are post-action snapshots, rather
than a verified pre-action transition dataset.

Consequently original-data BC runs are explicitly **diagnostic_only**. They test
the genuine pretrained model, loader, normalization, optimizer, checkpointing
and inference. A falling imitation loss is not full weighing success. The loader
requires `--allow-legacy-diagnostic` for this file and documents its approximate
one-frame action alignment. It never silently converts it into validated data.

The immutable audit/split/statistics manifest is in
`docs/pilot/legacy_data_audit/data_manifest.json`. Powder 5 is withheld from BC,
normalization and later online training. Fresh seen-powder simulation seeds,
not the held-out powder, must be used for checkpoint selection.

## Implementation

- `chemlab_rl/scrape_env.py`: full-reset bimanual environment, with shared
  action-triggered grasp and finite powder-transfer dynamics. The scale is
  explicitly synthetic mass accounting with lag and independent measurement
  noise; it is not the old puck contact-force sensor. A settled, stopped, clean
  in-band episode is required for success.
- `scripts/collect_scrape_v2.py`: corrected expert/collection and replay work.
  Full-task demonstrations must pass actual action replay before use.
- `chemlab_rl/vla_bc.py`: pretrained `lerobot/smolvla_base`, LeRobot 0.7.0,
  three cameras, 14-dimensional proprioception, target-bearing language,
  and 14-dimensional servo targets. Powder vectors and subgoal bits are not
  actor inputs. It freezes the VLM and trains the action expert.
- `scripts/evaluate_vla.py`: full-reset frozen-policy rollout evaluation, with
  success, true synthetic mass error, scale error, overshoot, spill, stage dones,
  seeds, targets, timings, traces, action arrays and optional videos.
- `scripts/modal_train.py`: one A100 80 GB, no configured retries, fixed absolute
  deadline across restarts, checkpoint/log volume and separate model cache.
  It checks the authenticated workspace before upload/compute.

Local simulation dependencies are already installed in
`chem_rl_sim/.venv/bin/python`. The separate root `.venv` contains Modal and
data-inspection tools. Modal credentials are in `/tmp/chemlab-modal.toml`, outside
the repository and source uploads, with a 24-hour lifetime and mode 0600.

## Gates and interpretation

1. Pretrained-weight load, finite forward/backward, checkpoint reload and bounded
   execution must pass before a longer BC job.
2. Corrected expert motion and transfer must work through the same environment
   interface the learner uses. An unsupported floating regrasp is not allowed.
3. Record pre-action observations and the exact executed action; check replay.
4. Measure BC in closed-loop rollouts. Report incomplete/failed episodes.
5. Start OGPO–VaDar only with a verified environment and BC actor integration.
   The existing low-dimensional JAX vial-placement launcher is not a VLA trainer.
6. Freeze the selected policy before evaluating held-out powders. Use the same
   targets/seeds and success conditions for subsequent methods.

SmolVLA is flow-based, but its BC interface alone does not supply OGPO's
stochastic denoising path likelihoods. That is an additional implementation and
verification step; it must not be replaced by a Gaussian action-head update and
reported as paper OGPO.

## Verified references

- SmolVLA: https://huggingface.co/docs/lerobot/smolvla
- Base checkpoint: https://huggingface.co/lerobot/smolvla_base
- LeRobot release: https://pypi.org/project/lerobot/0.7.0/
- Modal volumes: https://modal.com/docs/guide/volumes
- Modal timeout/retry semantics: https://modal.com/docs/guide/functions
- SkillWeaver release status: https://github.com/Ada4321/SkillWeaver#release-status

## Run outcomes

This section is updated from measured runs, not planned outcomes. At initial
setup, eight data-interface unit tests pass and the original dataset passes the
explicit diagnostic audit. No full weighing policy success has been established.
