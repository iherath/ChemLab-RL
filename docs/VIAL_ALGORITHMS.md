# SmolVLA comparison on native vial placement

This is a separate rigid-body vial-placement pilot. Its task is to open the
draft shield, move the 20 g vial onto the balance, and close the shield. It
does not measure powder-transfer performance or held-out powder generalization.
All methods begin from the same task-specific SmolVLA BC checkpoint.

## Methods and reported objectives

| Method | Actor training | Critic target | Action selection |
|---|---|---|---|
| BC, native ODE | None after BC | None | Native SmolVLA ODE sampler |
| BC, matched stochastic sampler | None after BC | None | Same K=10 stochastic flow chain as the RL actor |
| Frozen-BC best-of-N | BC actor stays fixed | Scalar discounted native success | Draw N=8 BC chunks and choose the highest mean target-critic value |
| Scalar OGPO | Full-chain clipped likelihood objective | Scalar discounted native success | One chunk from the EMA target actor |
| VaDar + OGPO | Same OGPO objective with next-future value aggregation | Four auxiliary subgoal targets | One chunk from the EMA target actor |

[Lecture 13, page 34](../../Lectures+recitations/lec-13-gcps.pdf), explicitly
calls critic-ranked best-of-N sampling "Q-chunking" or "BoN." The implementation
here is a frozen-BC BoN baseline following that mechanism. Its Bellman targets
also select the best of N candidates, matching its rollout policy. The lecture's
generic displayed target samples from the policy; using greedy BoN in the
bootstrap is an explicit implementation choice. No full reproduction of a
separate Q-chunking paper is claimed.

The native environment reward is zero until the first successful control,
then one. All methods use this same success definition for evaluation. Native
success checks vial position on the pan, uprightness, the measured reading,
and all three doors. A successful transition terminates immediately.

For scalar BoN and scalar OGPO, the two-control target is
`sum_i gamma^i r_i + gamma^h (1 - terminated) mean(Q_target(s_next, a_next))`,
where `h` is the actual number of executed controls. A terminal on the first
control has `h=1`. Time-limit truncations bootstrap from their final observation.
The discount defaults to 0.999 per control, matching the 20 Hz time scale.

VaDar does not replace that environment reward with a sum of stage bonuses.
It learns auxiliary reaching values for open door, grasp/lift vial, place vial,
and close shield. The implementation reuses the verified reference target:
completed heads use reward +5 and an absorbing value cap of 15; unfinished
heads use -1 per executed chunk. Bootstrap discounts use the actual chunk
length. A completed terminal head targets 15. These auxiliary targets differ
from the scalar success objective by design and must be stated when reporting
the comparison. The implementation does not normalize the reference next-future
weights or silently describe the auxiliary sum as native task return.

| Completed prefix | Next-future weights |
|---|---|
| 0000 | 0111 |
| 1000 | 0011 |
| 1100 | 0001 |
| 1110 | 0001 |
| 1111 | 0001 |

The grasp/lift label is a measured gripper-closure and vial-height proxy, not
a contact-force certificate. Labels are used for critic targets and training
advantages. They never enter the deployed actor or its frozen image encoder.

## Shared model, action, and data contract

- Inputs: three RGB views, 14 measured robot joint coordinates, measured scale
  reading in grams, and the fixed task instruction. Vial pose and stage labels
  are excluded from actor inputs.
- Output: 14 normalized absolute actuator commands, mapped to the native
  simulator's actuator ranges. The saved action postprocessor is reused.
- Flow latent: the checkpoint's complete 50 by 32 tensor. Only the first two
  actions and 14 action channels are executed. K defaults to 10 transitions.
- Likelihood: the joint density of all augmented-chain coordinates, accumulated
  in float64. It is not a tractable marginal density of the two robot commands.
- OGPO advantages: draw fresh target-policy groups at replay observations,
  scalarize values, subtract each critic member's candidate-group mean, and
  keep only advantages whose sign is supported by every ensemble member.
- Critics: two independently initialized MLPs on frozen contextual SmolVLM
  features plus measured state and the executed action prefix. This differs
  from the reference architecture and large paper hyperparameters.
- Replay starts with training demonstrations only, then receives actual robot
  rollouts. With the original file, demos 0 through 6 train, demo 7 is held out,
  and demo 8 is excluded for excessive balance force during placement. Import
  retains 7,428 controls in 3,716 chunks, ending each trajectory at first success.
- Reset geometry is fixed. Different rollout seeds change policy noise, not
  initial vial position. The held-out trajectory is not a held-out physical scene.

The actor and critic use EMA targets. Online inference and final RL evaluation
use the same stochastic actor sampler. Native ODE BC results are kept separate
from the matched stochastic BC baseline. BoN pays for N candidates per decision;
report N, K, wall time, controls, and updates alongside success rates.

## Numerical checks and limits

FP32 model forwards are the default. `--bf16` enables matching CUDA autocast
in chain sampling, chain rescoring, and frozen feature extraction. Model/Adam
parameters remain FP32, and likelihood accumulation remains FP64. Every CLI run
first checks sampled versus recomputed full-chain log densities at identical
microbatch sizes; a discrepancy above 1e-7 stops the run. `--density-check-only`
performs this check without an optimizer step or robot rollout. It does not
establish that BF16 and FP32 policies have identical actions.

Full-chain likelihood ratios cover 16,000 latent coordinates at K=10 even
though only 28 command coordinates are executed. Actor updates can therefore
have substantial ratio variance. Learning rate defaults to 1e-6, PPO clipping
to 0.01, and updates log density-ratio and critic-consensus diagnostics. This
pilot does not implement OGPO+, extra BC regularization, or success-replay
oversampling. Sparse scalar rewards are rare in uniformly sampled replay, so
short critic warmup may produce weak BoN rankings. No improvement is assumed.

The original demonstrations lack the new collection pipeline's exhaustive
contact audit. Native contact dynamics and force-based scale readings are
retained. The simulator reports post-control penetration diagnostics; those
do not certify every physics substep in these historical demonstrations.

Additional collection audits each 2 ms step during recorded controls. It
rejects penetration above 50 µm for robot/static contacts, opposite-arm
contacts, and non-pad robot contacts with movable objects. This includes
forbidden wrist, camera-mount and gripper-shell contact with the shield.
Finger-pad contacts with movable objects and other allowed support contacts
have a 1 mm penetration limit. Absolute scale load above 100 g is allowed
only for a continuous interval of at most 0.1 seconds. The original sensor
reading is retained. Hybrid dataset validation requires the serialized audit,
matching source and trajectory hashes, finite extrema and complete substep
coverage. Historical episodes retain their original hashes and quality filter.

The bounded runner checks time between chunks and updates. Model loading,
demonstration import, density checks, and final evaluation require additional
time. The Modal launcher must supply an external absolute deadline.

## Implementation and verification

`chemlab_rl/vial_ogpo_training.py` contains replay, targets, optimizers, collection,
and evaluation. `scripts/train_vial_ogpo.py` exposes `bon`, `ogpo`, and `vadar`.
`--method ogpo --baseline-only` evaluates unchanged BC with the matched sampler.
All result files identify the sampler, precision, data hashes, simulation source,
method, objective, and available update/rollout counts.

Run these from the repository root in the project's CUDA/LeRobot environment.
Replace `/path/to/shared_bc/policy` with the saved vial BC checkpoint directory.
The dataset must have exactly the hash and train split recorded in that checkpoint.
Each output directory must be new unless using its own saved checkpoint with
`--resume`; the replay database is part of the resume state.

Matched stochastic BC baseline, without training:

```bash
python scripts/train_vial_ogpo.py --checkpoint /path/to/shared_bc/policy --dataset data/teleop/weigh_vial_grok_v2.hdf5 --output docs/pilot/vial_bc_matched --method ogpo --baseline-only --flow-steps 10 --eval-episodes 5 --eval-max-steps 1600
```

Real-model optimizer/robot smoke only, with deliberately incomplete rollouts:

```bash
python scripts/train_vial_ogpo.py --checkpoint /path/to/shared_bc/policy --dataset data/teleop/weigh_vial_grok_v2.hdf5 --output docs/pilot/vial_ogpo_smoke --method ogpo --batch-size 2 --group-size 2 --actor-microbatch-size 1 --critic-warmup-updates 1 --max-updates 3 --max-environment-steps 4 --max-seconds 600 --eval-max-steps 4
```

Example bounded scalar run after measuring the smoke's update cost. Set
`--method bon` or `--method vadar` in separate output directories for the other
variants; retain the shared BC checkpoint, seed, data split, and evaluation seeds.
BoN's N=8 target selection makes its critic updates more expensive.

```bash
python scripts/train_vial_ogpo.py --checkpoint /path/to/shared_bc/policy --dataset data/teleop/weigh_vial_grok_v2.hdf5 --output docs/pilot/vial_ogpo --method ogpo --seed 0 --critic-warmup-updates 500 --max-updates 1300 --max-environment-steps 1600 --max-seconds 3600 --flow-steps 10 --eval-episodes 5 --eval-max-steps 1600
```

The example is a budget, not a completion-time estimate. The runner logs
warmup duration, update duration, and sampled terminal counts. With seven
terminal rows among 3,716 chunks and batch size four, 100 uniform warmup updates
expect only 0.75 terminal draws; 500 expect 3.77 and 1,000 expect 7.53. Even
1,000 updates do not guarantee useful value propagation across the long task.

Focused CPU tests cover actual Adam updates, the frozen BoN
actor, scalar and vector targets, terminal/truncation handling, first-success
collection, checkpoint restoration, density consistency, and held-out exclusion.
These tests use a small actor and a contract environment. Separate A100 checks
also completed actual optimizer updates on the pretrained SmolVLA checkpoint:

| Verified smoke | Critic / actor updates | Online / evaluation controls | Wall seconds | Density error / ratios |
|---|---:|---:|---:|---|
| Scalar OGPO | 3 / 2 | 4 / 4 | 114.82 | 0 / exactly 1 |
| VaDar + OGPO | 3 / 2 | 4 / 4 | 109.36 | 0 / exactly 1 |

Both checks used FP32, a K = 10 flow chain, batch size 2 and group size 2,
after importing the same 7 training demonstrations. Their saved evidence is
[`smolvla-vial-optimizer-smoke-001`](pilot/smolvla-vial-optimizer-smoke-001/wrapper_result.json)
and [`smolvla-vial-vadar-smoke-001`](pilot/smolvla-vial-vadar-smoke-001/wrapper_result.json).
These demonstrate real SmolVLA loading, likelihood consistency, critic updates,
actor gradients and interaction with the native robot environment. Neither
short rollout completed a task episode; no full-task success rate or RL
improvement follows from these checks. Frozen-BC BoN has CPU coverage, while
a full real-model BoN rollout and the longer comparison remain unmeasured.

The artifacts retain the exact tested data and code hashes. Later handoff
edits tightened dataset-audit validation; the flow and optimizer implementation
was unchanged. A new run must honor the launcher's exact-hash smoke gate.

## Primary references

- [OGPO, arXiv:2605.03065v4](https://arxiv.org/abs/2605.03065v4).
- [Pinned official VaDar design](https://github.com/simchowitzlabpublic/vadar-public/blob/bbd629ab388136a71fac706f7558a31e90a3e2b6/docs/DESIGN.md),
  including the distinction between VaDar auxiliary targets and VaDar-Shaped.
- [Pinned VaDar target/weighting code](https://github.com/simchowitzlabpublic/vadar-public/blob/bbd629ab388136a71fac706f7558a31e90a3e2b6/vadar/agents/modules/vadar_helper.py).
- [Existing port derivation and source pins](SMOLVLA_OGPO.md).
