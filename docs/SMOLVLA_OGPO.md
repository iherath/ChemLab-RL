# Experimental SmolVLA OGPO–VaDar integration

**Status: mathematical primitives, an online trainer with CPU contract tests,
and a successful real pretrained gradient smoke on an A100. No OGPO robot
fine-tuning run or weighing-policy improvement has been established.** The CPU
tests use a small stand-in actor. The
first A100 diagnostic loaded the real checkpoint but failed before backward
because the critic incorrectly assumed its actor state projection was frozen.
That assumption and a prefix-prefill issue were corrected. The second diagnostic
passed feature extraction but stopped at a 5.78e-5 full-chain log-density
discrepancy before backward. A CPU MLP reproduction traced this to strided chain
slices selecting differently rounded float32 Linear operations; the scorer now
uses contiguous slices matching the sampler. The third A100 diagnostic passed
with exactly zero sampled/re-scored log-density difference under the unchanged
1e-7 guard, then produced finite nonzero gradients. It used the legacy 14-D
checkpoint/data solely to verify the real model API; no optimizer step or robot
rollout ran. Results are in `docs/pilot/ogpo_gradient_smoke_003/`.
Failure evidence is in `docs/pilot/ogpo_gradient_smoke_001/` and
`docs/pilot/ogpo_gradient_smoke_002/`.

Implementation: `chemlab_rl/smolvla_ogpo.py`, `chemlab_rl/ogpo_training.py`, and
`scripts/train_ogpo_vadar.py`. Tests: `tests/test_smolvla_ogpo.py` and
`tests/test_ogpo_training.py`, plus `tests/test_ogpo_demo_replay.py`. This is a PyTorch port for a pretrained SmolVLA flow
model, not the repository's existing low-dimensional JAX vial-placement agent.

## Pinned primary sources

- OGPO paper, *Sample Efficient Full-Finetuning of Generative Control Policies*,
  [arXiv:2605.03065v4](https://arxiv.org/abs/2605.03065v4), particularly conservative
  advantages Eq. 4.5 and the stochastic-interpolant appendix.
- VaDar reference code at commit
  [`bbd629ab388136a71fac706f7558a31e90a3e2b6`](https://github.com/simchowitzlabpublic/vadar-public/tree/bbd629ab388136a71fac706f7558a31e90a3e2b6):
  `vadar/agents/modules/pg_helper.py`, `vadar/agents/modules/vadar_helper.py`,
  and `vadar/agents/ogpo_vadar.py`. The source copies inspected in this session
  were `/tmp/chemlab-pg_helper.py`, `/tmp/chemlab-vadar-helper.py`, and
  `/tmp/chemlab-vadar-agent.py`.
- LeRobot 0.7.0, commit
  [`410c095eacc0011d998a73383eb94abed2715af0`](https://github.com/huggingface/lerobot/tree/410c095eacc0011d998a73383eb94abed2715af0),
  `src/lerobot/policies/smolvla/modeling_smolvla.py`.
- Pretrained checkpoint `lerobot/smolvla_base`, revision
  `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`, as pinned by the BC manifest.

## Velocity convention and stochastic chain

Native SmolVLA uses time `t=1` at Gaussian noise and `t=0` at the action.
This port uses forward time `tau=0` at noise and `tau=1` at the action:

```text
v_forward(x, tau) = -v_smol(x, 1 - tau)
dt = 1 / K
tau_k = k / K, k = 0, ..., K - 1
b(x, tau) = v(x, tau) + 0.5 c² (tau v(x, tau) - x)
mean_k = x_k + dt b(x_k, tau_k)
variance_k = c² (1 - tau_k) dt
x_(k+1) ~ Normal(mean_k, variance_k I)
```

The correction follows `score=(tau*v-x)/(1-tau)` and diffusion
`sigma(tau)=c*sqrt(1-tau)`. Their factors cancel before numerical evaluation.
Every transition has positive variance, including the last (`c²/K²`). The
schedule never evaluates a transition beginning at `tau=1`. The correction
preserves ODE marginals only with an exact flow in continuous time; a learned
velocity and finite Euler steps introduce approximation error.

`SmolVLAVelocity` follows the pinned model's prefix/suffix attention path and
returns the negated native action projection. It recomputes the prefix rather
than sharing a mutable KV cache. Policy dropout stays disabled while gradients
remain enabled. The caller supplies the BC tokenizer/normalization output.
The CPU wrapper test uses a tiny stand-in; it does **not** verify the real
pretrained model's backward pass on a GPU.

## Likelihood and actor objective

Store the entire sampled tensor `[B,K+1,H,32]`, including unused action channels
and future actions beyond the executed horizon. Its augmented joint density is
the initial standard-normal density plus the sum of every transition log density
over all `K,H,32` coordinates. This is **not** the marginal density of the final
14-dimensional robot command.

`ChainSample.executed_actions(14, h)` is a deterministic projection. The caller
must unnormalize it with the same BC action statistics before stepping the
environment. Dropping auxiliary coordinates from the density, averaging logs
over time/channels, or replacing the full chain with a Gaussian action head
changes the estimator.

The current-policy density scores fixed, detached target-policy draws. The
initial prior cancels in the ratio. Logs accumulate in float64:

```text
ratio = exp(log p_current(chain | observation) - log p_target(chain | observation))
loss_actor = -mean(min(ratio*A, clip(ratio, 1-eps, 1+eps)*A))
```

Old log densities and critic advantages are detached. An overflowing ratio
raises an error; it is not silently log-clipped. High-dimensional full-chain
ratios may still have high variance or underflow. Conservative update sizes,
ratio diagnostics remain necessary. The trainer records min/max ratios, clipping
fraction, effective sample size, and maximum absolute log-ratio. Nonfinite loss
or gradient raises before that optimizer step; overflowing ratios also raise.
Scoring makes each stored time slice contiguous before the VLA forward, matching
the sampler's memory layout. This prevents a reproducible float32 Linear
kernel-rounding discrepancy; it does not relax likelihood precision or alter
which coordinates are summed. The real-model smoke records per-transition mean
errors in latent units and transition standard deviations, together
with their implied log-density/ratio differences.

## VaDar targets and conservative advantages

The three subgoal flags are latched and form an ordered completed prefix.
Next-future weighting matches the pinned reference, including the final-head
fallback:

| Completion flags | Next-future weights |
|---|---|
| `000` | `011` |
| `100` | `001` |
| `110` | `001` |
| `111` | `001` |

“All unfinished goals” instead describes current-future weighting; it is not
the default next-future rule. Non-prefix flags such as `010` are rejected.
No subgoal flags enter the deployed actor observation.

Given target-critic values `[B,N,M,G]`, first scalarize with these weights over
goals. Then subtract each critic member's mean across the `N` freshly sampled
candidates. If every member's centered advantage is positive, take the smallest;
if every member is negative, take the largest; otherwise use zero. There is no
group-standard-deviation normalization. Reversing goal aggregation and ensemble
reduction generally changes the result.

For every replay transition, train **all** critic heads. With completion flags
after the actually executed chunk, ensemble-mean next values, cap `C=15`, reward
`5` for completed heads and `-1` otherwise:

```text
bootstrap_g = d_g*C + (1-d_g)*mean_m Q_target,m(next_observation, next_action, g)
y_g = min(reward_g + gamma_g^h * (1-terminated) * bootstrap_g, C)
y_g = C if terminated and d_g=1
```

`h` is the actual number of executed environment steps, including early endings.
A time-limit truncation uses `terminated=False` and bootstraps from the final
observation, not the reset observation. Completed terminal heads target `C`
according to the reference-code convention. The critic loss sums squared errors
over heads, then averages over the replay batch and ensemble.

## Explicit deviations from the reference implementation

| Area | This port | Pinned reference options |
|---|---|---|
| Transition noise | Brownian variance `c²(1-tau)dt` | Sampler passes scheduled sigma directly as Normal std, without `sqrt(dt)` |
| Noise support | Untruncated Gaussian | Samples can be clipped to `mean ± 3 sigma` while Normal density is scored |
| Final transition | Stochastic, positive variance, included in density | Constant-noise option makes the final step deterministic and omits its log density |
| Latent/action clipping | None in the chain sampler | Intermediate and final actions can be clipped |
| Ratio horizon | All steps and all latent coordinates | Partial fine-tuning horizon and normalization options exist |
| Actor | Differentiable pretrained SmolVLA velocity | JAX flow actors |
| VaDar structure | Sequential task-completion heads | Also supports DAG weightings and cost heads |
| Critic representation | Frozen contextual SmolVLM prefix mean-pooling, concatenated with normalized measured sensors, then independent vector MLP critics | Reference encoder/critic architectures and other optional critic forms |
| Trainable actor scope | Pretrained action expert and projection parameters according to the BC configuration; VLM frozen, actor state projection can train | Paper studies full fine-tuning; released code supports multiple configurations |
| Optional BC anchor | Off by default; when enabled, fixed initial-BC augmented-chain forward KL | Reference has several regularizers including a slow-reference policy and flow-matching variants; this fixed chain-KL option is not claimed equivalent |
| Demonstration replay | Off by default; optional exact-BC physical demonstrations seed the ordinary replay and TD critic | OGPO discusses both online-only and demonstration-augmented settings, BC rollout initialization, and TD/MC warmup; this importer is not its success-buffer or regularization recipe |

These are deliberate estimator choices. Results from this port must identify
them; it is not a bit-for-bit reproduction of the released training recipe.

## Implemented online lifecycle

`OGPOTrainer` initializes frozen actor/critic targets from the online modules.
Targets use `target <- tau*online + (1-tau)*target`, with distinct actor/critic
rates. A replay minibatch produces one all-head vector critic update, then its
critic EMA update. After the configured critic warmup, the trainer draws fresh
target-actor chains at those replay **states**, computes target-critic group
advantages, performs one actor update, and finally updates the actor EMA. The
target actor is fixed for the entire sampled candidate/update group. It never
uses stored behavior actions as PPO chain samples.

Actor sampling/scoring uses matching microbatches (default two candidates).
Advantages still use the complete candidate group. Each microbatch contributes
its correctly weighted share of the mean objective, then gradients accumulate
before one optimizer step and one EMA update. This avoids retaining all
`batch_size × group_size × flow_steps` VLA forward graphs simultaneously without
discarding any latent dimension or changing the group baseline. CPU tests compare
the accumulated gradient/update with the joint candidate objective. Real online
GPU memory/runtime for the ten-transition default has not been benchmarked.

`PersistentReplay` stores pre-action and pre-reset final observations, actual
physical commands, executed-horizon masks, terminal/truncation flags, and the
before/after latched subgoal vectors in a bounded SQLite database. Arrays use
NPZ with `allow_pickle=False`. Transactions are durable; oldest records are
evicted at capacity. Material IDs remain audit metadata and material 5 is
rejected from training replay. Actor inputs are allowlisted cameras, measured
14-D proprio, task language, and—by default—the separate measured scale channel.
The resulting 15-D state uses the same BC manifest/tokenizer/normalization.
Neither true powder mass, material parameters, nor subgoal labels enter the actor.

### Optional demonstration replay

`--demo-replay ACCEPTED_PHYSICAL.hdf5` explicitly changes the experiment to
**demonstration-seeded off-policy** training. It is off by default. The importer
requires the exact dataset SHA256 in the BC manifest, the successful five- or
30-episode merge audit, all training materials 0–4, no held-out episodes, the
15-D measured-scale contract, the current complete collection/source-asset hash
map, the current material bank, matching grain count and MuJoCo version, and
10-Hz controls. It reruns the physical episode audit, including conservation,
recorded collision tolerances, final force–mass agreement and successful ending.
These are file/audit consistency checks; they do not independently re-simulate
the recorded trajectory.

The importer additionally checks every adjacent camera pair, robot action
bounds, ordered and latched head flags, and the 0.1-second discount time base.
It stores nonoverlapping two-control chunks: `obs[start]`, the two original
physical commands, `next_obs[stop-1]`, head flags immediately before and after
the chunk, and the real ending. An odd terminal tail has horizon one and a
masked zero action suffix. Initial flags are `000`, as specified by the hashed
full closed-shield reset. It does not use the HDF reward array or invent latent
chains for behavior actions. The existing vector TD equations reconstruct the
reward from completion flags; actor updates still sample fresh target-policy
chains at replay states. Imported actor observations contain only the same
deployable cameras, proprio, measured scale, and instruction as online data.

Images stream in fixed windows of at most 32 controls, retaining HDF dataset
handles to avoid repeatedly opening gzip chunks. Every adjacent pair is checked,
including pairs across window boundaries. With three 224×224 RGB cameras, both
observation streams occupy at most 28.9 MB of explicit decoded window buffers;
this excludes HDF/compression working memory and model weights. Numerical
validation is bounded to one episode, capped by `--max-episode-steps`. The complete import
uses one SQLite transaction into an initially empty replay; capacity must fit
the complete seed. A failure even on the last transition rolls everything back.
The import report preserves the dataset hash, full physical/material provenance,
original source paths and file hashes, per-episode trajectory hashes, targets,
seeds and corresponding replay row ranges. It is written both to SQLite and
`demonstration_import.json`. Timing phases, encoded payload bytes and maximum
RGB window size are reported; per-episode progress explicitly remains pending
until the single transaction commits. Repeating the same import on resume is a no-op;
changed provenance is rejected. Resume must repeat `--demo-replay` because the
experiment mode and dataset hash are part of the trainer/replay provenance.

The current runner still collects an online chunk before each TD update. This
option supplies successful replay states for warmup; it does **not** add an
offline pretraining phase, MC targets, special success sampling or extra warmup
updates. Ordinary bounded replay can eventually evict demonstrations. A few TD
updates on successful data do not establish a calibrated long-horizon critic.

Bounded local storage measurements found that the original repeated-frame HDF
reads could exhaust a short job before training. The windowed importer and the
merger's independent RGB frame chunks address that I/O bottleneck. The estimated
image I/O plus serialization cost for five complete demonstrations is roughly
2.8–3.2 minutes at 2,000 controls/demo, or 6.9–8.1 minutes at 5,000 controls/demo,
excluding full numeric audits, hashes, model setup and exports. These combine
small local CPU measurements; they are not measured Modal import throughput.
The first 1,200-second engineering job therefore keeps the no-demo default;
optionally seeded runs use a separate ID and retain the same external cap.
Methods, memory/disk measurements and caveats are recorded in
`docs/pilot/replay_import_cost/README.md`. No legacy pixels from those storage
fixtures entered physical training replay.

`PhysicalActionAdapter` unnormalizes the generated two-action prefix with the BC
postprocessor and clips to robot actuator limits. Exactly this executable prefix
is used for both rollout and critic candidate scoring. Critic actions are scaled
by robot limits. A shortened chunk carries an explicit validity mask; ignored
commands cannot influence its Q value. All 50×32 latent coordinates still enter
the actor density. This deterministic projection does not create a tractable
marginal action density.

`FrozenSmolVLAFeatures` calls the pinned model's prefix-only contextual forward
path with a fresh `use_cache=True` prefill, discards that temporary cache,
mean-pools valid tokens, and appends normalized measured sensors. A fresh prefill
is required to route prefix-only inputs through self-attention in the released
cross-attention implementation. No cache is shared with actor likelihood calls.
The encoder shares only the frozen VLM and privately snapshots the small learned
BC state projection; it creates no third full VLA weight copy or changes to actor
trainability. LeRobot's `train_expert_only` freezes the VLM, while
`train_state_proj` independently controls the actor state projection. The real
diagnostic checkpoint sets the latter to true. These contracts follow the pinned
[VLM/expert implementation](https://github.com/huggingface/lerobot/blob/410c095eacc0011d998a73383eb94abed2715af0/src/lerobot/policies/smolvla/smolvlm_with_expert.py)
and [state projection implementation](https://github.com/huggingface/lerobot/blob/410c095eacc0011d998a73383eb94abed2715af0/src/lerobot/policies/smolvla/modeling_smolvla.py).
Frozen VLM parameters are checked; the real 14-D checkpoint feature/backward
path passed the third smoke. The corrected physical 15-D checkpoint still needs
its own verification once trained. Candidate groups
reuse these detached state features, but no actor chain or mutable KV cache is
reused. The MLP critics each have their own weights and three output heads.

Trainer checkpoints atomically save online/target/optional-anchor weights,
the private frozen critic state-projection snapshot, both Adam states,
dedicated sampling generator state, replay sampling RNG,
counters, action bounds, configuration, and provenance. Exact optimizer/sampler
resume requires the same replay revision. Simulator state is **not** serialized;
an online restart begins a new physical episode. The CLI exports the updated
actor in the existing BC evaluator's `save_pretrained` format.

The serial runner has environment-step, update-count and wall-time limits. Wall
time is checked between chunks/model calls; an in-flight call can exceed that
limit, so a remote job timeout must still bound the process. No job launches on
import. No success-buffer regularization, asynchronous rollout workers, or
published-paper hyperparameter reproduction is claimed.

## Task time scale and actual subgoals

Current environment heads are: (1) **any** shield door joint opens more than
80 mm, (2) actual contained receiver grain mass exceeds 50 mg after head 1,
and (3) target mass/scale tolerance plus physical force–mass agreement, low
spill, and the required settling interval are satisfied. Head 1 does not mean
all doors are fully open. Labels are latched. There is no reward term based on
hidden material parameters, spatula height, or invented transfer mass.

Default math remains the pinned convention: `r_done=5`, `r_step=-1` once per
executed chunk, cap 15, and discount 0.99 **per 0.1-second control**. A two-control
chunk bootstraps with `0.99**2`. This can make very long tasks difficult: after
2,000 controls, `0.99**2000` is about 1.9e-9. The upper cap does not prevent large
negative step-cost values; distant candidate actions can become hard to separate.
The defaults have not been tuned for a roughly 200-second bimanual sequence.

The first physical engineering pilot explicitly uses **discount 0.999 per
control, `r_step=0`, `r_done=5`, cap 15**. The discount is supported as a
long-horizon choice by the OGPO paper's ToolHang (1,000 controls) and Transport
(800 controls) experiments, which use 0.999; their action chunks are eight, not
our two. At 10 Hz, 0.999 has a discounted half-life of about 69.3 seconds and
retains 0.135 of terminal value after 2,000 controls, or 0.00672 after the
5,000-control pilot episode limit. A cap-15 terminal contribution is thus approximately
2.03 or 0.101, respectively, before considering intermediate head completions.
This is a task-scale rationale, not evidence that these values are optimal.

Zero unfinished-head cost is an explicit **reward deviation** from the pinned
VaDar convention. Physical collision failures terminate the episode. With
negative time costs and zero bootstrap on true failure, ending immediately can
avoid a long stream of penalties. For example, a cost of -0.01 per control with
gamma 0.999 gives approximately -6.62 for a final-head success 2,000 controls
away, versus -0.01 for an immediate unfinished failure. Merely rescaling a
negative cost does not generally remove that incentive. Zero cost removes this
particular incentive; it does not supply dense progress rewards, prevent critic
extrapolation error, or guarantee learning. A future negative-cost design would
need an explicitly justified absorbing-failure value/tail before use.

CLI options expose `--discount`, `--r-step`, `--r-done` and `--value-cap`. The
optional `--step-reward-per-control` replaces the unfinished-head chunk cost
with `cost * sum(gamma**i for i in range(actual_h))`; it stays **unset** in this
pilot. Completion rewards, next-future weighting, all-head targets, true terminal
versus time-limit treatment, actual-horizon discounting, and cap 15 are retained.
The optional BC anchor also stays off. Mathematical CLI defaults remain the
reference values, so the pilot command must specify these task choices.

## First physical online pilot, recorded before launch

The prerequisite is the accepted physical-data BC5 checkpoint and its data
manifest, following `PILOT_PROTOCOL_2026-11-18.md`: 200 grains, five training
materials, 0.5 g target, three cameras, measured 15-D state, 50 predicted controls
and two executed at 10 Hz. The five-demo BC budget is 2,000 updates/30 minutes;
the separate thirty-demo baseline is 5,000 updates/60 minutes. A lower BC loss
or the old 14-D gradient diagnostic does not establish full-task control.

Start with the bounded **optimizer/integration check** below. It uses batch 2,
group 2, two denoising transitions and actor microbatch 2; one critic-only warmup
update precedes at most three actor updates. The runner stops at the first of
20 environment controls, four updates, or 600 seconds after entering the runner.
Use a 1,200-second external process cap to include model load, optional replay
import, environment construction and final saves. The inner clock does not
bound those stages or interrupt an in-flight simulator/model operation.

`chem_rl_sim/.venv/bin/python scripts/train_ogpo_vadar.py --checkpoint PHYSICAL_BC5 --output ONLINE_ENGINEERING --device cuda --grain-count 200 --powder-ids 0 --target-g 0.5 --max-episode-steps 5000 --batch-size 2 --group-size 2 --actor-microbatch-size 2 --flow-steps 2 --critic-warmup-updates 1 --discount 0.999 --r-step 0 --r-done 5 --value-cap 15 --bc-anchor-weight 0 --max-environment-steps 20 --max-updates 4 --max-seconds 600 --checkpoint-every 1`

The 5,000-step episode bound matches the physical demonstration collector. The
20-control integration budget above still ends the check much earlier.

The prepared Modal wrapper defaults to the same engineering configuration and
no demonstration replay, verifies the physical checkpoint/data contract before
the GPU job, and keeps a 1,200-second cap with no retries. The command below is
launch guidance, not a record that an online pilot has run:

`MODAL_CONFIG_PATH=/tmp/chemlab-modal.toml chem_rl_sim/.venv/bin/python -m modal run scripts/modal_train.py::ogpo_pilot --checkpoint-run PHYSICAL_BC5 --checkpoint-step 2000 --dataset BATCH-pilot5.hdf5 --run-id ogpo-engineering-001`

The wrapper's optional `--demo-replay` is a boolean selecting its already
verified dataset; the direct Python trainer's flag instead takes the HDF path.
Use a distinct run ID for that separately labeled experiment. `--mode learning`
selects the explicit longer profile with a 3,900-second cap; it does not silently
extend the engineering budget. These commands use the verified config path and
do not presume a profile name. The trainer writes `pipeline_timings.json` and
phase start/end logs for model load, environment construction, trainer setup,
optional import, online work and export, so setup time cannot be mistaken for
the 600-second online budget.

The 3,900-second learning profile is still an engineering pilot: its 65-minute
total cap is shorter than the roughly 90-minute runtime estimated for a complete
precise physical episode. Finishing this budget cannot establish full-task
completion or a success rate. Frozen evaluation has a separate one-A100 cap of
120 minutes and 5,000 controls; wall-time-limited or missing episode reports are
explicitly labeled incomplete.

The Modal demonstration batch keeps eight concurrent CPU containers by default.
An explicit `--parallelism 16` raises the shared collection/merge pool to at most
16 containers after the successful reference and five-demo pilot gates. Each
container requests and is limited to two CPU cores and 12 GiB, giving a maximum
of 32 CPU cores and 192 GiB at parallelism 16. Modal measures CPU in physical-core
units. The batch remains limited to 30 episodes, 120 minutes per collection call,
six hours overall, zero retries, and no GPU. Scheduling does not change seeds,
material splits, targets, or acceptance checks. [Modal resource limits](https://modal.com/docs/guide/resources),
[Modal autoscaling](https://modal.com/docs/guide/scale).

To run the separately labeled demonstration-seeded variant, append
`--demo-replay ACCEPTED_PHYSICAL_BC5.hdf5` to that single-line command. The
importer rejects legacy scripted data and any data/source/BC mismatch. No-demo
and seeded runs require separate output directories and separate reporting.

Only after finite losses/gradients, sensible ratios, a valid save/resume, and
measured runtime/memory for the corrected 15-D model should a longer pilot be
considered. A candidate is batch 2, group 4, ten flow transitions, microbatch 2,
100 critic warmup updates, at most 2,000 controls/1,000 updates/3,600 runner
seconds, and a 3,900-second external cap. This is a proposed bounded experiment,
not a completed throughput measurement or validated warmup length. Starting
that larger configuration requires a fresh output; exact resume cannot change
its sampling configuration. Record actual work completed under the cap.

| Candidate | B | Group N | Flow K | Approximate sample-forward work per critic + actor update |
|---|---:|---:|---:|---:|
| Initial engineering check | 2 | 2 | 2 | 20 |
| Conditional small learning pilot | 2 | 4 | 10 | 180 |
| Current general CLI defaults | 8 | 4 | 10 | 720 |

The count is `B*K*(1+2*N)`: next-action sampling, candidate sampling and candidate
scoring. It excludes rollout sampling, critic feature forwards, backpropagation
and simulator/rendering cost, so it is **not a latency estimate**. Microbatch 2
limits simultaneously retained actor forward graphs to about `2*K`, without
changing group-relative advantages or the full 50×32 likelihood. It does not
make the batch-8 configuration four times cheaper. Actual ten-transition GPU
memory/throughput remains unmeasured; the passing real-model smoke used K=2.

The recorded local 200-grain precise inert-hold benchmark took 1.96 wall seconds
for 0.1 simulated seconds, plus 13.45 seconds for construction, without camera
rendering. Multiplying that one hold by 2,000 controls gives about 65 minutes of
physics alone, an **extrapolation**, not a measured full task or Modal runtime.
The evidence is `docs/pilot/physics_validation/granular_transfer/robot_200_inert_precise.json`.
GPU forward speed cannot remove this CPU physics cost. The serial runner also
performs one update per collected chunk and stops when its update cap is met;
four updates are not a 2,000-control episode. These bounds can yield an incomplete
rollout, which must be reported as incomplete rather than a task success/failure
rate. Prioritize accepted physics, BC5 and its baseline rollout, then BC30; launch
a larger RL pilot only if its measured cost fits the remaining nightly budget.
No dollar cost or end-to-end speed is inferred from unmeasured GPU utilization.

The paper's larger recipes (for example batch 256, group 32 and ten critics)
are not reproduced by these budgets. Small groups, two MLP critics, the frozen
VLM feature representation, shortened engineering flow chain and optional demo
replay must remain explicit in any result. Improvement and held-out interpolation
claims require completed, matched baseline/updated-policy evaluations; neither
the gradient check nor a four-update run supplies those results.

## Runnable checks and entry point

Use a real BC checkpoint and its manifest-matching dataset for the gradient
smoke. This computes contextual critic features, samples/re-scores the complete
50×32 chain, verifies a finite nonzero pretrained actor gradient, and checks the
BC action unnormalizer. It performs no optimizer step or online rollout. Legacy
14-D diagnostic checkpoints are accepted only in this diagnostic mode.

`chem_rl_sim/.venv/bin/python scripts/train_ogpo_vadar.py --checkpoint CHECKPOINT_DIR --output OUTPUT_DIR --gradient-smoke --dataset DATASET.hdf5 --device cuda --flow-steps 2`

Online mode requires a corrected physical-grain BC checkpoint with the 15-D
measured-scale contract. It cycles specified training materials across episodes,
rejects held-out material 5, and always enables physical collision checks. Grain
count is explicit: 400 nominal-density 20 mg grains give 8 g; 200 give 4 g.
Other density variants have their corresponding physical initial mass. This is
not a runtime assignment of mass to match the desired target.

`chem_rl_sim/.venv/bin/python scripts/train_ogpo_vadar.py --checkpoint CHECKPOINT_DIR --output OUTPUT_DIR --device cuda --grain-count 200 --target-g 0.5 --max-episode-steps 5000 --discount 0.999 --r-step 0 --max-environment-steps 1000 --max-updates 500 --max-seconds 600`

The above bounded example will remain in critic warmup under the default 1,000
warmup updates; any deliberate shorter warmup must be explicit. It is not a
claim that a short job is enough to improve the VLA. A bounded online integration
run and held-out robot evaluation remain outstanding.

The completed real-model smoke used two denoising transitions and a latent
chain `[2,3,50,32]`, with critic features `[2,974]` and an action prefix
`[2,2,14]`. Sampled/recomputed means and log densities matched exactly. Gradient
norm was 21,752.17 across 155 tensors. VLM trainable parameters were zero; the
actor retained 31,680 trainable state-projection parameters and 99,880,992 total
trainable parameters. These measurements validate this interface/backward test,
not an optimizer update, the 10-transition default, or policy performance.

## Verification

`chem_rl_sim/.venv/bin/python -m pytest tests/test_smolvla_ogpo.py tests/test_ogpo_training.py -q`

Measured on 2026-11-18: **21 primitive, 21 trainer and 26 demo-import tests pass
(68 total)**. They cover independent Normal
densities, analytic score gradients, auxiliary dimensions, fixed target draws,
PPO clipping and gradient signs, next-future weighting, conservative reduction
order, TD terminal/truncation/horizon semantics, finite configuration rejection,
all-head critic supervision, and SmolVLA time/sign/dropout conventions. These
tests establish mathematical contracts only, not robot task performance. Trainer
tests additionally verify durable replay, split/input isolation, exact EMA and
optimizer/sampling resume, terminal versus time-limit bootstrap, actual-horizon
costs/masks, full-size fresh-chain actor updates, and bounded collection. Their
actor and contextual-prefix model are tiny test stand-ins; none loads real VLA
weights or evaluates a robot.
Demo-import tests use fabricated small HDF fixtures, not simulated expert
successes. They verify exact h=2 alignment, odd terminal tails, input/split
isolation, immutable source/BC contracts, complete audit provenance, idempotent
resume, bounded image-window reads and continuity at window boundaries, and rollback after late validation,
serialization or input-file-change failures.
The latest combined run passed **100 tests in 6.85 seconds on CPU**: the 68
OGPO/import tests plus 32 BC data/physical-merge tests, including lossless RGB
rechunking. No simulator rollout or GPU training was launched by this importer
implementation/check.
