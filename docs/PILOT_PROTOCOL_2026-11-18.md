# November 18 preliminary experiment protocol

This protocol is recorded before any accepted contact-driven robot dataset or
policy rollout result exists. The midnight report will distinguish completed
measurements from pending experiments. Failed attempts remain in the run records.

## Physical acceptance

Use the current MuJoCo contact-driven granular environment with 200 grains. Each
grain's mass follows its spherical volume and the material's synthetic density.
Only actuator commands may change the robot or material state after reset.
Initial packing rotation may depend on the recorded reset seed.

Every accepted demonstration must be a complete successful 200-grain robot
episode from reset, with no forbidden-contact failure, conserved inventory, a
settled receiving vial, and agreement within 0.010 g between contained mass and
tared pan force. Scheduling permission alone never establishes these results.
Record source hashes, material parameters, contact checks, solver statistics,
spills, and the measured balance reading. Maximum grain overlap is 0.300 mm;
this is an explicit numerical tolerance for 1.5 mm radius coarse grains, not a
claim of zero geometric overlap or real-powder calibration.

## Demonstrations and material split

- Training materials: IDs 0 through 4. Held-out material: ID 5.
- Material variation: density and sliding friction. The held-out vector is a
  distinct combination inside the training ranges, so this tests interpolation.
- First batch: five accepted demonstrations, one per training material, target
  0.5 g, seeds 1001 through 1005.
- Full batch: thirty accepted demonstrations, six per training material. Each
  material has two demonstrations at each of 0.3, 0.5, and 0.7 g. Seeds are
  1001 through 1030, assigned in the batch manifest.
- Collection defaults to eight workers and permits an explicit limit of thirty
  for the deadline. Each worker is limited to two physical CPU cores and 12 GiB;
  thirty workers therefore permit at most 60 cores and 360 GiB. Collection and
  merging share this pool. There are no automatic retries, with a
  120-minute limit per job, and a six-hour overall limit. Incomplete batches are
  not presented as a complete five- or thirty-demonstration experiment.

Before batch collection began, the user explicitly requested more compute to
meet the deadline while retaining the full physics. The initial acceleration
plan required a successful complete reference before starting the remaining
episodes. That reference instead stopped at its loading-position guard after
the complete approach, so it contributes no accepted demonstration.

The revised explicit `candidate_batch` mode can start after a saved-state
corrective continuation has validated the controller correction, loaded-tool
lift, and actual receiver transfer. Its source hashes must match current code;
its original failed parent and saved-state hashes are preserved. The unchanged
0.25 mm depth and 1 mm lateral loading guards must pass. The continuation must
transfer more than 0.050 g, settle, conserve inventory, spill zero grains, pass
contact checks, and show independent pan-force agreement within 0.010 g. It is
recorded as a partial-scope diagnostic and remains ineligible for training data.
It does not have to reach the complete task's 0.5 g target.

After that evidence is reviewed, all thirty new candidate episodes start from
the normal reset concurrently with a thirty-container pool. There is no
reference or partial-trajectory reuse. As soon as the first five individually
pass the complete episode audit, a shared CPU slot merges their pilot while
other candidates continue. A failure outside those five does not invalidate
their accepted pilot. Full30 publication requires every episode and the pilot
merge to pass. Failed attempts remain visible and are never retried. The
original reference-gated entrypoint remains available separately. Only the
scheduling prerequisite changes; full-episode physics acceptance, sample count,
material split, targets and seeds remain fixed. No batch launches while the
corrective continuation is incomplete or its source hash is stale.

At the published Function prices, a two-core/12-GiB worker costs approximately
$0.1902/hour. Thirty two-hour collections plus two one-hour merges cost
about $11.80 before startup overhead; a five-minute allowance for each call
brings the estimate to about $12.31 (allow $13). Running the same successful jobs concurrently
does not multiply their total compute duration. Starting all episodes before
the pilot finishes increases exposure to an early failure by at most about
$9.52 for the additional twenty-five two-hour jobs. These are compute estimates,
not a live invoice or confirmed remaining credit balance. The published Starter
limits of 100 containers and 10 concurrent GPUs accommodate this schedule;
workspace-specific overrides and actual allocation remain unverified. Scheduling
or quota failures are recorded without retries. Sources: [Modal pricing](https://modal.com/pricing)
and [resource limits](https://modal.com/docs/guide/resources).

The original thirty scripted demonstrations remain diagnostic-only. They do not
enter a physical-task BC dataset or count toward the replacement thirty.

## Behavior cloning

Use the pinned SmolVLA pretrained checkpoint recorded in the data/training
manifest. Freeze the vision-language backbone and train its action expert and
state projector. Inputs are three RGB cameras, 14 measured joint/gripper values,
the measured balance reading, and the target-mass instruction. Hidden contained
mass, material parameters, and subgoal flags are excluded from actor inputs.

The planned five-demo pilot uses 2,000 AdamW updates, batch size 8, learning rate
0.0001, weight decay 0.01, training seed 0, and a 30-minute wall-time limit.
The planned thirty-demo baseline starts from the same pretrained checkpoint and
uses 5,000 updates with the same settings and a 60-minute limit. Both predict
50 controls and execute two before replanning at 10 Hz. Save every 500 updates.
Record the actual completed update count if a deadline interrupts training.
Each accepted merge may start its corresponding BC run immediately. The two
distinct runs may overlap, using at most two A100-80GB BC slots, with the same
30- and 60-minute limits. Neither run is launched twice, and BC30 does not depend
on BC5 completing. Evaluation and RL compute have separate explicit budgets.

Use the final completed checkpoint for initial evaluation. A later change based
on training-material validation must be documented before held-out evaluation.
The five-demo pilot is an engineering result and is reported separately from the
thirty-demo baseline. Loss reduction alone does not establish control success.

## Rollout evaluation

First evaluate the five-demo policy on training material 0 at 0.5 g from the
complete closed-shield reset. Use a fresh seed outside the collection seed range.
After the full baseline is frozen, evaluate seen and held-out materials with the
same target list and fixed per-material seeds for every policy being compared.
Start with 0.5 g; expand to 0.3 and 0.7 g only as time allows. State the number of
completed episodes for every result, including a one-episode preliminary result.

The evaluator requires the checkpoint's recorded grain count, physical source
version, and material vectors to match the environment. Reset seeds and the
policy's stochastic noise seeds are paired across policies. A single rollout is
bounded at 5,000 controls and 120 minutes, including setup and saving time in
the remote wrapper.

Report task success within 0.05 g, true contained-mass error, measured-scale
error, spill, peak overshoot, failure stage, simulated duration, and wall time.
Invalid policy actions and contact violations are failures. A wall-time-truncated
rollout is incomplete; preserve and report it separately, without silently
dropping it from the requested evaluation count. Hold out material 5 from
training, normalization, replay, and checkpoint selection.

## RL and deadline

Run OGPO with VaDar only after a verified physical-data BC checkpoint exists.
The real-model likelihood/backward check has passed, but it is not an optimizer
or online-learning result. Task reward/discount settings and bounded online
budgets are specified separately in `SMOLVLA_OGPO.md` before an RL launch.

The first physical-model integration check is online-only, capped at 20 controls,
four updates, and 1,200 external seconds. If it passes, an optional 3,900-second
learning pilot starts again from the BC5 checkpoint, with its exact accepted
demonstrations explicitly seeding replay. This uses batch 2, group 4, ten flow
transitions, 100 critic warmup updates, and caps of 2,000 controls/1,000 updates.
Demonstrations expose late task states before an online episode completes, but
uniform replay rarely samples final successes during a short warmup. This pilot
therefore measures optimizer behavior and runtime; improvement still requires
separate completed evaluations. Record nonzero advantages and actor gradient
norms as well as optimizer-call counts. Training uses stochastic EMA-actor
chains; the initial matched BC/RL evaluation uses exported current weights with
the same native SmolVLA sampler for both policies.

Simulation throughput is the main schedule risk. The report must retain the
completed physical and pipeline results even if full BC or RL evaluation cannot
finish before midnight. Do not infer improvement or generalization from missing,
unfinished, or diagnostic-only runs.
