# Project outline

## Latest status: November 24, 2026

**Vial-placement QC completed at 175K updates with early stopping.** The selected 25K checkpoint and final 175K policy each achieved 20/20 on fresh stochastic rollout seeds in the same fixed scene. See [the final results](docs/VIAL_QC_FINAL_RESULTS_2026-11-24.md) for all plots, videos, checkpoints and limitations. There is no active QC job to resume.

**Subtask2 BC is pending its data audit.** The user requested a separate policy with a 500K maximum and early stopping. Branch `subtask2-solution-prep` at `1b5cb90` has no saved solution-prep demos and no successful full expert episode according to its latest notes. The user is checking with the colleague for the intended dataset/newer commit. [Audit and prerequisites](docs/SUBTASK2_BC_AUDIT_2026-11-24.md). No subtask2 BC compute has been launched.

## Current vial QC run: November 22, 2026

Michael completed BC400K training and supplied [the checkpoint handoff](docs/VIAL_BC400K_HANDOFF.md). The user subsequently authorized QC fine-tuning from that checkpoint, up to 500K updates with early stopping. QC run `vial-qc-bc400k-h100-main-20261122-003` reached 78,651 logged updates before an infrastructure restart exposed a coordinator recovery bug on November 23. The latest durable checkpoint is 75K. QC scored 10/10 at 25K, 50K and 75K, versus the 9/10 BF16 BC baseline. A later GPU fault interrupted training at 104,664 logged updates. It resumed from the 103,455 checkpoint in app `ap-luHK9PmImlzFeoQ1M8kLGz`, with 103,484 updates verified at 00:41 EST November 24. The 100K evaluation scored 9/10; the selected best remains 25K at 10/10. The original budget and algorithm are preserved. Read [the current QC live handoff](docs/VIAL_QC_LIVE_HANDOFF.md) for the active app, preserved budget, evaluation/video schedule, CPU plot finalizer, and monitoring/recovery details. Do not launch duplicate training. The older restriction below concerns retraining BC on the dataset, not this separately authorized QC run.

The November 18 powder handoff below remains the last recorded powder status; the vial result does not establish powder-weighing performance.

## New-agent handoff — November 18, 2026, 7:40 p.m. EST

**Start here. This handoff supersedes the original plan and older chronological statuses below.** Continue in this existing local checkout: `/Users/isuruherath/Desktop/CMU/Fall 2026/10-703 Deep RL/chemlab-rl-project`. A fresh GitHub clone does not contain the latest uncommitted powder-controller changes, diagnostic evidence, or midpoint report.

### Objective, authorization, and current status

The immediate objective is a physically valid full powder-weighing sequence, then SmolVLA behavior cloning and OGPO with VaDar, followed by evaluation on held-out synthetic material parameters. WAM comparison and SkillWeaver are deferred. Keep MuJoCo for this deadline. The user chose to retain full contact physics and report BC plus whatever RL completes. The midpoint report is due at **midnight EST on November 18**; do not promise completion by then without measured evidence.

**There are zero accepted complete powder demonstrations. No full-powder BC, online RL, or held-out-material evaluation has run.** The original 30 scripted powder episodes are diagnostic-only. Saved-state continuations and isolated fixtures below contribute zero training episodes. The actor/training pipeline exists, but the full robot's reliable scoop-and-pour controller remains the immediate blocker.

The independent vial-placement work is complete for this agent's assigned scope: 90 new image demonstrations plus 8 usable historical episodes, 98 total, with 97 training and 1 validation. Code and dataset are published to `iherath/ChemLab-RL/main` at `badeb254b6e881d35815da1034d202d5f9b781be`, verified against GitHub tree/blob hashes. All four new commits use Isuru's existing identity and have **no co-author trailers**. **Do not train on the expanded vial dataset: the user assigned that to a colleague.** Earlier BC on 7 episodes and short scalar-OGPO/VaDar optimizer checks are separately reported; they do not demonstrate full learned-policy task success. See [the vial runbook](docs/VIAL_VLA_RUNBOOK.md) and [dataset instructions](docs/VIAL_DATASET.md).

The user authorized bounded Modal work in workspace `ish70` using approximately $530 in credits and authorized publishing code/data without co-author attribution. Existing authorization need not be requested again. Never print credentials. All commands presented for the user to run must be on a single line.

### First actions for the incoming agent

1. Read this section, then [the physics takeover notes](docs/pilot/physics_validation/granular_scoop_servo_002/HANDOFF.md), and inspect `git status --short`. Preserve the dirty worktree and all failed-run evidence. The old physics workers received stop/acknowledgment-only instructions and interruption requests. Host checks found no active experiment or collection monitor at handoff; the agent registry still showed pending initialization, so do not interpret it as a completed worker acknowledgment. No fresh NoSlip3 pickup or corrected drain run has yet executed. Recheck process state before starting a duplicate job.
2. Diagnose test029's final 4.55 mm tool-position error against the unchanged 4 mm guard. Separate error already present on entry to the drain hold from subsequent drift. A bounded physical position correction under NoSlip3 is the next hypothesis, not a justification to widen the guard. The previous controller tuning under NoSlip0 did not solve actual tool slip.
3. Test a fresh physical pickup with NoSlip3 from the supported bench tool and held source. The historical loaded states have large tool-to-hand rotation from earlier slip. The route that works with those states may be unreachable with a fresh stable grasp. Validate actual full-scene clearance and transfer using the new measured grasp before broad collection.
4. Resolve/document the NoSlip3 density fixture's stricter 1 mg endpoint failure. The full transfer fixture passes, but production has **not** adopted NoSlip3. Keep the original failed result. Any solver change must be explicit, use fresh empty-receiver tare, preserve all physical checks, and record its actual compiled model and source.
5. Obtain a complete forward transfer, empty tool, no in-flight/spilled grains, return/retraction, and settled independent force agreement. Implement/package a real forward scheduling artifact against `_validate_loaded_transfer_feasibility` and the test contract in `tests/test_modal_launchers.py::loaded_transfer_fixture`. The existing `finalize_stage.py` produces an older pre-pickup schema and cannot package this new route. Review genuine captures and hashes before launch; never manufacture missing evidence. Each subsequent training demo must independently pass from reset. No passing scheduling artifact exists yet.
6. Launch bounded fresh candidates in parallel once those prerequisites pass; merge only accepted episodes, then start BC. Update the report with measured results. If time expires first, report the demonstrated scope and remaining failures accurately.

### Physical model and observations to preserve

- Production currently uses 200 dynamic spherical grains, radius 1.5 mm. At material0, each grain is 20 mg, density is 1414.710605 kg/m³, and the total is 4 g. Each grain's mass is density times volume. Transfer occurs through actual tool contact, gravity, and collision dynamics; inventory is derived from geometry. The tared pan force independently measures supported load.
- Production `chemlab_rl/granular_scene.py` is still pyramidal contacts, sparse CG, 400 iterations, tolerance `1e-8`, timestep `0.00025` s, implicitfast, **NoSlip0**, and grain contact dimension3. NoSlip3 and finger-contact dimension4 are diagnostic alternatives, not accepted production settings. Rolling/torsional grain coefficients are inactive under dimension3; do not claim validated cohesion or real-material calibration.
- No pinning, kinematic grasp attachments, sleeping grains, transition-time pose/velocity/mass assignments, or relaxed collision/penetration checks. Objects must remain supported or physically grasped. Diagnostic state restoration is initialization only and is never a full demonstration.
- The scoop is connected to its full 260 mm tool. Its cup has 14 mm outer diameter, 12 mm inner diameter, and 6 mm depth. The vial is 28 mm wide and 57 mm high; the faceted neck's minimum clearance is 21.911 mm. Cup-only clearance checks miss shaft, wrist, camera, held-source, and table collisions.
- Existing guards include 0.300 mm grain penetration, 4 mm tool tracking, 3 mm IK residual, 0.250 mm loading-depth error, and 1 mm lateral loading error. Settled mass/force agreement is within 10 mg. Scale settlement also requires stable inventory, no material in flight, support, quiet vertical motion, and force range within 5 mg over 0.5 s. The density fixture separately uses a stricter 1 mg endpoint check; distinguish it from the task criterion.
- The policy receives three camera views, measured joint/gripper state, the measured scale reading, and a target instruction. No true contained mass, material vector, or subgoal label enters the actor. The scripted expert may use privileged state for collection. Full-powder control is 10 Hz; SmolVLA predicts 50 actions and executes two before replanning. The scale model includes 0.25 s lag and 5 mg noise; raw force is used for validation.
- Training material IDs are 0–4; ID5 is held out for interpolation. The current study varies density and sliding friction. Pilot targets are 0.3/0.5/0.7 g, with the first prescribed five candidates at 0.5 g, one per training material. Larger original targets and unseen-material generalization are unverified.
- This initial task opens the shield, grasps/positions the source, picks up the spatula, performs repeated physical transfers, and settles at the target. Closing the shield, putting all tools away, and the remaining CV protocol are outside this experiment. A diagnostic pour's return/retraction is still required before declaring that transfer reusable for repeated scoops.

### Latest evidence and failed approaches

Paths in this table are under `docs/pilot/physics_validation/granular_scoop_servo_002/` unless otherwise stated. Read each `outcome.json`; absence of a final outcome does not mean success.

| Artifact | Measured result and implication |
| --- | --- |
| `pad_centered_pickup_suffix_002/` | Correcting the physical pad-patch placement by +8.5 mm along the handle enabled loading and withdrawal of 0.160 g, leaving 3.840 g in the source with zero spill. Peak grain penetration was 70.17 µm. The later pour failed IK. |
| `diagonal100_suffix_018/` | Reached the diagonal pour pose with 0.160 g. Rejected on 307.45 µm grain penetration against 300 µm. Real current-model captures exist, but this does not validate the historical prefix. |
| `diagonal88_suffix_020/` | Best settled robot partial transfer: 0.140 g received, raw force equivalent 0.1398901776 g (0.109822 mg error), 0.020 g retained, no spill or bare-pan grains, exact 4 g conservation. Peak penetration 262.828 µm. Failed on tool-grasp loss. The measured figure is `mass_force_diagnostic.png` beside its outcome. |
| `latched88_suffix_021/` | Holding the final joint command retained the grasp but exceeded the 4 mm tip guard. Its `drain_hold_c0_a0_k0.npz` includes scale tare/filter/noise state for short diagnostics; its receiver already contains powder, so it is not eligible as the initial state of the forward transfer gate. |
| `noslip0_stationary_027/` vs `noslip3_stationary_028/` | At the same drain state, NoSlip0 reached 18.37 mm drift and neck contact by 0.821 s. NoSlip3 held two seconds with 0.535 mm relative drift, 0.121° rotation, valid handle contacts, and no spill. This motivated test029. |
| `noslip3_transfer_029/` | Freshly settled empty-receiver tare under NoSlip3. Both grasps stayed intact and the tool emptied, but the test stopped at 4.55 mm drain tracking error. Receiver 0.140 g, in-flight 0.020 g, tool/spill0; peak penetration 280.429 µm. Force0.139269 g was not yet settled. Rejected after10.1 simulated seconds and315.6 wall seconds. |
| `noslip_force_fixtures_030/transfer_noslip3/` | Independent actuator-driven 200-grain fixture passed: 0.600 g received, settled raw reading0.599837 g, no retained/in-flight/spilled grains. The matched NoSlip0 fixture transferred0.500 g and measured0.500243 g. Solver settings change contact-driven yield, so prior loading yields must not be assumed unchanged. |
| `noslip_force_fixtures_030/density_noslip3/` | All four cases conserve and settle. Material2 measured0.070252827 g for0.069 g, failing the original1 mg endpoint threshold at1.253 mg error; the other three passed. Preserve this failure and consult the worker's final window/bias analysis. |
| `fresh_centered_condim4_022/` | Fresh pickup loaded0.100 g, then reported an approximately8 mm pad/handle penetration despite roughly50 µm physical motion. The worker audited the contact-distance discontinuity. This alternative is unadopted; read its final audit/handoff before drawing a conclusion. |

Do not repeat broad gain sweeps: live/frozen/predictive drain controllers under NoSlip0 all failed. Elliptic contact experiments and late-state dimension4 changes did not establish a stable grasp. The cup-opening rotation is real swing, not harmless axial spin. The selected low route uses absolute cup yaw150°, a diagonal outflow toward the open right side, an upstream12 mm pivot offset, height88 mm, and tilt94°. Its clearance was checked with historical measured grasps, so a new stable grasp still needs validation. Higher pours improved reach but worsened grain impact/spill. Relevant analysis is in `independent_pour_ik/` and `grasp_wrench_audit/`.

### Code, state restoration, and local tools

| File or directory | Responsibility |
| --- | --- |
| `scripts/collect_granular_demos.py` | Current physical expert: measured grasp, loading, reachable pour, drain, repeated batches. Uncommitted controller edits are intentional. |
| `scripts/collect_scrape_v2.py` | Native rigid-object approach and corrected `_pick_tool`; do not revert the measured pad-center offset. |
| `chemlab_rl/granular_env.py`, `granular_scene.py`, `granular_geometry.py`, `granular_material.py`, `granular_accounting.py` | Physical model, guards, inventory, force sensor, settlement, material split. |
| `docs/pilot/physics_validation/granular_scoop_servo_002/probe.py` | Current diagnostic entrypoint for pickup/pour continuations, with actual model/source/checkpoint captures. |
| Same directory: `drain_probe.py`, `solver_stationary.py`, `audit_boundary.py` | Short drift experiments and boundary evidence helpers. |
| Same directory: `finalize_stage.py` | Older pre-pickup stage producer. It does not implement the newer forward-only artifact contract; update or replace it before packaging that route. |
| `scripts/modal_collect_granular.py`, `tests/test_modal_launchers.py` | Uncommitted strict stage-gate implementation and tests; last verified70 tests passed. |
| `tests/test_granular_expert.py` | Current controller checks; last recorded27 tests passed. These do not replace dynamic validation. |
| `scripts/merge_granular_demos.py` | Strict acceptance/provenance merge; lossless one-frame RGB rechunking for random BC reads. |
| `chemlab_rl/vla_bc.py`, `scripts/train_vla_bc.py`, `scripts/modal_train.py`, `chemlab_rl/ogpo_training.py`, `scripts/train_ogpo_vadar.py` | Full-powder VLA/BC and OGPO implementation. The Modal BC wrapper invokes `chemlab_rl.vla_bc`. |

The parent source snapshot for trials020 and029 is archived with each run. Trial029 used `sha256:06d220596f3cf255f0481cba134bca97d07594d56356c688d27dd1107ff53035`; this is a historical identity, **not** permission to assume the current tree still matches. Freeze and recompute source identity before a new reviewed launch.

Useful restart boundaries are `docs/pilot/physics_validation/granular_regrasp_200/physical_raise80_speed06/final.npz` (source held, tool supported on bench), `pad_centered_pickup_suffix_002/loaded_and_lifted.npz` (under the scoop diagnostic directory), and `diagonal100_suffix_018/pre_pour_guard_c0_a0_k0.npz` (loaded tool, empty receiver; the actual test029 start). Restore full integration state, time, controls, warmstart, and controller metadata where present. Preserve tare/filter/RNG when continuing with existing received powder. A new solver needs an independently settled empty-receiver tare. Never tare away already received powder.

The ancestor CPU003 source archive was captured locally; its remote source hash was not independently verified. Historical002 lacks actual runtime model captures. Keep these provenance limits; do not reconstruct a fresh model and label it historical. Copies named `executed_probe.py` are archives: their relative `ROOT = ...parents[4]` can be wrong from the nested run folder, so use the current top-level probe or restore an archived source to a correctly rooted scratch location.

Local physics Python is `chem_rl_sim/.venv/bin/python` (MuJoCo3.16.0, Torch2.15). Root `.venv/bin/python` contains Modal1.6.4 and HDF5 tooling but no MuJoCo/Torch. Native imports may need `PYTHONPATH=.:chem_rl_sim:chem_rl_sim/third_party/urgantry_sim`. These are safe verification commands, each on one line, from the repository root:

```sh
PYTHONPATH=.:chem_rl_sim:chem_rl_sim/third_party/urgantry_sim chem_rl_sim/.venv/bin/python -m pytest tests/test_granular_expert.py -q
.venv/bin/python -m unittest tests.test_modal_launchers -q
.venv/bin/python docs/pilot/collection_bc_handoff_002/test_monitor.py -q
```

The full-reset local collector is `python -m scripts.collect_granular_demos`, using the physics interpreter and explicit `--grains 200` because its default is400. Its output, powder ID, seed, and target must be explicit. `--diagnostic` and `--approach-only` results cannot become accepted data. The current controller identifier is `guarded_complete_batch_v10_latched_drain`.

### Candidate collection, BC, and RL conditions

The prepared monitor is [collection_bc_handoff_002/monitor.py](docs/pilot/collection_bc_handoff_002/monitor.py); it has **never launched**. `scripts/modal_collect_granular.py::candidate_batch` accepts a separately reviewed stage artifact and collects30 **new, full-reset** candidates. No suffix or old reference is reused as data.

The two newer audit schemas are `saved_state_chained_corrective_continuation_v1` and `saved_state_loaded_transfer_feasibility_v1`; the validator also retains the older `saved_state_corrective_continuation` scope. The practical route for current historical states is the forward feasibility schema: loaded tool>50 mg, empty receiver/no spill/inflight at initialization, genuine production/initial/final compiled-model captures, exact native state audit, full forward transfer and retraction, conservation/contact checks, and settled force agreement. It explicitly records `historical_prefix_validated=false` and `accepted_demo_count=0`. The stronger chained schema cannot use prefix002 without missing historical captures. Source archives, actual model bytes, checkpoint hashes/times, and probe hashes must be genuine and agree with the currently adopted production model. Test029's NoSlip3 override cannot pass as production while production remains NoSlip0.

The monitor requires `--stage-validation` and `--expected-source-version`; use `--check-only` for validation before spending compute. It refuses an existing batch directory or `monitor_launch.lock`. Inspect prior state instead of deleting the lock or reusing run IDs blindly. Source identity is rechecked before each child launch; physical source and native assets are covered, while launchers, monitor, and docs are outside that physical-source digest.

Defaults are30 containers, each2 physical CPU cores/12 GiB;120-minute job limits and360-minute batch limit. The monitor hardcodes these limits. Review the remaining deadline before launch; the six-hour overall cap does not fit tonight if started late. No automatic retries. The first prescribed five (material0–4, target0.5 g) can merge while the other25 run; an arbitrary five successes do not satisfy that pilot. A later failure does not invalidate an accepted pilot. Full30 requires all30. BC waits for its own strict merge: BC5 is2000 updates/30-minute A100 budget; BC30 is5000/60 minutes. The two distinct datasets can train concurrently, once each. There is no evidence all full episodes fit120 minutes: the earlier failed approach alone took about one hour. Rendering was only about2.8% of measured runtime; hiding grains did not solve the physics cost.

Modal credentials are already configured in `/tmp/chemlab-modal.toml`; use `MODAL_CONFIG_PATH=/tmp/chemlab-modal.toml`. Do not add `--profile ish70` after a Modal entrypoint; that parsing attempt failed. Workspace verification is built into wrappers. Volumes are `chemlab-rl-data`, `chemlab-rl-runs`, and `chemlab-rl-model-cache`. No cloud powder collection or powder-training job is active at this handoff. Redirect SDK logs and redact access-token-like identifiers before showing them.

Powder OGPO primitives/trainer and real-model density/gradient checks exist. **A bounded full-powder online-RL Modal entrypoint still needs to be wired after genuine BC**; the existing wrapper supplies gradient smoke only. The trainer defaults are not this pilot: explicitly review200 grains, target0.5 g, discount0.999, and per-step reward0. The earlier rationale was that a negative per-step cost plus early collision termination can reward premature failure. Do not silently run the stale2.5 g/400-grain/discount0.99/per-step−1 defaults. See [SMOLVLA_OGPO.md](docs/SMOLVLA_OGPO.md). Algorithm comparison is BC, frozen-policy critic best-of8, scalar OGPO, and OGPO+VaDar; OGPO+ is deferred. Report optimization checks separately from full rollouts.

### Report, Git, and completion communication

The editable report is [report_body.tex](10-703_midpoint-report-template/report_body.tex), with [main.pdf](10-703_midpoint-report-template/main.pdf) as the latest built draft, six pages: five body pages plus references. Preserve `main.template.tex`. The abstract/body already describe the0.140 g robot partial transfer and the lack of complete powder demos. Michael Liang is credited for **BC and RL fine-tuning for vial placement**, as well as RealSense collection with Ron Sarma. Isuru leads this RL effort, Leo develops simulation, and Pranav Vignesh/Sarvesh Patil provide mentorship. Names/emails are in the report sources. Use plain writing without em dashes or canned contrasts. After meaningful changes, rebuild and visually inspect reflow.

From the repository root, the existing build command is:

```sh
cd '10-703_midpoint-report-template' && XDG_CACHE_HOME=/tmp/chemlab-tex-cache /tmp/chemlab-tectonic/tectonic main.tex
```

The report and most physics diagnostics remain local/untracked. The intentional tracked powder changes at handoff are `scripts/collect_granular_demos.py`, `scripts/modal_collect_granular.py`, `tests/test_granular_expert.py`, `tests/test_modal_launchers.py`, and `docs/SESSION_PROGRESS_2026-11-18.md`, plus this outline update. Do not blanket-add logs, credential files, or every diagnostic artifact. The published vial package is complete; avoid rewriting its history or retraining it. If publishing further authorized work, use the existing Git identity and no co-author trailers.

Tell the user when the **first complete demonstration passes** and when **full-powder BC actually starts**. Keep [the working record](docs/SESSION_PROGRESS_2026-11-18.md) and the report current. Distinguish genuine full-reset results from initialized diagnostics, and report failures and uncertainty directly.

## November 18 implementation update

The sections below preserve the original project plan. Today's audit supersedes
their descriptions of the powder transition and the original demonstrations as
training-ready. The original 30 episodes remain unchanged and are restricted to
diagnostics because their assigned mass, containment, and physical scale can
disagree, and their controller uses kinematic attachments.

The current weighing implementation uses frictional robot grasps and dynamic
coarse grains in MuJoCo. Grain mass follows volume and synthetic density; a tared
pan force sensor independently measures supported load. The actor also receives
the measured scale reading, as confirmed by Isuru. The active material study
varies density and sliding friction, with material IDs 0–4 for training and ID 5
held out for interpolation. This replaces the original eight-parameter transfer
rule for the current pilot.

The initial 200-grain collection targets are 0.3, 0.5, and 0.7 g. These smaller
targets support contact-model validation and early training; performance in the
original 1.3–4.3 g range has not been established. Closing the shield, returning
tools, and the remaining CV stages remain outside this initial three-subgoal
weighing experiment.

See [the live working record](docs/SESSION_PROGRESS_2026-11-18.md),
[the preliminary experiment protocol](docs/PILOT_PROTOCOL_2026-11-18.md), and
[the SmolVLA OGPO implementation notes](docs/SMOLVLA_OGPO.md) for current measured
results and launch conditions. Accepted full-task demonstrations and policy
success must be established before claiming BC or RL performance.

## Original project plan

The project is a VLM that switches short motor skills, with each skill an RL fine-tune of a base vision-language-action (VLA) policy. The research question is whether that split lets RL finish a 10–15 minute cyclic-voltammetry protocol. It is not whether a VLM can plan the assay, and it is not a single VaDar policy over the whole protocol.

## What stays outside the policy

Astra stays a planner. Nothing from it is distilled into the motor policy.

VaDar is what you would use if you wanted the sequence inside one policy: per-subgoal done signals, a vector Q, one subgoal-agnostic π. That has been shown out to about 1–1.5 minutes. A full CV run is an order of magnitude longer, and a real arm makes it longer still, so the sequence stays outside π.

## Stack

Cut the protocol into cartridges that each fit in that 1–1.5 minute window. The four stages already in the kinematic lab are the right kind of cut:

1. Weigh the solid.
2. Dispense electrolyte.
3. Seat the working, counter, and reference leads.
4. Run the sweep.

Weighing means scraping powder from a source container into the receiving vial until the scale is inside a mass band. The scripted demonstrations do that in Leo’s weigh scene. The training environment does not yet load those props or the transfer.

Each cartridge is its own vision-language-action policy, fine-tuned from the same base VLA:

- Collect at least 30 demonstrations of that subtask in the robot scene, with camera images and a language instruction.
- Behavior-clone the base VLA on those demonstrations.
- Fine-tune with OGPO + VaDar on the cartridge’s subgoal dones until the fine-tune beats that behavior-cloning checkpoint. Residual RL is the alternative when the base VLA already roughly executes the skill and only a small correction on a frozen trunk is needed.
- The deployed network maps images, proprioception, and the instruction to actions. It does not take a subgoal id. The VLM is what picks the cartridge.

The orchestrator is a VLM, called at stage boundaries rather than at control rate. GPT or Gemini if credits are available, otherwise a locally hosted Qwen3-8B. It sees the image and a short description of the latched stage, and it emits which cartridge to run until that stage’s done condition. It does not output joint commands, and its plans are not trained into the VLA.

## What to measure

The claim is: RL fine-tuning of a base VLA, with a VLM only choosing the cartridge, completes the long protocol.

On each cartridge, compare the base VLA, BC-only, and BC plus OGPO/VaDar or residual RL. End to end, compare an oracle stage index against the VLM switch. A flat VaDar run on the full 10–15 minutes is the negative control, and it is expected to fall outside the regime VaDar was validated on.

Full-protocol success is the number that answers the research question. Per-cartridge success is how you tell whether a miss came from a motor skill or from the switch.

## Weighing environment

Train in Leo’s MuJoCo bench (`chem_rl_sim`), not in Isaac Sim. The gantry, both grippers, the draft shield, the pan, and the receiving vial are already in the weigh layout. The spatula and the source vial are built in `scripts/collect_leo_scrape.py`. They are not yet in `LAYOUTS` in `chem_rl_sim/scene.py`, which is what the trainer loads.

The source is an open vial the same size as the receiving vial (57 mm tall, 28 mm across). The receiving vial starts on the pan. The left gripper opens the left draft door, carries the source in through the open top, and regrasps it through the left door so the mouth sits about 2 cm from the receiving vial. The right gripper opens the top and the right door, then scrapes down through the top opening. Front and rear panes stay shut. There is no stand.

Powder is a set of 0.02 g pucks, not a particle simulation. Mass is added while the spatula tip is over a mouth, and the LCD is the tared pan-force reading. The material vector is `flow_g_per_s`, `onset`, `delay_s`, `residual_g`, `lag_s`, `noise_g`, `spill_tilt`, `cohesion`. It is stored on each demo and is not part of the observation. Flow and cohesion change how fast mass is added. Lag and noise are recorded and are not applied to the LCD. The pan reading stops climbing near 4.5 g, so the kept targets run from 1.31 g to 4.36 g. On the heavier demos the accounted puck mass can sit above the scale reading. Success is the scale, within 0.05 g. The six powders in the file were not split into training and held-out sets.

`data/teleop/powder_weigh_bc.hdf5` is a tilt-only pilot of an older mass model. It has no arm, no images, and no spatula, so it is not the VLA training set. `data/teleop/scrape_vial_bc.hdf5` is the 30 scripted demonstrations, also stored one per file in `data/teleop/scrape_vial/` so they fit on GitHub: both arms, the balance, top1 and both wrist cameras, UR7e joint targets, the instruction, the done vector, and the powder vector. All 30 are the same scrape. None overshoot, and none pour the extra into the discard vial. `data/teleop/scrape_vial_bc.csv` is the per-demo table. Sample clips are `data/teleop/scrape_videos/weigh_full.mp4` and `demo_00_scoop.mp4`, `demo_05_scoop.mp4`, `demo_11_scoop.mp4`, and `demo_17_scoop.mp4`. The grok vial files only place a rigid 20 g vial. The RealSense bags are camera footage of the bench. They can be a visual reference for the spatula and the container. They have no actions.

VaDar’s dones for this cartridge, latched in order:

1. `open_door` — a draft-shield door is open.
2. `pour` — powder has started landing in the receiving vial.
3. `weigh` — the scrape has stopped and the reading has settled within 0.05 g of the target.

Closing the shield can be a fourth bit. The policy never receives these ids. `sparse_reward` turns the vector into one reward per critic head.

The VLA observation is the bench cameras, proprioception, and a language instruction that names the target mass (“scrape powder into the vial until the scale reads 2.50 g”). The action is the bimanual arm command. Targets stay in the band the pan can read, about 1.3 g to 4.3 g.

## Train the VLA

1. Move the spatula, the source vial, and the puck transfer from `scripts/collect_leo_scrape.py` into the weigh layout, and expose `info["subgoal_dones"]` with the three bits above, in the same shape `paper_ogpo/chem_vial_utils.py` already uses for the vial task.
2. Hold out at least one of the six powders before behavior cloning. The 30 demos cycle all six, so the held-out powder’s episodes have to come out of the cloning set. Targets stay in the 1.3–4.3 g band for both splits.
3. The warm-start set is the 30 successful demos in `data/teleop/scrape_vial_bc.hdf5` (30 total, not 30 per powder). Each one has images, proprioception, the instruction, actions, the done vector, and the powder vector. Failures were discarded. More demos, including a pour or an overshoot into the discard vial, can be added later. They are not required to start cloning.
4. Behavior-clone the base VLA on the training-powder demonstrations. This checkpoint is the supervised floor. The loss is action negative log-likelihood. No subgoal id enters the network.
5. Fine-tune with OGPO + VaDar on training powders only. Act with the target policy, store `(s, a, s', r, d(s))`, fit every critic head to the clipped TD target, and take the actor step on `Q_agg` with `weighting="next_future"`. The sum is the still-unfinished heads. Stop when the in-band rate on training powders beats the behavior-cloning checkpoint.
6. Freeze the policy. Evaluate on the held-out powders with no further updates. Record success, absolute mass error, overshoot, spilled mass, time, and which done bit failed.

The numbers to report come from `paper_ogpo/` (`ogpo_vadar`, flow policy, vector critic, `next_future`) once this environment is registered. `chemlab_rl/train.py` is the same loop with a one-step Gaussian, for checking that the dones, the reward, and the update run. It is not the VLA.

What this cartridge establishes is whether OGPO + VaDar, starting from behavior cloning, can hit a target mass for powders it was not trained on. Copy that recipe to electrolyte, lead seating, and the sweep. The VLM comparison waits until more than one cartridge works under an oracle stage index.

## Order of work

1. The first 30 scripted image demonstrations exist. Register the spatula, the source vial, the puck transfer, and the three subgoal dones on Leo’s weigh layout so the trainer can reset the same scene.
2. Hold one powder out of `data/teleop/scrape_vial_bc.hdf5` before cloning. The tilt-only HDF5 does not count.
3. Behavior-clone the base VLA on the remaining demos, then fine-tune with OGPO + VaDar until it beats that checkpoint on the training powders.
4. Evaluate the frozen policy on held-out powders.
5. Repeat the recipe for electrolyte, lead seating, and the sweep.
6. Add the VLM switch only after the cartridges work under an oracle index. That comparison is oracle stitching versus GPT or Gemini versus local Qwen3-8B, on full-protocol success.
7. A real arm comes after the sim protocol stitches. The cartridges stay separate on the arm.

## Where the current code sits

| Path | Role now |
| --- | --- |
| `chemlab_rl/env.py` | Kinematic sandbox of all four CV stages. Useful for the cartridge cut, not a robot simulator. |
| `chemlab_rl/vial_env.py` | Filled-vial placement over Leo’s MuJoCo env. Different skill from powder dispensing. |
| `chemlab_rl/powder_dispense.py` | Mass-accounting pour. The transfer model to attach to the scrape. Not the robot scene. |
| `data/teleop/powder_weigh_bc.hdf5` | Tilt-only pilot. Not the VLA training set. |
| `data/teleop/scrape_vial_bc.hdf5` | 30 scripted Leo-scene scrape demos. Vial-sized source, top1 and wrist images, UR7e joint targets, tared scale, done vector, powder vector. |
| `data/teleop/scrape_vial_bc.csv` | Per-demo table: approach, transfer, overshoot, correction, powder vector. |
| `paper_ogpo/` | Paper OGPO + VaDar (`ogpo_vadar`, flow policy, `next_future`). Register the new scene here for the reported run. |
| `chemlab_rl/train.py` | Same BC then OGPO + VaDar loop with a one-step Gaussian. Algorithm check, not the VLA. |
