# Project outline

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
