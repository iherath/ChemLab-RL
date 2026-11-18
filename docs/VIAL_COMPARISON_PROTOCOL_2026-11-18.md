# Separate vial-placement experiment

This experiment runs alongside full powder weighing. It does not replace the
granular collector, its physical acceptance gates, or the held-out material study.
The user selected SmolVLA and requested 90 additional BC demonstrations.

## Task and data

The native MuJoCo task opens the top shield, physically grasps a rigid 20 g vial,
places it on the balance, and closes the shield. Native success checks pan
position, height, uprightness, all doors, and a 20 ± 1 g force reading. There is
no powder in this experiment. Controls run at 20 Hz with the original 2 ms
physics timestep. The evaluation horizon is 1,600 controls, or 80 seconds.

The existing `weigh_vial_grok_v2.hdf5` has nine unique trajectories and 11,002
controls. It includes top, left-wrist, and right-wrist RGB images at 64 × 64.
The similarly named `_lowdim` file removes images. Camera/action alignment was
checked across all nine episodes. The first 344 action frames are shared, and
all resets are identical. This is a fixed-scene feasibility experiment.

The initial BC split trains on demo_0 through demo_6 and validates on demo_7.
Demo_8 is explicitly excluded: 22 consecutive recorded frames exceed 100 g,
peaking at 3,603.84 g during placement. That is a force transient caused by
contact, not vial mass. Every old episode is truncated at its first terminal
transition, removing the 173 subsequent success-labelled frames.

Actor inputs are the three RGB views, 14 measured robot coordinates, the
measured scale reading, and a task instruction. Actions are the recorded 14
normalized absolute actuator targets. The commands already include gravity
compensation; replay must not add it again. Object pose and subgoal labels are
excluded from the actor input. Normalization uses training episodes only.

The 90 new demonstrations vary the grasp yaw, small grasp offsets, and motion
speed through continuous recipe parameters. Each candidate starts from the
native fixed scene and retains the original physical dynamics. Each is checked
after every physics substep, including contact penetration, finite dynamics,
and sustained balance load. A candidate is saved only after native task success.
Rejected candidates are preserved as audit records. No duplicate trajectory may
enter the merged data. The original validation trajectory stays outside training.

## Comparison

All learning methods start from the same frozen BC checkpoint and use the same
training demonstrations, observation/action contract, environment horizon, and
final-success metric. Track environment controls, update count, wall time, GPU
time, inference latency, success, final reading, and failure stage separately.

1. BC: the pretrained SmolVLA policy after supervised flow matching.
2. Critic best-of-N: a frozen BC actor proposes action chunks; a learned scalar
   critic ranks them. The critic's Bellman target uses the same best-of-N rule.
3. OGPO: scalar critic with a sparse final-success reward, plus likelihood-ratio
   actor updates over stochastic flow trajectories.
4. OGPO with VaDar: the same actor update with four auxiliary subgoal critics and
   the reference future-head aggregation. Its auxiliary targets differ from
   scalar environment return; this difference is part of the method.

Use 50-action flow trajectories, execute two actions, and record the number of
flow steps and proposals. Compare native BC inference and the stochastic sampler
explicitly if they differ. Evaluation of RL checkpoints uses the same EMA
stochastic policy used during collection. OGPO+ is deferred to keep this a
focused test of critic selection, actor improvement, and value decomposition.

These are implementation ports. Terminology and differences from complete
published Q-chunking or OGPO recipes must be stated, including limited training,
critic ensemble size, candidate count, and demonstration-seeded replay. One
training seed and one fixed scene cannot establish robust algorithm superiority.

## Verified adapter result

Replaying existing demo_0 through the normalized action adapter reached native
success after 1,002 controls, exactly its recorded first-terminal index. The
final scale read 19.999999996 g. Maximum measured-joint deviation from the
recording was 2.38e-7 rad; execution and three-camera rendering took 30.34 seconds.
This is expert replay validation, not a learned-policy result. Evidence is in
`docs/pilot/vial_replay_001/result.json`.

Historical data did not record every-substep contact checks. The old replay's
successful outcome does not retroactively certify all of its contacts. New
collection checks identified that the old placement preset could push the vial
into the pan. The new expert computes the placement target from the measured
physical grasp and approaches the final height slowly; it never attaches or
teleports the vial.
