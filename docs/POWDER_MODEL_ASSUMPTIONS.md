# Powder-model assumptions and limits

Audited on 2026-11-18 against `PowderState` and `ScrapeWeighEnv._geometry()` in
`chemlab_rl/scrape_env.py`, the visuals in `chemlab_rl/weigh_scene.py`, and the
original scoop collector in `scripts/collect_leo_scrape.py`.

**The current mapping from spatula pose to transferred powder is a hand-designed
synthetic surrogate. It is not a calibrated model of real scraping, granular
flow, adhesion, or weighing.** MuJoCo simulates robot and container contact;
separate scalar rules change the powder inventory and displayed scale reading.
Passing rigid-body contact checks does not validate those powder rules.

The surrogate can support an experiment about control and adaptation within its
specified parameter family. It cannot establish generalization to real powders,
or identify actual material properties, without measured calibration and held-out
physical validation.

The user's subsequent requirement is stricter: transferred quantity must arise
from tool/powder contact with known synthetic density and mass, and the scale
must accurately reflect that quantity. The height-gated surrogate documented
below does **not** meet that requirement. This document records the audit, not
approval to train on that model.

## What the robot actually controls

The actor issues 14 arm-joint/gripper targets at 10 Hz. Ordinary MuJoCo dynamics
move the arms, tool and containers. At the end of each action interval, the
environment measures the actual tool-tip and vial poses and evaluates geometric
gates. Powder transfer is then updated once using `dt=0.1 s`.

The powder update does **not** inspect scraping velocity, acceleration, contact
force, vibration, blade angle, swept volume, wall friction, particle size,
particle collisions, density or adhesion. Two trajectories with the same
sequence of sampled geometric gates produce the same true mass evolution,
regardless of their intervening motion. The random seed affects sensor noise.

The target mass does not enter `PowderState.step()`. It enters the instruction,
success test, and privileged scripted expert's controller. The expert can use
true mass and the parameter vector; the VLA observation omits those privileged
inputs and sees the rendered scale through its cameras.

## Parameter meanings and numerical ranges

The eight fields are inherited from `Material` in
`chemlab_rl/powder_dispense.py`. That older module models a normalized hopper
tilt. The new scrape surrogate reuses the same names but changes the meaning of
`onset` to a height-derived discharge gate. None of these values were fitted to
laboratory powder measurements.

| Field | Current meaning | Units | Six configured vectors | Separate older `sample_material()` family |
|---|---|---|---|---|
| `flow_g_per_s` | Rate multiplier for loading/discharge; also source-tilt spill rate | g/s | 0.7–2.0 | 0.35–2.4 |
| `onset` | Threshold of the normalized height engagement `e` | dimensionless | 0.12–0.28 | 0.12–0.42 |
| `delay_s` | Continuous active time before discharge; also sets tail decay timescale | s | 0.12–0.40 | 0.10–0.70 |
| `residual_g` | Upper scale of a reserved post-stop tail, multiplied by discharge gate | g | 0.04–0.15 | 0–min(0.35, 0.25×target_g) |
| `lag_s` | First-order scale-filter time constant | s | 0.20–0.45 | 0.15–0.70 |
| `noise_g` | Standard deviation of independent Gaussian scale measurement noise | g | 0.006–0.012 | 0.004–0.020 |
| `spill_tilt` | Threshold of `(1-cos(source_angle))/2` | dimensionless | 0.84–0.92 | 0.78–0.95 |
| `cohesion` | Multiplicative rate reduction `1-cohesion`; not physical cohesive strength | dimensionless | 0.05–0.30 | 0–0.45 |

`lag_s` and `noise_g` are **sensor parameters**, not powder properties. The
remaining fields are phenomenological transfer parameters, not independently
measured constitutive material constants. The eight-dimensional vector is thus
a mixed environment-and-sensor parameter vector.

The six configured vectors, in the table's field order, are:

| ID | flow | onset | delay | residual | lag | noise | spill threshold | cohesion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.6 | .15 | .15 | .05 | .25 | .008 | .90 | .05 |
| 1 | 1.1 | .20 | .25 | .08 | .35 | .010 | .88 | .15 |
| 2 | .7 | .18 | .40 | .12 | .45 | .012 | .92 | .25 |
| 3 | 2.0 | .12 | .12 | .04 | .20 | .006 | .86 | .10 |
| 4 | .9 | .28 | .30 | .10 | .40 | .009 | .90 | .20 |
| 5 | 1.4 | .22 | .20 | .15 | .30 | .011 | .84 | .30 |

The corrected environment uses these fixed vectors by default. It does not call
the older random sampler. That sampler makes the residual distribution depend
on the target; using it for a new material-generalization protocol would need an
explicit decision about that target dependence. Custom `Material` values are
currently not range-validated, so arbitrary or nonfinite vectors are unsafe
inputs to the existing equations.

## Loading the tool

Initial inventories are 8 g in the source, 0 g on the tool, and 0 g in the
receiver. Tool capacity is an assumed 1 g.

The loading gate requires both the tool and source to be recognized as held by
real opposing pad contacts. In source-local coordinates, the sampled tool-tip
center must satisfy:

```text
radial_distance < 7.5 mm
8 mm < tip_height < min(51 mm, powder_surface_height + 1 mm)
powder_surface_height = 2 mm + 40 mm * source_g / 8 g
```

The radial margin accounts conservatively for the 3 mm blade radius inside the
nominal 11.05 mm inner neck. This is geometric overlap with an assumed powder
region, not simulated powder/blade contact. While the gate is true:

```text
loaded_g = max(0, min(source_g, 1 g - tool_g,
                      flow_g_per_s * (1-cohesion) * dt))
source_g -= loaded_g
tool_g += loaded_g
```

The tool need not move: holding it still in that region loads it automatically.
There is no scoop-volume or orientation model. Requiring the source to be held
also prevents loading from an otherwise identical source resting on the bench.
The 8 mm minimum tip height makes roughly the final 1 g geometrically
inaccessible as the assumed bed falls, without an explicit residual-bed model.

## Dispensing over the receiver

In receiver-local coordinates, let `h` be tool-tip height above its 57 mm rim.
The receiver gate requires a held tool, radial tip-center distance below 11 mm,
and `0 < h < 150 mm`. It does not test tool angle or powder flight direction.

```text
e = clip((50 mm - h) / 40 mm, 0, 1)
active = receiver_gate and e > onset and tool_g > 1e-8 g
gate = (e-onset) / (1-onset)
```

Continuous active time resets to zero whenever `active` becomes false. Once it
reaches `delay_s`:

```text
rate_g_per_s = flow_g_per_s * (1-cohesion) * gate
out_g = min(tool_g, rate_g_per_s * dt)
tool_g -= out_g
```

For the configured vectors, discharge begins below approximately 38.8–45.2 mm
above the rim. Engagement saturates at one at heights of 10 mm or less, while
remaining strictly above the rim. Thus the current control mechanism is
**height-controlled flow from a hovering loaded tip**; it is not a model of
scraping, shaking, or tipping powder off a real spatula.

Mass emitted while the receiver gate is true is immediately added to
`mass_g`; otherwise it is immediately added to `spilled_g`. There is no flight
time, particle trajectory, bounce, obstruction, or interception model.
The gate uses the tip center rather than the full falling powder footprint.
It is evaluated in the receiver's rotating frame, not against world gravity.

## Tail and spill rules

During discharge, a tail budget is reserved from remaining tool inventory:

```text
tail_g = max(previous_tail_g, min(residual_g * gate, remaining_tool_g))
```

This does not create additional mass. When `active` becomes false, each step
releases:

```text
tail_out_g = min(tool_g, tail_g,
                 max(0.005 g, tail_g * dt / max(dt, delay_s)))
```

The tail is deducted from both inventories and its destination uses the current
receiver gate. Its budget retains the largest gate encountered until it decays;
reducing an ongoing flow does not reduce that reservation. The 0.005 g minimum
per tail step is an arbitrary numerical rule and makes this part of the model
depend on the update frequency. A newly active interval that is still waiting
for its delay pauses the inactive-tail branch.

Source-vial tilt uses `u=(1-Rzz)/2`. When `u > spill_tilt`, source inventory is
removed at:

```text
spill_rate_g_per_s = flow_g_per_s * (u-spill_tilt)/(1-spill_tilt)
```

All of this source loss goes to `spilled_g`, even if the source mouth is over the
receiver. The configured thresholds correspond to approximately **133–147
degrees from upright**. An open vial can therefore reach ordinary large tilts
without spilling in this surrogate. No experimental evidence supports these
thresholds.

There is no corresponding receiver-tilt spill rule. Accounted receiver mass
remains there if the vial tips over. Likewise, loaded tool inventory can remain
on an arbitrarily oriented or dropped tool; only the discharge/tail rules remove
it. These are causal limitations of the current model, not properties of powder.

## Scale and visual representation

The scale is calculated from accounted receiver mass:

```text
alpha = 1-exp(-dt/lag_s)
filtered_g += alpha*(mass_g-filtered_g)
reading_g = max(0, filtered_g + Normal(0, noise_g))
```

Noise is independently resampled per control step and is not integrated into
true mass. Zero-clipping creates a positive measurement bias near an empty
receiver. The LCD displays this value rounded to two decimal places.

**This reading does not come from the MuJoCo pan force sensor.** It continues to
report receiver inventory if the receiver is lifted off the pan. The separate
success condition requires an upright, stable receiver physically supported on
the pan, but that condition does not change the displayed mass dynamics.

The source and receiving vial rigid bodies each retain their fixed 20 g glass
mass; the tool retains its fixed 46 g rigid-body mass. Loading or transferring
powder does not update their mass, inertia, grasp load or pan force. Powder fill
geometries have zero mass and collisions disabled.

The fill is a 10 mm-radius cylinder extending from the 2 mm vial bottom, with
height `40 mm * mass_g / 8 g`, capped at 40 mm. Its inferred density at 8 g is
about 0.637 g/mL, identical for all configured powders and never calibrated.
It rotates rigidly with the vial rather than maintaining a granular free
surface under gravity. Tool powder and spilled material are scalar inventories,
not rendered physical particles. The old floating puck visualization has been
removed from the corrected rigid-body scene.

## What improved relative to the original 30-demo collector

The original scoop collector performed a source-mouth and receiver-tip check,
then repeatedly held the robot still and incremented `scene.recv_mass_g`
directly outside the action-driven environment transition. Its increment was
`max(0.12, flow*(1-cohesion)*0.2)` g per dosing callback, further limited by a
target-dependent expression. It did not deduct finite source/tool inventory.

It represented assigned receiver mass as 0.02 g pucks and read tared pan contact
force. That force measurement could differ substantially from the accounted
mass because of puck placement and contacts. Storing delay, residual, lag,
noise, onset and spill parameters in those demos did not make them active in
the scoop-transfer equations: the principal rate used flow and cohesion.

The new surrogate makes transfer a shared consequence of the actual action
trajectory, uses finite source/tool/receiver/spill accounting, activates all
eight parameters in specified equations, and removes target-dependent mass
addition. These are software and causal-consistency improvements. They do not
turn either implementation into experimentally validated powder physics.

## Audit findings and interpretation of held-out powders

A direct numerical audit stored at `/tmp/chemlab-powder-causal-audit.json`
confirmed that every configured material loads 1 g in two seconds under a
stationary loading gate. After that, 0.5 s at engagement `e=.5` deposited
approximately .250, .105, .041, .311, .066 and .141 g respectively for IDs 0–5.
Those are outputs of the equations, not predictions validated against a robot.
Total `source_g + tool_g + mass_g + spilled_g` remained 8 g to floating-point
precision for these valid-parameter tests.

Two distinct vectors with `(flow,cohesion)=(1.6,.05)` and `(1.9,.20)`, with other
fields identical, produced **identical** loading, normal discharge and scale
traces. Both have `flow*(1-cohesion)=1.52 g/s`. These parameters cannot be
separately identified from normal loading/dosing in this model; source-tilt
spillage is the only implemented distinction. Held-out vector IDs therefore
need not represent distinct observable transfer dynamics.

Other implementation limitations identified by inspection:

- Geometry is sampled once per 100 ms action rather than integrated along the
  full trajectory. Brief visits or exits between samples can be missed.
- A collision failure can stop rigid-body stepping early, but the powder
  update still applies the full 100 ms interval. Failed-episode mass accounting
  can therefore exceed its actual simulated duration.
- Invalid custom parameter values are not rejected centrally and can cause
  negative rates/inventories, NaNs, or noise-sampling errors.
- The final success criterion uses both privileged accounted mass and noisy
  scale reading within 0.05 g, low spill, stopped transfer, and one second of
  settling. This is an evaluation rule for the surrogate, not a calibrated
  measurement uncertainty model.

Appropriate reporting is: **“Generalization to held-out parameters of an
explicit synthetic transfer-and-sensor model.”** Stronger claims about unseen
real powders, accurate mass deposited per scrape, physically faithful material
properties, or sim-to-real performance are unsupported by the current code.

## Independent audit of the original force sensor and pucks

The existing `balance_pan_force` sensor measures the force between the fixed pan
body and its parent, in the pan site's frame. It includes contact loads and
external disturbances; it is not intrinsically a mass sensor. At rest, with a
clear load path, dividing vertical load by gravity yields supported mass.
[MuJoCo force-sensor documentation](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-force)

The repository's `pan_reading_g()` subtracts the 150 g pan's weight and converts
newtons to grams using 9.81 m/s². The old collector then subtracts the measured
empty-receiver tare, approximately 20 g. This structure is suitable for an
independent scale check if tare, static settling and support conditions are
validated. Dynamic impacts, robot forces, and objects resting elsewhere on the
pan must not be mistaken for receiver powder mass.

Inspection of saved final states from the original HDF5 identified a concrete
reason for its mass/reading disagreement. The diagnostic loaded each saved
`qpos` into the original model, set velocities to zero, recomputed contact
forces, and compared assigned puck counts with their actual positions and
support forces. These are reconstructed static diagnostics, not exact playback
of the stored velocities (which were not recorded).

| Original episode | Assigned accounted mass | Assigned 20 mg pucks | Pucks with approximately full support force | Reconstructed summed support | Stored scale reading |
|---|---:|---:|---:|---:|---:|
| `demo_2` | 1.500 g | 75 | 75 | 1.500 g | 1.500 g |
| `demo_0` | 4.425 g | 221 | 199 | 3.980 g | 3.977 g |
| `demo_20` | 6.327 g | 316 | 218 | 4.358 g | 4.363 g |

The support-force sums explain most of the stored scale reading. The accounting
continued to include lost particles. In `demo_20`, some assigned pucks had fallen
hundreds of meters below the scene. Only 99 assigned puck centers remained above
the receiver's inner bottom; many other supported pucks were wedged beneath the
receiver between its bottom and pan, and the vial was lifted approximately
1.4 mm. A scale reading can therefore include powder outside the receiving vial.

Original puck collision masks are `contype=2, conaffinity=0`, while the receiver,
discard vial and pan accept bit 2. This means pucks do not collide with one
another or with the ordinary bench/floor. They collapse into coincident layers
instead of forming a supported granular pile, and escaped pucks can fall
through the bench. The state audit establishes escaping/under-vial particles;
it does not by itself identify every numerical mechanism that let them pass
the receiver bottom. Soft-contact tuning, initial placement and timestep require
separate testing before any physical-grain replacement is accepted.

Diagnostic artifacts: `/tmp/chemlab-puck-force-audit.py` and
`/tmp/chemlab-puck-force-audit.json`. No original dataset or production code was
modified for these probes.

### Minimum checks for a contact-driven replacement

1. Define particle or parcel volume and density explicitly. For physical grains,
   verify each compiled body's `mass = density × volume`, and the complete
   initial inventory sums to the declared 8 g. Grain resolution must be small
   enough for a 0.05 g target tolerance; quantization error is part of reporting.
2. Establish disjoint inventories from actual state: source, tool, receiver,
   airborne/transit and spilled. Their mass sum must remain 8 g through every
   action. Membership must not be incremented merely because a loading/dosing
   gate was visited. Count deposited receiver mass from grains actually inside
   its cavity, with appropriate geometric clearance and stable classification.
3. Require tool/powder contact and motion to produce transfer. A stationary tool
   that never contacts powder must not load; an empty tool hovering over the
   receiver must not create deposited mass. Mass must move continuously through
   the intended load path rather than being created, teleported or deleted.
4. For a grain model, enable grain–grain, grain–tool, grain–vial, grain–pan and
   grain–bench collisions from model compilation. Test no tunneling or lost
   particles across the actual timestep, contact settings and operating speeds.
   A physically useful scoop geometry is also required; the current thin rod
   does not automatically acquire a real spatula's carrying capacity.
5. Tare the settled pan with the empty receiving vial. Test known static loads
   across the intended 0–8 g range and across positions in the receiver. At
   rest, net force-derived mass should match independently summed receiver
   membership mass within a tolerance substantially tighter than 0.05 g.
6. Separately detect grains on the bare pan, grains under the receiver, and
   foreign/robot contacts. They can contribute load without being successfully
   dispensed. A matching force reading alone cannot certify in-vial mass.
7. Require the receiver to be supported and settled before mass agreement is
   evaluated. Lifting it must remove its load from the pan; tilting it must allow
   physical loss or trigger the model's explicit contact/volume spill rule.
   Preserve raw force readings before adding any optional sensor lag/noise.
8. Test one-grain transfers, whole-inventory conservation, repeated
   loading/withdrawal/deposition, no-contact controls, source/receiver tilt,
   tool release, and off-pan readings. Verify both settling convergence and
   membership stability over a time window, not a single sample.

A physical grain model with known synthetic density can satisfy internal mass
and sensor consistency while still being only a coarse approximation to real
powder. Dry friction and particle geometry do not automatically provide a
cohesive material law. Any new powder parameter vector should describe the
replacement's actual equations, with sensor settings reported separately.
