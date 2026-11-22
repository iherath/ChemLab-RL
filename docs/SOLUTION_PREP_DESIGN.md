# Subtask 2 — Solution prep: BC data generation design (PROPOSAL, not implemented)

Status: draft for review. No code has been written yet.

## Decisions taken (2026-11-22)

| Question | Choice |
| --- | --- |
| Dispensing tool | Syringe, **two-handed** (barrel arm + plunger arm). **Syringe size varies per episode** (e.g. 1 / 2 / 3 / 5 mL); total solution volume is always **3.0 mL**, so small syringes need multiple draw→dispense cycles |
| Mixing | One arm holds the vial body while the other screws the cap on tight. The holder raises the vial, the other arm takes it top-to-bottom and **shakes it linearly**, then hands it back, and the holder sets it in its stand |
| Liquid fidelity | Synthetic volume/concentration accounting (same philosophy as `scrape_env.py`) |
| Generator | SkillWeaver-lite: parameterized closed-loop skills + state verifiers + verifier-guided retry search. No LLM in the loop |
| Start state | Fresh reset: vial pre-loaded with a sampled powder mass, upright on bench |
| Varied / held out | Syringe size (sets number of draw cycles), solid dissolution rate, scene layout. Total volume fixed at 3.0 mL |
| Cap | **Screw cap** (helical thread, ~1 turn to tight). It replaced the snap cap on 2026-11-22 |
| First collection | 30 accepted pilot demos, then scale to ~100 |

SkillWeaver's own exploration code (VLM planner, MCTS, hierarchical verifier,
experience memory) is listed as "coming soon" in its repo, and its released NIS
skills are Isaac Lab / Franka. So this design re-implements the key ideas in
MuJoCo rather than depending on SkillWeaver code.

## ⚠ Gap to subtask 1 (open work item)

What is being built here for subtask 2 does **not** directly continue from what was built for
subtask 1. Subtask 2 starts from a fresh reset with a vial that already holds the solid, upright on
the bench. Subtask 1 so far covers (a) placing an empty vial on the balance pan with the shield
(vial-placement pilot) and (b) scraping powder into a receiver on the pan to a target reading
(`scrape_env.py`, synthetic mass accounting). Still missing to stitch the two cartridges:

1. **Weigh the solid into the tared vial on the balance**: vial placed on the pan → tare →
   transfer powder until the tared reading hits the target mass → settle. (Combines the
   vial-placement and scrape pieces; the tare step does not exist yet.)
2. **Remove the filled vial from the balance** through the draft-shield door and set it upright on
   the bench, i.e. produce the start state that subtask 2 assumes.
3. Make subtask 2's reset distribution match the end-state distribution of (1)–(2) (powder mass,
   vial pose), or switch subtask 2 to chain from those end states.

Until these exist, subtask-2 results are cartridge-level only and say nothing about the stitched
weigh → prep sequence.

## Task definition

Instruction (language, varies per episode):
"Use the syringe to add 3.0 mL of solvent to the vial, cap the vial, and swirl
until the solid has dissolved." (syringe size is visible, not stated)

Episode, from a fresh reset:
1. Pick the syringe (left = barrel) and bring the right gripper to the plunger flange.
2. Insert the needle into the solvent reservoir below the liquid line.
3. Bimanual draw: the right arm pulls the plunger to min(capacity, remaining volume).
4. Withdraw, move over the receiving vial, insert the tip into the mouth.
5. Bimanual dispense: push the plunger to empty. Repeat 2–5 until 3.0 mL is delivered, then set the syringe down.
6. Pick the cap and press it down onto the vial mouth until it snaps closed.
7. Pick the capped vial, swirl/invert until homogeneous, and set it upright on the bench.

VaDar subgoal dones (latched, never given to the actor):
1. `loaded`: first draw complete — syringe volume within ±0.05 mL of the planned draw, tip out of the reservoir
2. `dispensed`: vial solvent within ±0.05 mL of 3.0 mL, syringe < 0.05 mL, spill < 0.05 mL
3. `capped`: cap seated and engaged
4. `homogeneous`: undissolved < 1% of solid, homogeneity ≥ 0.98, vial upright on the bench, grippers released

Success = all four, no forbidden contact, conservation within tolerance.

## Scene (new module, Leo's `chem_rl_sim` left untouched)

`chemlab_rl/solution_scene.py` builds on `chem_rl_sim.scene.build_spec`
(Robotiq, no clutter) and adds:
- **Receiving vial**: the existing `vial_20ml_open`. A visual-only fill cylinder shows liquid height; a
  powder/turbidity layer shows undissolved solid, so the cameras can see progress.
- **Solvent reservoir**: a wider open vial/beaker (~50 mL), heavy or in a cork-ring holder so the needle
  insertion does not tip it.
- **Syringes**: new parametric asset built for each size (1/2/3/5 mL: barrel length and bore scale, graduations as texture). Barrel body (free joint), needle
  capsule, plunger body on a **slide joint** with damping and friction loss, so it holds its stroke when
  released, plus a finger flange the right gripper can pinch. Rests in a simple rack so the barrel is
  graspable.
- **Cap**: the existing `cap_24_400_white`.
- The balance stays in the scene as fixed set dressing with its doors closed. It is not used in this subtask.

Layout randomization: ±3 cm xy and ±20° yaw on the reservoir, the syringe rack, the vial and the cap,
under reachability/IK feasibility rejection.

## Liquid model — `SolutionState` (synthetic, auditable)

State: `reservoir_mL, syringe_liq_mL, syringe_air_mL, vial_mL, spilled_mL, solid_undissolved_g,
solid_dissolved_g, homogeneity`.

- **Draw**: plunger stroke × barrel area. Liquid is added only while the needle tip is below the
  reservoir liquid line; otherwise it is air.
- **Dispense**: plunger push expels air first, then liquid. Liquid goes to the vial if the tip is inside
  the vial-mouth cylinder, otherwise to `spilled`.
- **Dissolution**: `dm/dt = -k · (a0 + agitation) · m_undissolved · (1 - C/C_sat)`. Agitation comes from
  the vial's measured angular and linear acceleration. `k` is the hidden material parameter (IDs 0–4 for
  training, 5 held out).
- **Homogeneity** relaxes toward 1 at a rate proportional to agitation, and decays toward a
  stratified value when there is none.
- **Spill**: tilting an uncapped vial past the liquid-surface angle moves volume to `spilled`. A capped
  vial does not spill.
- Conservation: `reservoir + syringe + vial + spilled` is constant (checked every step).

Cap model (decided): **snap-to-close**. When the cap is seated within 1.5 mm / 5° of the mouth and pressed
down with ≥ a threshold force, an equality weld activates (a press-fit/snap vial). This is the only
constraint not driven by contact; it is recorded in the metadata.

**TODO (later): screw-on caps.** Replace the snap with thread engagement: require a ≥ 90° twist
under axial load to engage (and an untwist to release), or model threads as contact geometry.
The real CV vials are screw-cap (24-400), so this must be revisited before Sim2Real.

## Generator — SkillWeaver-lite

```
chemlab_rl/solution_skills/
  base.py        Skill(params) -> run(env) closed-loop; pre(env), post(env) verifiers
  library.py     grasp, move_to, insert_tip, withdraw, plunger_stroke (bimanual),
                 place, snap_cap, swirl, invert, release, home
  verifiers.py   state-based checks per skill + global safety (penetration, forbidden
                 contact, conservation, spill)
  search.py      verifier-guided search over skill parameters
```

- **Skills** are parameterized and closed-loop (IK servo on the measured pose each control step),
  reusing the IK/keyframe machinery that `collect_scrape_v2.py` already uses. The parameters are grasp
  offset and approach axis, speeds, insertion depth, stroke rate, swirl amplitude/frequency and
  number of cycles, and which arm holds the barrel (mirrored variants).
- **Verifier-guided search**: the task graph is fixed (draw → dispense → cap → mix), and each node is a
  skill with sampled parameters. Before each skill the search snapshots the MuJoCo and liquid state. If
  the post-verifier fails, it restores and resamples, for up to K tries. Then it backtracks one node.
- **Experience memory**: failure reasons per skill/parameter region are logged. Parameter sampling is
  adapted CEM-style toward regions that passed, while variance is kept for diversity.
- **Closed-loop mixing**: the expert swirls until `homogeneous`. So the swirl duration varies with the
  hidden dissolution rate, and the policy has to read it from the rendered turbidity.
- Because each accepted segment starts from the restored snapshot, accepted segments chain exactly.
  The stored episode is reset + accepted actions only.

## Acceptance gates (per episode)

1. Full action replay from reset with the same seed reproduces every subgoal and the final state.
2. Conservation error < 1e-6 mL; spill < 0.05 mL; final vial volume 3.0 ± 0.05 mL; no unintended contact; penetration < 1 mm.
3. Subgoal bits latch in order and never flip backward. This was a flaw found in the old powder data.
4. Diversity check: a new episode is rejected if its DTW distance to an accepted one with the same
   target is below a threshold. This addresses the "repeated trajectories" finding.

## Data format

The schema matches `scrape_v2`, so `vla_bc.py` needs minimal changes: 10 Hz, three 224×224 cameras
(`top1`, wrists), 14-D proprio, 14-D joint/gripper servo targets, pre-action obs aligned with the
action, next obs, the 4-bit done vector, the instruction, and hidden metadata (k, V, solid mass, layout
seed, skill params, verifier log). Splits are fixed before collection: k ID 5 held out, plus one held-out
syringe size (proposal: 2 mL, an interpolation case between 1 and 3 mL).

## Build order

0. Bootstrap `chem_rl_sim` + `.venv-vial` (`scripts/bootstrap_vial_training.sh`).
1. Syringe/reservoir assets + scene; IK reachability for both arms; plunger slide sanity check.
2. `SolutionState` + unit tests (conservation, draw/dispense, dissolution).
3. Per-skill fixtures with verifiers. Plunger stroke (esp. 1 mL syringe precision) is the riskiest.
4. One full expert episode + replay gate.
5. Search + experience memory; collect the 30-episode pilot; audit; render GIFs.
6. Package + 100-update SmolVLA BC smoke test.

Steps 1 and 2 can proceed in parallel. Step 3 can be split per skill once the scene is fixed.

## Build status (2026-11-22)

Done:
- `chemlab_rl/solution_state.py` + `tests/test_solution_state.py` (13 pass): syringes 1/2/3/5 mL
  (2 mL held out), six solids (ID 5 held out, k inside the training range), draw/dispense with
  air-first expulsion, tilt spill, saturating dissolution, homogeneity index, conservation checks.
  Deliberate choices: liquid pushed back while the tip is in the reservoir returns to it; liquid
  never "unmixes" without agitation; no randomness inside the model.
- `chemlab_rl/solution_scene.py`: reservoir vial (18 mL), syringe (barrel + child plunger on a
  damped slide joint) in a fixed rack, snap cap with an inactive `cap_snap` weld, fill/turbidity
  visuals. All four sizes settle stably (drift 0, tilt < 0.1°).
- Reachability: search over layouts and approach directions. The **plunger arm (right) must
  approach from the robot side (tool +y)**; from +x its forearm crosses the balance. The rack was
  raised so short barrels are grasped >= 9 cm above the bench. Every key pose is reachable for
  every size (IK <= 0.19 mm, no penetration).
- Dynamic plunger test (full physics, Session planner): two-handed lift 70 mm, pull 60 % stroke,
  push back. Stroke tracks the command within ~1.6 mm (<= 0.03 mL on the 1 mL syringe), the barrel
  moves <= 1.5 mm in the left grasp, and max pad penetration is 0.83 mm (< 1 mm limit).

Known issues for the skill layer:
- The right arm's pre-approach must avoid routing over the balance (use a front-left via point).
- During the two-arm lift the plunger can creep 1–2 mm (draws air). The expert must purge air
  in the reservoir before the final draw, and closed-loop stroke correction is needed for the
  ±0.05 mL tolerance.

- `chemlab_rl/solution_env.py` (`SolutionPrepEnv`, schema `solution_prep_v1`) with 8 tests in
  `tests/test_solution_env.py`. It has 14-D actions at 10 Hz, bilateral-pad grasp detection
  (the plunger is tracked as its own body), a per-substep contact audit (the needle may touch
  nothing), liquid stepping from measured tip/plunger/tilt/agitation, a 4-bit done vector,
  3 cameras, and syringe size / solid / layout jitter as reset options. Tests cover quiet reset and
  hold for all sizes, seeded layout jitter, exact action replay, snapping only when pressed, the
  snapped cap staying seated, and the cap following a tilted vial.
- Snap weld is set at the nominal seated pose, not the instantaneous one (the latter froze in
  press compression). The snap condition is checked every 50 ms.
- An impact press can punch a free cap through the 1.2 mm vial-wall hulls. Real presses are
  position-controlled and the audit catches deep penetration, so the skill must press slowly.
- Viewer: `scripts/view_solution_scene.py` (static scene, or `--demo` under `mjpython` to watch
  the two-handed syringe motion).

### Skill library (`chemlab_rl/solution_skills.py`) — first full expert episodes

Skills: `acquire_syringe` (left barrel from -x, right plunger rod from the front), `lift_syringe`,
`transit_to`, `draw` (purge -> draw -> closed-loop top-up -> air gap inside the reservoir -> out;
re-dips if short), `dispense`, `return_syringe`, `snap_cap` (left, top-down, 0.25 mm press steps),
`swirl` (top-down hold of the capped vial, circles in front of the rack until homogeneous),
`place_vial`. Each raises `SkillFailed` when its state verifier fails. Parameters (standoffs,
hover/insert depths, air gap, swirl radius/frequency/tilt, cap grip, ...) are sampled per episode.
`run_expert(env, seed, syringe, solid_id, solid_g)` runs one episode from reset.

Lessons that shaped the controller (all found by running episodes):
- Session follow mode has a 1 mm deadband. Plunger moves need a fine IK servo, and incremental
  targets must start from the COMMANDED pose, not the measured one, or the plunger jumps back by
  the hand's preload offset (this lost ~0.12 mL per draw).
- Independent arm lag during carries pumps the plunger. Carries are rigid: the right-hand target =
  the left path + an integral stroke-hold correction. Plus an air gap after each draw.
- Liquid expelled with the tip inside the reservoir mouth now drips back into it (state-model change).
- Cap: real 24-400 height (12.5 mm), grasped so the pads stay 4 mm above the bench/lip; cap geom
  priority 2 (pads are priority 1) to keep pad compression < 1 mm. Agitation unit recalibrated to
  1 m/s^2 (robot swirl); swirls now take ~9–25 s.
- Needle contact is allowed only with its own rack slot.

Measured (images off): ~13 / 27 episodes succeed across all four syringe sizes and five solids.
Every episode that completes dispensing ends at 2.993–3.002 mL with 0.000 mL spill, exact
conservation, and max pad compression <= 0.94 mm. A 1 mL episode is ~1750 steps (four
draw/dispense cycles), 3/5 mL ~1250–1450 steps, ~15–25 s wall each.

Remaining failures depend on layout seed, not syringe size (the same seeds fail for every size):
(a) the right arm's approach to the plunger is blocked (table/balance) for some rack jitters;
(b) the left pad grazes a rack block while capping; (c) the right wrist camera mount grazes the
balance's top glass/handle. All three are routing problems near the balance and rack.

Next: (1) re-run the layout search with full approach paths (not just end-pose IK) and move the
work area further from the balance; (2) the collector: verifier-guided retries with snapshot
restore, replay gate, diversity check and HDF5 writer in the scrape_v2 schema; (3) the 30-episode
pilot.

### Screw cap + handover + shake (2026-11-22, replaces the snap cap / swirl)

Sequence after the syringe is back in its rack:
1. `hold_vial`: the right hand grips the vial body from the front, tilted 35° down.
2. `screw_cap`: the left hand picks the cap top-down, lowers it onto the rim with a closed loop on
   cap-centre vs mouth until the thread engages, then turns it about 1 turn (wrist_3 plus a pitch
   descent) until tight.
3. `raise_vial`: the right lifts it straight out of the stand and slides it away from the parked syringe.
4. `axial_grasp`: the left takes it top-to-bottom (fingers on the cap top and under the base,
   approaching from −x).
5. The left slides the vial out of the right's opened fingers and the right parks.
6. `shake`: linear, 2–2.5 Hz, ±2–3 cm, until homogeneous.
7. `hand_back`: just above the stand, with a coordinated grip swap.
8. `place_in_stand`.

Model: `vial_a` carries a thread body (hinge coupled to a slide, 8 TPI, 1 turn). The weld
`cap_thread` engages it (the one non-contact constraint). Friction rises over the last 60°, and the
hinge end stop is "tight". The vial is sealed only when tight. Cap↔vial contact is excluded.
Subgoal 2 = tight.

Design changes this forced (all found by running episodes):
- **Vial stand** (60 mm platform, 4 mm locating recess). A horizontal Robotiq grasp on a 57 mm vial
  resting on the bench puts the gripper body into the table. Final placement is into the stand.
- Cap: real 24-400 height, 10 g. Contact and constraint stiffness scale with effective mass; at 4 g
  the pads compressed to ≥ 1 mm.
- Thread joints need real armature, and hinge ranges are in degrees in this spec (that was a bug).
- Wrist re-index by quarter turns (the cap is round) to get one full tightening turn within
  wrist_3's ±360°, keeping margin on both sides.
- Partial opens and coordinated grip swaps at handovers. A full Robotiq open sweeps ~45 mm per side
  into the other hand. "Partly open" depends on the part: ~140 on 28 mm parts, ~35 on the 66 mm
  top-to-bottom grasp.
- The gantry column boxes in the right wrist at the handover pose, so the left slides the vial out
  of the right's open fingers instead.

Measured: the full sequence succeeds end to end (3 mL seed 0: 2352 steps, 2.996 mL, cap 1.002 turns,
shake 26.8 s; 5 mL seed 0 too). Robustness is low: **2 / 17** episodes across sizes and seeds.
Failures, by class:
- the right arm's approach to the plunger is blocked (5, pre-existing, layout-dependent);
- the axial-grasp/handover region collides with the parked syringe, the balance post or the column (4);
- cap-pad compression > 1 mm under shaking/handover loads (2);
- the raised vial changes the syringe-insertion poses so the arms meet (1);
- cap not centred at engagement (1);
- the vial is pushed into the stand recess (1);
- a single-draw tolerance miss on the 5 mL syringe (1).

Next: a robustness pass. Re-run the planned-path layout search with the stand and the new
handover/hold poses (likely moving the rack and the reservoir further from the vial). Add per-skill
retries with snapshot restore in the collector. Then the 30-episode pilot.

### Robustness pass (2026-11-22, balance kept)

What the static obstacles really are: besides the balance (AABB x −0.02..0.22, y −0.13..0.21,
top z 1.13), the planner's "table" label covers ALL world geoms. That includes the **gantry column**
(x ±0.045, y −0.43..−0.34, up to z 1.555) and the gantry head the arms hang from. The right arm's
home is behind/right of the balance.

Layout findings (episode- and planner-based searches):
- Shifting the prep cluster left or forward (18 layouts × 4 sizes, 72 episodes) succeeded 0 times.
  The left arm can't side-grasp close to its own base, and forward positions run the right elbow
  into the column/gantry head.
- The strip in front of the balance (y −0.13..−0.30) is not better: right-hand targets left of
  x ≈ 0.05 fold the elbow into the column.
- So the prep area stays where it is. The fix was ROUTING: the right arm leaves home **high over the
  balance** (via (0.30, 0.10, 1.25), clean on 8/8 seeds), then a staging point
  rod + (0.08, −0.08, 0.08), and returns the same way. The direct vial-hold staging stays (it is
  clean from home).
- Other fixes: strict planner previews (any penetration); a closed loop on cap x/y/z while lowering
  onto the thread; deeper needle insertion and a lower carry over the raised vial.
- Moving the rack further away freed the handover but broke the barrel/plunger reach, so it was
  reverted.

Measured: **3 / 24** episodes succeed (was 2 / 17). The plunger approach and vial hold no longer
fail. The dominant remaining class is the top-to-bottom grasp (left gripper vs the parked syringe:
6 / 21 failures). The rest are spread over cap engagement, grip compression, hand-back clearances
and singles.

Assessment: hand-tuning one collision at a time oscillates in this cramped cell. Failures depend on
the sampled parameters and seed, not on a fixed layout, which is exactly what per-skill retries
(snapshot → resample → retry) are designed to absorb. Next: build the collector's retry search and
measure the yield per attempted episode.

### Single-arm carry (2026-11-22, in progress)

Change requested by the user: only the carrying arm holds the syringe while it moves. The other
arm touches it only to pull/push the plunger. Implemented: the left arm carries the barrel alone
(plunger drift during carries is now 0.000 mL, vs ±0.013 mL with the two-handed carry), and the
right arm `engage_plunger`s for each stroke and `release_plunger`s afterwards.

Blocker found: the right arm can only (re-)approach the plunger rod in one narrow pocket of this
cell, around the old rack position (−0.205, 0.08). Measured with in-episode planner probes and real
episodes:
- Further left (old reservoir, x −0.27) is out of reach.
- Forward (y < 0) folds its elbow into the gantry column.
- Behind the vial puts its wrist into the balance's left door.
- Approaching top-down from the over-balance high point fails at both the reservoir and the vial.

In the two-handed design the right hand grabbed the rod once in that pocket and then rode along,
which is why it worked. Partial progress: with the reservoir moved into the pocket and a raised
rack at the left-front (−0.30, −0.06, needle tip 55 mm up), picking up the syringe and drawing
succeed. Engaging the plunger above the vial is still blocked.

The current code is in this intermediate state (reservoir without a stand, default layout), so the
expert does not yet complete episodes. Proposed resolution: swap roles. The right hand carries the
barrel (grasped from the front, in the pocket where it already works, and moved locally by servo,
never re-planned from home), and the left hand, which reaches the whole left work area freely, works
the plunger.

## HANDOFF — where to resume (2026-11-22)

State of the code (nothing is committed; all subtask-2 files are new or modified in the working tree):
- `chemlab_rl/solution_state.py`, `solution_scene.py`, `solution_env.py`, `solution_skills.py`,
  `tests/test_solution_state.py`, `tests/test_solution_env.py` (22 tests pass),
  `scripts/view_solution_scene.py` (viewer: `--demo`, `--expert SEED --every N`),
  `scripts/solution_dev/` (episode runner, parallel batch runner, layout search, last batch logs).
- Run from `ChemLab-RL/`: `.venv-vial/bin/python scripts/solution_dev/run_episode.py 3mL 0` (one
  episode with its skill log) and `.venv-vial/bin/python scripts/solution_dev/batch_episodes.py
  1mL:0 3mL:1 ...` (parallel stats). The sim needs `scripts/bootstrap_vial_training.sh`, then
  `pip install --no-deps --config-settings editable_mode=compat -e chem_rl_sim`.

Current design (last user decisions):
- Screw cap; right hand holds the vial body while the left screws; the right raises the vial; the
  left takes it top-to-bottom, shakes linearly and hands it back; the right sets it in its stand.
- **Single-arm syringe carry, roles swapped (user decision):** the RIGHT hand carries the barrel
  (front grasp) and the LEFT hand engages the plunger rod (side pinch) only for each stroke, then
  parks at `LEFT_PARK`. The syringe is re-levelled in the hand before moves and racking. The
  reservoir sits on a 60 mm stand; the rack is raised (needle tip 50 mm up).

Measured:
- One full liquid-handling pass with the role swap worked on 3mL seed 0: both draws/dispenses, 2.993 mL,
  0 spill, syringe racked at 0.02° tilt.
- The 16-episode batch after the swap was **0/16**. Main blockers: the right hand's approach to the
  barrel in the rack (blocked by world geometry / touching the vial or reservoir, 8/16), the vial-hold
  staging (2), and the top-to-bottom grasp (left wrist, 3).
- The best result overall was the earlier two-handed carry, at 3/24
  (`scripts/solution_dev/last_batch_two_hand_carry.txt`).

Hard-won constraints (don't re-discover):
- The planner's "table" label covers ALL world geoms: the bench, the gantry **column**
  (x ±0.045, y −0.43..−0.34) and the gantry head.
- The right arm's home is behind/right of the balance. Leave it via (0.30, 0.10, 1.25), not a straight line.
- The right arm reaches a plunger/barrel from the front only near (−0.205, 0.08). Forward folds its
  elbow into the column; behind the vial hits the balance's left door; far left is out of reach.
- Robotiq: pads reach 14.6 mm below / ~23 mm above the pinch. A full open sweeps ~45 mm per side, so
  use partial opens at handovers. Command ≈ 255·(1 − gap/93 mm).
- MuJoCo: contact and constraint stiffness scale with effective mass, so the cap is 10 g and the
  thread joints carry armature. Hinge ranges in this spec are in DEGREES. Leo's planner allows 1 mm of
  preview penetration (overridden to 0).

Suggested next steps:
1. Re-place the syringe rack inside the right arm's front pocket (it moved when the reservoir was
   swapped in), and re-check the barrel grasp heights with the raised rack.
2. Build the collector's per-skill retry (snapshot → resample parameters → retry). Many failures depend
   on the sampled parameters, not the layout.
3. Then run the 30-episode pilot. Still open: the gap to subtask 1 (weighing into the tared vial on the
   balance, then moving it to the bench).
