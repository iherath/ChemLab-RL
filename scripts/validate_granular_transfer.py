"""Contact-only powder scoop validation in an isolated actuated test fixture.

The source is a fixed fixture; the receiver is fixed unless --force-pan is used.
The scoop has three actuated translation joints and one tilt joint. Every grain
remains a free body under gravity; after reset, only actuator controls change.
This validates granular transfer and optionally force weighing, not robot grasps.

The original successful proof used --legacy-vial-colliders. The default uses
profile-following colliders, including the visual neck narrowing. Never infer
full robot-task validity from this isolated experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "chem_rl_sim"), str(ROOT / "chem_rl_sim/third_party/urgantry_sim")]
from chem_rl_sim.assets import add_asset
from chem_rl_sim.balance import PAN_R, PAN_T, PAN_MASS, STEEL_RGBA
from chemlab_rl.granular_accounting import GranularAccounting
from chemlab_rl.granular_geometry import replace_vial_collision, scoop_membership

RADIUS = .0015
MASS_KG = .00002
CUP_OUTER, CUP_INNER, CUP_DEPTH = .007, .006, .006


def build_fixture(grain_count=400, timestep=.00025, iterations=200,
                  solver="cg", legacy_vial_colliders=False, force_pan=False, tolerance=1e-6):
    spec = mj.MjSpec()
    opt = spec.option
    opt.timestep = timestep
    opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    opt.cone = mj.mjtCone.mjCONE_PYRAMIDAL
    opt.jacobian = mj.mjtJacobian.mjJAC_SPARSE
    opt.solver = {"cg": mj.mjtSolver.mjSOL_CG, "newton": mj.mjtSolver.mjSOL_NEWTON}[solver]
    opt.iterations, opt.noslip_iterations, opt.tolerance = iterations, 0, tolerance
    spec.worldbody.add_geom(name="floor", type=mj.mjtGeom.mjGEOM_PLANE,
                           size=[.4, .4, .01], rgba=[.4, .4, .4, 1],
                           pos=[0, 0, -PAN_T if force_pan else 0],
                           solref=[.002, 1], solimp=[.95, .99, .0001, .5, 2])
    if force_pan:
        # The same pan body, geom, site and force sensor as add_balance, placed
        # on a bench without the surrounding draft shield for this fixture.
        pan = spec.worldbody.add_body(name="balance_pan", pos=[.08, 0, -PAN_T/2])
        pan.add_geom(name="balance_pan", type=mj.mjtGeom.mjGEOM_CYLINDER,
                     size=[PAN_R, PAN_T/2, 0], rgba=STEEL_RGBA, mass=PAN_MASS,
                     friction=[1., .005, .0001])
        pan.add_site(name="balance_pan_site", pos=[0, 0, PAN_T/2])
        spec.add_sensor(name="balance_pan_force", type=mj.mjtSensor.mjSENS_FORCE,
                        objtype=mj.mjtObj.mjOBJ_SITE, objname="balance_pan_site")
        spec.worldbody.add_geom(name="source_fixture_base", type=mj.mjtGeom.mjGEOM_CYLINDER,
                                size=[.020, PAN_T/2], pos=[0, 0, -PAN_T/2],
                                rgba=[.5, .5, .5, 1])
    for name, x in (("source", 0.), ("receiver", .08)):
        if force_pan and name == "receiver":
            continue  # Add its free joint after grains to preserve snapshot layout.
        vial = add_asset(spec, spec.worldbody, "vial_20ml_open", name,
                         [x, 0, 0], free=False, friction=(.4, .0001, .00002))
        if not legacy_vial_colliders:
            replace_vial_collision(spec, vial)
    cup = spec.worldbody.add_body(name="scoop")
    for i, (name, axis) in enumerate((("x", [1, 0, 0]), ("y", [0, 1, 0]),
                                     ("z", [0, 0, 1]), ("tilt", [0, 1, 0]))):
        cup.add_joint(name="scoop_" + name, axis=axis,
                      type=mj.mjtJoint.mjJNT_SLIDE if i < 3 else mj.mjtJoint.mjJNT_HINGE)
        kp, kd, limit = (10000., 20., 2.) if i < 3 else (1., .005, .1)
        spec.add_actuator(name="scoop_" + name, target="scoop_" + name,
                          trntype=mj.mjtTrn.mjTRN_JOINT, gaintype=mj.mjtGain.mjGAIN_FIXED,
                          gainprm=[kp] + [0]*9, biastype=mj.mjtBias.mjBIAS_AFFINE,
                          biasprm=[0, -kp, -kd] + [0]*7,
                          forcelimited=True, forcerange=[-limit, limit])
    # Same dimensions as the robot scoop, expressed in its simpler +Z-up frame.
    props = dict(friction=[.4, .0001, .00002], condim=3, solref=[.0005, 1],
                 solimp=[.95, .99, .0001, .5, 2], rgba=[.65, .7, .75, 1])
    cup.add_geom(name="scoop_bottom", type=mj.mjtGeom.mjGEOM_CYLINDER,
                 size=[CUP_OUTER, .0005], pos=[0, 0, .0005], **props)
    outer_plane = CUP_OUTER*math.cos(math.pi/16)
    for k in range(16):
        angle = 2*math.pi*k/16
        middle = (outer_plane+CUP_INNER)/2
        cup.add_geom(name=f"scoop_wall_{k}", type=mj.mjtGeom.mjGEOM_BOX,
                     size=[(outer_plane-CUP_INNER)/2, CUP_OUTER*math.sin(math.pi/16), CUP_DEPTH/2],
                     pos=[middle*math.cos(angle), middle*math.sin(angle), CUP_DEPTH/2],
                     quat=[math.cos(angle/2), 0, 0, math.sin(angle/2)], **props)
    cup.add_geom(name="scoop_stem", type=mj.mjtGeom.mjGEOM_CAPSULE,
                 fromto=[-.0055, 0, .005, -.0055, 0, .040], size=[.001], **props)
    spacing = .00305
    xy = [(spacing*(i+.5*(j % 2)), spacing*math.sqrt(3)/2*j)
          for j in range(-4, 5) for i in range(-4, 5)
          if (spacing*(i+.5*(j % 2)))**2 + (spacing*math.sqrt(3)/2*j)**2 <= .0106**2]
    for k in range(grain_count):
        layer, slot = divmod(k, len(xy))
        body = spec.worldbody.add_body(name=f"grain_{k}",
                    pos=[*xy[slot], .0021+RADIUS+spacing*layer])
        body.add_freejoint(name=f"grain_{k}")
        body.add_geom(name=f"grain_{k}", type=mj.mjtGeom.mjGEOM_SPHERE,
                      size=[RADIUS], mass=MASS_KG, priority=1,
                      friction=[.5, .0001, .00002], condim=3,
                      solref=[.0005, 1], solimp=[.95, .99, .0001, .5, 2],
                      rgba=[.72, .58, .32, 1])
    if force_pan:
        receiver = add_asset(spec, spec.worldbody, "vial_20ml_open", "receiver",
                             [.08, 0, .00001], free=True, friction=(.4, .0001, .00002))
        if not legacy_vial_colliders:
            replace_vial_collision(spec, receiver)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 800, 600
    model = spec.compile()
    data = mj.MjData(model)
    data.qpos[2] = data.ctrl[2] = .080  # Initial fixture pose only.
    mj.mj_forward(model, data)
    return model, data


def initialize_pouring_component(model, data, snapshot, retained_indices):
    """Initialize a focused receiver-impact replay from a recorded fixture state.

    The stationary source pile is physically disconnected during pouring. Its
    omission reduces runtime without removing contacts from the loaded cup.
    State writes occur only here, before the first replay step.
    """
    old_q, old_v = snapshot["qpos"], snapshot["qvel"]
    data.qpos[:4], data.qvel[:4] = old_q[:4], old_v[:4]
    data.ctrl[:] = snapshot["ctrl"] if "ctrl" in snapshot else [.073, 0, .068, 0]
    for new, old in enumerate(retained_indices):
        data.qpos[4+new*7:11+new*7] = old_q[4+old*7:11+old*7]
        data.qvel[4+new*6:10+new*6] = old_v[4+old*6:10+old*6]
    mj.mj_forward(model, data)


def retained_in_snapshot(snapshot):
    """Select the geometrically loaded grains in this fixture's pre-pour state."""
    q, v = snapshot["qpos"], snapshot["qvel"]
    if q.ndim != 1 or (len(q)-4) % 7 or len(v) != 4+(len(q)-4)//7*6:
        raise ValueError("Snapshot must contain this fixture's four joints and free spherical grains")
    angle = float(q[3])
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    local = (q[4:].reshape(-1, 7)[:, :3]-q[:3]) @ rotation
    keep = np.flatnonzero((np.linalg.norm(local[:, :2], axis=1) < CUP_INNER+RADIUS*.5)
                          & (local[:, 2] > 0) & (local[:, 2] < CUP_DEPTH+RADIUS*1.5))
    if not len(keep):
        raise ValueError("No loaded grains found in the pre-pour snapshot")
    return keep


class Diagnostics:
    def __init__(self, model, data, grain_count, output, force_pan=False):
        self.model, self.data, self.output = model, data, output
        self.grains = np.array([model.body(f"grain_{i}").id for i in range(grain_count)])
        self.grain_geom = np.array([model.geom(f"grain_{i}").id for i in range(grain_count)])
        self.is_grain = np.zeros(model.ngeom, bool)
        self.is_grain[self.grain_geom] = True
        self.is_cup = model.geom_bodyid == model.body("scoop").id
        self.names = [model.geom(g).name for g in range(model.ngeom)]
        self.max_pen = 0.
        self.worst = {}
        self.max_pen_by_type = {}
        self.over_limit_steps = self.streak = self.longest_streak = 0
        self.solver = dict(steps=0, cap_steps=0, max_iterations=0,
                           max_final_gradient=0., max_final_improvement=0.)
        self.trace, self.stages = [], []
        self.started = time.perf_counter()
        self.scale_rows = []
        self.accounting = None
        if force_pan:
            self.accounting = GranularAccounting(
                model, data, self.grains, "source", "receiver",
                tool_membership=scoop_membership(model, data, "scoop", tip_in_tool=[0, 0, 0], cup_frame=np.eye(3)),
                expected_total_mass_g=grain_count*MASS_KG*1000, lag_s=.25, noise_g=0.,
                mass_tolerance_g=.005, settling_s=.5, force_stability_g=.002)

    def step(self):
        m, d = self.model, self.data
        if d.ncon:
            pairs, distances = np.asarray(d.contact.geom), np.asarray(d.contact.dist)
            both = self.is_grain[pairs].all(axis=1)
            either = self.is_grain[pairs].any(axis=1)
            cup = self.is_cup[pairs].any(axis=1)
            categories = {"grain_grain": both, "grain_cup": either & cup,
                          "grain_other": either & ~both & ~cup, "rigid_rigid": ~either}
            for name, mask in categories.items():
                if mask.any():
                    self.max_pen_by_type[name] = max(self.max_pen_by_type.get(name, 0.),
                                                      float(-distances[mask].min()))
            worst = int(distances.argmin())
            penetration = float(-distances[worst])
            if penetration > self.max_pen:
                self.max_pen = penetration
                self.worst = dict(time=float(d.time), penetration_m=penetration,
                                  pair=[self.names[g] for g in pairs[worst]])
            bad = bool((distances[either] < -.2*RADIUS).any())
            self.over_limit_steps += int(bad)
            self.streak = self.streak+1 if bad else 0
            self.longest_streak = max(self.longest_streak, self.streak)
        iters = np.asarray(d.solver_niter)
        active = np.flatnonzero(iters > 0)
        if len(active):
            self.solver["steps"] += 1
            self.solver["cap_steps"] += int((iters >= m.opt.iterations).any())
            self.solver["max_iterations"] = max(self.solver["max_iterations"], int(iters.max()))
            gradient = np.asarray(d.solver.gradient).reshape(mj.mjNISLAND, mj.mjNSOLVER)
            improvement = np.asarray(d.solver.improvement).reshape(mj.mjNISLAND, mj.mjNSOLVER)
            for island in active:
                index = min(int(iters[island]), mj.mjNSOLVER)-1
                self.solver["max_final_gradient"] = max(self.solver["max_final_gradient"], abs(float(gradient[island, index])))
                self.solver["max_final_improvement"] = max(self.solver["max_final_improvement"], abs(float(improvement[island, index])))
        if self.accounting is not None and self.accounting.tare_force_n is not None:
            elapsed = float(d.time)-self.accounting._last_sim_time
            if elapsed >= .02-1e-10:
                state = self.accounting.update(elapsed)
                row = {name: value for name, value in state.items() if name != "labels"}
                row["time_s"] = float(d.time)
                self.scale_rows.append(row)

    def snapshot(self, label):
        d = self.data
        p = d.xpos[self.grains].copy()
        rotation = d.body("scoop").xmat.reshape(3, 3)
        local = (p-d.body("scoop").xpos) @ rotation
        cup = ((np.linalg.norm(local[:, :2], axis=1) < CUP_INNER+RADIUS*.5)
               & (local[:, 2] > 0) & (local[:, 2] < CUP_DEPTH+RADIUS*1.5))
        source = (np.linalg.norm(p[:, :2], axis=1) < .012) & (p[:, 2] > .001) & (p[:, 2] < .057) & ~cup
        receiver = (np.linalg.norm(p[:, :2]-[.08, 0], axis=1) < .012) & (p[:, 2] > .001) & (p[:, 2] < .057) & ~cup
        other = ~(source | receiver | cup)
        row = dict(stage=label, sim_seconds=float(d.time), wall_seconds=time.perf_counter()-self.started,
                   source_n=int(source.sum()), receiver_n=int(receiver.sum()), scoop_n=int(cup.sum()),
                   other_n=int(other.sum()), receiver_g=float(receiver.sum()*MASS_KG*1000),
                   total_grains=len(p), max_penetration_m=self.max_pen, worst_contact=self.worst.copy(),
                   ncon=d.ncon, nefc=d.nefc, solver=self.solver.copy())
        self.trace.append(row)
        if self.scale_rows:
            row["scale"] = self.scale_rows[-1].copy()
        np.savez(self.output/(label+".npz"), qpos=d.qpos, qvel=d.qvel, ctrl=d.ctrl)
        print(json.dumps(row), flush=True)

    def move(self, label, seconds, position, angle=0.):
        m, d = self.model, self.data
        start, a0 = d.ctrl[:3].copy(), float(d.ctrl[3])
        started = time.perf_counter()
        for k in range(round(seconds/m.opt.timestep)):
            u = (k+1)/round(seconds/m.opt.timestep)
            u = u*u*(3-2*u)
            d.ctrl[:3] = start+(np.asarray(position)-start)*u
            d.ctrl[3] = a0+(angle-a0)*u
            mj.mj_step(m, d)
            self.step()
        self.stages.append(dict(stage=label, wall_seconds=time.perf_counter()-started, sim_seconds=seconds))
        self.snapshot(label)


def plot_force_curve(rows, output):
    """Save measured force and independently counted receiver mass together."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    times = np.asarray([row["time_s"] for row in rows])
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(times, [row["raw_net_mass_g"] for row in rows], color="#c46c33", linewidth=.9,
                alpha=.8, label="Measured pan force minus empty tare / gravity")
    ax.plot(times, [row["filtered_mass_g"] for row in rows], color="#386fa4", linewidth=1.4,
                label="Force reading with 0.25 s instrument lag")
    ax.step(times, [row["receiver_g"] for row in rows], where="post", color="#202020",
                linewidth=1.4, label="Physical grain mass inside receiver")
    axes[0].set_ylabel("Net load (g)")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].set_title("Contact transfer into a free vial on a force-sensing pan")
    axes[0].legend(fontsize=8, loc="upper left")
    error = 1000*np.asarray([row["raw_net_mass_g"]-row["receiver_g"] for row in rows])
    final = times >= times[-1]-1
    axes[1].plot(times[final], error[final], color="#c46c33", linewidth=1.)
    axes[1].axhline(0, color="#202020", linewidth=.8)
    axes[1].axhspan(-5, 5, color="#43875d", alpha=.12, label="±5 mg mass agreement tolerance")
    axes[1].set_ylabel("Measured force − inventory (mg)")
    axes[1].set_xlim(max(times[0], times[-1]-1), times[-1])
    axes[1].set_title("Final second: raw-force residual (settling is checked separately)", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].set_xlabel("Simulation time (s)")
    last = rows[-1]
    settled = str(last.get("settled", False)).lower() == "true"
    fig.supxlabel(f"Final receiver inventory {last['receiver_g']:.3f} g; measured net load {last['raw_net_mass_g']:.5f} g; "
                  f"settled gate: {'pass' if settled else 'fail'}", fontsize=9)
    fig.savefig(output/"force_curve.png", dpi=180)
    fig.savefig(output/"force_curve.svg")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--grain-count", type=int, default=400)
    p.add_argument("--dt", type=float, default=.00025)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--tolerance", type=float, default=1e-6)
    p.add_argument("--settling-seconds", type=float, default=0., help="Extra physical hold after the final upright stage")
    p.add_argument("--solver", choices=("cg", "newton"), default="cg")
    p.add_argument("--legacy-vial-colliders", action="store_true")
    p.add_argument("--force-pan", action="store_true", help="Use a free receiver on the production force-sensing pan")
    p.add_argument("--adaptive-insertion", action="store_true", help="Set insertion floor from observed bed top, clamped to 6–16 mm")
    p.add_argument("--sweep", action="store_true", help="Add a slow ±3 mm lateral sweep at insertion depth")
    p.add_argument("--refine-from", type=Path,
                   help="Replay only the loaded cup/receiver component from a pre-pour NPZ state")
    p.add_argument("--stop-after", choices=("settled", "loaded", "lifted", "over_receiver", "upright"), default="upright")
    args = p.parse_args()
    if not 1 <= args.grain_count <= 400 or not .00005 <= args.dt <= .002 or not 1 <= args.iterations <= 200:
        p.error("Use 1–400 grains, 50 µs–2 ms timestep and 1–200 solver iterations")
    if not 0 < args.tolerance <= 1e-4 or not 0 <= args.settling_seconds <= 60:
        p.error("Use a positive solver tolerance at most 1e-4 and 0–60 extra settling seconds")
    retained = None
    snapshot = None
    if args.refine_from:
        if args.force_pan:
            p.error("--force-pan uses a full empty-tare rollout and cannot be combined with --refine-from")
        if args.stop_after != "upright":
            p.error("--refine-from requires --stop-after upright")
        with np.load(args.refine_from, allow_pickle=False) as saved:
            snapshot = {name: saved[name].copy() for name in saved.files}
        retained = retained_in_snapshot(snapshot)
        args.grain_count = len(retained)
    args.output.mkdir(parents=True, exist_ok=False)
    model, data = build_fixture(args.grain_count, args.dt, args.iterations, args.solver,
                               args.legacy_vial_colliders, args.force_pan, args.tolerance)
    if snapshot is not None:
        initialize_pouring_component(model, data, snapshot, retained)
    probe = Diagnostics(model, data, args.grain_count, args.output, args.force_pan)
    insertion = .016
    plan = [("settled", .8, [0, 0, .08], 0), ("inserted", 2.5, [0, 0, .016], 0),
            ("loaded", .4, [0, 0, .016], 0), ("lifted", 3., [0, 0, .075], 0),
            ("over_receiver", 2., [.073, 0, .068], 0),
            ("tipped", 2., [.073, 0, .068], np.deg2rad(110)),
            ("drained", 1., [.073, 0, .068], np.deg2rad(110)),
            ("upright", 1., [.073, 0, .068], 0)]
    if retained is not None:
        plan = plan[5:]
    for label, seconds, position, angle in plan:
        if label == "inserted" and args.adaptive_insertion:
            bed_top = float(data.xpos[probe.grains, 2].max()+RADIUS)
            insertion = max(.006, min(.016, bed_top-.014))
            position = [0, 0, insertion]
        elif label == "loaded" and args.adaptive_insertion:
            position = [0, 0, insertion]
        probe.move(label, seconds, position, angle)
        if label == "settled" and probe.accounting is not None:
            probe.accounting.tare_empty_receiver()
            print(json.dumps({"tare_force_n": probe.accounting.tare_force_n,
                              "empty_pan_and_receiver_g": probe.accounting.raw_force_n()/9.81*1000}), flush=True)
        if label == "inserted" and args.sweep:
            height = float(data.ctrl[2])
            for name, x, duration in (("sweep_left", -.003, .8), ("sweep_right", .003, 1.6), ("sweep_center", 0., .8)):
                probe.move(name, duration, [x, 0, height])
        if label == args.stop_after:
            break
    if args.stop_after == "upright" and args.settling_seconds:
        probe.move("final_settling", args.settling_seconds, data.ctrl[:3].copy(), float(data.ctrl[3]))
    result = dict(mujoco_version=mj.__version__, fixture="actuated_xyz_and_tilt_fixed_source_free_receiver_on_pan" if args.force_pan else "actuated_xyz_and_tilt_fixed_vials",
                  vial_colliders="legacy_straight_walls_missing_neck" if args.legacy_vial_colliders else "profile_following_24_segments",
                  complete=args.stop_after == "upright", timestep=args.dt,
                  grain_count=args.grain_count, grain_radius_m=RADIUS, grain_mass_kg=MASS_KG,
                  density_kg_m3=MASS_KG/(4/3*math.pi*RADIUS**3), initial_mass_g=args.grain_count*MASS_KG*1000,
                  cup_outer_diameter_m=.014, cup_inner_diameter_m=.012, cup_depth_m=.006,
                  maximum_penetration_m=probe.max_pen, penetration_by_type_m=probe.max_pen_by_type,
                  grain_penetration_limit_m=.2*RADIUS, violation_steps=probe.over_limit_steps,
                  longest_violation_s=probe.longest_streak*args.dt,
                  solver=probe.solver, trace=probe.trace, timing=probe.stages,
                  all_finite=bool(np.isfinite(data.qpos).all()),
                  grain_pose_writes_after_initialization=0,
                  inventory_note="Exclusive geometric categories at snapshot times; production accounting additionally checks support contacts.")
    result["configuration"] = dict(force_pan=args.force_pan, adaptive_insertion=args.adaptive_insertion,
                                    sweep=args.sweep, solver=args.solver, iterations=args.iterations,
                                    tolerance=args.tolerance, extra_settling_seconds=args.settling_seconds,
                                    insertion_depth_m=insertion if args.adaptive_insertion else .016,
                                    scoop_mass_kg=float(model.body_mass[model.body("scoop").id]))
    if probe.scale_rows:
        result["final_scale"] = probe.scale_rows[-1]
        result["tare_force_n"] = probe.accounting.tare_force_n
        with (args.output/"force_curve.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(probe.scale_rows[0]))
            writer.writeheader()
            writer.writerows(probe.scale_rows)
        plot_force_curve(probe.scale_rows, args.output)
    result["checks"] = dict(finite=bool(np.isfinite(data.qpos).all()),
                            grain_contact_tolerance=probe.over_limit_steps == 0,
                            final_grain_count_conserved=sum(probe.trace[-1][key] for key in
                                ("source_n", "receiver_n", "scoop_n", "other_n")) == args.grain_count,
                            no_final_unassigned_grains=probe.trace[-1]["other_n"] == 0)
    if probe.scale_rows:
        last = probe.scale_rows[-1]
        result["checks"].update(settled_force_matches_inventory=last["mass_force_agreement"],
                                no_bare_pan_grains=last["bare_pan_grain_mass_g"] == 0,
                                no_foreign_scale_contact=not (last["foreign_pan_contact"] or last["foreign_receiver_contact"]))
    result["fixture_checks_passed"] = all(result["checks"].values())
    if retained is not None:
        result["refinement"] = dict(snapshot=str(args.refine_from), original_grain_ids=retained.tolist(),
                                    scope="Loaded cup and receiver; disconnected stationary source pile omitted")
    (args.output/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    mj.mj_saveModel(model, str(args.output/"fixture.mjb"))


if __name__ == "__main__":
    main()
