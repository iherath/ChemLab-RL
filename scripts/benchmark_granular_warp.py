"""Bounded, non-production CPU/MJWarp granular contact benchmark.

The scoop is a prescribed mocap fixture, not a robot grasp. Grains are free
bodies throughout each rollout. Initialization is repeated for each backend;
no particle pose or velocity is prescribed after a rollout starts.

MJWarp API reference: https://mujoco.readthedocs.io/en/stable/mjwarp/index.html
Pinned release: mujoco-warp v3.16.0; runtime measurements exclude compilation,
model transfer, graph capture, snapshots, and output serialization.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import mujoco as mj
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "chem_rl_sim"), str(ROOT / "chem_rl_sim/third_party/urgantry_sim")]
from chem_rl_sim.assets import add_asset

try:
    import warp as wp
except ImportError:
    wp = None

RADIUS = .0015
MASS = .00002
CUP_RADIUS = .007
CUP_INNER = .006
CUP_DEPTH = .006

if wp is not None:
    @wp.kernel
    def move_fixture(mocap_pos: wp.array2d(dtype=wp.vec3),
                     step: wp.array(dtype=int), params: wp.array(dtype=float)):
        w = wp.tid()
        u = wp.min(1.0, float(step[w] + 1) / params[2])
        u = u * u * (3.0 - 2.0 * u)
        mocap_pos[w, 0] = wp.vec3(0.0, 0.0, params[0] + (params[1] - params[0]) * u)
        step[w] += 1

    @wp.kernel
    def audit_contacts(
        nacon: wp.array(dtype=int), dist: wp.array(dtype=float),
        geoms: wp.array(dtype=wp.vec2i), worlds: wp.array(dtype=int),
        is_grain: wp.array(dtype=int), max_pen: wp.array(dtype=float), max_fixture_pen: wp.array(dtype=float),
    ):
        i = wp.tid()
        if i < nacon[0]:
            g = geoms[i]
            if (g[0] >= 0 and is_grain[g[0]] != 0) or (g[1] >= 0 and is_grain[g[1]] != 0):
                wp.atomic_max(max_pen, worlds[i], wp.max(0.0, -dist[i]))
            else:
                wp.atomic_max(max_fixture_pen, worlds[i], wp.max(0.0, -dist[i]))

    @wp.kernel
    def audit_finite(q: wp.array2d(dtype=float), bad: wp.array(dtype=int)):
        w, j = wp.tid()
        if not wp.isfinite(q[w, j]):
            wp.atomic_max(bad, w, 1)


def event(name: str, **values) -> None:
    print(json.dumps({"event": name, **values}), flush=True)


def save(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class BenchmarkFailure(RuntimeError):
    """Preserve completed stages when a GPU validation guard aborts a run."""
    def __init__(self, message: str, partial: dict):
        super().__init__(message)
        self.partial = partial


def build_model(particles: int = 400, dt: float = .001, iterations: int = 200,
                condim: int = 3, adhesion: float = 0.0):
    s = mj.MjSpec()
    s.option.timestep = dt
    s.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    s.option.cone = mj.mjtCone.mjCONE_PYRAMIDAL
    s.option.solver = mj.mjtSolver.mjSOL_CG
    s.option.iterations = iterations
    s.option.jacobian = mj.mjtJacobian.mjJAC_SPARSE
    s.option.noslip_iterations = 0
    s.option.tolerance = 1e-6
    s.worldbody.add_geom(name="floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[.4, .4, .01])
    for name, pos in (("source", [0, 0, 0]), ("receiver", [.08, 0, 0])):
        body = add_asset(s, s.worldbody, "vial_20ml_open", name, pos,
                         free=False, friction=(.4, .0001, .00002))
        body.add_site(name=name + "_load_site", pos=[0, 0, 0], size=[.001] * 3)
        s.add_sensor(name=name + "_force", type=mj.mjtSensor.mjSENS_FORCE,
                     objtype=mj.mjtObj.mjOBJ_SITE, objname=name + "_load_site")
    cup = s.worldbody.add_body(name="scoop", pos=[0, 0, .08], mocap=True)
    kwargs = dict(friction=[.4, .0001, .00002], condim=condim,
                  solref=[.003, 1], solimp=[.95, .99, .0001, .5, 2])
    cup.add_geom(name="scoop_bottom", type=mj.mjtGeom.mjGEOM_CYLINDER,
                 size=[CUP_RADIUS, .0005], pos=[0, 0, .0005], **kwargs)
    for k in range(16):
        a = 2 * np.pi * k / 16
        outer = CUP_RADIUS * np.cos(np.pi / 16)
        radius = (outer + CUP_INNER) / 2
        cup.add_geom(name=f"scoop_wall_{k}", type=mj.mjtGeom.mjGEOM_BOX,
                     size=[(outer - CUP_INNER) / 2, CUP_RADIUS * np.sin(np.pi / 16), CUP_DEPTH / 2],
                     pos=[radius * np.cos(a), radius * np.sin(a), CUP_DEPTH / 2],
                     quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)], **kwargs)
    cup.add_geom(name="scoop_stem", type=mj.mjtGeom.mjGEOM_CAPSULE,
                 fromto=[-.0055, 0, .005, -.0055, 0, .040], size=[.001], **kwargs)
    spacing = .0032
    xy = []
    for j in range(-4, 5):
        for i in range(-4, 5):
            x, y = spacing * (i + .5 * (j % 2)), spacing * np.sqrt(3) / 2 * j
            if x * x + y * y <= .0093 ** 2:
                xy.append((x, y))
    for k in range(particles):
        layer, slot = divmod(k, len(xy))
        x, y = xy[slot]
        body = s.worldbody.add_body(name=f"grain_{k}", pos=[x, y, .0022 + RADIUS + spacing * layer])
        body.add_freejoint(name=f"grain_{k}")
        body.add_geom(name=f"grain_{k}", type=mj.mjtGeom.mjGEOM_SPHERE,
                      size=[RADIUS], mass=MASS, friction=[.5, .0001, .00002],
                      condim=condim, solref=[.003, 1], solimp=[.95, .99, .0001, .5, 2],
                      adhesion=adhesion, gap=.00001 if adhesion else 0.)
    model = s.compile()
    ids = np.array([model.body(f"grain_{k}").id for k in range(particles)], dtype=int)
    mask = np.zeros(model.ngeom, dtype=np.int32)
    for k in range(particles):
        mask[model.geom(f"grain_{k}").id] = 1
    return model, ids, mask


def stages(dt: float):
    # Same deterministic fixture path on CPU and all GPU worlds.
    return [("settled", round(.8 / dt), .08),
            ("approached", round(.75 / dt), .028),
            ("inserted", round(1.0 / dt), .0105),
            ("lifted", round(3.0 / dt), .06),
            ("rested", round(.3 / dt), .06)]


def snapshot(model, ids, qpos, qvel, xpos, sensor, cup_pos, tare, max_pen, max_fixture_pen):
    pos = xpos[ids]
    source = (np.linalg.norm(pos[:, :2], axis=1) < .012) & (pos[:, 2] > .001) & (pos[:, 2] < .057)
    receiver = (np.linalg.norm(pos[:, :2] - [.08, 0], axis=1) < .012) & (pos[:, 2] > .001) & (pos[:, 2] < .057)
    local = pos - cup_pos
    retained = ((np.linalg.norm(local[:, :2], axis=1) < CUP_INNER + .5 * RADIUS)
                & (local[:, 2] > 0) & (local[:, 2] < CUP_DEPTH + 1.5 * RADIUS))
    masses = model.body_mass[ids]
    source_force = float(sensor[model.sensor("source_force").adr[0] + 2] - tare[0])
    receiver_force = float(sensor[model.sensor("receiver_force").adr[0] + 2] - tare[1])
    return {
        "finite": bool(np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(sensor).all()),
        "particle_count": int(len(ids)), "total_particle_mass_g": float(masses.sum() * 1000),
        "source_n": int(source.sum()), "receiver_n": int(receiver.sum()), "scoop_vicinity_n": int(retained.sum()),
        "outside_containers_and_scoop_n": int((~(source | receiver | retained)).sum()),
        "escaped_scene_n": int(((np.abs(pos[:, :2]).max(axis=1) > .4) | (pos[:, 2] < 0)).sum()),
        "source_mass_g": float(masses[source].sum() * 1000),
        "receiver_mass_g": float(masses[receiver].sum() * 1000),
        "source_load_g": source_force / 9.81 * 1000,
        "receiver_load_g": receiver_force / 9.81 * 1000,
        "max_grain_penetration_m": float(max_pen),
        "max_fixture_penetration_m": float(max_fixture_pen),
        "grain_min_z_m": float(pos[:, 2].min()), "grain_max_z_m": float(pos[:, 2].max()),
        "grain_rms_velocity_m_s": float(np.sqrt(np.mean(qvel.reshape(-1, 6)[:, :3] ** 2))),
        "grain_center_mean_m": pos.mean(axis=0).tolist(),
    }


def cpu_run(model, ids, mask, chunk_steps: int):
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    tare = [float(data.sensordata[model.sensor(n + "_force").adr[0] + 2]) for n in ("source", "receiver")]
    rows, max_pen, max_fixture_pen, elapsed = [], 0., 0., 0.
    total_steps = 0
    max_solver_iterations = 0
    iteration_limit_steps = 0
    rollout_started = time.perf_counter()
    for label, nstep, target in stages(model.opt.timestep):
        start_z = float(data.mocap_pos[0, 2])
        for first in range(0, nstep, chunk_steps):
            count = min(chunk_steps, nstep - first)
            before = time.perf_counter()
            for offset in range(count):
                u = (first + offset + 1) / nstep
                u = u * u * (3 - 2 * u)
                data.mocap_pos[0, 2] = start_z + (target - start_z) * u
                mj.mj_step(model, data)
                observed_iterations = int(np.max(data.solver_niter))
                max_solver_iterations = max(max_solver_iterations, observed_iterations)
                iteration_limit_steps += int(observed_iterations >= model.opt.iterations)
                if data.ncon:
                    contact = data.contact
                    grain_contact = mask[contact.geom[:, 0]].astype(bool) | mask[contact.geom[:, 1]].astype(bool)
                    if grain_contact.any():
                        max_pen = max(max_pen, float(-contact.dist[grain_contact].min()))
                    if (~grain_contact).any():
                        max_fixture_pen = max(max_fixture_pen, float(-contact.dist[~grain_contact].min()))
            elapsed += time.perf_counter() - before
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("CPU produced non-finite state")
        total_steps += nstep
        row = snapshot(model, ids, data.qpos, data.qvel, data.xpos, data.sensordata,
                       data.mocap_pos[0], tare, max_pen, max_fixture_pen)
        row.update(stage=label, sim_seconds=float(data.time), ncon=data.ncon, nefc=data.nefc)
        rows.append(row)
        event("cpu_stage", **row, measured_wall_seconds=elapsed)
    warnings = data.warning.number.tolist()
    if any(warnings):
        raise RuntimeError(f"CPU simulation warnings: {warnings}")
    return dict(worlds=1, steps_per_world=total_steps, measured_wall_seconds=elapsed,
                rollout_wall_seconds=time.perf_counter() - rollout_started,
                world_steps_per_second=total_steps / elapsed,
                aggregate_sim_seconds_per_wall_second=total_steps * model.opt.timestep / elapsed,
                stages=rows, warnings=warnings, tare_force_z_N=tare,
                max_solver_iterations_observed=max_solver_iterations,
                iteration_limit_steps=iteration_limit_steps)


def gpu_run(model, ids, mask, nworld, chunk_steps, nconmax, njmax, njmax_nnz):
    import mujoco_warp as mjw
    if wp is None or not wp.is_cuda_available():
        raise RuntimeError("CUDA Warp device unavailable")
    preparation = time.perf_counter()
    with wp.ScopedDevice("cuda:0"):
        m = mjw.put_model(model)
        d = mjw.make_data(model, nworld=nworld, nconmax=nconmax, njmax=njmax,
                          njmax_nnz=njmax_nnz, nccdmax=min(nconmax, 1024))
        grain_mask = wp.array(mask, dtype=int)
        max_pen = wp.zeros(nworld, dtype=float)
        max_fixture_pen = wp.zeros(nworld, dtype=float)
        bad = wp.zeros(nworld, dtype=int)
        fixture_step = wp.zeros(nworld, dtype=int)
        fixture_params = wp.array([.08, .08, 1.], dtype=float)
        # Warm all kernels before capture. This temporary state is discarded.
        wp.launch(move_fixture, dim=nworld, inputs=[d.mocap_pos, fixture_step, fixture_params])
        mjw.step(m, d)
        wp.launch(audit_contacts, dim=d.naconmax,
                  inputs=[d.nacon, d.contact.dist, d.contact.geom, d.contact.worldid, grain_mask, max_pen, max_fixture_pen])
        wp.launch(audit_finite, dim=(nworld, model.nq), inputs=[d.qpos, bad])
        wp.launch(audit_finite, dim=(nworld, model.nv), inputs=[d.qvel, bad])
        wp.synchronize()
        with wp.ScopedCapture() as capture:
            for _ in range(chunk_steps):
                wp.launch(move_fixture, dim=nworld, inputs=[d.mocap_pos, fixture_step, fixture_params])
                mjw.step(m, d)
                wp.launch(audit_contacts, dim=d.naconmax,
                          inputs=[d.nacon, d.contact.dist, d.contact.geom, d.contact.worldid, grain_mask, max_pen, max_fixture_pen])
                wp.launch(audit_finite, dim=(nworld, model.nq), inputs=[d.qpos, bad])
                wp.launch(audit_finite, dim=(nworld, model.nv), inputs=[d.qvel, bad])
        # Reset is initialization of the measured rollout, never a dynamic correction.
        mjw.reset_data(m, d)
        max_pen.zero_()
        max_fixture_pen.zero_()
        bad.zero_()
        mjw.forward(m, d)
        wp.synchronize()
        sensor = d.sensordata.numpy()
        tares = [[float(sensor[w, model.sensor(n + "_force").adr[0] + 2])
                  for n in ("source", "receiver")] for w in range(nworld)]
        prepare_seconds = time.perf_counter() - preparation
        event("gpu_prepared", worlds=nworld, preparation_seconds=prepare_seconds,
              nconmax=nconmax, njmax=njmax, njmax_nnz=njmax_nnz)
        rows, elapsed, total_steps = [], 0., 0
        cup_pos = np.zeros((nworld, 1, 3), dtype=np.float32)
        cup_pos[:, 0, 2] = .08
        rollout_started = time.perf_counter()
        for label, nstep, target in stages(model.opt.timestep):
            if nstep % chunk_steps:
                raise ValueError("Each stage must be divisible by chunk_steps")
            start_z = float(cup_pos[0, 0, 2])
            fixture_step.zero_()
            fixture_params.assign(np.array([start_z, target, nstep], dtype=np.float32))
            wp.synchronize()
            for first in range(0, nstep, chunk_steps):
                before = time.perf_counter()
                wp.capture_launch(capture.graph)
                wp.synchronize()
                elapsed += time.perf_counter() - before
                overflow = d.overflow.numpy()
                if np.any(overflow):
                    flag_names = sorted({flag.name for value in overflow for flag in mjw.OverflowType
                                         if int(flag) and int(value) & int(flag)})
                    partial = dict(worlds=nworld, status="failed", preparation_seconds=prepare_seconds,
                                   measured_wall_seconds=elapsed,
                                   completed_steps_per_world=total_steps + first + chunk_steps,
                                   failed_stage=label, stages=rows, overflow=overflow.tolist(),
                                   overflow_flags=flag_names)
                    event("gpu_guard_failed", **partial)
                    raise BenchmarkFailure(f"GPU validation guard: {flag_names}; flags={overflow.tolist()}", partial)
                if np.any(bad.numpy()):
                    raise RuntimeError("GPU produced non-finite state; benchmark invalid")
            total_steps += nstep
            cup_pos[:, 0, 2] = target
            qpos, qvel, xpos = d.qpos.numpy(), d.qvel.numpy(), d.xpos.numpy()
            sensor, penetration = d.sensordata.numpy(), max_pen.numpy()
            fixture_penetration = max_fixture_pen.numpy()
            stage_rows = [snapshot(model, ids, qpos[w], qvel[w], xpos[w], sensor[w],
                                   cup_pos[w, 0], tares[w], penetration[w], fixture_penetration[w]) for w in range(nworld)]
            for w, row in enumerate(stage_rows):
                row.update(stage=label, world=w)
            rows.extend(stage_rows)
            event("gpu_stage", worlds=nworld, stage=label, measured_wall_seconds=elapsed,
                  first_world=stage_rows[0], all_worlds_finite=all(r["finite"] for r in stage_rows))
        return dict(worlds=nworld, steps_per_world=total_steps, measured_wall_seconds=elapsed,
                    rollout_wall_seconds=time.perf_counter() - rollout_started,
                    preparation_seconds=prepare_seconds, world_steps_per_second=nworld * total_steps / elapsed,
                    aggregate_sim_seconds_per_wall_second=nworld * total_steps * model.opt.timestep / elapsed,
                    stages=rows, overflow=d.overflow.numpy().tolist(), tare_force_z_N=tares)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--particles", type=int, default=400)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--condim", type=int, choices=[3, 4, 6], default=3)
    p.add_argument("--adhesion", type=float, default=0.)
    p.add_argument("--max-seconds", type=int, default=840)
    p.add_argument("--chunk-steps", type=int, default=10)
    p.add_argument("--nconmax", type=int, default=4096)
    p.add_argument("--njmax", type=int, default=32768)
    p.add_argument("--njmax-nnz", type=int, default=524288)
    args = p.parse_args()
    if not 1 <= args.particles <= 400 or not 1 <= args.max_seconds <= 900:
        raise ValueError("Benchmark bounded to 400 particles and 900 seconds")
    if args.chunk_steps not in (1, 5, 10):
        raise ValueError("Use 1, 5, or 10 substeps per graph")
    def expired(*_):
        raise TimeoutError("Absolute benchmark runtime limit reached")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.max_seconds)
    result = {"status": "running", "purpose": "isolated prescribed-scoop physics benchmark; not a robot policy result",
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "versions": {"mujoco": mj.__version__, "numpy": np.__version__},
              "geometry": {"particles": args.particles, "particle_radius_m": RADIUS,
                           "particle_mass_kg": MASS, "material_density_kg_m3": MASS / (4 * math.pi * RADIUS ** 3 / 3),
                           "total_mass_g": args.particles * MASS * 1000},
              "timing_scope": "Stepping plus contact/finiteness audit; excludes compile/capture/transfers/reporting. CPU is one world; GPU1 and GPU8 are batched throughput.",
              "particle_pose_writes_during_rollout": 0}
    save(args.output, result)
    try:
        model, ids, mask = build_model(args.particles, iterations=args.iterations,
                                       condim=args.condim, adhesion=args.adhesion)
        result["model"] = {"nq": model.nq, "nv": model.nv, "ngeom": model.ngeom,
                           "dt": model.opt.timestep, "solver": "CG", "jacobian": "sparse", "cone": "pyramidal"}
        event("model_ready", **result["model"])
        result["cpu"] = cpu_run(model, ids, mask, args.chunk_steps)
        save(args.output, result)
        if not args.cpu_only:
            result["versions"].update({name: importlib.metadata.version(name) for name in ("mujoco-warp", "warp-lang")})
            if result["versions"]["mujoco"] != "3.16.0" or result["versions"]["mujoco-warp"] != "3.16.0":
                raise RuntimeError("Benchmark requires pinned MuJoCo and MJWarp 3.16.0")
            wp.init()
            result["device"] = str(wp.get_device("cuda:0"))
            result["gpu"] = []
            for worlds in (1, 8):
                event("gpu_preparing", worlds=worlds)
                run = gpu_run(model, ids, mask, worlds, args.chunk_steps, args.nconmax, args.njmax, args.njmax_nnz)
                run["aggregate_speedup_over_cpu1"] = run["world_steps_per_second"] / result["cpu"]["world_steps_per_second"]
                result["gpu"].append(run)
                save(args.output, result)
        checks = []
        for backend, rows in [("cpu", result["cpu"]["stages"])] + [(f"gpu{g['worlds']}", g["stages"]) for g in result.get("gpu", [])]:
            settled = [r for r in rows if r["stage"] == "settled"]
            checks.append({"backend": backend, "finite": all(r["finite"] for r in rows),
                           "declared_particle_mass_correct": all(abs(r["total_particle_mass_g"] - args.particles * MASS * 1000) < 1e-8 for r in rows),
                           "no_floor_tunneling": all(r["grain_min_z_m"] >= .5 * RADIUS for r in rows),
                           "no_escaped_particles": all(r["escaped_scene_n"] == 0 for r in rows),
                           "penetration_below_half_radius": all(r["max_grain_penetration_m"] < .5 * RADIUS for r in rows),
                           "fixture_penetration_below_100um": all(r["max_fixture_penetration_m"] < .0001 for r in rows),
                           "settled_scale_within_10_percent": all(abs(r["source_load_g"] - r["source_mass_g"]) < max(.05, .1 * r["source_mass_g"]) for r in settled)})
            if backend == "cpu":
                checks[-1]["solver_iteration_limit_not_reached"] = result["cpu"]["iteration_limit_steps"] == 0
        result["checks"] = checks
        result["physics_checks_passed"] = all(all(v for k, v in c.items() if k != "backend") for c in checks)
        result["status"] = "complete" if result["physics_checks_passed"] else "complete_with_physics_failures"
        save(args.output, result)
        event("benchmark_complete", status=result["status"], checks=checks)
    except BaseException as exc:
        if isinstance(exc, BenchmarkFailure):
            result.setdefault("gpu", []).append(exc.partial)
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(args.output, result)
        raise
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
