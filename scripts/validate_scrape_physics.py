"""Audit corrected scene dimensions/support and optionally render inspection views.

This certifies the reset scene only. A full expert episode still requires the
environment's continuous physical validation and a separate accepted rollout.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemlab_rl.scrape_env import ScrapeWeighEnv


def mesh_bounds(env, prefix, kind):
    m = env.model
    points = []
    for g in range(m.ngeom):
        if not (m.geom(g).name or "").startswith(f"{prefix}_{kind}_"):
            continue
        mesh = int(m.geom_dataid[g])
        start, count = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
        rotation = np.zeros(9)
        env.mj.mju_quat2Mat(rotation, m.geom_quat[g])
        points.append(m.mesh_vert[start:start + count] @ rotation.reshape(3, 3).T + m.geom_pos[g])
    points = np.concatenate(points)
    return {"min_m": points.min(0).tolist(), "max_m": points.max(0).tolist(),
            "extents_m": np.ptp(points, axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/pilot/physics_validation")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = ScrapeWeighEnv(images=False)
    try:
        env.reset(seed=20261118)
        names = ("source", "vial_a", "spatula", "discard")
        start = {n: env.data.body(n).xpos.copy() for n in names}
        for _ in range(10):
            _, _, terminated, _, info = env.step(env.hold_action())
            if terminated:
                raise RuntimeError(info["failure"])
        drift = {n: float(np.linalg.norm(env.data.body(n).xpos - start[n])) for n in names}
        assert max(drift.values()) < 0.003
        dimensions = {n: {k: mesh_bounds(env, n, k) for k in ("visual", "collision")}
                      for n in ("source", "vial_a")}
        for n in dimensions:
            np.testing.assert_allclose(dimensions[n]["visual"]["extents_m"], [0.028, 0.028, 0.057], atol=1e-6)
            np.testing.assert_allclose(dimensions[n]["collision"]["extents_m"], [0.02898774, 0.02898774, 0.057], atol=1e-6)
        report = {"scope": "reset scene only; not full expert rollout certification", "passed": True,
                  "dimensions": dimensions, "passive_one_second_drift_m": drift,
                  "receiver_supported": env._receiver_supported(),
                  "physics_checks": env.physics_checks, "contact_minima_m": env.contact_minima,
                  "grasp_model": "mujoco_friction_only", "free_body_pose_writes_after_reset": False,
                  "powder_model": "synthetic finite mass, bounded visual fill; not granular simulation",
                  "collision_hulls": "canonical conservative hulls extend ~0.494 mm beyond visual radius"}
        (args.output / "reset_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.render:
            from PIL import Image, ImageDraw
            with env.mj.Renderer(env.model, height=600, width=800) as renderer:
                images = []
                for label, lookat, distance, azimuth, elevation in (
                    ("Whole scene", [0, 0, 0.85], 2.05, 135, -25),
                    ("Weighing bench", [0.02, 0.05, 0.91], 0.95, -60, -40),
                    ("Source and powder fill", [-0.34, 0.06, 0.81], 0.18, -50, -35),
                    ("Connected spatula", [0.40, 0.12, 0.80], 0.40, -90, -45),
                ):
                    camera = env.mj.MjvCamera()
                    camera.lookat[:], camera.distance = lookat, distance
                    camera.azimuth, camera.elevation = azimuth, elevation
                    renderer.update_scene(env.data, camera=camera, scene_option=env._render_option)
                    frame = Image.fromarray(renderer.render().copy())
                    ImageDraw.Draw(frame).text((15, 15), label, fill="black", stroke_width=1, stroke_fill="white")
                    images.append(frame)
                sheet = Image.new("RGB", (1600, 1200), "white")
                for i, frame in enumerate(images):
                    sheet.paste(frame, ((i % 2) * 800, (i // 2) * 600))
                sheet.save(args.output / "reset_inspection.png")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
