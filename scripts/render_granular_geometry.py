"""Render four views of a robot reset or a saved physical simulation state.

Optional state restoration is for visualization only, never collection or
evaluation. After reset/restoration, rendering leaves all physical state intact.
"""
import argparse
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chemlab_rl.granular_env import GranularWeighEnv
from chemlab_rl.granular_geometry import SCOOP_TIP
from chemlab_rl.granular_material import ScaleInstrument

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--state", type=Path, help="Saved qpos/qvel/ctrl NPZ; visualization only")
parser.add_argument("--grains", type=int, default=200)
parser.add_argument("--target", type=float, default=1.)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
env = GranularWeighEnv(grain_count=args.grains, images=False, check_collisions=True,
                       instrument=ScaleInstrument(noise_g=0.), target_g=args.target)
env.reset(seed=args.seed)
model, data = env.model, env.data
if args.state is not None:
    with np.load(args.state, allow_pickle=False) as saved:
        for name in ("qpos", "qvel", "ctrl"):
            array = saved[name]
            current = getattr(data, name)
            if array.shape != current.shape or not np.isfinite(array).all():
                raise ValueError(f"Saved {name} is incompatible with the requested scene")
            current[:] = array
        mj.mj_forward(model, data)
        if "reading_g" in saved:
            env.scene.show(float(saved["reading_g"]))
frozen = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), float(data.time))
model.vis.global_.offwidth, model.vis.global_.offheight = 1200, 900
model.vis.headlight.ambient[:] = [.65, .65, .65]
model.vis.headlight.diffuse[:] = [.8, .8, .8]
option = mj.MjvOption()
option.sitegroup[:] = 0
source = data.body("source").xpos.copy()
receiver = data.body("vial_a").xpos.copy()
tool = data.body("spatula").xpos.copy()
cup = tool + data.body("spatula").xmat.reshape(3, 3) @ SCOOP_TIP
views = {
    "whole_bench": (np.array([0., .05, env.legacy.BOARD_TOP+.15]), 1.8, -65., -30.),
    "source": (source+[0, 0, .025], .13, 115., -35.),
    "scoop": (cup, .08, 10., -20.),
    "receiver": (receiver+[0, 0, .025], .15, 115., -30.),
}
renderer = mj.Renderer(model, 900, 1200)
try:
    for label, (target, distance, azimuth, elevation) in views.items():
        camera = mj.MjvCamera()
        camera.lookat[:] = target
        camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
        renderer.update_scene(data, camera=camera, scene_option=option)
        Image.fromarray(renderer.render()).save(args.output/(label+".png"))
finally:
    renderer.close()
for original, actual in zip(frozen[:3], (data.qpos, data.qvel, data.ctrl)):
    np.testing.assert_array_equal(original, actual)
assert frozen[3] == data.time
metadata = dict(scope=("Saved physical simulation state restored for visualization; no dynamics after restoration"
                       if args.state is not None else
                       "Current complete robot scene immediately after reset; no object edits afterward"),
    restored_state_path=str(args.state.resolve()) if args.state is not None else None,
    rendering_is_validation_run=False,
    seed=args.seed, grain_count=args.grains, grain_mass_g=env.material.grain_mass_g,
    source_mass_kg=float(model.body_mass[model.body("source").id]),
    receiver_mass_kg=float(model.body_mass[model.body("vial_a").id]),
    tool_mass_kg=float(model.body_mass[model.body("spatula").id]),
    source_origin_m=source.tolist(), receiver_origin_m=receiver.tolist(), cup_origin_m=cup.tolist(),
    timestep_s=float(model.opt.timestep), solver_tolerance=float(model.opt.tolerance),
    collision_failure=env.scene.bad, qpos_qvel_ctrl_time_unchanged_by_render=True,
    views={label: dict(lookat=target.tolist(), distance=distance, azimuth=azimuth, elevation=elevation)
           for label, (target, distance, azimuth, elevation) in views.items()})
(args.output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
env.close()
print(json.dumps(metadata), flush=True)
