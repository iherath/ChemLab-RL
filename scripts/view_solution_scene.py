"""Open the subtask-2 (solution prep) scene in MuJoCo's interactive viewer.

Static scene (drag objects with Ctrl+right-click, rotate camera with the mouse):
    .venv-vial/bin/python scripts/view_solution_scene.py --syringe 3mL

Watch the scripted two-handed syringe motion (grasp barrel + rod, lift,
pull plunger, push back) in real time. On macOS the live viewer needs mjpython:
    .venv-vial/bin/mjpython scripts/view_solution_scene.py --syringe 3mL --demo

Watch a FULL expert episode (draw/dispense cycles, return syringe, snap cap,
swirl, place) through the real env, with live liquid/subgoal printouts:
    .venv-vial/bin/mjpython scripts/view_solution_scene.py --syringe 1mL --expert 6 --speed 3

This is a viewing tool only: no data is recorded.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from chemlab_rl.solution_scene import SolutionScene  # noqa: E402
from chemlab_rl.solution_state import SYRINGES  # noqa: E402


def _face(z, y):
    z = np.asarray(z, float); z /= np.linalg.norm(z)
    y = np.asarray(y, float); y = y - z * np.dot(y, z); y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


LEFT = _face([1, 0, 0], [0, 1, 0])     # barrel arm, from -x
RIGHT = _face([0, 1, 0], [1, 0, 0])    # plunger arm, from the robot side


class _ViewerBridge:
    """Minimal env for Leo's keyframe Session that steps physics and syncs the viewer."""

    def __init__(self, scene, viewer, speed):
        self.model, self.data, self.hand = scene.model, scene.data, "robotiq"
        self.n_substeps, self._step_count = scene.n_substeps, 0
        self.viewer, self.speed = viewer, speed

    def task_status(self):
        return {"success": False}

    def reset(self, seed=None):
        return None, self.task_status()

    def step(self, ctrl):
        t0 = time.time()
        self.data.ctrl[:] = ctrl
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        self.viewer.sync()
        dt = self.n_substeps * self.model.opt.timestep / self.speed
        time.sleep(max(0.0, dt - (time.time() - t0)))
        return None, 0.0, False, False, self.task_status()


def run_demo(scene, viewer, speed):
    from chem_rl_sim.teleop.keyframe import Session
    ses = Session(_ViewerBridge(scene, viewer, speed))
    g = scene.geom

    def drain():
        while ses.tick() and viewer.is_running():
            pass

    def go(side, pos):
        other = "right" if side == "left" else "left"
        ses.target_to_current(other)
        ses.set_target(side, np.asarray(pos, float), LEFT if side == "left" else RIGHT)
        if ses.plan().status == "blocked":
            print(f"[skip] {side} waypoint blocked: {ses.plan().reasons}")
            return
        ses.execute(force=True)
        drain()

    def follow(targets, max_ticks=200):
        for side in ("left", "right"):
            if side in targets:
                ses.set_target(side, targets[side], LEFT if side == "left" else RIGHT)
            else:
                ses.target_to_current(side)
        ses.follow = True
        n = 0
        while ses.tick() and n < max_ticks and viewer.is_running():
            n += 1
        ses.follow = False

    base = scene.data.body("syringe").xpos.copy()
    lp, rp = base + [0, 0, g.grip_z], base + [0, 0, g.rod_grip_z]
    print("left arm: grasp barrel");  [go("left", p) for p in (lp + [-0.08, 0, 0.03], lp + [-0.06, 0, 0], lp)]
    ses.grip("left", True); drain()
    print("right arm: pinch plunger rod"); [go("right", p) for p in (rp + [0, -0.06, 0], rp)]
    ses.grip("right", True); drain()
    print("lift syringe out of the rack"); follow({"left": lp + [0, 0, 0.07], "right": rp + [0, 0, 0.07]})
    pull = 0.6 * g.stroke
    print(f"pull plunger {pull * 1000:.0f} mm"); follow({"right": rp + [0, 0, 0.07 + pull]})
    print(f"  stroke now {scene.plunger_stroke() * 1000:.1f} mm")
    print("push plunger back"); follow({"right": rp + [0, 0, 0.07]})
    print(f"  stroke now {scene.plunger_stroke() * 1000:.1f} mm. Close the window to exit.")
    while viewer.is_running():
        viewer.sync(); time.sleep(0.05)


def run_expert_view(syringe: str, seed: int, speed: float, every: int = 1):
    from chemlab_rl.solution_env import SolutionPrepEnv
    from chemlab_rl.solution_skills import run_expert
    env = SolutionPrepEnv(images=False)
    env.reset(seed=seed, options={"syringe": syringe})          # builds this syringe's model
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [-0.17, 0.03, 0.85]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.75, 210, -25
        step, last = env.step, [None]

        def watched(action):
            t0 = time.time()
            out = step(action)
            if env.steps % every:                     # redraw only every `every` steps
                return out
            viewer.sync()
            dones = "".join(str(int(b)) for b in env.subgoal_dones)
            line = (f"step {env.steps:5d}  vial {env.liquid.vial_mL:5.3f} mL  syringe {env.liquid.syringe_liquid_mL:5.3f} mL"
                    f"  undissolved {env.liquid.undissolved_g:6.4f} g  dones {dones}")
            if env.steps % 10 < every:
                print(line, flush=True)
            time.sleep(max(0.0, every * env.dt / speed - (time.time() - t0)))
            if not viewer.is_running():
                raise KeyboardInterrupt
            return out

        env.step = watched
        r = run_expert(env, seed=seed, syringe=syringe, solid_id=seed % 5, solid_g=0.1, record=False)
        print(f"outcome: {r['outcome']}  vial {r['info']['vial_mL']:.4f} mL  spill {r['info']['spilled_mL']:.4f} mL")
        while viewer.is_running():
            viewer.sync(); time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--syringe", default="3mL", choices=sorted(SYRINGES))
    ap.add_argument("--demo", action="store_true", help="run the scripted two-handed syringe motion")
    ap.add_argument("--expert", type=int, metavar="SEED", help="run the full expert episode with this seed")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed for --demo/--expert (1 = real time)")
    ap.add_argument("--every", type=int, default=1, help="--expert: redraw every N control steps (faster playback)")
    args = ap.parse_args()
    if args.expert is not None:
        return run_expert_view(args.syringe, args.expert, args.speed, max(1, args.every))
    scene = SolutionScene(SYRINGES[args.syringe])
    if not args.demo:
        mujoco.viewer.launch(scene.model, scene.data)
        return
    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        viewer.cam.lookat[:] = scene.data.body("syringe").xpos + [0, 0, 0.08]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 0.6, 200, -20
        run_demo(scene, viewer, args.speed)


if __name__ == "__main__":
    main()
