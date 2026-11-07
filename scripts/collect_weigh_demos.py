"""Record scripted keyframe weigh() episodes into Leo's OGPO HDF5 layout.

Uses chem_rl_sim.teleop.keyframe_script.weigh and Recorder. Nothing here is a
learned policy. Output defaults to data/teleop/weigh_vial.hdf5.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

ROOT = Path(__file__).resolve().parents[1]
# <project>/chem_rl_sim is the clone; put it ahead of cwd so `import chem_rl_sim`
# hits the inner package, not a namespace folder at the project root.
sys.path.insert(0, str(ROOT / "chem_rl_sim"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "data" / "teleop" / "weigh_vial.hdf5")
    p.add_argument("--n-demos", type=int, default=3)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from chem_rl_sim.env import VialWeighingEnv
    from chem_rl_sim.teleop.keyframe import Session
    from chem_rl_sim.teleop.keyframe_script import weigh
    from chem_rl_sim.teleop.recorder import Recorder

    if args.out.exists():
        args.out.unlink()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        image_size=args.image_size,
        control_hz=20.0,
        hand="robotiq",
        layout="weigh",
        clutter=False,
        max_episode_steps=10**9,
        randomize=True,
    )
    env = VialWeighingEnv(**env_kwargs)
    rec = Recorder(
        args.out,
        {
            "env_name": "ChemVialWeighing-v0",
            "type": "chem_rl_sim",
            "env_kwargs": {**env_kwargs, "normalized_actions": True},
            "collector": "keyframe_script.weigh",
        },
        20.0,
        "robotiq",
    )
    sess = Session(env, rec)

    print(f"recording {args.n_demos} demos -> {args.out}")
    for i in range(args.n_demos):
        sess.begin_episode(seed=args.seed + i)
        statuses = weigh(sess)
        n = sess.save_episode()
        rec.wait()
        if rec.last_error:
            raise RuntimeError(rec.last_error)
        bad = {k: v for k, v in statuses.items() if v != "ok"}
        print(
            f"demo {i}  samples={n}  success={int(sess.info['success'])}  "
            f"balance={sess.info['balance_display']}  "
            f"plans_not_ok={bad or 'none'}  saved={rec.last_saved}"
        )
        if not sess.info["success"]:
            raise SystemExit(f"demo {i} did not reach env success")

    rec.wait()
    env.close()
    _summarize(args.out)


def _summarize(path: Path) -> None:
    import h5py

    with h5py.File(path, "r") as f:
        names = list(f["data"].keys())
        print(f"\nHDF5 {path}  demos={len(names)}")
        for name in names:
            g = f[f"data/{name}"]
            print(
                f"  {name}: actions={g['actions'].shape}  "
                f"success={g.attrs['success']}  "
                f"T={g.attrs['num_samples']}  "
                f"keys={list(g['obs'].keys())}"
            )
            act = g["actions"][:]
            print(
                f"           action range [{act.min():.3f}, {act.max():.3f}]  "
                f"object={g['obs/object'].shape}  joint_pos={g['obs/joint_pos'].shape}"
            )


if __name__ == "__main__":
    sys.exit(main() or 0)
