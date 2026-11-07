"""Replay grok HDF5 qpos trajectories from an operator 3/4 view into mp4 + gif."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chem_rl_sim" / "third_party" / "urgantry_sim"))
sys.path.insert(0, str(ROOT / "chem_rl_sim"))

import imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _label_frames(rgb: np.ndarray, title: str) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, img.width, 36), fill=(20, 20, 20))
    draw.text((12, 6), title, fill=(240, 240, 240), font=font)
    return np.asarray(img)


def render_demo(env, states: np.ndarray, title: str, every: int) -> list[np.ndarray]:
    m, d = env.model, env.data
    nq, nv = m.nq, m.nv
    renderer = mujoco.Renderer(m, height=480, width=720)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.10, 0.04, 0.88)
    cam.distance, cam.azimuth, cam.elevation = 1.15, 125.0, -35.0
    frames = []
    for i in range(0, states.shape[0], every):
        st = states[i]
        d.time = float(st[0])
        d.qpos[:] = st[1 : 1 + nq]
        d.qvel[:] = st[1 + nq : 1 + nq + nv]
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=cam)
        frames.append(_label_frames(renderer.render(), title))
    renderer.close()
    return frames


def write_media(frames: list[np.ndarray], mp4: Path, gif: Path) -> None:
    mp4.parent.mkdir(parents=True, exist_ok=True)
    gif.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(mp4, fps=10, macro_block_size=1)
    for fr in frames:
        writer.append_data(fr)
    writer.close()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    vf = (
        "fps=10,scale=640:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a"
    )
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4), "-vf", vf, str(gif)], check=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", type=Path, required=True)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "grok_demos")
    p.add_argument("--every", type=int, default=8)
    p.add_argument("--max-gifs", type=int, default=8)
    args = p.parse_args()

    import h5py
    from chem_rl_sim.env import VialWeighingEnv

    names = {}
    if args.log and args.log.exists():
        for e in json.loads(args.log.read_text()):
            if e.get("saved"):
                names[e["saved"]] = e["recipe"]["name"]

    env = VialWeighingEnv(layout="weigh", clutter=False, image_size=64, max_episode_steps=10**9)
    written = []
    with h5py.File(args.hdf5, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for name in demos[: args.max_gifs]:
            g = f[f"data/{name}"]
            title = names.get(name, name)
            print(f"rendering {name} ({title}) T={g['actions'].shape[0]}")
            frames = render_demo(env, np.asarray(g["states"]), f"{name}: {title}", args.every)
            mp4 = args.out_dir / f"{name}_{title}.mp4"
            gif = args.out_dir / f"{name}_{title}.gif"
            write_media(frames, mp4, gif)
            written.append(str(gif))
            print(f"  {gif}  {gif.stat().st_size / 1e6:.1f} MB")
    env.close()
    print("wrote", len(written), "gifs")


if __name__ == "__main__":
    main()
