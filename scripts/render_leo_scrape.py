"""Replay Leo-scene BC demos. Each video is the three cameras stored for the VLA."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT), str(ROOT / "chem_rl_sim"),
                str(ROOT / "chem_rl_sim" / "third_party" / "urgantry_sim")]

from collect_leo_scrape import Scene, _restore

CAMERAS = (("top1", "top1"), ("left wrist", "left_grip_wrist"), ("right wrist", "right_grip_wrist"))


def choose(rows: list[dict]) -> list[int]:
    clean = [i for i, row in enumerate(rows) if row["mode"] == "clean"]
    picks = [
        min(clean, key=lambda i: rows[i]["steps"]),
        max(clean, key=lambda i: rows[i]["steps"]),
        next(i for i, row in enumerate(rows) if row["mode"] == "spatula"),
        next(i for i, row in enumerate(rows) if row["mode"] == "pour"),
    ]
    out = []
    for i in picks:
        if i not in out:
            out.append(i)
    return out


def label(image: np.ndarray, text: str) -> np.ndarray:
    bar = np.zeros((28, image.shape[1], 3), dtype=np.uint8)
    # Tiny built-in glyphs, enough for the camera name and the mode.
    glyph = {
        " ": ["00000"] * 7,
        "a": ["00000", "01110", "00001", "01111", "10001", "10001", "01111"],
        "c": ["00000", "01110", "10000", "10000", "10000", "10001", "01110"],
        "e": ["00000", "01110", "10001", "11111", "10000", "10001", "01110"],
        "f": ["00110", "01000", "11100", "01000", "01000", "01000", "01000"],
        "h": ["10001", "10001", "11111", "10001", "10001", "10001", "10001"],
        "i": ["01110", "00100", "00100", "00100", "00100", "00100", "01110"],
        "l": ["01000", "01000", "01000", "01000", "01000", "01000", "01110"],
        "n": ["00000", "10110", "11001", "10001", "10001", "10001", "10001"],
        "o": ["00000", "01110", "10001", "10001", "10001", "10001", "01110"],
        "p": ["00000", "11110", "10001", "11110", "10000", "10000", "10000"],
        "r": ["00000", "10110", "11001", "10000", "10000", "10000", "10000"],
        "s": ["00000", "01111", "10000", "01110", "00001", "00001", "11110"],
        "t": ["00100", "11111", "00100", "00100", "00100", "00100", "00011"],
        "u": ["00000", "10001", "10001", "10001", "10001", "10011", "01101"],
        "w": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
        "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    }
    x = 8
    for ch in text.lower():
        rows = glyph.get(ch, glyph[" "])
        for row, bits in enumerate(rows):
            for col, bit in enumerate(bits):
                if bit == "1":
                    bar[3 + row * 3:6 + row * 3, x + col * 3:x + col * 3 + 2] = 235
        x += 16
    return np.concatenate([bar, image], axis=0)


def main() -> None:
    src = ROOT / "data" / "teleop" / "scrape_vial_bc.hdf5"
    out = ROOT / "data" / "teleop" / "scrape_videos"
    out.mkdir(parents=True, exist_ok=True)
    scene = Scene()
    renderer = mujoco.Renderer(scene.model, 360, 480)
    with h5py.File(src, "r") as handle:
        names = sorted(handle["data"].keys(), key=lambda n: int(n.split("_")[1]))
        rows = []
        for name in names:
            g = handle["data"][name]
            rows.append({
                "name": name,
                "mode": g.attrs["mode"],
                "steps": int(g.attrs["num_samples"]),
                "target": float(g.attrs["target_g"]),
                "reading": float(g.attrs["reading_g"]),
            })
        for index in choose(rows):
            row = rows[index]
            group = handle["data"][row["name"]]
            qpos = group["qpos"][:]
            mocap_pos = group["mocap_pos"][:]
            mocap_quat = group["mocap_quat"][:]
            reading = group["obs/lowdim"][:, -3]
            path = out / f"{row['name']}_{row['mode']}.mp4"
            with imageio.get_writer(path, fps=10, codec="libx264", quality=8, macro_block_size=1) as writer:
                for i in range(len(qpos)):
                    frame = {
                        "qpos": qpos[i], "mocap_pos": mocap_pos[i], "mocap_quat": mocap_quat[i],
                        "reading": float(reading[i]),
                    }
                    _restore(scene, frame)
                    panels = []
                    for title, cam in CAMERAS:
                        renderer.update_scene(scene.data, camera=cam)
                        panels.append(label(renderer.render(), title))
                    writer.append_data(np.concatenate(panels, axis=1))
            print(f"wrote {path}  {row['mode']}  target {row['target']:.2f}  reading {row['reading']:.2f}")


if __name__ == "__main__":
    main()
