"""Replay a sample of scrape demos at a watchable resolution."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import collect_scrape_demos as collect

_GLYPH = {
    " ": ["00000"] * 7,
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "g": ["00000", "01110", "10001", "10001", "01111", "00001", "01110"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
}


def stamp(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw a short caption. The reading changes every frame, so it cannot be a single ffmpeg label."""
    image = frame.copy()
    image[:28, : 8 + 16 * len(text)] = (0, 0, 0)
    x = 8
    for char in text:
        rows = _GLYPH.get(char, _GLYPH[" "])
        for row, bits in enumerate(rows):
            for col, bit in enumerate(bits):
                if bit == "1":
                    image[4 + row * 3 : 7 + row * 3, x + col * 3 : x + col * 3 + 2] = 235
        x += 16
    return image


def pick(rows: list[dict]) -> list[int]:
    """Shortest, longest, lightest target, and heaviest target, without repeats."""
    order = [
        min(range(len(rows)), key=lambda i: rows[i]["steps"]),
        max(range(len(rows)), key=lambda i: rows[i]["steps"]),
        min(range(len(rows)), key=lambda i: rows[i]["target_g"]),
        max(range(len(rows)), key=lambda i: rows[i]["target_g"]),
    ]
    chosen = []
    for index in order:
        if index not in chosen:
            chosen.append(index)
    return chosen


def place(model, data, row: np.ndarray) -> None:
    data.mocap_pos[0] = row[0:3]
    data.mocap_quat[0] = collect._quat_y(float(row[3]))
    data.mocap_pos[1] = row[4:7]
    data.mocap_quat[1] = collect._quat_y(float(row[7]))
    data.qpos[model.joint("door").qposadr[0]] = float(row[8])
    reading, target = float(row[9]), max(float(row[10]), 0.5)
    data.qpos[model.joint("fill").qposadr[0]] = 0.04 * np.clip(reading / target, 0.0, 1.0)
    mujoco.mj_forward(model, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=ROOT / "data" / "teleop" / "scrape_vial_bc.hdf5")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "teleop" / "scrape_videos")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_string(collect.XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    with h5py.File(args.src, "r") as handle:
        names = sorted(handle["data"].keys(), key=lambda name: int(name.split("_")[1]))
        rows = []
        for name in names:
            group = handle["data"][name]
            rows.append(
                {
                    "name": name,
                    "steps": int(group.attrs["num_samples"]),
                    "target_g": float(group.attrs["target_g"]),
                    "reading_g": float(group.attrs["reading_g"]),
                    "powder_id": int(group.attrs["powder_id"]),
                }
            )
        for index in pick(rows):
            row = rows[index]
            lowdim = handle["data"][row["name"]]["obs/lowdim"][:]
            path = args.out / f"{row['name']}_p{row['powder_id']}.mp4"
            with imageio.get_writer(path, fps=10, codec="libx264", quality=8, macro_block_size=1) as writer:
                for state in lowdim:
                    place(model, data, state)
                    renderer.update_scene(data, camera="cam")
                    text = f"{state[9]:.2f}g {state[10]:.2f}g"
                    writer.append_data(stamp(renderer.render(), text))
            print(f"wrote {path}  final {row['reading_g']:.2f}/{row['target_g']:.2f} g  {len(lowdim)} frames")


if __name__ == "__main__":
    main()
