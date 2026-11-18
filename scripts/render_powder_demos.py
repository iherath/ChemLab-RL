"""Render a few powder-to-target demonstrations as mp4 clips.

The demonstrations are tilt commands and a scale reading, not a MuJoCo
scene. Each frame draws the hopper at the logged tilt and the pan fill at
the logged reading.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]


def _pick(summary: list[dict], n: int) -> list[str]:
    """Shortest, longest, lightest target, heaviest target, then fill."""
    by_steps = sorted(summary, key=lambda d: d["steps"])
    by_target = sorted(summary, key=lambda d: d["target_g"])
    ordered = [by_steps[0], by_steps[-1], by_target[0], by_target[-1]]
    seen = {d["demo"] for d in ordered}
    for demo in summary:
        if len(ordered) >= n:
            break
        if demo["demo"] not in seen:
            ordered.append(demo)
            seen.add(demo["demo"])
    return [d["demo"] for d in ordered[:n]]


def _frame(ax_scene, ax_plot, tilt, reading, target, history, t, times, title):
    ax_scene.clear()
    ax_scene.set_xlim(-1.4, 1.6)
    ax_scene.set_ylim(-0.2, 2.4)
    ax_scene.set_aspect("equal")
    ax_scene.axis("off")
    ax_scene.set_title(title, loc="left", fontsize=11)

    pan = FancyBboxPatch(
        (-0.7, 0.0), 1.4, 0.12, boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor="#4a4e57", edgecolor="#1c1e24", linewidth=1.2,
    )
    ax_scene.add_patch(pan)
    fill_h = 0.9 * float(np.clip(reading / max(target, 1e-6), 0.0, 1.15))
    if fill_h > 0.01:
        ax_scene.add_patch(
            Rectangle((-0.55, 0.12), 1.1, fill_h, facecolor="#d8d2c4", edgecolor="none")
        )
    ax_scene.plot([-0.7, 0.7], [0.12 + 0.9, 0.12 + 0.9], color="#2a6f4e", lw=1.2, ls="--")
    ax_scene.text(0.75, 0.12 + 0.9, f"target {target:.2f} g", color="#2a6f4e", fontsize=8, va="center")

    angle = np.deg2rad(-70.0 * float(np.clip(tilt, 0.0, 1.0)))
    hopper = np.array([[-0.28, 0.0], [0.28, 0.0], [0.18, 0.55], [-0.18, 0.55]])
    pivot = np.array([-0.15, 1.55])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    poly = (hopper @ rot.T) + pivot
    ax_scene.add_patch(Polygon(poly, closed=True, facecolor="#8a9099", edgecolor="#222"))
    ax_scene.add_patch(Circle(pivot, 0.04, facecolor="#222"))
    ax_scene.text(-1.35, 2.15, f"tilt {tilt:.2f}", fontsize=9, color="#333")
    ax_scene.text(-1.35, 0.35, f"{reading:.2f} g", fontsize=12, color="#222")

    if tilt > 0.2 and (len(history) < 2 or history[-1] >= history[-2] - 1e-3):
        mouth = poly[1]
        for i, grain in enumerate(np.linspace(0.15, 0.85, 5)):
            y = mouth[1] - grain * (mouth[1] - 0.12 - fill_h)
            x = mouth[0] + 0.15 * grain
            ax_scene.add_patch(Circle((x, y), 0.035, facecolor="#c4b8a0", edgecolor="none"))

    ax_plot.clear()
    ax_plot.axhspan(target - 0.05, target + 0.05, color="#2a6f4e", alpha=0.15)
    ax_plot.axhline(target, color="#2a6f4e", lw=1.0, ls="--")
    ax_plot.plot(times[: t + 1], history[: t + 1], color="#1f4b82", lw=1.8)
    ax_plot.scatter([times[t]], [history[t]], color="#1f4b82", s=18, zorder=3)
    ax_plot.set_xlim(0, max(times[-1], 0.1))
    ymax = max(target * 1.25, max(history) * 1.05, 0.2)
    ax_plot.set_ylim(0, ymax)
    ax_plot.set_xlabel("time (s)")
    ax_plot.set_ylabel("scale reading (g)")
    ax_plot.set_title("reading vs target band (±0.05 g)", fontsize=11)


def render_one(obs: np.ndarray, attrs: dict, out: Path, fps: int) -> None:
    tilt = obs[:, 0]
    reading = obs[:, 1]
    target = float(attrs["target_g"])
    times = np.arange(len(obs)) * 0.1
    title = (
        f"{attrs['name']}   target {target:.2f} g   "
        f"final {float(attrs['reading_g']):.2f} g   "
        f"flow {float(attrs['flow_g_per_s']):.2f} g/s"
    )
    fig, (ax_scene, ax_plot) = plt.subplots(
        1, 2, figsize=(9.6, 4.8), dpi=100, gridspec_kw={"width_ratios": [1.0, 1.15]}
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=2000)
    with writer.saving(fig, str(out), dpi=100):
        for t in range(len(obs)):
            _frame(ax_scene, ax_plot, tilt[t], reading[t], target, reading, t, times, title)
            writer.grab_frame()
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hdf5", type=Path, default=ROOT / "data" / "teleop" / "powder_weigh_bc.hdf5")
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "teleop" / "powder_videos")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()

    summary = json.loads(args.hdf5.with_suffix(".json").read_text())["demos"]
    names = _pick(summary, args.n)
    by_name = {d["demo"]: d for d in summary}
    with h5py.File(args.hdf5, "r") as f:
        for name in names:
            obs = np.asarray(f[f"data/{name}/obs"])
            meta = dict(by_name[name])
            meta["name"] = name
            out = args.out_dir / f"{name}.mp4"
            render_one(obs, meta, out, args.fps)
            print(f"wrote {out}  ({len(obs)} frames, target {meta['target_g']:.2f} g)")


if __name__ == "__main__":
    sys.exit(main() or 0)
