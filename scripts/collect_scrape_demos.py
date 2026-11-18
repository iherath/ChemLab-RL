"""Thirty scripted scrapes of powder from a source vial into the receiving vial.

The bench is a small MuJoCo scene: draft-shield door, pan, receiving vial,
source vial, and spatula. Both tools are mocap bodies the script moves.
Powder transfer is mass accounting while the spatula scrapes the source mouth
over the receiving vial. Images come from the overhead camera.

These are warm-start demonstrations for that scene. They are not UR7e joint
targets and not a particle pour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chemlab_rl.powder_dispense import TOL_G, Material

DT = 0.1
MAX_STEPS = 280
IMAGE = 96

# Six training powders. Five targets and approach offsets each -> 30 demos.
POWDERS = [
    Material(1.6, 0.15, 0.15, 0.05, 0.25, 0.008, 0.9, 0.05),
    Material(1.1, 0.20, 0.25, 0.08, 0.35, 0.010, 0.88, 0.15),
    Material(0.7, 0.18, 0.40, 0.12, 0.45, 0.012, 0.92, 0.25),
    Material(2.0, 0.12, 0.12, 0.04, 0.20, 0.006, 0.86, 0.10),
    Material(0.9, 0.28, 0.30, 0.10, 0.40, 0.009, 0.9, 0.20),
    Material(1.4, 0.22, 0.20, 0.15, 0.30, 0.011, 0.84, 0.30),
]

XML = """
<mujoco>
  <visual>
    <global offwidth="640" offheight="480"/>
    <rgba haze="0.15 0.15 0.16 1"/>
  </visual>
  <asset>
    <material name="glass" rgba="0.75 0.88 0.92 0.45"/>
    <material name="powder" rgba="0.82 0.74 0.58 1"/>
    <material name="wood" rgba="0.45 0.28 0.14 1"/>
    <material name="steel" rgba="0.72 0.73 0.76 1"/>
    <material name="shield" rgba="0.78 0.90 0.88 0.28"/>
  </asset>
  <worldbody>
    <light pos="0.2 -0.2 1.2" dir="0 0.2 -1" diffuse="0.9 0.9 0.9"/>
    <light pos="-0.3 0.4 0.8" dir="0.2 -0.2 -1" diffuse="0.4 0.4 0.45"/>
    <camera name="cam" pos="0.0 -0.42 0.40" xyaxes="1 0 0 0 0.50 0.87" fovy="52"/>
    <geom name="bench" type="box" size="0.45 0.35 0.02" pos="0 0 -0.02" rgba="0.16 0.16 0.17 1"/>
    <body name="balance" pos="0 0 0">
      <geom type="box" size="0.11 0.15 0.045" pos="0 0.02 0.045" rgba="0.90 0.89 0.84 1"/>
      <geom name="pan" type="cylinder" size="0.04 0.004" pos="0 0 0.095" material="steel"/>
      <body name="door" pos="0 0 0.20">
        <joint name="door" type="slide" axis="0 1 0" range="0 0.16" damping="2"/>
        <geom type="box" size="0.105 0.13 0.004" material="shield"/>
        <geom type="box" size="0.012 0.004 0.006" pos="0 -0.1 0.008" rgba="0.15 0.15 0.16 1"/>
      </body>
    </body>
    <body name="recv" pos="0 0 0.10">
      <geom type="cylinder" size="0.013 0.028" material="glass" contype="0" conaffinity="0"/>
    </body>
    <body name="fill" pos="0 0 0.078">
      <joint name="fill" type="slide" axis="0 0 1" range="0 0.045"/>
      <geom type="cylinder" size="0.010 0.003" material="powder" contype="0" conaffinity="0"/>
    </body>
    <body name="source" mocap="true" pos="-0.16 0.08 0.14">
      <geom type="cylinder" size="0.011 0.022" material="glass" contype="0" conaffinity="0"/>
      <site name="mouth" pos="0 0 0.022" size="0.004"/>
    </body>
    <body name="spatula" mocap="true" pos="0.14 -0.02 0.16">
      <geom type="box" size="0.018 0.007 0.005" pos="-0.02 0 0" material="wood" contype="0" conaffinity="0"/>
      <geom type="capsule" fromto="0 0 0 0.10 0 0" size="0.0025" rgba="0.78 0.79 0.82 1" contype="0" conaffinity="0"/>
      <site name="tip" pos="0.10 0 0" size="0.003"/>
    </body>
  </worldbody>
</mujoco>
"""


def _quat_y(angle: float) -> np.ndarray:
    h = 0.5 * angle
    return np.array([np.cos(h), 0.0, np.sin(h), 0.0], dtype=np.float64)


def _site(model, data, name: str) -> np.ndarray:
    return data.site(name).xpos.copy()


class ScrapeEnv:
    def __init__(self, material: Material, target_g: float, rng: np.random.Generator):
        self.material = material
        self.target_g = float(target_g)
        self.rng = rng
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=IMAGE, width=IMAGE)
        self.mass_g = 0.0
        self.reading_g = 0.0
        self.spilled_g = 0.0
        self._delay = 0.0
        self._loaded = 0.0
        self.scraping = False
        self.door = 0.0
        self.src = np.array([-0.16, 0.08, 0.16], dtype=np.float64)
        self.src_tilt = 0.0
        self.sp = np.array([0.14, -0.02, 0.18], dtype=np.float64)
        self.sp_pitch = -0.4
        self.t = 0
        self._place()

    def _place(self) -> None:
        self.data.mocap_pos[0] = self.src
        self.data.mocap_quat[0] = _quat_y(self.src_tilt)
        self.data.mocap_pos[1] = self.sp
        self.data.mocap_quat[1] = _quat_y(self.sp_pitch)
        self.data.qpos[self.model.joint("door").qposadr[0]] = self.door
        fill = 0.04 * np.clip(self.mass_g / max(self.target_g, 0.5), 0.0, 1.0)
        self.data.qpos[self.model.joint("fill").qposadr[0]] = fill
        mujoco.mj_forward(self.model, self.data)

    def _transfer(self) -> None:
        m = self.material
        over = np.linalg.norm(self.src[:2]) < 0.03 and self.src[2] > 0.12 and self.src_tilt > 0.6
        # Leave room for the spout residual so it can fall into the vial, not onto the bench.
        if self.scraping and over:
            self._delay += DT
            if self._delay >= m.delay_s and self.mass_g + m.residual_g < self.target_g - 0.01:
                rate = m.flow_g_per_s * (1.0 - m.cohesion)
                self._loaded = m.residual_g
                step = max(0.0, rate) * DT
                room = max(0.0, self.target_g - 0.01 - m.residual_g - self.mass_g)
                self.mass_g += min(step, room)
        elif self._loaded > 0.0 and over and self.mass_g < self.target_g:
            room = self.target_g - self.mass_g
            dump = min(self._loaded, 0.03, room)
            self.mass_g += dump
            self._loaded -= dump
        else:
            self._delay = 0.0
            if self._loaded > 0.0 and not over:
                self.spilled_g += self._loaded
                self._loaded = 0.0
        lag = max(m.lag_s, DT)
        self.reading_g += (DT / lag) * (self.mass_g - self.reading_g)
        self.reading_g += float(self.rng.normal(0.0, m.noise_g * 0.25))

    def drive(self, door, src, tilt, sp, pitch, scraping: bool) -> np.ndarray:
        """Move the tools toward a pose. The returned action is the clipped delta."""
        door_a = float(np.clip((door - self.door) / 0.04, -1.0, 1.0))
        src_a = np.clip((np.asarray(src) - self.src) / np.array([0.02, 0.02, 0.02]), -1.0, 1.0)
        tilt_a = float(np.clip((tilt - self.src_tilt) / 0.12, -1.0, 1.0))
        sp_a = np.clip((np.asarray(sp) - self.sp) / np.array([0.02, 0.02, 0.02]), -1.0, 1.0)
        pitch_a = float(np.clip((pitch - self.sp_pitch) / 0.15, -1.0, 1.0))
        self.door = float(np.clip(self.door + 0.04 * door_a, 0.0, 0.16))
        self.src = self.src + src_a * 0.02
        self.src_tilt = float(np.clip(self.src_tilt + 0.12 * tilt_a, 0.0, 1.3))
        self.sp = self.sp + sp_a * 0.02
        self.sp_pitch = float(np.clip(self.sp_pitch + 0.15 * pitch_a, -1.4, 0.6))
        self.scraping = scraping
        self._place()
        self._transfer()
        self.t += 1
        return np.array([door_a, *src_a, tilt_a, *sp_a, pitch_a], dtype=np.float32)

    def image(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera="cam")
        return self.renderer.render().copy()

    def dones(self) -> np.ndarray:
        door_open = self.door > 0.08
        pouring = self.mass_g > 0.05
        weighed = (
            abs(self.reading_g - self.target_g) <= TOL_G
            and self.src_tilt < 0.25
            and np.linalg.norm(self.sp) > 0.08
            and self.spilled_g <= 0.05
        )
        return np.array([door_open, pouring, weighed], dtype=np.float32)

    def obs(self) -> dict:
        return {
            "image": self.image(),
            "lowdim": np.array(
                [
                    *self.src,
                    self.src_tilt,
                    *self.sp,
                    self.sp_pitch,
                    self.door,
                    self.reading_g,
                    self.target_g,
                    self.reading_g - self.target_g,
                ],
                dtype=np.float32,
            ),
            "dones": self.dones(),
            "reading_g": self.reading_g,
            "mass_g": self.mass_g,
            "spilled_g": self.spilled_g,
        }


def _mouth_pose(env: ScrapeEnv, wobble: float) -> tuple[np.ndarray, float]:
    """Spatula pose whose tip sits in the source mouth, plus a small scrape."""
    pitch = -0.9
    c, s = np.cos(pitch), np.sin(pitch)
    tip_local = np.array([0.10 * c, 0.0, -0.10 * s])
    mouth = env.src + np.array([0.0, wobble, 0.02])
    return mouth - tip_local, pitch


def scripted(env: ScrapeEnv, phase_t: int) -> np.ndarray:
    """Open the shield, carry the source over the vial, scrape, then pull away."""
    home_src = np.array([-0.16, 0.08, 0.16])
    over = np.array([0.0, 0.0, 0.16])
    m = env.material
    full = env.mass_g + m.residual_g >= env.target_g - 0.01 or env.mass_g >= env.target_g - 0.01
    if env.door < 0.14:
        return env.drive(0.16, home_src, 0.0, env.sp, env.sp_pitch, False)
    if full and env._loaded > 1e-4:
        sp, pitch = _mouth_pose(env, 0.0)
        return env.drive(0.16, over, 0.9, sp, pitch, False)
    if full:
        away = np.array([0.16, -0.06, 0.20])
        return env.drive(0.16, over + np.array([0.0, 0.0, 0.02]), 0.0, away, -0.3, False)
    if np.linalg.norm(env.src - over) > 0.025 or env.src_tilt < 0.85:
        sp, pitch = _mouth_pose(env, 0.0)
        return env.drive(0.16, over, 1.0, sp, pitch, False)
    wobble = 0.008 if (phase_t // 4) % 2 == 0 else -0.008
    sp, pitch = _mouth_pose(env, wobble)
    return env.drive(0.16, over, 1.05, sp, pitch, True)


def rollout(material: Material, target_g: float, seed: int) -> dict | None:
    rng = np.random.default_rng(seed)
    env = ScrapeEnv(material, target_g, rng)
    images, lowdim, actions, dones = [], [], [], []
    for t in range(MAX_STEPS):
        before = env.obs()
        action = scripted(env, t)
        after = env.obs()
        images.append(before["image"])
        lowdim.append(before["lowdim"])
        actions.append(action)
        dones.append(before["dones"])
        if after["dones"][2] > 0.5 and t > 20:
            images.append(after["image"])
            return {
                "image": np.stack(images[:-1]),
                "next_image": np.stack(images[1:]),
                "lowdim": np.stack(lowdim).astype(np.float32),
                "actions": np.stack(actions).astype(np.float32),
                "dones": np.stack(dones).astype(np.float32),
                "target_g": target_g,
                "reading_g": float(after["reading_g"]),
                "mass_g": float(after["mass_g"]),
                "spilled_g": float(after["spilled_g"]),
                "seed": seed,
                "material": material.as_dict(),
            }
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "data" / "teleop" / "scrape_vial_bc.hdf5")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    kept = []
    attempts = 0
    while len(kept) < args.n and attempts < args.n * 20:
        attempts += 1
        powder_i = len(kept) % len(POWDERS)
        material = POWDERS[powder_i]
        target = float(rng.uniform(1.2, 6.0))
        seed = int(rng.integers(0, 2**31 - 1))
        demo = rollout(material, target, seed)
        if demo is None:
            print(f"miss powder {powder_i} target {target:.2f}")
            continue
        demo["powder_id"] = powder_i
        kept.append(demo)
        err = abs(demo["reading_g"] - target)
        print(
            f"kept {len(kept):02d}/{args.n} powder={powder_i} target={target:.2f} "
            f"reading={demo['reading_g']:.3f} err={err:.3f} T={len(demo['actions'])}"
        )
    if len(kept) < args.n:
        raise SystemExit(f"only {len(kept)} successes in {attempts} attempts")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    instruction = "Scrape powder from the source vial into the receiving vial until the scale reads the target mass."
    with h5py.File(args.out, "w") as f:
        f.attrs["collector"] = "mujoco_spatula_scrape"
        f.attrs["tol_g"] = TOL_G
        f.attrs["instruction"] = instruction
        f.attrs["note"] = (
            "Mocap spatula and source vial over a receiving vial. "
            "Powder transfer is mass accounting. Actions are normalized tool deltas, not UR7e joints."
        )
        data = f.create_group("data")
        for i, demo in enumerate(kept):
            g = data.create_group(f"demo_{i}")
            g.create_dataset("obs/image", data=demo["image"], compression="gzip", compression_opts=4)
            g.create_dataset("next_obs/image", data=demo["next_image"], compression="gzip", compression_opts=4)
            g.create_dataset("obs/lowdim", data=demo["lowdim"])
            g.create_dataset("actions", data=demo["actions"])
            g.create_dataset("subgoal_dones", data=demo["dones"])
            g.attrs["success"] = 1
            g.attrs["num_samples"] = int(len(demo["actions"]))
            g.attrs["target_g"] = demo["target_g"]
            g.attrs["reading_g"] = demo["reading_g"]
            g.attrs["mass_g"] = demo["mass_g"]
            g.attrs["spilled_g"] = demo["spilled_g"]
            g.attrs["powder_id"] = demo["powder_id"]
            g.attrs["seed"] = demo["seed"]
            g.attrs["instruction"] = f"{instruction} Target {demo['target_g']:.2f} g."
            for key, value in demo["material"].items():
                g.attrs[key] = value
    summary = [
        {
            "demo": f"demo_{i}",
            "powder_id": d["powder_id"],
            "target_g": d["target_g"],
            "reading_g": d["reading_g"],
            "abs_error_g": abs(d["reading_g"] - d["target_g"]),
            "steps": int(len(d["actions"])),
            **d["material"],
        }
        for i, d in enumerate(kept)
    ]
    args.out.with_suffix(".json").write_text(
        json.dumps({"attempts": attempts, "tol_g": TOL_G, "n_powders": len(POWDERS), "demos": summary}, indent=2)
    )
    errors = np.array([row["abs_error_g"] for row in summary])
    print(
        f"wrote {args.out}\nkept {len(kept)}/{attempts}  "
        f"abs error mean {errors.mean():.4f} max {errors.max():.4f}"
    )


if __name__ == "__main__":
    main()
