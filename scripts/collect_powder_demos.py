"""Write 30 successful powder-to-target demonstrations.

Each episode draws a target mass and a synthetic material, then runs the
closed-loop demonstrator in chemlab_rl.powder_dispense. Failed draws are
discarded. The HDF5 observation has the scale, the tilt, and the target.
Material parameters are attributes, not policy inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chemlab_rl.powder_dispense import OBS_NAMES, TOL_G, rollout, sample_material


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "teleop" / "powder_weigh_bc.hdf5",
    )
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    kept = []
    attempts = 0
    while len(kept) < args.n and attempts < args.n * 40:
        attempts += 1
        target = float(rng.uniform(1.0, 8.0))
        material = sample_material(rng, target)
        ep_seed = int(rng.integers(0, 2**31 - 1))
        demo = rollout(material, target, ep_seed)
        if demo is not None:
            kept.append(demo)
            print(
                f"kept {len(kept):02d}/{args.n}  target={target:.2f} g  "
                f"reading={demo['reading_g']:.3f}  T={len(demo['actions'])}  "
                f"flow={material.flow_g_per_s:.2f}  delay={material.delay_s:.2f}s"
            )
    if len(kept) < args.n:
        raise SystemExit(f"only {len(kept)} successes in {attempts} attempts")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    with h5py.File(args.out, "w") as f:
        f.attrs["collector"] = "mass_accounting_powder"
        f.attrs["tol_g"] = TOL_G
        f.attrs["obs_names"] = json.dumps(OBS_NAMES)
        f.attrs["note"] = (
            "Synthetic materials. Scale reading is lagged mass in the vessel, "
            "not a PhysX or MuJoCo contact sensor."
        )
        data = f.create_group("data")
        for i, demo in enumerate(kept):
            g = data.create_group(f"demo_{i}")
            g.create_dataset("obs", data=demo["obs"])
            g.create_dataset("actions", data=demo["actions"])
            g.create_dataset("next_obs", data=demo["next_obs"])
            g.attrs["success"] = 1
            g.attrs["num_samples"] = int(len(demo["actions"]))
            g.attrs["target_g"] = demo["target_g"]
            g.attrs["reading_g"] = demo["reading_g"]
            g.attrs["mass_g"] = demo["mass_g"]
            g.attrs["spilled_g"] = demo["spilled_g"]
            g.attrs["seed"] = demo["seed"]
            g.attrs["abs_error_g"] = abs(demo["reading_g"] - demo["target_g"])
            for key, value in demo["material"].items():
                g.attrs[key] = value

    summary = []
    for i, demo in enumerate(kept):
        summary.append(
            {
                "demo": f"demo_{i}",
                "target_g": demo["target_g"],
                "reading_g": demo["reading_g"],
                "abs_error_g": abs(demo["reading_g"] - demo["target_g"]),
                "spilled_g": demo["spilled_g"],
                "steps": int(len(demo["actions"])),
                **demo["material"],
            }
        )
    sidecar = args.out.with_suffix(".json")
    sidecar.write_text(json.dumps({"attempts": attempts, "tol_g": TOL_G, "demos": summary}, indent=2))
    errors = np.array([row["abs_error_g"] for row in summary])
    targets = np.array([row["target_g"] for row in summary])
    flows = np.array([row["flow_g_per_s"] for row in summary])
    print(
        f"\nwrote {args.out}\n"
        f"kept {len(kept)}/{attempts} attempts\n"
        f"target {targets.min():.2f}–{targets.max():.2f} g   "
        f"flow {flows.min():.2f}–{flows.max():.2f} g/s\n"
        f"abs error mean {errors.mean():.4f} g  max {errors.max():.4f} g  tol {TOL_G} g"
    )


if __name__ == "__main__":
    main()
