"""Validate and merge successful contact-driven demos, rejecting duplicate episodes.

The default output requires 30 episodes. Use --expected-episodes 5 for an
explicit five-episode pilot. Every declared training material must be present;
held-out material 5 is always excluded. A rejected input aborts the entire merge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemlab_rl.granular_material import GranularMaterial
from chemlab_rl.vla_bc import CAMERAS, _text, file_sha256, inspect_dataset, observed_state

RGB_COPY_FRAMES = 32


def _copy_attributes(source, destination):
    """Preserve attribute values, HDF5 dtypes and shapes during group creation."""
    for name in source.attrs:
        attribute = source.attrs.get_id(name)
        try:
            destination.attrs.create(name, source.attrs[name], shape=attribute.shape, dtype=attribute.dtype)
        finally:
            attribute.close()


def _copy_rgb_dataset(source, parent, name):
    """Lossless, bounded-memory copy into one compressed RGB frame per chunk.

    h5py's automatic chunks can span many timesteps, amplifying random BC reads.
    Contiguous 32-frame source reads amortize old temporal chunk decompression;
    the destination layout makes each later random frame independently readable.
    """
    if source.ndim != 4 or source.shape[-1] != 3 or source.dtype != np.uint8:
        raise ValueError(f"{source.name}: RGB rechunking requires uint8 [T,H,W,3]")
    destination = parent.create_dataset(
        name, shape=source.shape, dtype=source.dtype, maxshape=source.maxshape,
        chunks=(1, *source.shape[1:]), compression="gzip", compression_opts=4,
        shuffle=source.shuffle, fletcher32=source.fletcher32, fillvalue=source.fillvalue,
    )
    _copy_attributes(source, destination)
    for start in range(0, len(source), RGB_COPY_FRAMES):
        stop = min(start + RGB_COPY_FRAMES, len(source))
        destination[start:stop] = source[start:stop]
    return destination


def _copy_episode(source, parent, name):
    """Copy every episode value; only RGB chunk layout/compression is changed."""
    destination = parent.create_group(name)
    _copy_attributes(source, destination)
    for key in source:
        if key not in ("obs", "next_obs"):
            source.copy(source[key], destination, name=key)
            continue
        observations = destination.create_group(key)
        _copy_attributes(source[key], observations)
        for channel in source[key]:
            if channel in CAMERAS:
                _copy_rgb_dataset(source[key][channel], observations, channel)
            else:
                source.copy(source[key][channel], observations, name=channel)
    return destination


def _json_attribute(handle, name):
    if name not in handle.attrs:
        raise ValueError(f"Missing required {name} metadata")
    try:
        return json.loads(_text(handle.attrs[name]))
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid JSON metadata {name}") from error


def _metadata(handle):
    if _text(handle.attrs.get("powder_dynamics", "")) != "physical_grains_v1":
        raise ValueError("Only physical_grains_v1 demonstrations may be merged")
    bank = _json_attribute(handle, "powder_parameter_vectors")
    if not isinstance(bank, list) or len(bank) != 6 or not all(isinstance(m, dict) for m in bank):
        raise ValueError("Expected the complete six-material parameter bank")
    materials = [GranularMaterial(**entry) for entry in bank]
    if any(material.as_dict() != entry for material, entry in zip(materials, bank)):
        raise ValueError("Material bank must explicitly specify every physical parameter")
    train = _json_attribute(handle, "training_powder_ids")
    heldout = _json_attribute(handle, "held_out_powder_ids")
    if sorted(train) != [0, 1, 2, 3, 4] or heldout != [5]:
        raise ValueError("Required split is training materials 0--4 and held-out material 5")
    count = handle.attrs.get("grain_count", 0)
    if not isinstance(count, (int, np.integer)) or not 1 <= count <= 400:
        raise ValueError("Missing or invalid grain_count")
    sources = _json_attribute(handle, "source_files_sha256")
    if (not isinstance(sources, dict) or not sources
            or not all(isinstance(k, str) and isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v)
                       for k, v in sources.items())):
        raise ValueError("Invalid source_files_sha256 provenance")
    version = "sha256:" + hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    if _text(handle.attrs.get("source_version", "")) != version:
        raise ValueError("source_version does not match its source_files_sha256 manifest")
    control_hz = float(handle.attrs.get("control_hz", 0))
    mujoco_version = _text(handle.attrs.get("mujoco_version", ""))
    if not math.isfinite(control_hz) or control_hz <= 0 or not mujoco_version:
        raise ValueError("Missing valid control_hz or mujoco_version")
    return {"grain_count": int(count), "powder_parameter_vectors": bank,
            "training_powder_ids": sorted(train), "held_out_powder_ids": heldout,
            "source_version": version, "source_files_sha256": sources,
            "control_hz": control_hz, "mujoco_version": mujoco_version}


def _channel(demo, name, length):
    if name not in demo:
        raise ValueError(f"{demo.name}: missing {name}")
    value = np.asarray(demo[name])
    if value.shape != (length,) or not np.isfinite(value).all():
        raise ValueError(f"{demo.name}/{name}: expected {length} finite scalar samples")
    return value


def _fingerprint(demo):
    """Identify repeated physical trajectories despite renamed IDs/sensor noise."""
    digest = hashlib.sha256()
    names = ["actions", "obs/proprio", "physics/qpos", "physics/qvel"]
    names += [f"physics/{key}" for key in ("mass_g", "source_g", "tool_g", "spilled_g", "inflight_g", "raw_net_mass_g")]
    for name in names:
        data = demo[name]
        digest.update(json.dumps([name, data.shape, data.dtype.str]).encode())
        for start in range(0, len(data), 256):
            digest.update(np.ascontiguousarray(data[start:start + 256]).tobytes())
    return digest.hexdigest()


def _validate_episode(demo, metadata):
    length = len(demo["actions"])
    powder = int(demo.attrs["powder_id"])
    if powder not in metadata["training_powder_ids"] or powder == 5:
        raise ValueError(f"{demo.name}: held-out or undeclared training material {powder}")
    material = GranularMaterial(**metadata["powder_parameter_vectors"][powder])
    if (demo.attrs.get("success") != 1 or demo.attrs.get("physical_valid") != 1
            or _text(demo.attrs.get("physical_validation_scope", "")) != "rigid_bodies_and_grains"
            or _text(demo.attrs.get("powder_dynamics", "")) != "physical_grains_v1"
            or _text(demo.attrs.get("scale_model", "")) != "physical_pan_force_with_instrument_lag_and_noise"):
        raise ValueError(f"{demo.name}: missing complete physical-grain validation")
    if demo.attrs.get("grain_count") != metadata["grain_count"] or int(demo.attrs.get("num_samples", -1)) != length:
        raise ValueError(f"{demo.name}: grain/sample count metadata disagrees")
    if demo.attrs.get("control_hz") != metadata["control_hz"]:
        raise ValueError(f"{demo.name}: control frequency differs from dataset metadata")
    for key, expected in (("grain_radius_m", material.radius_m), ("grain_mass_g", material.grain_mass_g)):
        if not math.isclose(float(demo.attrs.get(key, float("nan"))), expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"{demo.name}: {key} does not follow the declared material")
    audit = _json_attribute(demo, "physics_audit")
    if not isinstance(audit, dict) or audit.get("material") != material.as_dict():
        raise ValueError(f"{demo.name}: audit material differs from the declared bank")
    vector = np.asarray(demo["material"])
    expected_vector = np.asarray(list(material.as_dict().values()))
    if vector.shape != expected_vector.shape or not np.allclose(vector, expected_vector, rtol=1e-6, atol=1e-12):
        raise ValueError(f"{demo.name}: stored material vector disagrees with the declared bank")
    contact_minima = audit.get("contact_minima_m")
    if (not isinstance(contact_minima, dict) or not isinstance(audit.get("grasp_events"), list)
            or any(not math.isfinite(float(value)) or float(value) < -.001 - 1e-12
                   for value in contact_minima.values())):
        raise ValueError(f"{demo.name}: incomplete or invalid rigid contact audit")
    for key, maximum in (("intentional_contact_tolerance_m", .001), ("forbidden_contact_tolerance_m", .00005),
                         ("grain_contact_tolerance_m", .2 * material.radius_m), ("mass_force_tolerance_g", .010)):
        tolerance = float(audit.get(key, float("nan")))
        if not math.isfinite(tolerance) or not 0 < tolerance <= maximum + 1e-12:
            raise ValueError(f"{demo.name}: invalid or excessively loose {key}")
    checks = _channel(demo, "physics/checks", length)
    substeps = int(demo.attrs["physics_steps_per_action"])
    if (checks.dtype.kind not in "iu" or np.any(np.diff(np.r_[0, checks]) < substeps)
            or int(demo.attrs["physics_checks"]) != checks[-1]):
        raise ValueError(f"{demo.name}: incomplete per-action physics checks")
    for name in ("physics/qpos", "physics/qvel"):
        value = np.asarray(demo[name])
        if value.ndim != 2 or value.shape[0] != length or not np.isfinite(value).all():
            raise ValueError(f"{demo.name}: invalid recorded {name}")
    for prefix in ("obs", "next_obs"):
        observed_state(demo[prefix], include_measured_scale=True)
        if demo[f"{prefix}/proprio"].shape != (length, 14):
            raise ValueError(f"{demo.name}: invalid measured proprio shape")
        for camera in CAMERAS:
            image = demo[f"{prefix}/{camera}"]
            if image.shape != demo[f"obs/{camera}"].shape or image.dtype != np.uint8:
                raise ValueError(f"{demo.name}: inconsistent next-observation camera {camera}")
    for key in ("proprio", "scale_reading_g"):
        if not np.array_equal(demo[f"next_obs/{key}"][:-1], demo[f"obs/{key}"][1:]):
            raise ValueError(f"{demo.name}: observation transitions are discontinuous at {key}")
    dones = _channel(demo, "dones", length)
    timeouts = _channel(demo, "timeouts", length)
    _channel(demo, "rewards", length)
    elapsed = _channel(demo, "physics/elapsed_s", length)
    if elapsed[0] <= 0 or np.any(np.diff(elapsed) <= 0):
        raise ValueError(f"{demo.name}: recorded simulation time is not increasing")
    if dones.dtype.kind != "b" or timeouts.dtype.kind != "b" or not dones[-1] or dones[:-1].any() or timeouts.any():
        raise ValueError(f"{demo.name}: requires a complete successful episode without timeout")
    subgoals = np.asarray(demo["subgoal_dones"])
    if (subgoals.shape != (length, 3) or not np.isin(subgoals, [0, 1]).all()
            or not np.all(subgoals[-1] == 1) or np.any(np.diff(subgoals.astype(int), axis=0) < 0)):
        raise ValueError(f"{demo.name}: incomplete or reversing subgoals")
    ledger = {key: _channel(demo, f"physics/{key}", length)
              for key in ("mass_g", "source_g", "tool_g", "spilled_g", "inflight_g")}
    total = metadata["grain_count"] * material.grain_mass_g
    mass_tolerance = max(1e-8, total * 1e-8)
    for name, values in ledger.items():
        if (np.any(values < -mass_tolerance)
                or not np.allclose(values / material.grain_mass_g, np.rint(values / material.grain_mass_g), atol=1e-6, rtol=0)):
            raise ValueError(f"{demo.name}: {name} is not an inventory of whole declared grains")
    error = sum(ledger.values()) - total
    recorded_error = _channel(demo, "physics/conservation_error_g", length)
    if np.any(np.abs(error) > mass_tolerance) or not np.allclose(error, recorded_error, atol=mass_tolerance, rtol=0):
        raise ValueError(f"{demo.name}: mass conservation failed at a recorded step")
    penetration = _channel(demo, "physics/max_grain_contact_penetration_m", length)
    if np.any(penetration < 0) or np.any(np.diff(penetration) < -1e-12) or np.any(penetration > .2 * material.radius_m + 1e-12):
        raise ValueError(f"{demo.name}: grain penetration exceeds 0.2 radius or cumulative audit is invalid")
    raw = _channel(demo, "physics/raw_net_mass_g", length)
    force_error = _channel(demo, "physics/mass_force_error_g", length)
    if not np.allclose(raw - ledger["mass_g"], force_error, atol=1e-8, rtol=0) or abs(force_error[-1]) > .010 + 1e-12:
        raise ValueError(f"{demo.name}: final raw force and contained mass disagree by more than 10 mg")
    reading = _channel(demo, "physics/balance_g", length)
    if not np.allclose(demo["next_obs/scale_reading_g"][:, 0], reading, atol=1e-6, rtol=1e-6):
        raise ValueError(f"{demo.name}: observed scale differs from recorded instrument reading")
    for name, values in (("mass_g", ledger["mass_g"]), ("reading_g", reading), ("spilled_g", ledger["spilled_g"])):
        if not math.isclose(float(demo.attrs.get(name, float("nan"))), float(values[-1]), abs_tol=1e-8, rel_tol=1e-8):
            raise ValueError(f"{demo.name}: final {name} metadata disagrees with its trace")
    target = float(demo.attrs["target_g"])
    if (not math.isfinite(target) or not 0 < target <= total or abs(ledger["mass_g"][-1] - target) > .05 + 1e-9
            or abs(reading[-1] - target) > .05 + 1e-9 or ledger["inflight_g"][-1] > mass_tolerance
            or ledger["spilled_g"][-1] > .05 + 1e-9):
        raise ValueError(f"{demo.name}: final weighing outcome does not meet the 50 mg band")
    seed = demo.attrs.get("seed")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError(f"{demo.name}: missing integer episode seed")
    return (powder, int(seed), target), _fingerprint(demo)


def merge_demos(inputs, output, expected_episodes=30):
    """All-or-nothing local merge; no model loading, training, or cloud calls."""
    if expected_episodes <= 0:
        raise ValueError("expected_episodes must be positive")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    inputs = [Path(path).resolve(strict=True) for path in inputs]
    if not inputs or len(set(inputs)) != len(inputs):
        raise ValueError("Supply distinct input files; duplicate paths are rejected")
    metadata = None
    episodes = []
    identities, trajectories, file_hashes = set(), set(), set()
    for path in inputs:
        digest = file_sha256(path)
        if digest in file_hashes:
            raise ValueError(f"Duplicate input file contents: {path}")
        file_hashes.add(digest)
        audit = inspect_dataset(path, (5,))
        if audit["use"] != "synthetic_coarse_grain_control_data" or not audit["num_episodes"]:
            raise ValueError(f"{path}: no successful physical-grain episodes")
        with h5py.File(path, "r") as handle:
            current = _metadata(handle)
            if metadata is None:
                metadata = current
            elif current != metadata:
                raise ValueError(f"{path}: incompatible grain count, material bank, split, or source version")
            for row in audit["episodes"]:
                identity, fingerprint = _validate_episode(handle["data"][row["name"]], metadata)
                if identity in identities or fingerprint in trajectories:
                    raise ValueError(f"{path}/{row['name']}: duplicate episode identity or numerical trajectory")
                identities.add(identity)
                trajectories.add(fingerprint)
                episodes.append({"path": str(path), "episode": row["name"], "file_sha256": digest,
                                 "trajectory_sha256": fingerprint, "powder_id": identity[0]})
    if len(episodes) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes, received {len(episodes)}; no output written")
    if {row["powder_id"] for row in episodes} != set(metadata["training_powder_ids"]):
        raise ValueError("Every declared training material must be represented")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.pending")
    try:
        with h5py.File(temporary, "x") as merged:
            with h5py.File(inputs[0], "r") as first:
                _copy_attributes(first, merged)
            merged.attrs.update(merged_episode_count=len(episodes), expected_episode_count=expected_episodes,
                                merge_validation="physical_grain_episode_audit_v1",
                                rgb_storage_layout="one_frame_per_chunk_gzip4",
                                rgb_copy_window_frames=RGB_COPY_FRAMES,
                                merge_provenance=json.dumps(episodes, sort_keys=True))
            data = merged.create_group("data")
            with h5py.File(inputs[0], "r") as first:
                _copy_attributes(first["data"], data)
            for index, row in enumerate(episodes):
                with h5py.File(row["path"], "r") as source:
                    _copy_episode(source["data"][row["episode"]], data, name=f"demo_{index}")
            for path, digest in {(row["path"], row["file_sha256"]) for row in episodes}:
                if file_sha256(path) != digest:
                    raise ValueError(f"Input changed during merge: {path}")
        audit = inspect_dataset(temporary, (5,))
        # Atomic publication without overwriting a destination created meanwhile.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    audit["dataset_path"] = str(output.resolve())
    audit["dataset_sha256"] = file_sha256(output)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=30)
    args = parser.parse_args(argv)
    print(json.dumps(merge_demos(args.inputs, args.out, args.expected_episodes), indent=2))


if __name__ == "__main__":
    main()
