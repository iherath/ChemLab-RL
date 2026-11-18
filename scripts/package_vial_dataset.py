"""Package the vial HDF5 as Git-sized per-demo files and reconstruct it locally."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import h5py
import numpy as np

SCHEMA = "vial_hdf_shards_v1"
MAX_FILE_BYTES = 95 * 1024**2  # Below GitHub's 100 MiB individual-file limit.
TASK_ID = "vial_placement_native_v1"
ORIGIN_KEYS = ("collector", "source_dataset_sha256", "source_episode", "source_version",
               "collector_sha256", "physics_validation", "physical_valid", "seed")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_scalar(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _copy_attributes(source, target):
    # Explicit dtype preserves fixed strings, integer widths, arrays and opaque
    # values rather than re-inferring types from their Python representation.
    for name in source.attrs:
        target.attrs.create(name, source.attrs[name], dtype=source.attrs.get_id(name).dtype)


def _new_hdf(path):
    properties = h5py.h5p.create(h5py.h5p.FILE_CREATE)
    properties.set_obj_track_times(False)
    return h5py.File(h5py.h5f.create(os.fsencode(path), h5py.h5f.ACC_EXCL, fcpl=properties))


def _new_group(parent, name):
    properties = h5py.h5p.create(h5py.h5p.GROUP_CREATE)
    properties.set_obj_track_times(False)
    return h5py.Group(h5py.h5g.create(parent.id, name.encode(), gcpl=properties))


def _contains_reference(dtype):
    if h5py.check_dtype(ref=dtype) is not None:
        return True
    if dtype.fields:
        return any(_contains_reference(entry[0]) for entry in dtype.fields.values())
    return dtype.subdtype is not None and _contains_reference(dtype.subdtype[0])


def _validate_copy_contract(source):
    """Refuse uncommon cross-file constructs rather than silently changing them."""
    addresses = {}
    def check(obj):
        for name in obj.attrs:
            if _contains_reference(obj.attrs.get_id(name).dtype):
                raise ValueError(f"HDF5 reference attribute unsupported: {obj.name}/{name}")
        if isinstance(obj, h5py.Dataset):
            if _contains_reference(obj.dtype) or obj.is_virtual or obj.external:
                raise ValueError(f"Reference/virtual/external dataset unsupported: {obj.name}")
        if isinstance(obj, h5py.Group):
            for name in obj:
                link = obj.get(name, getlink=True)
                if not isinstance(link, h5py.HardLink):
                    raise ValueError(f"Only self-contained hard links are supported: {obj.name}/{name}")
                child = obj[name]
                address = h5py.h5o.get_info(child.id).addr
                if address in addresses:
                    raise ValueError(f"Aliased HDF5 objects unsupported: {child.name}")
                addresses[address] = child.name
                check(child)
    check(source)


def _validate_split(split, names):
    keys = ("train_episodes", "heldout_episodes", "excluded_episodes")
    if set(split) != set(keys) or not all(isinstance(split[key], list) for key in keys):
        raise ValueError("Split must contain train_episodes, heldout_episodes and excluded_episodes lists")
    all_names = [name for key in keys for name in split[key]]
    if any(not isinstance(name, str) for name in all_names) or len(set(all_names)) != len(all_names):
        raise ValueError("Split episode IDs must be unique strings")
    if set(all_names) != set(names):
        raise ValueError("Split must partition every packaged demonstration exactly once")
    return {key: list(split[key]) for key in keys}


def _default_split(names):
    expected = {f"demo_{i}" for i in range(98)}
    if set(names) != expected:
        raise ValueError("Default package requires 98 demos; provide an explicit split for another dataset")
    return dict(train_episodes=[f"demo_{i}" for i in [*range(7), *range(8, 98)]],
                heldout_episodes=["demo_7"], excluded_episodes=[])


def _record_file(path, relative):
    size = path.stat().st_size
    if size >= MAX_FILE_BYTES:
        raise ValueError(f"{relative} is {size} bytes, exceeding the per-file package budget")
    return dict(path=relative, bytes=size, sha256=sha256(path))


def package_from_combined(input_path, outdir, *, split=None):
    """Stream-copy each demonstration, preserving HDF5 datasets and attributes.

    The manifest has no timestamps or absolute machine paths. One metadata HDF5
    preserves root/data attributes and all non-demonstration groups. HDF5 copies
    retain chunk layout, compression and dataset types without loading movies.
    """
    input_path, outdir = Path(input_path), Path(outdir)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if outdir.exists():
        raise FileExistsError(f"Package output already exists: {outdir}")
    outdir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{outdir.name}.build-", dir=outdir.parent))
    try:
        (staging / "shards").mkdir()
        combined_sha = sha256(input_path)
        with h5py.File(input_path, "r") as source:
            if "data" not in source or not isinstance(source["data"], h5py.Group):
                raise ValueError("Combined HDF5 requires a data group")
            names = sorted((name for name in source["data"] if re.fullmatch(r"demo_\d+", name)),
                           key=lambda name: int(name.split("_")[1]))
            if not names or any(not isinstance(source["data"][name], h5py.Group) for name in names):
                raise ValueError("Expected nonempty demonstration groups")
            split = _validate_split(_default_split(names) if split is None else split, names)
            _validate_copy_contract(source)
            metadata = staging / "metadata.hdf5"
            with _new_hdf(metadata) as target:
                _copy_attributes(source, target)
                for name in source:
                    if name != "data":
                        source.copy(name, target, name=name)
                data = _new_group(target, "data")
                _copy_attributes(source["data"], data)
                for name in source["data"]:
                    if name not in names:
                        source.copy(source["data"][name], data, name=name)
            records = []
            for index, name in enumerate(names):
                group = source["data"][name]
                relative = f"shards/demo_{index:03d}.hdf5"
                path = staging / relative
                with _new_hdf(path) as shard:
                    source.copy(group, _new_group(shard, "data"), name=name)
                record = dict(episode=name, **_record_file(path, relative),
                              origin={key: _json_scalar(group.attrs[key]) for key in ORIGIN_KEYS if key in group.attrs},
                              split=next(key.removesuffix("_episodes") for key in split if name in split[key]))
                if "actions" in group:
                    record["recorded_controls"] = len(group["actions"])
                if "dones" in group:
                    first = np.flatnonzero(np.asarray(group["dones"], bool))
                    record["controls_through_first_terminal"] = int(first[0])+1 if len(first) else None
                records.append(record)
        manifest = dict(schema=SCHEMA, task_id=TASK_ID, source_combined_file=input_path.name,
                        source_combined_sha256=combined_sha, source_combined_bytes=input_path.stat().st_size,
                        episode_count=len(names), fixed_scene=True,
                        initial_condition_generalization=False,
                        contact_scope="historical episodes lack exhaustive substep audit; new episodes retain their acceptance metadata",
                        metadata=_record_file(metadata, "metadata.hdf5"), episodes=records, split=split,
                        file_byte_limit=MAX_FILE_BYTES,
                        assembly_identity="HDF5 semantic round trip; assembled byte hash is recorded after reconstruction")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
        (staging / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True)+"\n")
        staging.rename(outdir)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _member_path(package_dir, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Package member path must be relative")
    root = Path(package_dir).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Package member escapes its directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def verify(package_dir):
    """Verify every component hash and manifest/split completeness before use."""
    package_dir = Path(package_dir)
    manifest = json.loads((package_dir / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("task_id") != TASK_ID:
        raise ValueError("Unknown vial package schema/task")
    episodes = manifest["episodes"]
    names = [r["episode"] for r in episodes]
    if len(names) != manifest["episode_count"] or len(set(names)) != len(names):
        raise ValueError("Manifest episode list is incomplete or duplicated")
    if any(not re.fullmatch(r"demo_\d+", name) for name in names):
        raise ValueError("Invalid episode ID")
    _validate_split(manifest["split"], names)
    if json.loads((package_dir / "split.json").read_text()) != manifest["split"]:
        raise ValueError("Package split.json differs from manifest")
    records = [manifest["metadata"], *episodes]
    if len({r["path"] for r in records}) != len(records):
        raise ValueError("Duplicate component paths")
    for record in records:
        path = _member_path(package_dir, record["path"])
        if (path.stat().st_size != record["bytes"] or path.stat().st_size >= MAX_FILE_BYTES
                or sha256(path) != record["sha256"]):
            raise ValueError(f"Package checksum/size mismatch: {record['path']}")
    return manifest


def assemble(package_dir, output):
    """Reconstruct all groups/attributes and record the newly assembled file hash."""
    package_dir, output = Path(package_dir), Path(output)
    sidecar = output.with_suffix(output.suffix + ".assembly.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Assembly output or identity sidecar already exists")
    manifest = verify(package_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{output.name}.assemble-", suffix=".hdf5",
                                     dir=output.parent, delete=False) as stream:
        staging = Path(stream.name)
    try:
        shutil.copyfile(_member_path(package_dir, manifest["metadata"]["path"]), staging)
        with h5py.File(staging, "r+") as target:
            if "data" not in target or not isinstance(target["data"], h5py.Group):
                raise ValueError("Package metadata lacks data group")
            for record in manifest["episodes"]:
                name = record["episode"]
                if name in target["data"]:
                    raise ValueError(f"Duplicate assembled episode: {name}")
                with h5py.File(_member_path(package_dir, record["path"]), "r") as shard:
                    if set(shard) != {"data"} or set(shard["data"]) != {name}:
                        raise ValueError(f"Unexpected shard contents: {record['path']}")
                    shard.copy(shard["data"][name], target["data"], name=name)
        identity = dict(schema=SCHEMA, task_id=TASK_ID, assembled_file=output.name,
                        assembled_sha256=sha256(staging), assembled_bytes=staging.stat().st_size,
                        source_combined_sha256=manifest["source_combined_sha256"],
                        package_manifest_sha256=sha256(package_dir / "manifest.json"),
                        episode_count=manifest["episode_count"], split=manifest["split"],
                        fixed_scene=True, verification="all component SHA256 values verified before assembly")
        staging.rename(output)
        sidecar.write_text(json.dumps(identity, indent=2, sort_keys=True)+"\n")
        return identity
    except BaseException:
        staging.unlink(missing_ok=True)
        raise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("package")
    packing.add_argument("--input", required=True, type=Path)
    packing.add_argument("--output-dir", required=True, type=Path)
    packing.add_argument("--split", type=Path)
    unpacking = commands.add_parser("assemble")
    unpacking.add_argument("--package-dir", required=True, type=Path)
    unpacking.add_argument("--output", required=True, type=Path)
    checking = commands.add_parser("verify")
    checking.add_argument("--package-dir", required=True, type=Path)
    args = p.parse_args(argv)
    if args.command == "package":
        split = json.loads(args.split.read_text()) if args.split else None
        result = package_from_combined(args.input, args.output_dir, split=split)
        summary = dict(package=str(args.output_dir), episodes=result["episode_count"],
                       source_combined_sha256=result["source_combined_sha256"],
                       component_bytes=sum(r["bytes"] for r in [result["metadata"], *result["episodes"]]))
    elif args.command == "assemble":
        summary = assemble(args.package_dir, args.output)
    else:
        result = verify(args.package_dir)
        summary = dict(verified=True, package=str(args.package_dir), episodes=result["episode_count"])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
