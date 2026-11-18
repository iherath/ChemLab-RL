"""Lossless HDF5 shard/assembly checks, without model training or cloud access."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.package_vial_dataset import assemble, package_from_combined, sha256, verify


def assert_equal_hdf(left, right):
    def attrs(a, b):
        assert set(a.attrs) == set(b.attrs)
        for key in a.attrs:
            assert a.attrs.get_id(key).dtype == b.attrs.get_id(key).dtype
            np.testing.assert_array_equal(a.attrs[key], b.attrs[key])
    def compare(a, b):
        attrs(a, b)
        assert isinstance(a, h5py.Group) == isinstance(b, h5py.Group)
        if isinstance(a, h5py.Group):
            assert set(a) == set(b)
            for key in a:
                compare(a[key], b[key])
        else:
            assert a.dtype == b.dtype and a.shape == b.shape
            assert (a.compression, a.compression_opts, a.chunks, a.shuffle, a.fletcher32) == (b.compression, b.compression_opts, b.chunks, b.shuffle, b.fletcher32)
            np.testing.assert_array_equal(a[()], b[()])
    with h5py.File(left, "r") as a, h5py.File(right, "r") as b:
        compare(a, b)


def fixture_file(path):
    with h5py.File(path, "w") as f:
        f.attrs["root_note"] = "fixed scene with μg display"
        f.attrs.create("wide_array", np.array([1, 2, 65535], dtype=np.uint16))
        f.create_dataset("calibration/bias", data=np.array([1.0, 2.0], np.float64))
        f["calibration"].attrs["units"] = "grams"
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"collector": "hybrid_vial_98_v1"})
        data.attrs["total"] = np.int64(12)
        data.create_dataset("summary", data=np.array(123, np.int32))
        for i in range(3):
            group = data.create_group(f"demo_{i}")
            group.attrs["num_samples"] = 4
            group.attrs["source_episode"] = f"demo_{i}"
            group.attrs["source_dataset_sha256"] = "a"*64
            group.attrs["physical_valid"] = np.bool_(i > 0)
            group.attrs["collector"] = "scripted"
            group.create_dataset("actions", data=np.full((4, 14), i, np.float32))
            group.create_dataset("dones", data=np.array([0, 0, 1, 1], np.uint8))
            image = np.arange(4*8*8*3, dtype=np.uint8).reshape(4, 8, 8, 3)
            pixels = group.create_dataset("obs/image", data=image, chunks=(1, 8, 8, 3), compression="gzip", compression_opts=3, shuffle=True, fletcher32=True)
            pixels.attrs["channel_order"] = np.bytes_("RGB")
    return dict(train_episodes=["demo_0", "demo_2"], heldout_episodes=["demo_1"], excluded_episodes=[])


def test_roundtrip_preserves_all_hdf_values_types_attrs_and_compression(tmp_path):
    source = tmp_path / "combined.hdf5"
    split = fixture_file(source)
    package = tmp_path / "package"
    manifest = package_from_combined(source, package, split=split)
    assert manifest["episode_count"] == 3 and manifest["fixed_scene"]
    assert manifest["episodes"][0]["controls_through_first_terminal"] == 3
    assert manifest["episodes"][1]["split"] == "heldout"
    assert manifest["episodes"][2]["origin"]["physical_valid"] is True
    assert not str(tmp_path) in (package / "manifest.json").read_text()
    assert verify(package) == manifest
    with h5py.File(package / "metadata.hdf5", "r") as metadata:
        assert not metadata["/"].id.get_create_plist().get_obj_track_times()
        assert not metadata["data"].id.get_create_plist().get_obj_track_times()
    output = tmp_path / "assembled.hdf5"
    identity = assemble(package, output)
    assert identity["assembled_sha256"] == sha256(output)
    assert identity["source_combined_sha256"] == sha256(source)
    assert json.loads(output.with_suffix(".hdf5.assembly.json").read_text()) == identity
    assert_equal_hdf(source, output)


def test_corruption_fails_before_creating_output(tmp_path):
    source = tmp_path / "combined.hdf5"
    split = fixture_file(source)
    package = tmp_path / "package"
    package_from_combined(source, package, split=split)
    with (package / "shards/demo_000.hdf5").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        assemble(package, tmp_path / "assembled.hdf5")
    assert not (tmp_path / "assembled.hdf5").exists()


def test_existing_outputs_and_bad_split_are_refused(tmp_path):
    source = tmp_path / "combined.hdf5"
    split = fixture_file(source)
    package = tmp_path / "package"
    with pytest.raises(ValueError, match="partition"):
        package_from_combined(source, package, split={**split, "heldout_episodes": []})
    assert not package.exists()
    package_from_combined(source, package, split=split)
    with pytest.raises(FileExistsError):
        package_from_combined(source, package, split=split)
    output = tmp_path / "assembled.hdf5"
    assemble(package, output)
    with pytest.raises(FileExistsError):
        assemble(package, output)


def test_external_links_and_reference_attributes_are_explicitly_rejected(tmp_path):
    for kind in ("external", "reference"):
        source = tmp_path / (kind+".hdf5")
        split = fixture_file(source)
        with h5py.File(source, "a") as f:
            if kind == "external":
                f["elsewhere"] = h5py.ExternalLink("private.hdf5", "/data")
            else:
                f.attrs["pointer"] = f["data"].ref
        with pytest.raises(ValueError, match="unsupported|self-contained"):
            package_from_combined(source, tmp_path / kind, split=split)


def test_manifest_paths_cannot_escape_package(tmp_path):
    source = tmp_path / "combined.hdf5"
    split = fixture_file(source)
    package = tmp_path / "package"
    package_from_combined(source, package, split=split)
    manifest = json.loads((package / "manifest.json").read_text())
    manifest["metadata"]["path"] = "../combined.hdf5"
    (package / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        verify(package)
