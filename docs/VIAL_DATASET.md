# Vial-placement dataset package

The combined dataset has 98 trajectories: eight usable historical demonstrations
and 90 additional demonstrations collected with physics-substep contact checks.
Historical `demo_8` is excluded because its balance reading exceeded 100 g during
placement. Historical `demo_7` is the held-out trajectory. The remaining 97
trajectories train. The split is recorded in
[`configs/vial_hybrid98_split.json`](../configs/vial_hybrid98_split.json).

This is the native rigid-body vial task: open the draft shield, move the 20 g
vial onto the scale, and close the shield. Initial scene geometry is fixed.
The held-out trajectory does not establish generalization to different scenes
or powders. The eight historical trajectories lack the newer collection
pipeline's exhaustive contact audit; their origin remains explicit.

## Repository format and assembly

`data/teleop/vial_placement_v1/` contains:

- `manifest.json`: component SHA256 values, source combined-file SHA256,
  per-episode provenance, counts, split, and fixed-scene scope.
- `split.json`: the same 97/1 split, bundled with the data.
- `metadata.hdf5`: every root/data attribute and every non-demo group from the
  original combined file.
- `shards/demo_000.hdf5` through `shards/demo_097.hdf5`: one intact episode per
  file. Each component must be below 95 MiB. Internal names remain `demo_0`
  through `demo_97`; numeric filename padding is only for directory sorting.

These are regular Git files; Git LFS is not required. Reconstruct the HDF5 from
the repository root after installing the environment below:

```bash
.venv-vial/bin/python scripts/package_vial_dataset.py assemble --package-dir data/teleop/vial_placement_v1 --output data/teleop/weigh_vial_hybrid98.hdf5
```

Assembly verifies every component's size and SHA256 before copying. It preserves
all recorded arrays, shapes, dtypes, attributes, chunk layouts, and compression.
It refuses existing output files. The output's
`weigh_vial_hybrid98.hdf5.assembly.json` sidecar records the reconstructed file's
new byte hash, original combined-file hash, package-manifest hash, and split.
HDF5 binary layout may change during reconstruction, so a new BC data manifest
must use the reconstructed file's hash. Per-demo payload hashes and origin
metadata remain unchanged.

To verify an existing package without assembling:

```bash
.venv-vial/bin/python scripts/package_vial_dataset.py verify --package-dir data/teleop/vial_placement_v1
```

For maintainers packaging the collected combined file:

```bash
.venv-vial/bin/python scripts/package_vial_dataset.py package --input data/teleop/vial90-audited-001_combined98.hdf5 --output-dir data/teleop/vial_placement_v1 --split configs/vial_hybrid98_split.json
```

The packager streams HDF5 object copies and checks each resulting file size.
It refuses overwrite, overlapping/incomplete splits, external/soft links,
reference-valued objects, aliased objects, and externally stored or virtual
datasets. Those uncommon constructs are absent from the intended dataset;
refusal prevents a supposedly lossless package from silently losing references.
No source paths outside the package or local credentials are stored in the
generated JSON metadata.

## Reproducible native code and training environment

The parent repository does not vendor Leo's native simulator. The audited source
checkout was clean at these commits:

| Repository | Required commit |
|---|---|
| [chem_rl_sim](https://github.com/leo01110111/chem_rl_sim) | `dc71b11c7675b17c59aff9e88305e7e45acc93cc` |
| [urgantry_sim](https://github.com/leo01110111/urgantry_sim) | `46a784ac2cb59080205e722d91084bb244195092` |

No native patch is needed. The bootstrap clones the exact simulator commit,
initializes its pinned robot submodule, creates `.venv-vial`, installs the
training requirements, and installs both native packages in editable mode.
It also makes this repository importable without a manually set `PYTHONPATH`.
Existing native checkouts with another revision or tracked edits are left
unchanged and cause a clear error.

Use Python 3.12, Git, an NVIDIA driver compatible with the installed CUDA PyTorch
wheel, and the Linux rendering libraries `libegl1`, `libgl1`, `libosmesa6` plus
`ffmpeg`. The bootstrap does not install system packages or GPU drivers.

```bash
bash scripts/bootstrap_vial_training.sh
```

The critical versions match the Modal image: LeRobot 0.7.0, MuJoCo 3.16.0,
Gymnasium 1.4.0, and Trimesh 5.1.1. NumPy uses the 2.x range required by this
runtime. Use [`requirements-vial.txt`](../requirements-vial.txt), which includes
[`requirements-vla.txt`](../requirements-vla.txt); the toy project's
`requirements.txt` has a conflicting NumPy 1.x constraint.

Remaining dependency versions are resolved by pip and saved to
`.venv-vial/installed-requirements.txt` after installation. This snapshot records
the installed environment; the repository does not claim a complete
platform-independent transitive lock. The bootstrap checks package versions,
imports BC/RL modules, compiles the committed native assets into a 14-actuator
MuJoCo model, checks for the balance force sensor, and prints CUDA availability.
It starts no model training and downloads no pretrained weights.

To repeat only those setup checks:

```bash
bash scripts/bootstrap_vial_training.sh --check-only
```

Use `.venv-vial/bin/python` for the commands in
[`VIAL_VLA_RUNBOOK.md`](VIAL_VLA_RUNBOOK.md). That runbook covers BC, checkpoint
verification, native ODE evaluation, the matched stochastic baseline, and
OGPO/VaDar. [`VIAL_ALGORITHMS.md`](VIAL_ALGORITHMS.md) explains the algorithm
differences and current evidence limits. All BC/RL runs must use the saved
checkpoint's exact reconstructed dataset hash and train split.

## Packaging checks

Five synthetic round-trip tests verify all dataset values, attributes and
attribute dtypes, image compression/chunking, root/non-demo metadata, overwrite
refusal, corruption detection, split validity, and path containment. An actual
historical demonstration was also packaged and reconstructed: its 29.8 MB HDF5
episode retained semantic SHA256
`926774c953558382d3201a7cf6ef6a21c6fb6d64c980f907478ce84215ecad54`.
These checks establish packaging fidelity, not policy performance.
