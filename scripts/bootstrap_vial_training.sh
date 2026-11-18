#!/usr/bin/env bash
# Native source + isolated Python environment. This does not start training.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VIAL_NATIVE="$ROOT/chem_rl_sim"
VIAL_ENV="${VIAL_ENV:-$ROOT/.venv-vial}"
VIAL_PYTHON="${VIAL_PYTHON:-python3.12}"
NATIVE_COMMIT=dc71b11c7675b17c59aff9e88305e7e45acc93cc
ROBOT_COMMIT=46a784ac2cb59080205e722d91084bb244195092
CHECK_ONLY=0
if [[ "${1:-}" == "--check-only" && "$#" == 1 ]]; then
    CHECK_ONLY=1
elif [[ "$#" != 0 ]]; then
    echo "Usage: bash scripts/bootstrap_vial_training.sh [--check-only]" >&2
    exit 2
fi

if [[ "$CHECK_ONLY" == 0 ]]; then
    "$VIAL_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
    if [[ ! -e "$VIAL_NATIVE" ]]; then
        git clone --no-checkout https://github.com/leo01110111/chem_rl_sim.git "$VIAL_NATIVE"
        git -C "$VIAL_NATIVE" checkout --detach "$NATIVE_COMMIT"
    fi
fi

if [[ ! -d "$VIAL_NATIVE/.git" && ! -f "$VIAL_NATIVE/.git" ]]; then
    echo "Missing native Git checkout: $VIAL_NATIVE" >&2
    exit 1
fi
if [[ "$(git -C "$VIAL_NATIVE" rev-parse HEAD)" != "$NATIVE_COMMIT" ]]; then
    echo "Native checkout must be at $NATIVE_COMMIT; existing checkout left unchanged." >&2
    exit 1
fi
if [[ -n "$(git -C "$VIAL_NATIVE" status --porcelain --untracked-files=no)" ]]; then
    echo "Native checkout has tracked modifications; refusing to change it." >&2
    exit 1
fi
if [[ "$CHECK_ONLY" == 0 ]]; then
    git -C "$VIAL_NATIVE" submodule update --init --recursive
fi
if [[ "$(git -C "$VIAL_NATIVE/third_party/urgantry_sim" rev-parse HEAD)" != "$ROBOT_COMMIT" ]]; then
    echo "Robot submodule does not match $ROBOT_COMMIT" >&2
    exit 1
fi
if [[ -n "$(git -C "$VIAL_NATIVE/third_party/urgantry_sim" status --porcelain --untracked-files=no)" ]]; then
    echo "Robot submodule has tracked modifications; refusing to continue." >&2
    exit 1
fi

if [[ "$CHECK_ONLY" == 0 ]]; then
    if [[ ! -x "$VIAL_ENV/bin/python" ]]; then
        "$VIAL_PYTHON" -m venv "$VIAL_ENV"
    fi
    "$VIAL_ENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing environment is not Python 3.12"'
    "$VIAL_ENV/bin/python" -m pip install -r "$ROOT/requirements-vial.txt"
    # Runtime dependencies are installed above; no dependency re-resolution is
    # needed when registering these two committed native packages.
    "$VIAL_ENV/bin/python" -m pip install --no-deps -e "$VIAL_NATIVE/third_party/urgantry_sim" -e "$VIAL_NATIVE"
    "$VIAL_ENV/bin/python" - "$ROOT" <<'PY'
import pathlib, site, sys
pathlib.Path(site.getsitepackages()[0], "chemlab_project.pth").write_text(sys.argv[1]+"\n")
PY
    "$VIAL_ENV/bin/python" -m pip freeze > "$VIAL_ENV/installed-requirements.txt"
fi

if [[ ! -x "$VIAL_ENV/bin/python" ]]; then
    echo "Missing isolated environment: $VIAL_ENV" >&2
    exit 1
fi
if [[ "$(uname -s)" != Darwin ]]; then
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi
export TOKENIZERS_PARALLELISM=false
"$VIAL_ENV/bin/python" - "$ROOT" <<'PY'
import importlib.metadata as md
import json
from pathlib import Path
import sys
import torch
from chem_rl_sim.scene import build_model
from chemlab_rl.vial_vla_data import TASK_ID
from chemlab_rl.vial_ogpo_training import VialOGPOConfig
from scripts.package_vial_dataset import SCHEMA
expected = {"lerobot": "0.7.0", "mujoco": "3.16.0", "gymnasium": "1.4.0", "trimesh": "5.1.1"}
versions = {name: md.version(name) for name in [*expected, "torch", "numpy", "h5py"]}
for name, version in expected.items():
    if versions[name] != version:
        raise RuntimeError(f"Unexpected {name} version: {versions[name]}, expected {version}")
model = build_model(layout="weigh", hand="robotiq", clutter=False)
assert model.nu == 14 and model.sensor("balance_pan_force").id >= 0
assert VialOGPOConfig().chain.steps == 10
print(json.dumps({"ready": True, "task_id": TASK_ID, "package_schema": SCHEMA,
                  "versions": versions, "cuda_available": torch.cuda.is_available(),
                  "native_actuators": model.nu, "python": sys.executable}, indent=2))
PY
echo "Setup complete. Activate .venv-vial or use its python executable. No training was started."
