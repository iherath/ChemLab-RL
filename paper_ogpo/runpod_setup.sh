#!/bin/bash
# One-time RunPod setup for paper OGPO + VaDaR on ChemVialWeighing.
#
#   bash paper_ogpo/runpod_setup.sh
#
# Expects this repo at CHEMLAB_ROOT (default: directory containing paper_ogpo/).
# Demos ship as data/teleop/weigh_vial_grok_v2_lowdim.hdf5 (images stripped for git).
set -euo pipefail

CHEMLAB_ROOT=${CHEMLAB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VADAR_ROOT=${VADAR_ROOT:-$(cd "$CHEMLAB_ROOT/.." && pwd)/vadar-public}
CHEM_RL_SIM=${CHEM_RL_SIM:-$CHEMLAB_ROOT/chem_rl_sim}
CHEMVIAL_DATASET=${CHEMVIAL_DATASET:-$CHEMLAB_ROOT/data/teleop/weigh_vial_grok_v2_lowdim.hdf5}
CUDA=${CUDA:-cuda12}

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

if command -v apt-get >/dev/null; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq libosmesa6 libgl1 >/dev/null
fi

if [ ! -f "$VADAR_ROOT/vadar/main.py" ]; then
    git clone --depth 1 https://github.com/simchowitzlabpublic/vadar-public.git "$VADAR_ROOT"
fi
if [ ! -f "$CHEM_RL_SIM/chem_rl_sim/env.py" ]; then
    git clone --recurse-submodules https://github.com/leo01110111/chem_rl_sim.git "$CHEM_RL_SIM"
fi
if [ ! -f "$CHEMVIAL_DATASET" ]; then
    echo "missing $CHEMVIAL_DATASET — git pull should include weigh_vial_grok_v2_lowdim.hdf5" >&2
    exit 1
fi

cd "$VADAR_ROOT"
if ! UV_PROJECT_ENVIRONMENT="$VADAR_ROOT/.venv" uv sync --frozen --extra "$CUDA"; then
    UV_PROJECT_ENVIRONMENT="$VADAR_ROOT/.venv" uv sync --extra "$CUDA"
fi
UV_PROJECT_ENVIRONMENT="$VADAR_ROOT/.venv" uv pip install mujoco

"$VADAR_ROOT/.venv/bin/python" "$CHEMLAB_ROOT/paper_ogpo/patch_vadar.py" "$VADAR_ROOT"

cat > "$CHEMLAB_ROOT/paper_ogpo/.env.sh" <<EOF
export CHEMLAB_ROOT="$CHEMLAB_ROOT"
export VADAR_ROOT="$VADAR_ROOT"
export CHEM_RL_SIM="$CHEM_RL_SIM"
export CHEMVIAL_DATASET="$CHEMVIAL_DATASET"
export PYTHONPATH="$VADAR_ROOT:$CHEMLAB_ROOT:$CHEM_RL_SIM:$CHEM_RL_SIM/third_party/urgantry_sim"
export MUJOCO_GL=osmesa
export WANDB_MODE=\${WANDB_MODE:-online}
EOF
echo "wrote $CHEMLAB_ROOT/paper_ogpo/.env.sh"
echo "next: source paper_ogpo/.env.sh && bash paper_ogpo/run_chem_vial.sh"
