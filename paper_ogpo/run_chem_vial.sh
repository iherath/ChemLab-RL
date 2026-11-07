#!/bin/bash
# Paper OGPO + VaDaR (next-future, vector critic) on chem_vial-low_dim.
# Hyperparameters follow vadar-public scripts/paper/{fork_pins,tasks/toolhang_ogpo,methods/vadar_nf}.
#
#   source paper_ogpo/.env.sh
#   bash paper_ogpo/run_chem_vial.sh
#   VRAM_GB=48 bash paper_ogpo/run_chem_vial.sh          # A6000 / A100-40
#   FAST=1 bash paper_ogpo/run_chem_vial.sh              # short compile check
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck disable=SC1091
[ -f "$HERE/.env.sh" ] && source "$HERE/.env.sh"

: "${VADAR_ROOT:?run paper_ogpo/runpod_setup.sh first}"
: "${CHEMVIAL_DATASET:?}"
: "${PYTHONPATH:?}"

VRAM_GB=${VRAM_GB:-24}
FAST=${FAST:-0}
seed=${seed:-0}
device=${device:-0}
SAVE_DIR=${SAVE_DIR:-$CHEMLAB_ROOT/docs/ogpo_vadar}

if [ "$VRAM_GB" -ge 40 ]; then
    jax_frac=${jax_frac:-0.45}
    n_eval_envs=${n_eval_envs:-32}
    buffer_size=${buffer_size:-2000000}
else
    jax_frac=${jax_frac:-0.65}
    n_eval_envs=${n_eval_envs:-8}
    buffer_size=${buffer_size:-1000000}
fi

# Paper tool-hang OGPO column, with VaDaR NF / vector critic.
bc_pi_steps=300000
q_warmup_steps=20000
online_steps=3000000
start_training=10000
eval_interval=20000
eval_interval_bc=20000
group_num_samples=16
if [ "$FAST" = "1" ]; then
    bc_pi_steps=2000
    q_warmup_steps=200
    online_steps=200
    start_training=50
    n_eval_envs=2
    eval_interval=200
    eval_interval_bc=200
    buffer_size=20000
fi

export CUDA_VISIBLE_DEVICES=$device
export XLA_PYTHON_CLIENT_MEM_FRACTION=$jax_frac
export MUJOCO_GL=${MUJOCO_GL:-egl}

cd "$VADAR_ROOT"
mkdir -p "$SAVE_DIR" logs
echo "=== ogpo_vadar chem_vial | VRAM_GB=$VRAM_GB jax_frac=$jax_frac n_eval_envs=$n_eval_envs FAST=$FAST ==="

"$VADAR_ROOT/.venv/bin/python" vadar/main.py \
    --algo=ogpo_vadar \
    --experiment.project=chem-vial \
    --experiment.run_group=ogpo_vadar \
    --experiment.run_name=chem_vial_ogpo_vadar \
    --experiment.wandb_name=chem_vial_ogpo_vadar \
    --experiment.seed=$seed \
    --experiment.log=true \
    --experiment.save_dir=$SAVE_DIR \
    --env.name=chem_vial-low_dim \
    --env.goal_obs=null \
    --env.reward_mode=sparse \
    --training.bc_pi_steps=$bc_pi_steps \
    --training.bc_q_steps=0 \
    --training.calql_steps=0 \
    --training.bc_refine_steps=0 \
    --training.q_warmup_steps=$q_warmup_steps \
    --training.online_steps=$online_steps \
    --training.bc_only=false \
    --training.gpu_mode=single \
    --training.force_jax_sync=true \
    --buffer.size=$buffer_size \
    --buffer.start_training=$start_training \
    --buffer.utd_warmup=1 \
    --buffer.utd_online=1 \
    --buffer.utd_q=1 \
    --buffer.utd_pi=1 \
    --buffer.offline_ratio=0.0 \
    --buffer.q_warmup_mode=interleaved \
    --buffer.warmup_offline_ratio=0.0 \
    --eval.n_eval_envs=$n_eval_envs \
    --eval.eval_episodes=2 \
    --eval.plot_q_vs_mc=false \
    --intervals.log_interval=5000 \
    --intervals.eval_interval=$eval_interval \
    --intervals.eval_interval_bc=$eval_interval_bc \
    --intervals.save_interval=500000 \
    --bc.clip_bc=true \
    --bc.clip_bc_threshold=0.25 \
    --bc.clip_bc_wrt=ODE \
    --success_buffer.use_success_buffer=true \
    --agent.horizon_length=8 \
    --agent.discount=0.999 \
    --agent.tau=0.05 \
    --agent.actor_tau=0.005 \
    --agent.best_of_n=8 \
    --agent.num_qs=10 \
    --agent.q_agg=mean \
    --agent.subsample_bon=true \
    --agent.flow_steps=10 \
    --agent.ft_flow_steps=0 \
    --agent.time_embedding=scalar \
    --agent.bc_coeff=0.1 \
    --agent.adv_strategy=vanilla \
    --agent.group_num_samples=$group_num_samples \
    --agent.use_bc_regularization=true \
    --agent.use_denoiser=true \
    --agent.use_constant_noise=true \
    --agent.error_correct_sde_to_ode=true \
    --agent.normalize_denoising_horizon=true \
    --agent.normalize_act_space_dimension=true \
    --agent.ppo_batch_size=256 \
    --agent.batch_size=256 \
    --agent.clip_epsilon=0.01 \
    --agent.entropy_coeff=0.0 \
    --agent.constant_noise_std=0.005 \
    --agent.min_noise_std=0.005 \
    --agent.max_noise_std=0.005 \
    --agent.actor_hidden_dims=[512,512,512,512] \
    --agent.value_hidden_dims=[512,512,512,512] \
    --agent.weighting=next_future \
    --agent.vadar_critic=vector \
    --agent.use_subgoal_discounts=true \
    --agent.subgoal_discounts=[0.9,0.99,0.995,0.999] \
    --agent.use_subgoal_dag=false \
    --agent.bon_weighting=sum
