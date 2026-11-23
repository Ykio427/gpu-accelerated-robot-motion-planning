#!/usr/bin/env bash
set -eo pipefail

PROJECT=/data/yezhengfu/robot_motion_control
cd "$PROJECT"
source .venv-curobo/bin/activate

export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.9
export CUROBO_USE_LRU_CACHE=1
export MPLBACKEND=Agg
export MUJOCO_GL=egl

python experiments/curobo_obstacle_visualization.py

echo
echo "Obstacle visualization artifacts:"
ls -lh experiments/results/curobo_obstacle_motion.mp4
ls -lh experiments/results/curobo_obstacle_motion.png
ls -lh experiments/results/curobo_obstacle_visual_trajectory.csv
ls -lh experiments/results/curobo_obstacle_visual_metrics.json
