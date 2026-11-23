#!/usr/bin/env bash
set -eo pipefail

PROJECT=/data/yezhengfu/robot_motion_control
cd "$PROJECT"
source .venv-curobo/bin/activate

export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.9
export CUROBO_USE_LRU_CACHE=1
export MPLBACKEND=Agg

python experiments/curobo_obstacle_benchmark.py --trials 10

echo
echo "Obstacle benchmark artifacts:"
ls -lh experiments/results/curobo_obstacle_runs.csv
ls -lh experiments/results/curobo_obstacle_metrics.json
ls -lh experiments/results/curobo_obstacle_benchmark.png
ls -lh experiments/results/CUROBO_OBSTACLE_BENCHMARK.md
