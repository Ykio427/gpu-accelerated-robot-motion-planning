#!/usr/bin/env bash
set -eo pipefail

cd "$(dirname "$0")"
source .venv-curobo/bin/activate

export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.9
export CUROBO_USE_LRU_CACHE=1
export MUJOCO_GL=egl
export MPLBACKEND=Agg

python experiments/curobo_dynamic_replanning.py

echo
echo "Dynamic replanning artifacts:"
ls -lh   experiments/results/curobo_dynamic_replanning.mp4   experiments/results/curobo_dynamic_replanning.png   experiments/results/curobo_dynamic_replanning_metrics.json   experiments/results/curobo_dynamic_replanning_trajectory.csv
