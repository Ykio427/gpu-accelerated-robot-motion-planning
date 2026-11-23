#!/usr/bin/env bash
set -eo pipefail

PROJECT=/data/yezhengfu/robot_motion_control
cd "$PROJECT"
source .venv-curobo/bin/activate

export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.9
export CUROBO_USE_LRU_CACHE=1
export MPLBACKEND=Agg

if [[ ! -f experiments/results/ompl_metrics.json ]]; then
  echo "Missing OMPL result. Run: bash run_ompl_baseline.sh"
  exit 1
fi

python experiments/curobo_benchmark.py --runs 20
python experiments/compare_planners.py

echo
echo "Comparison artifacts:"
ls -lh experiments/results/planner_comparison.*
ls -lh experiments/results/PLANNER_COMPARISON.md
