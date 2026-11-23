#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

source .venv-curobo/bin/activate

export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.9
export CUROBO_USE_LRU_CACHE=1
export MPLBACKEND=Agg
export MUJOCO_GL=egl

echo "=== Multi-scene benchmark preflight ==="
python -m py_compile experiments/curobo_multiscene_benchmark.py

echo "=== Running 30 deterministic GPU planning scenes ==="
python experiments/curobo_multiscene_benchmark.py

echo
echo "Multi-scene benchmark artifacts:"
ls -lh   experiments/results/multiscene_runs.csv   experiments/results/multiscene_summary.json   experiments/results/multiscene_benchmark.png   experiments/results/MULTISCENE_BENCHMARK.md
