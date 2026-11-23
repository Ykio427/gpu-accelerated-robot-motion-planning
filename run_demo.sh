#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data/yezhengfu/robot_motion_control
cd "$PROJECT"

if [[ -z "${SLURM_JOB_ID:-}" ]] || [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "ERROR: no Slurm GPU allocation detected."
  echo "Run: srun --gpus=1 --pty bash"
  echo "Then run this script again."
  exit 2
fi

source .venv-curobo/bin/activate
export CUDA_HOME=/usr/local/cuda
export TORCH_CUDA_ARCH_LIST=8.9
export CUROBO_USE_LRU_CACHE=1
export MUJOCO_GL=egl
export MPLBACKEND=Agg

python - <<'PY'
import torch
import warp as wp
import mujoco
import curobo

wp.init()
assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
assert "cuda:0" in [str(device) for device in wp.get_devices()], "Warp cannot access CUDA"
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))
print("Warp:", wp.__version__)
print("MuJoCo:", mujoco.__version__)
print("cuRobo:", curobo.__file__)
PY

python experiments/panda_curobo_mujoco.py "$@"

echo
echo "Artifacts:"
ls -lh experiments/results/panda_curobo_*
