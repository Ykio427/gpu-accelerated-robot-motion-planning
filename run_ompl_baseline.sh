#!/usr/bin/env bash
set -eo pipefail

PROJECT=/data/yezhengfu/robot_motion_control
cd "$PROJECT"

apptainer exec --fakeroot \
  --bind "${PROJECT}:/workspace" \
  --pwd /workspace/ros2_ws \
  "${PROJECT}/container/rootfs/ubuntu22" \
  bash -lc '
set -eo pipefail
source /opt/ros/humble/setup.bash

printf "[1/2] Building planner_benchmark...\n"
colcon build --packages-select planner_benchmark --symlink-install
source install/setup.bash

printf "[2/2] Running 20 OMPL/RRTConnect trials...\n"
ros2 launch planner_benchmark ompl_benchmark.launch.py

printf "\nResult summary:\n"
cat /workspace/experiments/results/ompl_metrics.json
'
