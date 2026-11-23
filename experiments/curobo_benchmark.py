#!/usr/bin/env python3
"""Repeatable cuRobo benchmark for the same Panda task used by OMPL."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

PROJECT = Path("/data/yezhengfu/robot_motion_control")
RESULT_DIR = PROJECT / "experiments/results"
RUNS_PATH = RESULT_DIR / "curobo_runs.csv"
METRICS_PATH = RESULT_DIR / "curobo_benchmark_metrics.json"
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
START_OFFSET = [0.45, -0.25, 0.35, 0.18, -0.35, -0.22, 0.45]


def summarize(result):
    if not bool(result.success.item()):
        return {"success": False, "latency_ms": float(result.solve_time) * 1000.0}
    trajectory = result.get_interpolated_plan()
    positions = trajectory.position.detach().cpu().numpy()
    dt = float(result.interpolation_dt)
    segment_length = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return {
        "success": True,
        "latency_ms": float(result.solve_time) * 1000.0,
        "path_length_rad": float(segment_length.sum()),
        "duration_s": float(max(positions.shape[0] - 1, 0) * dt),
        "trajectory_points": int(positions.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run inside a Slurm GPU job.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    config = MotionGenConfig.load_from_robot_config(
        "franka.yml", "collision_table.yml", tensor_args,
        interpolation_dt=0.02, interpolation_steps=10000, use_cuda_graph=True,
    )
    planner = MotionGen(config)
    print("Warming up cuRobo on the GPU...")
    planner.warmup()

    retract = planner.get_retract_config()
    goal_kinematics = planner.compute_kinematics(
        JointState.from_position(retract.view(1, -1))
    )
    goal_pose = Pose(
        goal_kinematics.ee_pos_seq.squeeze(),
        quaternion=goal_kinematics.ee_quat_seq.squeeze(),
    )
    start = JointState.from_position(
        retract.view(1, -1).clone(), joint_names=JOINT_NAMES
    )
    start.position[0] += torch.tensor(
        START_OFFSET, device=start.position.device, dtype=start.position.dtype
    )

    results = []
    for index in range(args.runs):
        result = planner.plan_single(
            start, goal_pose,
            MotionGenPlanConfig(max_attempts=5, timeout=10.0, time_dilation_factor=0.5),
        )
        row = summarize(result)
        results.append(row)
        print(
            f"run {index + 1:02d}/{args.runs:02d} "
            f"success={row['success']} latency={row['latency_ms']:.2f} ms"
        )

    fields = [
        "run", "success", "latency_ms", "path_length_rad",
        "duration_s", "trajectory_points",
    ]
    with RUNS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(results, start=1):
            writer.writerow({"run": index, **row})

    successful = [row for row in results if row["success"]]
    latencies = np.array([row["latency_ms"] for row in successful], dtype=float)
    denominator = max(len(results), 1)
    metrics = {
        "planner": "cuRobo 0.7.8",
        "robot": "Franka Panda",
        "gpu": torch.cuda.get_device_name(0),
        "runs": args.runs,
        "successful_runs": len(successful),
        "success_rate_percent": 100.0 * len(successful) / denominator,
        "planning_latency_mean_ms": float(latencies.mean()) if len(latencies) else 0.0,
        "planning_latency_p50_ms": float(np.percentile(latencies, 50)) if len(latencies) else 0.0,
        "planning_latency_p95_ms": float(np.percentile(latencies, 95)) if len(latencies) else 0.0,
        "mean_joint_path_length_rad": float(np.mean([row["path_length_rad"] for row in successful])) if successful else 0.0,
        "mean_trajectory_duration_s": float(np.mean([row["duration_s"] for row in successful])) if successful else 0.0,
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Runs: {RUNS_PATH}")
    print(f"Metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
