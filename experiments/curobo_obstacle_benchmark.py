#!/usr/bin/env python3
"""Collision-aware multi-target benchmark for cuRobo and Franka Panda."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

PROJECT = Path("/data/yezhengfu/robot_motion_control")
RESULT_DIR = PROJECT / "experiments/results"
RUNS_PATH = RESULT_DIR / "curobo_obstacle_runs.csv"
METRICS_PATH = RESULT_DIR / "curobo_obstacle_metrics.json"
PLOT_PATH = RESULT_DIR / "curobo_obstacle_benchmark.png"
REPORT_PATH = RESULT_DIR / "CUROBO_OBSTACLE_BENCHMARK.md"
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
START_OFFSET = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
SCENARIOS = {
    "center": [0.00, 0.00, 0.00],
    "forward": [-0.15, 0.00, 0.00],
    "left": [0.00, 0.12, 0.00],
    "right": [0.00, -0.12, 0.00],
    "high": [0.00, 0.00, 0.10],
}


def percentile(values, q):
    return float(np.percentile(values, q)) if values else 0.0


def summarize_rows(rows):
    successful = [row for row in rows if row["success"]]
    latencies = [row["latency_ms"] for row in successful]
    paths = [row["path_length_rad"] for row in successful]
    durations = [row["duration_s"] for row in successful]
    return {
        "runs": len(rows),
        "successful_runs": len(successful),
        "success_rate_percent": 100.0 * len(successful) / max(len(rows), 1),
        "planning_latency_mean_ms": float(np.mean(latencies)) if latencies else 0.0,
        "planning_latency_p50_ms": percentile(latencies, 50),
        "planning_latency_p95_ms": percentile(latencies, 95),
        "mean_joint_path_length_rad": float(np.mean(paths)) if paths else 0.0,
        "mean_trajectory_duration_s": float(np.mean(durations)) if durations else 0.0,
    }


def make_goal(base_position, base_quaternion, offset, tensor_args):
    position = base_position.clone() + tensor_args.to_device(offset)
    return Pose(position, quaternion=base_quaternion.clone())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run inside a Slurm GPU job.")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    config = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        "collision_cubby.yml",
        tensor_args,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=True,
        interpolation_dt=0.02,
        interpolation_steps=10000,
        num_ik_seeds=30,
        num_trajopt_seeds=12,
        num_graph_seeds=4,
    )
    planner = MotionGen(config)
    print("Warming up collision-aware cuRobo planner...")
    planner.warmup()

    retract = planner.get_retract_config()
    retract_state = JointState.from_position(retract.view(1, -1))
    kinematics = planner.compute_kinematics(retract_state)
    base_position = kinematics.ee_pos_seq.squeeze()
    base_quaternion = kinematics.ee_quat_seq.squeeze()
    start = JointState.from_position(
        retract.view(1, -1).clone(), joint_names=JOINT_NAMES
    )
    start.position[0] += tensor_args.to_device(START_OFFSET)
    plan_config = MotionGenPlanConfig(
        max_attempts=5,
        timeout=10.0,
        time_dilation_factor=0.5,
        enable_graph=True,
    )

    rows = []
    total = len(SCENARIOS) * args.trials
    run_index = 0
    for scenario, offset in SCENARIOS.items():
        goal = make_goal(base_position, base_quaternion, offset, tensor_args)
        for trial in range(1, args.trials + 1):
            run_index += 1
            result = planner.plan_single(start, goal, plan_config)
            success = bool(result.success.item())
            row = {
                "run": run_index,
                "scenario": scenario,
                "trial": trial,
                "success": success,
                "status": str(result.status),
                "latency_ms": float(result.solve_time) * 1000.0,
                "path_length_rad": 0.0,
                "duration_s": 0.0,
                "trajectory_points": 0,
            }
            if success:
                trajectory = result.get_interpolated_plan()
                positions = trajectory.position.detach().cpu().numpy()
                row["path_length_rad"] = float(
                    np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()
                )
                row["duration_s"] = float(
                    max(positions.shape[0] - 1, 0) * float(result.interpolation_dt)
                )
                row["trajectory_points"] = int(positions.shape[0])
            rows.append(row)
            print(
                f"run {run_index:02d}/{total:02d} scenario={scenario:<7} "
                f"success={success} latency={row['latency_ms']:.2f} ms"
            )

    fields = [
        "run", "scenario", "trial", "success", "status", "latency_ms",
        "path_length_rad", "duration_s", "trajectory_points",
    ]
    with RUNS_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    per_scenario = {
        name: summarize_rows([row for row in rows if row["scenario"] == name])
        for name in SCENARIOS
    }
    metrics = {
        "planner": "cuRobo 0.7.8",
        "robot": "Franka Panda",
        "world": "collision_cubby.yml",
        "gpu": torch.cuda.get_device_name(0),
        "goal_offsets_m": SCENARIOS,
        "overall": summarize_rows(rows),
        "scenarios": per_scenario,
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    names = list(SCENARIOS)
    success_rates = [per_scenario[name]["success_rate_percent"] for name in names]
    p50 = [per_scenario[name]["planning_latency_p50_ms"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bars = axes[0].bar(names, success_rates, color="#2CA02C")
    axes[0].set_title("Collision-aware planning success rate")
    axes[0].set_ylabel("percent")
    axes[0].set_ylim(0, 110)
    axes[0].bar_label(bars, fmt="%.1f")
    bars = axes[1].bar(names, p50, color="#36A2EB")
    axes[1].set_title("Planning latency P50 by target")
    axes[1].set_ylabel("ms")
    axes[1].bar_label(bars, fmt="%.2f")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.tick_params(axis="x", rotation=20)
    fig.suptitle("Franka Panda cuRobo Obstacle Benchmark")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)

    table_rows = []
    for name in names:
        item = per_scenario[name]
        table_rows.append(
            f"| {name} | {item['successful_runs']}/{item['runs']} | "
            f"{item['success_rate_percent']:.1f}% | "
            f"{item['planning_latency_p50_ms']:.3f} ms | "
            f"{item['planning_latency_p95_ms']:.3f} ms | "
            f"{item['mean_joint_path_length_rad']:.4f} rad |"
        )
    table = chr(10).join(table_rows)
    overall = metrics["overall"]
    report = f"""# cuRobo Collision-Aware Multi-Target Benchmark

## Experiment

- Robot: Franka Panda
- Planner: cuRobo 0.7.8
- GPU: {metrics["gpu"]}
- Collision world: collision_cubby.yml
- Targets: {len(SCENARIOS)}
- Trials per target: {args.trials}
- Total planning requests: {len(rows)}

## Aggregate Results

- Success rate: {overall["success_rate_percent"]:.1f}%
- Planning latency P50: {overall["planning_latency_p50_ms"]:.3f} ms
- Planning latency P95: {overall["planning_latency_p95_ms"]:.3f} ms
- Mean joint path length: {overall["mean_joint_path_length_rad"]:.4f} rad
- Mean trajectory duration: {overall["mean_trajectory_duration_s"]:.4f} s

## Per-Target Results

| Target | Success | Rate | P50 latency | P95 latency | Mean path length |
|---|---:|---:|---:|---:|---:|
{table}

![Obstacle benchmark](curobo_obstacle_benchmark.png)
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Runs: {RUNS_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Plot: {PLOT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
