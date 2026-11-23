#!/usr/bin/env python3
"""Run deterministic multi-scene cuRobo planning and obstacle replanning benchmarks."""

import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from curobo.geom.types import WorldConfig
from curobo.types.math import Pose
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

from curobo_dynamic_replanning import (
    DYNAMIC_WORLD_PATH,
    JOINT_NAMES,
    RESULT_DIR,
    START_OFFSET,
    end_effector_positions,
    joint_state,
    make_planner,
)

N_SCENES = 30
RANDOM_SEED = 20261123
TRIGGER_FRACTION = 0.05
TARGET_RANGES = np.array(
    [
        [0.14, 0.28],
        [-0.14, 0.14],
        [0.04, 0.16],
    ],
    dtype=np.float64,
)
CANDIDATE_FRACTIONS = (0.45, 0.55, 0.65, 0.75, 0.82)
CANDIDATE_SIZES = (0.030, 0.040, 0.050)

RUNS_PATH = RESULT_DIR / "multiscene_runs.csv"
SUMMARY_PATH = RESULT_DIR / "multiscene_summary.json"
PLOT_PATH = RESULT_DIR / "multiscene_benchmark.png"
REPORT_PATH = RESULT_DIR / "MULTISCENE_BENCHMARK.md"


def valid_flag(value):
    return bool(value.item()) if hasattr(value, "item") else bool(value)


def load_base_world():
    return yaml.safe_load(DYNAMIC_WORLD_PATH.read_text(encoding="utf-8"))


def update_world(planner, world_dict):
    planner.update_world(WorldConfig.from_dict(world_dict))


def solve(planner, start, goal, attempts):
    result = planner.plan_single(
        start,
        goal,
        MotionGenPlanConfig(
            max_attempts=attempts,
            timeout=15.0,
            time_dilation_factor=0.5,
            enable_graph=True,
        ),
    )
    if not bool(result.success.item()):
        return None, str(result.status)

    trajectory = result.get_interpolated_plan()
    positions = trajectory.position.detach().cpu().numpy()
    return (
        {
            "positions": positions,
            "dt": float(result.interpolation_dt),
            "latency_ms": float(result.solve_time) * 1000.0,
        },
        "SUCCESS",
    )


def path_metrics(positions, dt):
    if len(positions) < 2:
        return {
            "path_length_rad": 0.0,
            "duration_s": 0.0,
            "max_speed_rad_s": 0.0,
            "max_acceleration_rad_s2": 0.0,
        }

    delta = np.diff(positions, axis=0)
    velocity = delta / dt
    acceleration = (
        np.diff(velocity, axis=0) / dt
        if len(velocity) > 1
        else np.zeros_like(velocity)
    )
    return {
        "path_length_rad": float(np.linalg.norm(delta, axis=1).sum()),
        "duration_s": float((len(positions) - 1) * dt),
        "max_speed_rad_s": float(np.abs(velocity).max()),
        "max_acceleration_rad_s2": float(np.abs(acceleration).max()),
    }


def build_goal(planner, tensor_args, retract, offset):
    retract_state = joint_state(tensor_args, retract.detach().cpu().numpy())
    kinematics = planner.compute_kinematics(retract_state)
    position = (
        kinematics.ee_pos_seq.squeeze()
        + tensor_args.to_device(offset.tolist())
    )
    quaternion = kinematics.ee_quat_seq.squeeze()
    return Pose(position, quaternion=quaternion)


def terminal_goal(planner, tensor_args, terminal_positions):
    state = joint_state(tensor_args, terminal_positions)
    kinematics = planner.compute_kinematics(state)
    return Pose(
        kinematics.ee_pos_seq.squeeze(),
        quaternion=kinematics.ee_quat_seq.squeeze(),
    )


def select_blocker(
    planner,
    tensor_args,
    base_world,
    initial_positions,
    initial_ee,
    trigger_index,
):
    replan_start = joint_state(
        tensor_args,
        initial_positions[trigger_index],
    )
    diagnostics = []

    for size in CANDIDATE_SIZES:
        for fraction in CANDIDATE_FRACTIONS:
            candidate_index = min(
                len(initial_positions) - 2,
                max(
                    trigger_index + 2,
                    int((len(initial_positions) - 1) * fraction),
                ),
            )
            position = initial_ee[candidate_index].copy()
            dims = np.array(
                [size, size, max(0.055, size * 1.5)],
                dtype=np.float64,
            )

            world_dict = load_base_world()
            world_dict.setdefault("cuboid", {})["benchmark_blocker"] = {
                "pose": [
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "dims": [float(value) for value in dims],
            }
            update_world(planner, world_dict)

            start_valid, start_status = planner.check_start_state(replan_start)
            blocked_state = joint_state(
                tensor_args,
                initial_positions[candidate_index],
            )
            future_valid, future_status = planner.check_start_state(
                blocked_state
            )
            diagnostics.append(
                {
                    "size": size,
                    "fraction": fraction,
                    "start": str(start_status),
                    "future": str(future_status),
                }
            )

            if valid_flag(start_valid) and not valid_flag(future_valid):
                return {
                    "position": position,
                    "dims": dims,
                    "path_index": candidate_index,
                    "path_fraction": (
                        candidate_index / (len(initial_positions) - 1)
                    ),
                    "world": world_dict,
                    "replan_start": replan_start,
                }

    update_world(planner, base_world)
    return None


def empty_row(scene_id, offset, status):
    return {
        "scene_id": scene_id,
        "target_x_m": round(float(offset[0]), 5),
        "target_y_m": round(float(offset[1]), 5),
        "target_z_m": round(float(offset[2]), 5),
        "success": False,
        "status": status,
        "base_latency_ms": math.nan,
        "replan_latency_ms": math.nan,
        "replan_path_length_rad": math.nan,
        "replan_duration_s": math.nan,
        "max_speed_rad_s": math.nan,
        "max_acceleration_rad_s2": math.nan,
        "min_ee_center_distance_m": math.nan,
        "blocker_path_fraction": math.nan,
        "blocker_x_m": math.nan,
        "blocker_y_m": math.nan,
        "blocker_z_m": math.nan,
        "blocker_dx_m": math.nan,
        "blocker_dy_m": math.nan,
        "blocker_dz_m": math.nan,
        "trajectory_points": 0,
        "error_message": "",
    }


def run_scene(planner, tensor_args, retract, offset, scene_id):
    base_world = load_base_world()
    update_world(planner, base_world)

    start = joint_state(
        tensor_args,
        (retract + tensor_args.to_device(START_OFFSET)).detach()
        .cpu()
        .numpy(),
    )
    goal = build_goal(planner, tensor_args, retract, offset)
    base_plan, base_status = solve(planner, start, goal, attempts=18)
    if base_plan is None:
        return empty_row(
            scene_id,
            offset,
            "BASE_" + base_status,
        )

    initial = base_plan["positions"]
    initial_ee = end_effector_positions(planner, tensor_args, initial)
    trigger_index = max(
        1,
        int((len(initial) - 1) * TRIGGER_FRACTION),
    )

    blocker = select_blocker(
        planner,
        tensor_args,
        base_world,
        initial,
        initial_ee,
        trigger_index,
    )
    if blocker is None:
        return empty_row(
            scene_id,
            offset,
            "NO_VALID_BLOCKER",
        )

    update_world(planner, blocker["world"])
    goal_exact = terminal_goal(
        planner,
        tensor_args,
        initial[-1],
    )
    replanned, replan_status = solve(
        planner,
        blocker["replan_start"],
        goal_exact,
        attempts=32,
    )
    if replanned is None:
        row = empty_row(
            scene_id,
            offset,
            "REPLAN_" + replan_status,
        )
        row["base_latency_ms"] = round(base_plan["latency_ms"], 4)
        row["blocker_path_fraction"] = round(
            blocker["path_fraction"],
            5,
        )
        return row

    positions = replanned["positions"]
    ee_positions = end_effector_positions(
        planner,
        tensor_args,
        positions,
    )
    min_distance = float(
        np.linalg.norm(
            ee_positions - blocker["position"],
            axis=1,
        ).min()
    )
    metrics = path_metrics(positions, replanned["dt"])

    return {
        "scene_id": scene_id,
        "target_x_m": round(float(offset[0]), 5),
        "target_y_m": round(float(offset[1]), 5),
        "target_z_m": round(float(offset[2]), 5),
        "success": True,
        "status": "SUCCESS",
        "base_latency_ms": round(base_plan["latency_ms"], 4),
        "replan_latency_ms": round(replanned["latency_ms"], 4),
        "replan_path_length_rad": round(
            metrics["path_length_rad"],
            6,
        ),
        "replan_duration_s": round(metrics["duration_s"], 6),
        "max_speed_rad_s": round(
            metrics["max_speed_rad_s"],
            6,
        ),
        "max_acceleration_rad_s2": round(
            metrics["max_acceleration_rad_s2"],
            6,
        ),
        "min_ee_center_distance_m": round(min_distance, 6),
        "blocker_path_fraction": round(
            blocker["path_fraction"],
            5,
        ),
        "blocker_x_m": round(float(blocker["position"][0]), 6),
        "blocker_y_m": round(float(blocker["position"][1]), 6),
        "blocker_z_m": round(float(blocker["position"][2]), 6),
        "blocker_dx_m": round(float(blocker["dims"][0]), 6),
        "blocker_dy_m": round(float(blocker["dims"][1]), 6),
        "blocker_dz_m": round(float(blocker["dims"][2]), 6),
        "trajectory_points": int(len(positions)),
        "error_message": "",
    }


def percentile(values, quantile):
    return float(np.percentile(np.asarray(values), quantile))


def build_summary(rows):
    successful = [row for row in rows if row["success"]]
    latencies = [
        float(row["replan_latency_ms"])
        for row in successful
    ]
    paths = [
        float(row["replan_path_length_rad"])
        for row in successful
    ]
    clearances = [
        float(row["min_ee_center_distance_m"])
        for row in successful
    ]
    durations = [
        float(row["replan_duration_s"])
        for row in successful
    ]

    summary = {
        "planner": "cuRobo 0.7.8",
        "robot": "Franka Panda",
        "gpu": torch.cuda.get_device_name(0),
        "random_seed": RANDOM_SEED,
        "requested_scenes": N_SCENES,
        "successful_scenes": len(successful),
        "success_rate_percent": round(
            100.0 * len(successful) / len(rows),
            3,
        ),
        "failed_status_counts": {},
    }

    for row in rows:
        if row["success"]:
            continue
        status = str(row["status"])
        summary["failed_status_counts"][status] = (
            summary["failed_status_counts"].get(status, 0) + 1
        )

    if successful:
        summary.update(
            {
                "replanning_latency_mean_ms": round(
                    float(np.mean(latencies)),
                    4,
                ),
                "replanning_latency_p50_ms": round(
                    percentile(latencies, 50),
                    4,
                ),
                "replanning_latency_p95_ms": round(
                    percentile(latencies, 95),
                    4,
                ),
                "mean_path_length_rad": round(
                    float(np.mean(paths)),
                    6,
                ),
                "mean_trajectory_duration_s": round(
                    float(np.mean(durations)),
                    6,
                ),
                "mean_min_ee_center_distance_m": round(
                    float(np.mean(clearances)),
                    6,
                ),
                "minimum_clearance_across_scenes_m": round(
                    float(np.min(clearances)),
                    6,
                ),
            }
        )
    return summary


def write_csv(rows):
    with RUNS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, summary):
    scene_ids = [int(row["scene_id"]) for row in rows]
    successful = [row for row in rows if row["success"]]
    success_ids = [int(row["scene_id"]) for row in successful]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "cuRobo Multi-Scene Dynamic Obstacle Benchmark",
        fontsize=16,
        fontweight="bold",
    )

    axes[0, 0].bar(
        ["Success", "Failure"],
        [
            summary["successful_scenes"],
            summary["requested_scenes"] - summary["successful_scenes"],
        ],
        color=["#31c48d", "#ef4444"],
    )
    axes[0, 0].set_title(
        f"Success Rate: {summary['success_rate_percent']:.1f}%"
    )
    axes[0, 0].set_ylabel("Scenes")

    if successful:
        axes[0, 1].plot(
            success_ids,
            [row["replan_latency_ms"] for row in successful],
            marker="o",
            linewidth=1.5,
            color="#3b82f6",
        )
        axes[0, 1].axhline(
            summary["replanning_latency_p50_ms"],
            color="#f59e0b",
            linestyle="--",
            label="P50",
        )
        axes[0, 1].axhline(
            summary["replanning_latency_p95_ms"],
            color="#ef4444",
            linestyle=":",
            label="P95",
        )
        axes[0, 1].legend()
        axes[0, 1].set_title("Online Replanning Latency")
        axes[0, 1].set_ylabel("Latency (ms)")

        axes[1, 0].plot(
            success_ids,
            [row["replan_path_length_rad"] for row in successful],
            marker="o",
            linewidth=1.5,
            color="#8b5cf6",
        )
        axes[1, 0].set_title("Joint-Space Path Length")
        axes[1, 0].set_ylabel("Path length (rad)")

        axes[1, 1].bar(
            success_ids,
            [
                100.0 * row["min_ee_center_distance_m"]
                for row in successful
            ],
            color="#06b6d4",
        )
        axes[1, 1].set_title("Minimum EE-Center Clearance")
        axes[1, 1].set_ylabel("Clearance (cm)")
    else:
        for axis in (axes[0, 1], axes[1, 0], axes[1, 1]):
            axis.text(
                0.5,
                0.5,
                "No successful scenes",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )

    for axis in axes.flat:
        axis.set_xlabel("Scene ID")
        axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(summary):
    failures = summary.get("failed_status_counts", {})
    failure_text = (
        ", ".join(f"{key}: {value}" for key, value in failures.items())
        if failures
        else "None"
    )
    report = f"""# cuRobo Multi-Scene Benchmark

## Configuration

- Robot: Franka Panda
- Planner: cuRobo 0.7.8
- GPU: {summary["gpu"]}
- Deterministic random seed: {summary["random_seed"]}
- Requested scenes: {summary["requested_scenes"]}
- Each scene: a reachable target candidate plus an automatically validated dynamic blocker

## Aggregate Results

| Metric | Result |
|---|---:|
| Successful scenes | {summary["successful_scenes"]} / {summary["requested_scenes"]} |
| Success rate | {summary["success_rate_percent"]:.2f}% |
| Replanning latency mean | {summary.get("replanning_latency_mean_ms", float("nan")):.3f} ms |
| Replanning latency P50 | {summary.get("replanning_latency_p50_ms", float("nan")):.3f} ms |
| Replanning latency P95 | {summary.get("replanning_latency_p95_ms", float("nan")):.3f} ms |
| Mean joint path length | {summary.get("mean_path_length_rad", float("nan")):.4f} rad |
| Mean trajectory duration | {summary.get("mean_trajectory_duration_s", float("nan")):.4f} s |
| Mean minimum EE-center clearance | {summary.get("mean_min_ee_center_distance_m", float("nan")):.4f} m |
| Worst successful-scene clearance | {summary.get("minimum_clearance_across_scenes_m", float("nan")):.4f} m |

Failed status counts: {failure_text}

## Interpretation

The benchmark changes the Cartesian target and constructs a collision obstacle on the original future path for every scene. A blocker is accepted only when the current execution state remains valid and a later original-trajectory state becomes invalid. Therefore, successful replanning results represent actual online path replacement rather than replay of an unchanged trajectory.

![Multi-scene benchmark](multiscene_benchmark.png)
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this benchmark inside a Slurm GPU job."
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)
    offsets = rng.uniform(
        TARGET_RANGES[:, 0],
        TARGET_RANGES[:, 1],
        size=(N_SCENES, 3),
    )

    print("[1/4] Building and warming up the cuRobo GPU planner...")
    planner, tensor_args = make_planner()
    retract = planner.get_retract_config()

    rows = []
    print(f"[2/4] Running {N_SCENES} deterministic scenes...")
    for index, offset in enumerate(offsets, start=1):
        try:
            row = run_scene(
                planner,
                tensor_args,
                retract,
                offset,
                index,
            )
        except Exception as error:
            row = empty_row(
                index,
                offset,
                "EXCEPTION_" + type(error).__name__,
            )
            row["error_message"] = str(error)
        rows.append(row)
        latency = (
            f"{row['replan_latency_ms']:.2f} ms"
            if row["success"]
            else "-"
        )
        print(
            f"  scene {index:02d}/{N_SCENES} "
            f"success={row['success']} "
            f"status={row['status']} "
            f"replan={latency}"
        )

    print("[3/4] Writing CSV, JSON, plot and Markdown report...")
    write_csv(rows)
    summary = build_summary(rows)
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    plot_results(rows, summary)
    write_report(summary)

    print("[4/4] DONE")
    print(json.dumps(summary, indent=2))
    print(f"Runs:    {RUNS_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Plot:    {PLOT_PATH}")
    print(f"Report:  {REPORT_PATH}")


if __name__ == "__main__":
    main()
