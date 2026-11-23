#!/usr/bin/env python3
"""Compare repeatable cuRobo and MoveIt2 OMPL benchmark results."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path("/data/yezhengfu/robot_motion_control")
RESULT_DIR = PROJECT / "experiments/results"
CUROBO_PATH = RESULT_DIR / "curobo_benchmark_metrics.json"
OMPL_PATH = RESULT_DIR / "ompl_metrics.json"
JSON_PATH = RESULT_DIR / "planner_comparison.json"
CSV_PATH = RESULT_DIR / "planner_comparison.csv"
PLOT_PATH = RESULT_DIR / "planner_comparison.png"
REPORT_PATH = RESULT_DIR / "PLANNER_COMPARISON.md"


def load(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    curobo = load(CUROBO_PATH)
    ompl = load(OMPL_PATH)
    planners = [curobo["planner"], ompl["planner"]]
    metrics = {
        "success_rate_percent": [
            curobo["success_rate_percent"], ompl["success_rate_percent"]],
        "planning_latency_p50_ms": [
            curobo["planning_latency_p50_ms"], ompl["planning_latency_p50_ms"]],
        "planning_latency_p95_ms": [
            curobo["planning_latency_p95_ms"], ompl["planning_latency_p95_ms"]],
        "mean_joint_path_length_rad": [
            curobo["mean_joint_path_length_rad"], ompl["mean_joint_path_length_rad"]],
        "mean_trajectory_duration_s": [
            curobo["mean_trajectory_duration_s"], ompl["mean_trajectory_duration_s"]],
    }
    comparison = {
        "task": "Franka Panda identical start configuration and goal pose",
        "planners": planners,
        "metrics": metrics,
    }
    JSON_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", *planners])
        for name, values in metrics.items():
            writer.writerow([name, *values])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    colors = ["#36A2EB", "#FF9F40"]
    panels = [
        ("planning_latency_p50_ms", "Planning latency P50", "ms"),
        ("mean_joint_path_length_rad", "Mean joint path length", "rad"),
        ("mean_trajectory_duration_s", "Mean trajectory duration", "s"),
    ]
    for axis, (key, title, unit) in zip(axes, panels):
        values = metrics[key]
        bars = axis.bar(["cuRobo GPU", "OMPL CPU"], values, color=colors)
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.25)
        axis.bar_label(bars, fmt="%.3f")
    fig.suptitle("Franka Panda Motion Planning Benchmark")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)

    speedup = ompl["planning_latency_p50_ms"] / max(
        curobo["planning_latency_p50_ms"], 1e-9
    )
    report = f"""# Planner Comparison

## Experiment

- Robot: Franka Panda
- Task: identical start configuration and goal pose
- Trials: cuRobo {curobo["runs"]}, OMPL {ompl["runs"]}
- GPU: {curobo.get("gpu", "unknown")}
- Planners: cuRobo 0.7.8 and MoveIt2 OMPL RRTConnect

## Results

| Metric | cuRobo GPU | OMPL CPU |
|---|---:|---:|
| Success rate | {curobo["success_rate_percent"]:.1f}% | {ompl["success_rate_percent"]:.1f}% |
| Planning latency P50 | {curobo["planning_latency_p50_ms"]:.3f} ms | {ompl["planning_latency_p50_ms"]:.3f} ms |
| Planning latency P95 | {curobo["planning_latency_p95_ms"]:.3f} ms | {ompl["planning_latency_p95_ms"]:.3f} ms |
| Mean joint path length | {curobo["mean_joint_path_length_rad"]:.4f} rad | {ompl["mean_joint_path_length_rad"]:.4f} rad |
| Mean trajectory duration | {curobo["mean_trajectory_duration_s"]:.4f} s | {ompl["mean_trajectory_duration_s"]:.4f} s |

## Conclusion

cuRobo achieved a P50 planning-latency speedup of **{speedup:.2f}x** over OMPL on this task.

![Planner comparison](planner_comparison.png)
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Plot: {PLOT_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
