#!/usr/bin/env python3
"""GPU motion planning for Franka Panda with cuRobo and MuJoCo rendering."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

PROJECT = Path("/data/yezhengfu/robot_motion_control")
MODEL_PATH = PROJECT / "experiments/generated/franka/franka_panda_mujoco.urdf"
RESULT_DIR = PROJECT / "experiments/results"
CSV_PATH = RESULT_DIR / "panda_curobo_trajectory.csv"
VIDEO_PATH = RESULT_DIR / "panda_curobo_demo.mp4"
POSTER_PATH = RESULT_DIR / "panda_curobo_poster.png"
PLOT_PATH = RESULT_DIR / "panda_curobo_joint_plot.png"
METRICS_PATH = RESULT_DIR / "panda_curobo_metrics.json"
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
START_OFFSET = [0.45, -0.25, 0.35, 0.18, -0.35, -0.22, 0.45]


def plan_with_curobo():
    print("[1/4] Loading Panda and warming up cuRobo on the GPU...")
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    config = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        "collision_table.yml",
        tensor_args,
        interpolation_dt=0.02,
        interpolation_steps=10000,
        use_cuda_graph=True,
    )
    planner = MotionGen(config)
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
    offset = torch.tensor(
        START_OFFSET, device=start.position.device, dtype=start.position.dtype
    )
    start.position[0] += offset

    result = planner.plan_single(
        start,
        goal_pose,
        MotionGenPlanConfig(
            max_attempts=5,
            timeout=10.0,
            time_dilation_factor=0.5,
        ),
    )
    if not bool(result.success.item()):
        raise RuntimeError(f"cuRobo planning failed: {result.status}")

    trajectory = result.get_interpolated_plan()
    positions = trajectory.position.detach().cpu().numpy()
    dt = float(result.interpolation_dt)
    solve_time = float(result.solve_time)
    print(f"      Planning succeeded: {solve_time * 1000:.2f} ms")
    print(f"      Trajectory: {positions.shape[0]} points, dt={dt:.4f} s")
    return positions, dt, solve_time


def compute_metrics(positions, dt, solve_time):
    velocity = np.diff(positions, axis=0) / dt
    acceleration = np.diff(velocity, axis=0) / dt if len(velocity) > 1 else np.zeros_like(velocity)
    segment_length = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    joint_range = np.ptp(positions, axis=0)
    return {
        "planner": "cuRobo 0.7.8",
        "robot": "Franka Panda",
        "gpu": torch.cuda.get_device_name(0),
        "success": True,
        "planning_time_ms": round(solve_time * 1000.0, 3),
        "trajectory_points": int(positions.shape[0]),
        "trajectory_duration_s": round((positions.shape[0] - 1) * dt, 4),
        "joint_path_length_rad": round(float(segment_length.sum()), 4),
        "max_joint_speed_rad_s": round(float(np.abs(velocity).max()), 4),
        "max_joint_acceleration_rad_s2": round(float(np.abs(acceleration).max()), 4),
        "joint_motion_range_rad": {
            name: round(float(value), 4) for name, value in zip(JOINT_NAMES, joint_range)
        },
        "interpolation_dt_s": dt,
    }


def save_results(positions, dt, metrics):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    times = np.arange(positions.shape[0], dtype=np.float64) * dt
    table = np.column_stack([times, positions])
    np.savetxt(
        CSV_PATH,
        table,
        delimiter=",",
        header="time_s," + ",".join(JOINT_NAMES),
        comments="",
    )
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(10, 5.6))
    for index, name in enumerate(JOINT_NAMES):
        ax.plot(times, positions[:, index], linewidth=2, label=name)
    ax.set_title("cuRobo GPU-Planned Panda Joint Trajectory")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint angle (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)

    print(f"[2/4] CSV:     {CSV_PATH}")
    print(f"      Metrics: {METRICS_PATH}")
    print(f"      Plot:    {PLOT_PATH}")


def resample_for_video(positions, dt, fps, playback_scale):
    source_times = np.arange(positions.shape[0], dtype=np.float64) * dt
    duration = max(float(source_times[-1]) * playback_scale, 1.0 / fps)
    video_times = np.arange(0.0, duration + 0.5 / fps, 1.0 / fps)
    plan_times = np.clip(video_times / playback_scale, source_times[0], source_times[-1])
    return np.column_stack(
        [np.interp(plan_times, source_times, positions[:, j]) for j in range(7)]
    )


def render_with_mujoco(positions, dt, fps=30, playback_scale=2.0):
    print("[3/4] Rendering Panda motion with MuJoCo EGL...")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    joint_addresses = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"MuJoCo joint not found: {name}")
        joint_addresses.append(int(model.jnt_qposadr[joint_id]))

    finger_addresses = []
    for name in ("panda_finger_joint1", "panda_finger_joint2"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"MuJoCo joint not found: {name}")
        finger_addresses.append(int(model.jnt_qposadr[joint_id]))

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.45]
    camera.distance = 1.8
    camera.azimuth = 135.0
    camera.elevation = -18.0

    renderer = mujoco.Renderer(model, height=480, width=640)
    video_positions = resample_for_video(positions, dt, fps, playback_scale)

    def render_frame(q):
        data.qpos[joint_addresses] = q
        data.qpos[finger_addresses] = 0.02
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        return renderer.render().copy()

    first_frame = render_frame(video_positions[0])
    imageio.imwrite(POSTER_PATH, first_frame)
    with imageio.get_writer(
        VIDEO_PATH,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for _ in range(fps):
            writer.append_data(first_frame)
        for q in video_positions:
            writer.append_data(render_frame(q))
        final_frame = render_frame(video_positions[-1])
        for _ in range(fps):
            writer.append_data(final_frame)
    renderer.close()
    print(f"      Video:  {VIDEO_PATH}")
    print(f"      Poster: {POSTER_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-video", action="store_true", help="skip MuJoCo video rendering")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--playback-scale", type=float, default=2.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this inside a Slurm GPU job.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    positions, dt, solve_time = plan_with_curobo()
    metrics = compute_metrics(positions, dt, solve_time)
    save_results(positions, dt, metrics)
    if not args.no_video:
        render_with_mujoco(positions, dt, fps=args.fps, playback_scale=args.playback_scale)

    print("[4/4] DONE")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
