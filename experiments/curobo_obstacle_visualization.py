#!/usr/bin/env python3
"""Plan a collision-aware Panda motion with cuRobo and render the exact cubby world."""

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

PROJECT = Path("/data/yezhengfu/robot_motion_control")
MODEL_PATH = PROJECT / "experiments/generated/franka/franka_panda_mujoco.urdf"
WORLD_PATH = PROJECT / "third_party/curobo/src/curobo/content/configs/world/collision_cubby.yml"
RESULT_DIR = PROJECT / "experiments/results"
VIDEO_PATH = RESULT_DIR / "curobo_obstacle_motion.mp4"
POSTER_PATH = RESULT_DIR / "curobo_obstacle_motion.png"
CSV_PATH = RESULT_DIR / "curobo_obstacle_visual_trajectory.csv"
METRICS_PATH = RESULT_DIR / "curobo_obstacle_visual_metrics.json"
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
START_OFFSET = [0.25] * 7
GOAL_OFFSET = [0.0, 0.12, 0.0]
TARGET_NAME = "left"


def quaternion_matrix_wxyz(quaternion):
    w, x, y, z = [float(value) for value in quaternion]
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_obstacles():
    world = yaml.safe_load(WORLD_PATH.read_text(encoding="utf-8"))
    obstacles = []
    for name, specification in world["cuboid"].items():
        pose = specification["pose"]
        obstacles.append(
            {
                "name": name,
                "dims": np.asarray(specification["dims"], dtype=np.float64),
                "position": np.asarray(pose[:3], dtype=np.float64),
                "rotation": quaternion_matrix_wxyz(pose[3:7]),
            }
        )
    return obstacles


def plan_motion():
    print("[1/3] Planning through collision_cubby.yml on the GPU...")
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    config = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        "collision_cubby.yml",
        tensor_args,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        interpolation_dt=0.02,
        interpolation_steps=10000,
        use_cuda_graph=True,
        num_ik_seeds=30,
        num_trajopt_seeds=12,
        num_graph_seeds=4,
    )
    planner = MotionGen(config)
    planner.warmup()
    retract = planner.get_retract_config()
    retract_state = JointState.from_position(retract.view(1, -1))
    kinematics = planner.compute_kinematics(retract_state)
    goal_position = kinematics.ee_pos_seq.squeeze() + tensor_args.to_device(GOAL_OFFSET)
    goal_quaternion = kinematics.ee_quat_seq.squeeze()
    goal = Pose(goal_position, quaternion=goal_quaternion)
    start = JointState.from_position(
        retract.view(1, -1).clone(), joint_names=JOINT_NAMES
    )
    start.position[0] += tensor_args.to_device(START_OFFSET)
    result = planner.plan_single(
        start,
        goal,
        MotionGenPlanConfig(
            max_attempts=5,
            timeout=10.0,
            time_dilation_factor=0.5,
            enable_graph=True,
        ),
    )
    if not bool(result.success.item()):
        raise RuntimeError(f"cuRobo planning failed: {result.status}")
    trajectory = result.get_interpolated_plan()
    positions = trajectory.position.detach().cpu().numpy()
    dt = float(result.interpolation_dt)
    latency_ms = float(result.solve_time) * 1000.0
    goal_position_np = goal_position.detach().cpu().numpy()
    print(f"      success=True, latency={latency_ms:.2f} ms, points={len(positions)}")
    return positions, dt, latency_ms, goal_position_np


def resample_trajectory(positions, dt, fps=30, playback_scale=2.5):
    source_times = np.arange(len(positions), dtype=np.float64) * dt
    duration = max(float(source_times[-1]) * playback_scale, 1.0 / fps)
    video_times = np.arange(0.0, duration + 0.5 / fps, 1.0 / fps)
    plan_times = np.clip(video_times / playback_scale, source_times[0], source_times[-1])
    return np.column_stack(
        [np.interp(plan_times, source_times, positions[:, index]) for index in range(7)]
    )


def add_scene_geom(scene, geom_type, size, position, rotation, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        geom_type,
        np.asarray(size, dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.asarray(rotation, dtype=np.float64).reshape(9),
        np.asarray(rgba, dtype=np.float32),
    )
    scene.ngeom += 1


def annotate(frame, progress, latency_ms):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((14, 14, 415, 76), radius=9, fill=(8, 15, 28, 210))
    draw.text((28, 25), "cuRobo Collision-Aware Motion Planning", fill=(255, 255, 255, 255))
    draw.text(
        (28, 50),
        f"Target: {TARGET_NAME} | GPU plan: {latency_ms:.1f} ms | {progress:3.0f}%",
        fill=(102, 224, 255, 255),
    )
    return np.asarray(image)


def render_video(positions, dt, latency_ms, goal_position, obstacles, fps=30):
    print("[2/3] Rendering Panda, cubby obstacles, goal and end-effector trail...")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    joint_addresses = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        joint_addresses.append(int(model.jnt_qposadr[joint_id]))
    finger_addresses = []
    for name in ("panda_finger_joint1", "panda_finger_joint2"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        finger_addresses.append(int(model.jnt_qposadr[joint_id]))
    hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
    if hand_body_id < 0:
        hand_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_link7")

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.45]
    camera.distance = 2.1
    camera.azimuth = 180.0
    camera.elevation = -55.0
    renderer = mujoco.Renderer(model, height=480, width=640)
    video_positions = resample_trajectory(positions, dt, fps=fps)
    trail = []

    def render_frame(q, index):
        data.qpos[joint_addresses] = q
        data.qpos[finger_addresses] = 0.02
        mujoco.mj_forward(model, data)
        trail.append(data.xpos[hand_body_id].copy())
        renderer.update_scene(data, camera=camera)
        for obstacle in obstacles:
            color = [0.30, 0.32, 0.36, 0.28] if obstacle["name"] == "table" else [0.95, 0.31, 0.08, 0.10]
            add_scene_geom(
                renderer.scene,
                mujoco.mjtGeom.mjGEOM_BOX,
                obstacle["dims"] * 0.5,
                obstacle["position"],
                obstacle["rotation"],
                color,
            )
        add_scene_geom(
            renderer.scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            [0.045, 0.0, 0.0],
            goal_position,
            np.eye(3),
            [0.15, 1.0, 0.25, 1.0],
        )
        for point in trail[-30::2]:
            add_scene_geom(
                renderer.scene,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                [0.012, 0.0, 0.0],
                point,
                np.eye(3),
                [0.1, 0.8, 1.0, 0.82],
            )
        progress = 100.0 * index / max(len(video_positions) - 1, 1)
        return annotate(renderer.render().copy(), progress, latency_ms)

    frames = []
    for index, q in enumerate(video_positions):
        frames.append(render_frame(q, index))
    imageio.imwrite(POSTER_PATH, frames[len(frames) // 2])
    with imageio.get_writer(
        VIDEO_PATH,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for _ in range(fps):
            writer.append_data(frames[0])
        for frame in frames:
            writer.append_data(frame)
        for _ in range(fps):
            writer.append_data(frames[-1])
    renderer.close()
    print(f"      video:  {VIDEO_PATH}")
    print(f"      poster: {POSTER_PATH}")


def save_results(positions, dt, latency_ms, obstacle_count):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    times = np.arange(len(positions), dtype=np.float64) * dt
    np.savetxt(
        CSV_PATH,
        np.column_stack([times, positions]),
        delimiter=",",
        header="time_s," + ",".join(JOINT_NAMES),
        comments="",
    )
    metrics = {
        "planner": "cuRobo 0.7.8",
        "robot": "Franka Panda",
        "gpu": torch.cuda.get_device_name(0),
        "collision_world": "collision_cubby.yml",
        "target": TARGET_NAME,
        "success": True,
        "planning_latency_ms": round(latency_ms, 3),
        "trajectory_points": int(len(positions)),
        "trajectory_duration_s": round((len(positions) - 1) * dt, 4),
        "obstacles_rendered": obstacle_count,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run inside a Slurm GPU job.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    positions, dt, latency_ms, goal_position = plan_motion()
    obstacles = load_obstacles()
    metrics = save_results(positions, dt, latency_ms, len(obstacles))
    render_video(positions, dt, latency_ms, goal_position, obstacles)
    print("[3/3] DONE")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
