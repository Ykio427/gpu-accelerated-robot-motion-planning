#!/usr/bin/env python3
"""Demonstrate online cuRobo replanning when an obstacle enters the Panda path."""

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
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from curobo_obstacle_visualization import (
    JOINT_NAMES,
    MODEL_PATH,
    PROJECT,
    RESULT_DIR,
    WORLD_PATH,
    add_scene_geom,
    quaternion_matrix_wxyz,
    resample_trajectory,
)

VIDEO_PATH = RESULT_DIR / "curobo_dynamic_replanning.mp4"
POSTER_PATH = RESULT_DIR / "curobo_dynamic_replanning.png"
CSV_PATH = RESULT_DIR / "curobo_dynamic_replanning_trajectory.csv"
METRICS_PATH = RESULT_DIR / "curobo_dynamic_replanning_metrics.json"
DYNAMIC_WORLD_NAME = "collision_table.yml"
DYNAMIC_WORLD_PATH = (
    PROJECT
    / "third_party/curobo/src/curobo/content/configs/world"
    / DYNAMIC_WORLD_NAME
)
START_OFFSET = [0.25] * 7
GOAL_OFFSET = [0.25, 0.15, 0.10]
TRIGGER_FRACTION = 0.05
BLOCK_FRACTION = 0.55
BLOCKER_DIMS = np.array([0.07, 0.07, 0.12], dtype=np.float64)


def load_dynamic_obstacles():
    world = yaml.safe_load(DYNAMIC_WORLD_PATH.read_text(encoding="utf-8"))
    obstacles = []
    for name, specification in world.get("cuboid", {}).items():
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


def make_planner():
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    config = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        DYNAMIC_WORLD_NAME,
        tensor_args,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        interpolation_dt=0.02,
        interpolation_steps=10000,
        collision_cache={"obb": 16, "mesh": 8},
        use_cuda_graph=True,
        num_ik_seeds=30,
        num_trajopt_seeds=12,
        num_graph_seeds=4,
    )
    planner = MotionGen(config)
    planner.warmup()
    return planner, tensor_args


def plan_once(planner, start, goal, attempts=8):
    result = planner.plan_single(
        start,
        goal,
        MotionGenPlanConfig(
            max_attempts=attempts,
            timeout=12.0,
            time_dilation_factor=0.5,
            enable_graph=True,
        ),
    )
    if not bool(result.success.item()):
        raise RuntimeError(f"cuRobo planning failed: {result.status}")
    trajectory = result.get_interpolated_plan()
    return (
        trajectory.position.detach().cpu().numpy(),
        float(result.interpolation_dt),
        float(result.solve_time) * 1000.0,
    )


def joint_state(tensor_args, positions):
    tensor = torch.as_tensor(
        positions,
        device=tensor_args.device,
        dtype=tensor_args.dtype,
    )
    if tensor.ndim == 1:
        tensor = tensor.view(1, -1)
    return JointState.from_position(tensor, joint_names=JOINT_NAMES)


def end_effector_positions(planner, tensor_args, positions):
    state = joint_state(tensor_args, positions)
    kinematics = planner.compute_kinematics(state)
    return (
        kinematics.ee_pos_seq.detach()
        .cpu()
        .numpy()
        .reshape(-1, 3)
    )


def plan_with_dynamic_obstacle():
    print("[1/4] Building the GPU planner and generating the initial path...")
    planner, tensor_args = make_planner()
    retract = planner.get_retract_config()

    retract_state = JointState.from_position(retract.view(1, -1))
    goal_kinematics = planner.compute_kinematics(retract_state)
    goal_position = (
        goal_kinematics.ee_pos_seq.squeeze()
        + tensor_args.to_device(GOAL_OFFSET)
    )
    goal_quaternion = goal_kinematics.ee_quat_seq.squeeze()
    goal = Pose(goal_position, quaternion=goal_quaternion)

    start = JointState.from_position(
        retract.view(1, -1).clone(),
        joint_names=JOINT_NAMES,
    )
    start.position[0] += tensor_args.to_device(START_OFFSET)

    initial, dt, initial_latency_ms = plan_once(planner, start, goal)
    initial_ee = end_effector_positions(planner, tensor_args, initial)
    trigger_index = max(1, int((len(initial) - 1) * TRIGGER_FRACTION))
    replan_start = joint_state(
        tensor_args,
        initial[trigger_index],
    )

    print(
        f"      initial plan: {initial_latency_ms:.2f} ms, "
        f"{len(initial)} points"
    )
    print(
        f"[2/4] Obstacle enters at {TRIGGER_FRACTION:.0%}; "
        "searching a valid blocked segment and replanning..."
    )

    blocker_position = None
    blocker_dims = None
    blocked_index = None
    candidate_fractions = (0.55, 0.65, 0.75, 0.85)
    candidate_sizes = (0.035, 0.045, 0.055)
    last_start_status = None
    last_path_status = None

    for size in candidate_sizes:
        for fraction in candidate_fractions:
            candidate_index = min(
                len(initial) - 2,
                max(
                    trigger_index + 2,
                    int((len(initial) - 1) * fraction),
                ),
            )
            candidate_position = initial_ee[candidate_index].copy()
            candidate_dims = np.array(
                [size, size, max(0.06, size * 1.5)],
                dtype=np.float64,
            )
            world_dict = yaml.safe_load(
                DYNAMIC_WORLD_PATH.read_text(encoding="utf-8")
            )
            world_dict.setdefault("cuboid", {})["dynamic_blocker"] = {
                "pose": [
                    float(candidate_position[0]),
                    float(candidate_position[1]),
                    float(candidate_position[2]),
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "dims": [float(value) for value in candidate_dims],
            }
            planner.update_world(WorldConfig.from_dict(world_dict))

            start_valid, start_status = planner.check_start_state(
                replan_start
            )
            blocked_state = joint_state(
                tensor_args,
                initial[candidate_index],
            )
            path_valid, path_status = planner.check_start_state(
                blocked_state
            )
            start_is_valid = (
                bool(start_valid.item())
                if hasattr(start_valid, "item")
                else bool(start_valid)
            )
            path_is_valid = (
                bool(path_valid.item())
                if hasattr(path_valid, "item")
                else bool(path_valid)
            )
            last_start_status = start_status
            last_path_status = path_status
            if start_is_valid and not path_is_valid:
                blocker_position = candidate_position
                blocker_dims = candidate_dims
                blocked_index = candidate_index
                break
        if blocker_position is not None:
            break

    if blocker_position is None:
        raise RuntimeError(
            "No dynamic blocker candidate preserved the current state "
            "while invalidating the future path: "
            f"start={last_start_status}, future={last_path_status}"
        )

    print(
        f"      selected blocker at {blocked_index / (len(initial) - 1):.0%} "
        f"with size {blocker_dims.round(3).tolist()} m"
    )
    terminal_state = joint_state(
        tensor_args,
        initial[-1],
    )
    terminal_kinematics = planner.compute_kinematics(terminal_state)
    replan_goal = Pose(
        terminal_kinematics.ee_pos_seq.squeeze(),
        quaternion=terminal_kinematics.ee_quat_seq.squeeze(),
    )
    replanned, replan_dt, replan_latency_ms = plan_once(
        planner,
        replan_start,
        replan_goal,
        attempts=32,
    )
    replanned_ee = end_effector_positions(
        planner,
        tensor_args,
        replanned,
    )

    initial_min_distance = float(
        np.linalg.norm(initial_ee - blocker_position, axis=1).min()
    )
    replanned_min_distance = float(
        np.linalg.norm(replanned_ee - blocker_position, axis=1).min()
    )
    print(
        f"      replan succeeded: {replan_latency_ms:.2f} ms, "
        f"{len(replanned)} points"
    )
    print(
        f"      EE center distance: initial={initial_min_distance:.3f} m, "
        f"replanned={replanned_min_distance:.3f} m"
    )

    return {
        "planner": planner,
        "tensor_args": tensor_args,
        "initial": initial,
        "initial_ee": initial_ee,
        "initial_dt": dt,
        "initial_latency_ms": initial_latency_ms,
        "trigger_index": trigger_index,
        "replanned": replanned,
        "replanned_ee": replanned_ee,
        "replan_dt": replan_dt,
        "replan_latency_ms": replan_latency_ms,
        "goal_position": goal_position.detach().cpu().numpy(),
        "blocker_position": blocker_position,
        "blocker_dims": blocker_dims,
        "blocked_index": blocked_index,
        "initial_min_distance": initial_min_distance,
        "replanned_min_distance": replanned_min_distance,
    }


def annotate(frame, phase, detail, progress):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    phase_colors = {
        "INITIAL EXECUTION": (74, 180, 255, 255),
        "OBSTACLE ENTERING": (255, 178, 54, 255),
        "GPU REPLANNING": (255, 85, 85, 255),
        "REPLANNED EXECUTION": (74, 230, 150, 255),
        "GOAL REACHED": (112, 255, 128, 255),
    }
    draw.rounded_rectangle((14, 14, 500, 86), radius=10, fill=(8, 15, 28, 220))
    draw.text(
        (28, 25),
        "Dynamic Obstacle Detection and Online Replanning",
        fill=(255, 255, 255, 255),
    )
    draw.text(
        (28, 49),
        phase,
        fill=phase_colors.get(phase, (255, 255, 255, 255)),
    )
    draw.text((205, 49), detail, fill=(190, 220, 238, 255))
    bar_left, bar_top, bar_right = 28, 72, 472
    draw.rounded_rectangle(
        (bar_left, bar_top, bar_right, 78),
        radius=3,
        fill=(52, 65, 80, 255),
    )
    draw.rounded_rectangle(
        (
            bar_left,
            bar_top,
            bar_left + int((bar_right - bar_left) * np.clip(progress, 0.0, 1.0)),
            78,
        ),
        radius=3,
        fill=phase_colors.get(phase, (80, 200, 255, 255)),
    )
    return np.asarray(image)


def render_video(result, obstacles, fps=30):
    print("[3/4] Rendering the detect-replan-execute sequence with MuJoCo...")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    joint_addresses = [
        int(
            model.jnt_qposadr[
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    name,
                )
            ]
        )
        for name in JOINT_NAMES
    ]
    finger_addresses = [
        int(
            model.jnt_qposadr[
                mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    name,
                )
            ]
        )
        for name in ("panda_finger_joint1", "panda_finger_joint2")
    ]
    hand_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "panda_hand",
    )
    if hand_body_id < 0:
        hand_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "panda_link7",
        )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.45]
    camera.distance = 2.1
    camera.azimuth = 180.0
    camera.elevation = -55.0
    renderer = mujoco.Renderer(model, height=480, width=640)

    trigger_index = result["trigger_index"]
    pre_positions = result["initial"][: trigger_index + 1]
    pre_video = resample_trajectory(
        pre_positions,
        result["initial_dt"],
        fps=fps,
        playback_scale=3.0,
    )
    post_video = resample_trajectory(
        result["replanned"],
        result["replan_dt"],
        fps=fps,
        playback_scale=3.0,
    )
    blocker_final = result["blocker_position"]
    blocker_start = blocker_final + np.array([0.0, -0.55, 0.0])
    incoming = np.linspace(blocker_start, blocker_final, fps + 1)
    trail = []

    def draw_frame(q, blocker_position, phase, detail, progress, path_points):
        data.qpos[joint_addresses] = q
        data.qpos[finger_addresses] = 0.02
        mujoco.mj_forward(model, data)
        trail.append(data.xpos[hand_body_id].copy())
        renderer.update_scene(data, camera=camera)

        for obstacle in obstacles:
            color = (
                [0.30, 0.32, 0.36, 0.22]
                if obstacle["name"] == "table"
                else [0.95, 0.31, 0.08, 0.07]
            )
            add_scene_geom(
                renderer.scene,
                mujoco.mjtGeom.mjGEOM_BOX,
                obstacle["dims"] * 0.5,
                obstacle["position"],
                obstacle["rotation"],
                color,
            )

        if blocker_position is not None:
            add_scene_geom(
                renderer.scene,
                mujoco.mjtGeom.mjGEOM_BOX,
                result["blocker_dims"] * 0.5,
                blocker_position,
                np.eye(3),
                [1.0, 0.05, 0.05, 0.78],
            )
            add_scene_geom(
                renderer.scene,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                [0.060, 0.0, 0.0],
                blocker_position,
                np.eye(3),
                [1.0, 0.08, 0.08, 0.22],
            )

        add_scene_geom(
            renderer.scene,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            [0.045, 0.0, 0.0],
            result["goal_position"],
            np.eye(3),
            [0.15, 1.0, 0.25, 1.0],
        )

        for point in path_points[:: max(1, len(path_points) // 24)]:
            add_scene_geom(
                renderer.scene,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                [0.009, 0.0, 0.0],
                point,
                np.eye(3),
                [0.05, 0.85, 1.0, 0.72],
            )

        for point in trail[-36::3]:
            add_scene_geom(
                renderer.scene,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                [0.013, 0.0, 0.0],
                point,
                np.eye(3),
                [0.15, 0.95, 1.0, 0.92],
            )

        return annotate(
            renderer.render().copy(),
            phase,
            detail,
            progress,
        )

    frames = []
    start_q = pre_video[0]
    for _ in range(fps):
        frames.append(
            draw_frame(
                start_q,
                None,
                "INITIAL EXECUTION",
                f"initial GPU plan {result['initial_latency_ms']:.1f} ms",
                0.0,
                result["initial_ee"],
            )
        )

    for index, q in enumerate(pre_video):
        frames.append(
            draw_frame(
                q,
                None,
                "INITIAL EXECUTION",
                "following the original collision-free path",
                index / max(len(pre_video) - 1, 1),
                result["initial_ee"],
            )
        )

    trigger_q = pre_video[-1]
    for index, blocker_position in enumerate(incoming):
        frames.append(
            draw_frame(
                trigger_q,
                blocker_position,
                "OBSTACLE ENTERING",
                "the original future path is now blocked",
                index / max(len(incoming) - 1, 1),
                result["initial_ee"],
            )
        )

    for index in range(fps):
        frames.append(
            draw_frame(
                trigger_q,
                blocker_final,
                "GPU REPLANNING",
                f"new plan computed in {result['replan_latency_ms']:.1f} ms",
                index / max(fps - 1, 1),
                result["replanned_ee"],
            )
        )

    for index, q in enumerate(post_video):
        frames.append(
            draw_frame(
                q,
                blocker_final,
                "REPLANNED EXECUTION",
                "executing the updated collision-free trajectory",
                index / max(len(post_video) - 1, 1),
                result["replanned_ee"],
            )
        )

    for _ in range(fps):
        frames.append(
            draw_frame(
                post_video[-1],
                blocker_final,
                "GOAL REACHED",
                "dynamic obstacle avoided successfully",
                1.0,
                result["replanned_ee"],
            )
        )

    imageio.imwrite(
        POSTER_PATH,
        frames[min(len(frames) - 1, len(frames) // 2 + fps)],
    )
    with imageio.get_writer(
        VIDEO_PATH,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)
    renderer.close()
    print(f"      video:  {VIDEO_PATH}")
    print(f"      poster: {POSTER_PATH}")


def save_results(result):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = result["initial"][: result["trigger_index"] + 1]
    combined = np.vstack([prefix, result["replanned"][1:]])
    phase_ids = np.concatenate(
        [
            np.zeros(len(prefix), dtype=np.float64),
            np.ones(len(result["replanned"]) - 1, dtype=np.float64),
        ]
    )
    times = np.arange(len(combined), dtype=np.float64) * result["initial_dt"]
    np.savetxt(
        CSV_PATH,
        np.column_stack([times, phase_ids, combined]),
        delimiter=",",
        header="time_s,phase_id," + ",".join(JOINT_NAMES),
        comments="",
    )

    metrics = {
        "planner": "cuRobo 0.7.8",
        "robot": "Franka Panda",
        "gpu": torch.cuda.get_device_name(0),
        "scenario": "dynamic obstacle enters the future path",
        "success": True,
        "trigger_fraction": TRIGGER_FRACTION,
        "initial_planning_latency_ms": round(
            result["initial_latency_ms"],
            3,
        ),
        "replanning_latency_ms": round(
            result["replan_latency_ms"],
            3,
        ),
        "initial_trajectory_points": int(len(result["initial"])),
        "replanned_trajectory_points": int(len(result["replanned"])),
        "initial_path_min_ee_center_distance_m": round(
            result["initial_min_distance"],
            4,
        ),
        "replanned_path_min_ee_center_distance_m": round(
            result["replanned_min_distance"],
            4,
        ),
        "dynamic_obstacle_dims_m": result["blocker_dims"].tolist(),
        "dynamic_obstacle_path_fraction": round(
            result["blocked_index"] / (len(result["initial"]) - 1),
            4,
        ),
        "avoidance_clearance_gain_m": round(
            result["replanned_min_distance"]
            - result["initial_min_distance"],
            4,
        ),
        "combined_execution_points": int(len(combined)),
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run inside a Slurm GPU job.")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result = plan_with_dynamic_obstacle()
    metrics = save_results(result)
    render_video(result, load_dynamic_obstacles())
    print("[4/4] DONE")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()