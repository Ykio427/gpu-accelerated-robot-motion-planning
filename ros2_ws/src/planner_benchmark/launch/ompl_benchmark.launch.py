from launch import LaunchDescription
from launch.actions import EmitEvent, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("moveit_resources_panda").to_moveit_configs()

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "allow_trajectory_execution": False,
                "publish_robot_description_semantic": True,
            },
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[moveit_config.robot_description],
    )

    static_world = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        output="log",
        arguments=["0", "0", "0", "0", "0", "0", "world", "panda_link0"],
    )

    benchmark = Node(
        package="planner_benchmark",
        executable="ompl_benchmark",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            {
                "runs": 20,
                "output_dir": "/workspace/experiments/results",
            },
        ],
    )

    delayed_benchmark = TimerAction(period=3.0, actions=[benchmark])
    stop_when_done = RegisterEventHandler(
        OnProcessExit(
            target_action=benchmark,
            on_exit=[EmitEvent(event=Shutdown(reason="OMPL benchmark complete"))],
        )
    )

    return LaunchDescription(
        [robot_state_publisher, static_world, move_group, delayed_benchmark, stop_when_done]
    )
