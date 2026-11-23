#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

using SteadyClock = std::chrono::steady_clock;

struct RunResult
{
  bool success{false};
  double latency_ms{0.0};
  double path_length_rad{0.0};
  double duration_s{0.0};
  std::size_t points{0};
};

double percentile(std::vector<double> values, double q)
{
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(
    std::round(q * static_cast<double>(values.size() - 1)));
  return values.at(index);
}

RunResult summarize(
  const moveit::planning_interface::MoveGroupInterface::Plan & plan,
  bool success, double latency_ms)
{
  RunResult result;
  result.success = success;
  result.latency_ms = latency_ms;
  if (!success) return result;

  const auto & points = plan.trajectory_.joint_trajectory.points;
  result.points = points.size();
  for (std::size_t i = 1; i < points.size(); ++i) {
    double squared = 0.0;
    for (std::size_t j = 0; j < points[i].positions.size(); ++j) {
      const double delta = points[i].positions[j] - points[i - 1].positions[j];
      squared += delta * delta;
    }
    result.path_length_rad += std::sqrt(squared);
  }
  if (!points.empty()) {
    const auto & stamp = points.back().time_from_start;
    result.duration_s = static_cast<double>(stamp.sec) +
      static_cast<double>(stamp.nanosec) * 1e-9;
  }
  return result;
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared(
    "ompl_benchmark",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));
  const int runs = node->get_parameter_or<int>("runs", 20);
  const std::string output_dir = node->get_parameter_or<std::string>(
    "output_dir", "/workspace/experiments/results");

  const std::vector<double> goal_joints{
    0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398};
  const std::vector<double> start_joints{
    0.45, -1.035398, 0.35, -2.176194, -0.35, 1.350796, 1.235398};

  auto joint_state_publisher =
    node->create_publisher<sensor_msgs::msg::JointState>("/joint_states", 10);
  auto joint_state =
    std::make_shared<sensor_msgs::msg::JointState>();
  joint_state->name = {
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7",
    "panda_finger_joint1", "panda_finger_joint2"};
  joint_state->position = start_joints;
  joint_state->position.push_back(0.035);
  joint_state->position.push_back(0.035);
  auto joint_state_timer = node->create_wall_timer(
    std::chrono::milliseconds(20),
    [node, joint_state_publisher, joint_state]() {
      joint_state->header.stamp = node->now();
      joint_state_publisher->publish(*joint_state);
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  std::thread spinner([&executor]() {executor.spin();});
  int return_code = 0;

  try {
    moveit::planning_interface::MoveGroupInterface move_group(node, "panda_arm");
    move_group.setPlanningPipelineId("ompl");
    move_group.setPlannerId("RRTConnectkConfigDefault");
    move_group.setPlanningTime(2.0);
    move_group.setNumPlanningAttempts(1);
    move_group.setMaxVelocityScalingFactor(1.0);
    move_group.setMaxAccelerationScalingFactor(1.0);

    const auto model = move_group.getRobotModel();
    const auto * joint_group = model->getJointModelGroup("panda_arm");
    if (joint_group == nullptr) {
      throw std::runtime_error("panda_arm joint group was not found");
    }

    moveit::core::RobotState start_state(model);
    start_state.setToDefaultValues();
    start_state.setJointGroupPositions(joint_group, start_joints);
    start_state.update();
    moveit::core::RobotState goal_state(model);
    goal_state.setToDefaultValues();
    goal_state.setJointGroupPositions(joint_group, goal_joints);
    goal_state.update();

    const Eigen::Isometry3d & goal_transform =
      goal_state.getGlobalLinkTransform("panda_hand");
    const Eigen::Quaterniond goal_quaternion(goal_transform.rotation());
    geometry_msgs::msg::Pose goal_pose;
    goal_pose.position.x = goal_transform.translation().x();
    goal_pose.position.y = goal_transform.translation().y();
    goal_pose.position.z = goal_transform.translation().z();
    goal_pose.orientation.x = goal_quaternion.x();
    goal_pose.orientation.y = goal_quaternion.y();
    goal_pose.orientation.z = goal_quaternion.z();
    goal_pose.orientation.w = goal_quaternion.w();

    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    move_group.setStartState(start_state);
    move_group.setPoseTarget(goal_pose, "panda_hand");
    RCLCPP_INFO(node->get_logger(), "Warming up OMPL/RRTConnect...");
    moveit::planning_interface::MoveGroupInterface::Plan warmup_plan;
    const bool warmup_ok =
      move_group.plan(warmup_plan) == moveit::core::MoveItErrorCode::SUCCESS;
    if (!warmup_ok) throw std::runtime_error("OMPL warmup planning failed");

    std::vector<RunResult> results;
    results.reserve(static_cast<std::size_t>(runs));
    for (int i = 0; i < runs; ++i) {
      move_group.setStartState(start_state);
      move_group.setPoseTarget(goal_pose, "panda_hand");
      moveit::planning_interface::MoveGroupInterface::Plan plan;
      const auto begin = SteadyClock::now();
      const bool success =
        move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS;
      const auto end = SteadyClock::now();
      const double latency_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
      results.push_back(summarize(plan, success, latency_ms));
      RCLCPP_INFO(
        node->get_logger(),
        "run %02d/%02d success=%s latency=%.2f ms points=%zu",
        i + 1, runs, success ? "true" : "false",
        latency_ms, results.back().points);
    }

    const auto successes = static_cast<std::size_t>(std::count_if(
      results.begin(), results.end(),
      [](const RunResult & result) {return result.success;}));
    std::vector<double> latencies;
    double path_sum = 0.0;
    double duration_sum = 0.0;
    for (const auto & result : results) {
      if (result.success) {
        latencies.push_back(result.latency_ms);
        path_sum += result.path_length_rad;
        duration_sum += result.duration_s;
      }
    }

    const double denominator = successes == 0 ? 1.0 : static_cast<double>(successes);
    const double success_rate = 100.0 * static_cast<double>(successes) /
      static_cast<double>(results.size());
    const double p50_ms = percentile(latencies, 0.50);
    const double p95_ms = percentile(latencies, 0.95);
    const double mean_path = path_sum / denominator;
    const double mean_duration = duration_sum / denominator;

    std::filesystem::create_directories(output_dir);
    const std::string csv_path = output_dir + "/ompl_runs.csv";
    const std::string json_path = output_dir + "/ompl_metrics.json";
    std::ofstream csv(csv_path);
    csv << "run,success,latency_ms,path_length_rad,duration_s,trajectory_points"
        << std::endl;
    csv << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < results.size(); ++i) {
      const auto & result = results[i];
      csv << i + 1 << ',' << (result.success ? 1 : 0) << ','
          << result.latency_ms << ',' << result.path_length_rad << ','
          << result.duration_s << ',' << result.points << std::endl;
    }

    const char quote = 34;
    std::ofstream json(json_path);
    json << std::fixed << std::setprecision(4);
    json << '{' << std::endl
         << "  " << quote << "planner" << quote << ": " << quote
         << "MoveIt2 OMPL RRTConnect" << quote << ',' << std::endl
         << "  " << quote << "robot" << quote << ": " << quote
         << "Franka Panda" << quote << ',' << std::endl
         << "  " << quote << "runs" << quote << ": " << runs << ',' << std::endl
         << "  " << quote << "successful_runs" << quote << ": "
         << successes << ',' << std::endl
         << "  " << quote << "success_rate_percent" << quote << ": "
         << success_rate << ',' << std::endl
         << "  " << quote << "planning_latency_p50_ms" << quote << ": "
         << p50_ms << ',' << std::endl
         << "  " << quote << "planning_latency_p95_ms" << quote << ": "
         << p95_ms << ',' << std::endl
         << "  " << quote << "mean_joint_path_length_rad" << quote << ": "
         << mean_path << ',' << std::endl
         << "  " << quote << "mean_trajectory_duration_s" << quote << ": "
         << mean_duration << std::endl
         << '}' << std::endl;

    RCLCPP_INFO(node->get_logger(), "OMPL benchmark complete");
    RCLCPP_INFO(node->get_logger(), "success rate: %.1f%%", success_rate);
    RCLCPP_INFO(node->get_logger(), "latency P50/P95: %.2f / %.2f ms", p50_ms, p95_ms);
    RCLCPP_INFO(node->get_logger(), "metrics: %s", json_path.c_str());
    RCLCPP_INFO(node->get_logger(), "per-run CSV: %s", csv_path.c_str());
  } catch (const std::exception & error) {
    RCLCPP_ERROR(node->get_logger(), "benchmark failed: %s", error.what());
    return_code = 1;
  }

  joint_state_timer->cancel();
  executor.cancel();
  if (spinner.joinable()) spinner.join();
  rclcpp::shutdown();
  return return_code;
}
