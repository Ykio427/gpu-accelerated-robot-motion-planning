cmake_minimum_required(VERSION 3.8)
project(planner_benchmark)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)

add_executable(ompl_benchmark src/ompl_benchmark.cpp)
target_compile_features(ompl_benchmark PUBLIC cxx_std_17)
ament_target_dependencies(
  ompl_benchmark
  rclcpp
  geometry_msgs
  sensor_msgs
  moveit_core
  moveit_ros_planning_interface
)

install(TARGETS ompl_benchmark DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
