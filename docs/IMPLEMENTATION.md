# GPU 加速机械臂运动规划：实现与复现记录

## 1. 项目目标

本项目实现一条完整且可复现的机器人运动规划链路：

1. 使用 Franka Panda 作为机械臂模型。
2. 使用 cuRobo 在 RTX 4090 上进行 GPU 加速运动规划。
3. 将规划结果标准化为按时间排列的 7 关节轨迹。
4. 将同一轨迹映射到 MuJoCo 模型并离屏渲染。
5. 输出视频、海报、关节曲线、CSV 和量化指标 JSON。
6. 后续加入 MoveIt2/OMPL CPU 基线，进行公平对比。

当前项目不依赖 HaMeR，也没有修改 HaMeR 目录。

## 2. 技术栈

- Ubuntu 22.04
- Python 3.10
- NVIDIA RTX 4090
- CUDA 12.4
- PyTorch 2.4.0+cu124
- cuRobo v0.7.8
- NVIDIA Warp 1.7.2
- MuJoCo 3.2.7（EGL 离屏渲染）
- ROS 2 Humble
- MoveIt2 / OMPL
- Apptainer
- Slurm
- NumPy、Matplotlib、imageio、FFmpeg

## 3. 当前目录结构

    robot_motion_control/
    ├── .venv-curobo/                 # 独立 Python 与 CUDA 环境
    ├── container/rootfs/ubuntu22/    # ROS 2 Humble + MoveIt2 容器
    ├── third_party/curobo/           # cuRobo v0.7.8 源码
    ├── experiments/
    │   ├── generated/franka/         # MuJoCo 可加载的 Panda 模型和网格
    │   ├── results/                  # 实验输出
    │   └── panda_curobo_mujoco.py    # 主实验程序
    ├── ros2_ws/                      # ROS 2 工作空间
    ├── docs/IMPLEMENTATION.md        # 本文档
    └── run_demo.sh                   # 一键验证与运行

## 4. 数据流

    起始关节角 + 目标末端位姿 + 碰撞场景
                         │
                         ▼
               cuRobo GPU 运动规划
                         │
                         ▼
              N x 7 关节轨迹 + 时间步长
                 │                 │
                 ▼                 ▼
          CSV / 指标 / 曲线图    MuJoCo 关节映射
                                      │
                                      ▼
                               MP4 视频 + PNG 海报

cuRobo 负责运动学、碰撞检测和轨迹优化；MuJoCo 在当前阶段负责模型验证与轨迹可视化。二者通过相同的 Panda 关节名称连接。

## 5. 关键实现

### 5.1 GPU 规划

主程序读取 cuRobo 的 franka.yml 与 collision_table.yml，构造 MotionGen，并先执行 warmup。正式规划从一个多关节偏移姿态返回 Panda 的 retract 目标位姿。偏移覆盖 7 个机械臂关节，运动幅度明显大于早期只改变 joint1 的测试。

### 5.2 轨迹输出

规划成功后读取 result.get_interpolated_plan()，获得：

- position：N x 7 关节角矩阵；
- interpolation_dt：相邻轨迹点时间间隔；
- solve_time：GPU 求解时间。

程序进一步计算关节空间路径长度、最大关节速度、最大关节加速度和每个关节运动范围。

### 5.3 MuJoCo 映射

MuJoCo 模型含 9 个自由度：7 个机械臂关节和 2 个手指关节。程序按关节名称查询 qpos 地址，将 cuRobo 的 7 维轨迹逐帧写入对应位置，手指保持 0.02 rad。

服务器没有桌面显示，因此使用 MUJOCO_GL=egl 进行无窗口 GPU 渲染。视频播放时间被放慢到规划时间的 2 倍，前后各保留 1 秒静止画面，便于观察。

## 6. 一键运行

必须先取得一张 Slurm GPU：

    srun --gpus=1 --pty bash

进入作业后执行：

    cd /data/yezhengfu/robot_motion_control
    bash run_demo.sh

只验证规划、跳过视频：

    bash run_demo.sh --no-video

改变视频帧率和播放速度：

    bash run_demo.sh --fps 30 --playback-scale 2.5

run_demo.sh 会先检查 Slurm 和 CUDA，再检查 PyTorch、Warp、MuJoCo、cuRobo，最后运行完整实验。

## 7. 输出文件

运行成功后，experiments/results/ 中应包含：

- panda_curobo_trajectory.csv：时间戳和 7 关节轨迹；
- panda_curobo_metrics.json：规划时间、轨迹点数、路径长度、速度和加速度；
- panda_curobo_joint_plot.png：7 个关节随时间变化曲线；
- panda_curobo_demo.mp4：MuJoCo 动画；
- panda_curobo_poster.png：视频首帧海报。

## 8. 已解决的问题

### GitHub TLS 下载失败

git clone 多次出现 GnuTLS recv error。改用 GitHub codeload 归档下载 cuRobo v0.7.8，并进行压缩包校验和解压。

### Warp 与驱动不兼容

最新版 Warp 1.19 要求更高 CUDA 驱动，服务器驱动只能提供 CUDA 12.4。将 warp-lang 固定为 1.7.2 后，RTX 4090 被正常识别。

### MuJoCo URDF 加载失败

原始 URDF 的网格路径与惯性矩阵不满足 MuJoCo 导入要求。将碰撞网格复制到生成目录，并生成移除 8 个不兼容 inertial 标签的 MuJoCo 专用 URDF。模型最终以 nq=9、nv=9 成功加载。

### 离屏帧缓冲尺寸错误

MuJoCo 默认离屏帧缓冲为 640 x 480，早期请求 960 x 720 导致异常。将渲染尺寸统一为 640 x 480。

## 9. 当前完成度

已完成：

- CUDA / PyTorch / cuRobo / Warp 环境；
- cuRobo CUDA 扩展验证；
- Panda GPU 运动规划；
- 多关节轨迹输出；
- MuJoCo 模型加载与轨迹映射；
- EGL 视频与海报生成；
- 曲线图和指标 JSON；
- 一键运行入口。

下一阶段：

1. 用 MoveIt2/OMPL 运行同一批起点、目标和场景；
2. 批量生成 100 至 1000 个请求；
3. 对比成功率、P50/P95 延迟、路径长度、平滑性和最小间隙；
4. 将 cuRobo 包装为 ROS 2 Action Server；
5. 通过 trajectory_msgs/JointTrajectory 接入 ros2_control；
6. 增加动态障碍物与在线重规划；
7. 确认实验室真机型号后实现关节名称、限位和控制器适配。

## 10. 简历表述（完成基线对比后使用）

项目名称：GPU-Accelerated Robot Motion Planning for Franka Panda

- 基于 cuRobo、MoveIt2/OMPL、ROS 2 Humble 与 MuJoCo 构建机械臂运动规划和仿真验证链路，在 Slurm 管理的 RTX 4090 上完成 GPU 加速规划。
- 设计统一轨迹与评测接口，统计成功率、P50/P95 延迟、关节空间路径长度、速度、加速度及碰撞间隙，并实现 CPU/GPU 规划器公平对比。
- 完成 cuRobo 轨迹到 MuJoCo 和 ros2_control 的关节映射，支持离屏视频、指标导出及后续真实机械臂适配。
- 定位并解决 CUDA 驱动与 Warp 版本冲突、URDF 网格路径和惯性矩阵兼容性、无桌面服务器 EGL 渲染等工程问题。

注意：第二条中的批量对比数字必须等 OMPL 基线与批量实验实际完成后再填写，不能提前编造。


## 10. 已验证的 cuRobo 实测结果

在 RTX 4090 上运行 `bash run_demo.sh`，已得到以下真实结果：

- 规划器：cuRobo 0.7.8
- 机器人：Franka Panda
- 规划成功：是
- 单次规划时间：37.532 ms
- 轨迹点数：62
- 轨迹时长：1.22 s
- 关节空间路径长度：0.7125 rad
- 最大关节速度：0.7779 rad/s
- 最大关节加速度：4.8489 rad/s²

生成物位于 `experiments/results/`：轨迹 CSV、指标 JSON、关节曲线、海报图和 MuJoCo 演示视频。

## 11. MoveIt2 / OMPL 对照基准

为了让项目不只是“跑通一个 GPU 示例”，新增了 ROS 2 C++ 包 `planner_benchmark`。它在同一台 Franka Panda 上使用 MoveIt2 的 OMPL/RRTConnect，对固定起点和固定末端位姿目标进行 1 次预热和 20 次正式规划。

统计指标包括：

- 规划成功率
- 规划延迟 P50 / P95
- 平均关节空间路径长度
- 平均轨迹时长
- 每次实验的轨迹点数与原始结果

运行方式（该基准不需要 GPU，也不需要 RViz）：

```bash
cd /data/yezhengfu/robot_motion_control
bash run_ompl_baseline.sh
```

结果将生成到：

- `experiments/results/ompl_metrics.json`
- `experiments/results/ompl_runs.csv`

这样项目形成了“GPU 并行优化规划器 cuRobo + 传统采样规划器 OMPL”的可量化对照，而不只是可视化演示。
