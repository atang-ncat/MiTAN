# MiTAN — Autonomous Lane-Following Vehicle

Real-time autonomous lane-following system for the **Yahboom Ackermann** vehicle on **NVIDIA Jetson AGX Orin** using **HybridNets** (custom-trained), **Isaac ROS**, and **Intel RealSense D455**.

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  RealSense  │     │   cuVSLAM    │     │  slam_toolbox   │
│  D455       │────▶│  (6-DOF      │────▶│  (2D Mapping)   │
│ Stereo IR ──┤     │   Pose)      │     └─────────────────┘
│             │     └──────────────┘
│ RGB ────────┤     ┌──────────────┐     ┌─────────────────┐
│             │────▶│  HybridNets  │────▶│  Lane Follower  │
│             │     │  (TensorRT)  │     │  (PD Controller) │
│             │     │              │     └────────┬────────┘
│             │     │ Lane Lines   │              │ /cmd_vel_auto
│             │     │ Road Seg.    │     ┌────────▼────────┐
│             │     │ Detections   │     │  cmd_vel_mux    │
│             │     └──────────────┘     │  (gamepad prio) │
└─────────────┘                          └────────┬────────┘
                                                  │ /cmd_vel
┌─────────────┐     ┌──────────────┐     ┌────────▼────────┐
│  RPLiDAR A3 │────▶│  slam_toolbox│     │  serial_bridge  │──▶ Arduino
└─────────────┘     └──────────────┘     └─────────────────┘
```

## Hardware

| Component | Details |
|---|---|
| **Compute** | NVIDIA Jetson AGX Orin 64GB |
| **Camera** | Intel RealSense D455 (USB 3.2) |
| **LiDAR** | RPLiDAR A3 (360°, 10 Hz) |
| **Chassis** | Yahboom Ackermann (servo steering + 2× DC motors) |
| **Controller** | Arduino Mega 2560 |
| **Storage** | HP EX900 Plus 512GB NVMe SSD |
| **Power Mode** | MAXN (`nvpmodel -m 0` + `jetson_clocks`) |

## Software Stack

| Component | Version |
|---|---|
| **ROS 2** | Humble |
| **Isaac ROS** | 3.2 |
| **cuVSLAM** | 12.6 |
| **TensorRT** | 10.3 |
| **HybridNets** | Custom-trained (Quanser studio map) |
| **Container** | `isaac_ros_dev` (aarch64) |

## HybridNets Model

Fine-tuned on the Quanser studio driving map for multi-task perception:

| Task | Classes | Metric |
|------|---------|--------|
| **Detection** | person, robot, traffic_light_green, traffic_light_off, traffic_light_red | mAP@0.5: 73.3% |
| **Road Segmentation** | drivable area | IoU: 94.1% |
| **Lane Segmentation** | lane lines | IoU: 76.9% |

Input: 384 × 640 (H × W) · ONNX opset 18 · TensorRT FP16

## Files

| File | Description |
|---|---|
| `lane_follow.launch.py` | **Lane-following stack** (camera + HybridNets + mux + serial) |
| `vslam_yolov10_realsense.launch.py` | Perception stack (VSLAM + YOLOv10 + LiDAR + SLAM) |
| `teleop.launch.py` | Gamepad teleop (joy + serial bridge) |
| `hybridnets_lane_follower.py` | Lane-following ROS 2 node (tiered guidance + PD control) |
| `cmd_vel_mux.py` | Priority-based velocity mux (teleop > autonomous) |
| `serial_bridge.py` | ROS 2 → Arduino serial bridge (`S<speed>,A<angle>`) |
| `hybridnets_deploy/` | HybridNets model, config, TRT conversion, inference code |
| `ackermann_drive/` | Arduino firmware for Yahboom chassis |
| `LANE_FOLLOWING_LOGIC.md` | Detailed lane-following algorithm documentation |
| `DRIVING_AND_MAPPING.md` | Guide for mapping with gamepad + SLAM |

## Quick Start — Lane Following

```bash
# 1. Set Jetson to MAX performance (on host)
sudo nvpmodel -m 0 && sudo jetson_clocks

# 2. Launch Isaac ROS container
cd /mnt/ssd/ws/robot_ws && isaac_dev

# 3. Build TensorRT engine (one-time, inside container)
cd /workspaces/isaac_ros-dev/hybridnets_deploy
bash convert_hybridnets_trt.sh

# 4. Launch lane-following stack
source /opt/ros/humble/setup.bash
sudo chmod 666 /dev/ttyACM0
ros2 launch /workspaces/isaac_ros-dev/lane_follow.launch.py

# 5. Hold LB on gamepad to override autonomous control at any time
```

## Quick Start — Perception + Mapping

```bash
# Inside container:
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/lidar_ws/install/setup.bash
ros2 launch /workspaces/isaac_ros-dev/vslam_yolov10_realsense.launch.py

# On host — teleop:
ros2 launch teleop.launch.py serial_port:=/dev/ttyACM0
```

## Lane-Following Logic

The system uses a **4-tier guidance extraction** from HybridNets output:

| Tier | Condition | Method | Speed |
|------|-----------|--------|-------|
| 1 (best) | Two lane lines | Midpoint between left/right | 0.15 m/s |
| 2 (good) | One lane line | Offset by assumed width | 0.15 m/s |
| 3 (fallback) | Road mask only | Drivable area centroid | 0.08 m/s |
| 4 (fail) | Nothing visible | Stop after 1.5s timeout | 0 m/s |

Additional features:
- **Curve speed reduction** — slows proportionally to steering angle (60% at max turn)
- **Traffic light response** — stops on red, goes on green
- **Road mask safety clamp** — constrains steering target within drivable area
- **Gamepad override** — hold LB for manual control at any time

See [LANE_FOLLOWING_LOGIC.md](LANE_FOLLOWING_LOGIC.md) for full algorithm details.

## Gamepad Controls

| Control | Action |
|---|---|
| LB (hold) | Deadman switch — must hold to drive manually |
| Left stick ↑↓ | Forward / reverse |
| Right stick ←→ | Steer left / right |
| Release LB | Stop (autonomous takes over if running) |

## Tuning Parameters

Adjustable at runtime via `ros2 param set`:

```bash
ros2 param set /hybridnets_lane_follower cruise_speed 0.2
ros2 param set /hybridnets_lane_follower kp 0.008
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cruise_speed` | 0.15 | Forward speed on straights (m/s) |
| `kp` | 0.005 | Proportional steering gain |
| `kd` | 0.002 | Derivative steering gain (dampens oscillation) |
| `max_angular_z` | 0.8 | Max steering output (rad/s) |
| `curve_slowdown_factor` | 0.6 | Speed reduction on curves (0–1) |
| `lane_width_px` | 200 | Assumed lane width for single-line fallback |

## Known Issues

- **RealSense IMU**: HID Motion Sensor failure on JetPack 6 — IMU disabled
- **Container serial**: The `admin` user needs `chmod 666 /dev/ttyACM0` on each restart
- **ONNX model size**: 55MB (GitHub warns about large files; consider Git LFS)
- **LiDAR permissions**: RPLiDAR needs `chmod 666 /dev/ttyUSB0` inside container

## Roadmap

- [x] cuVSLAM stereo tracking at 30 FPS
- [x] RT-DETR → YOLOv10m object detection pipeline
- [x] Combined VSLAM + Detection + LiDAR SLAM launch
- [x] RPLiDAR A3 integration with slam_toolbox
- [x] Gamepad teleop with Ackermann serial bridge
- [x] 2D occupancy grid mapping
- [x] HybridNets fine-tuning on Quanser map dataset
- [x] HybridNets TensorRT deployment
- [x] Lane-following node with tiered guidance
- [x] Traffic light response
- [x] Curve speed reduction
- [x] cmd_vel mux with gamepad override
- [ ] Live track validation and PID tuning
- [ ] Nav2 integration with lane constraints
- [ ] Depth-based obstacle avoidance
