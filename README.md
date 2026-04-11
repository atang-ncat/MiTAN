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
│             │     │  (TensorRT)  │     │  (PID Control)  │
│             │     │              │     └────────┬────────┘
│             │     │ Lane Lines   │     /cmd_vel_auto    │
│             │     │ Road Seg.    │     guidance_source  │
│             │     │ Detections   │     road_width       │
│             │     └──────────────┘              │
└─────────────┘              ┌─────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────┐
│  Intersection Controller (state machine)                   │
│  /cmd_vel_intersection                                     │
└────────────────────────────────────────────────────────────┘
                             │
┌────────────────────────────▼───────────────────────────────┐
│  cmd_vel_mux (teleop > intersection > auto)                │
└────────────────────────────┬───────────────────────────────┘
                             │ /cmd_vel
┌─────────────┐     ┌───────▼─────────┐
│  RPLiDAR A3 │     │  serial_bridge  │──▶ Arduino
└─────────────┘     └─────────────────┘
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
| `lane_follow.launch.py` | **Lane-following + intersection stack** (camera + HybridNets + intersection + mux + serial) |
| `vslam_yolov10_realsense.launch.py` | Perception stack (VSLAM + YOLOv10 + LiDAR + SLAM) |
| `teleop.launch.py` | Gamepad teleop (joy + serial bridge) |
| `hybridnets_lane_follower.py` | Lane-following ROS 2 node (two-pass guidance + PID control) |
| `intersection_controller.py` | **Intersection navigation** (state-machine + timed maneuvers) |
| `cmd_vel_mux.py` | Priority-based velocity mux (teleop > intersection > autonomous) |
| `serial_bridge.py` | ROS 2 → Arduino serial bridge (`S<speed>,A<angle>`) |
| `hybridnets_deploy/` | HybridNets model, config, TRT conversion, inference code |
| `ackermann_drive/` | Arduino firmware for Yahboom chassis |
| `LANE_FOLLOWING_LOGIC.md` | Detailed lane-following algorithm documentation |
| `INTERSECTION_NAVIGATION.md` | **Intersection navigation system documentation** |
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

The lane follower uses a **two-pass global priority system** built on
HybridNets segmentation and HSV color detection:

### Pass 1 — Detect

Scan 10 horizontal rows (45%–85% of frame height) and classify every
lane-mask cluster by color using HSV analysis:

- **Yellow** → center lane line
- **White / Unknown** → edge lane line

Yellow and white detections are collected **separately** across all rows.

### Pass 2 — Decide (globally, never mixed)

| Priority | Condition | Steering Target | Speed |
|----------|-----------|-----------------|-------|
| 1 (best) | Yellow found in **any** row | Drive to the right of the yellow line (offset ≈ 35–70 px) | cruise (0.20 m/s) |
| 2 (fallback) | **No yellow at all** | Stay left of the rightmost white line | cruise (0.20 m/s) |
| 3 (emergency) | No lane lines at all | Road-mask drivable area, right-biased | reduced (0.06 m/s) |
| 4 (fail) | Nothing visible for 1.5 s | Full stop | 0 m/s |

> **Key design choice:** Yellow and white are **never mixed** in the same
> frame.  If yellow is visible in even a single scan row, all white
> detections are ignored.  This prevents the white edge line from
> contaminating the steering path at corners where yellow curves but
> white goes straight.

### Smoothing & Control

- **Quadratic polyfit** on center points preserves curve shape (degree-2, not linear)
- **Look-ahead weighting** — top scan rows (further ahead) weighted more heavily to anticipate curves
- **Adaptive EMA** — smoothing factor increases on curves for faster response
- **PID controller** — proportional + derivative steering with rate limiter to prevent oscillation
- **Curve speed reduction** — slows proportionally to steering angle (60% at max turn)

### Additional Features

- **Traffic light response** — stops on red, goes on green (currently disabled)
- **Gamepad override** — hold LB for manual control at any time

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
ros2 param set /hybridnets_lane_follower cruise_speed 0.25
ros2 param set /hybridnets_lane_follower kp 1.0
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cruise_speed` | 0.08 | Forward speed on straights (m/s), overridden to 0.20 via launch |
| `min_curve_speed` | 0.16 | Minimum speed on tight curves (m/s) |
| `kp` | 0.8 | Proportional steering gain |
| `kd` | 0.20 | Derivative steering gain (dampens oscillation) |
| `max_angular_z` | 0.8 | Max steering output (rad/s) |
| `max_steer_rate` | 0.30 | Max steering change per frame (prevents oscillation) |
| `curve_slowdown_factor` | 0.6 | Speed reduction on curves (0–1) |
| `lane_width_px` | 200 | Assumed lane width for offset calculation |
| `scan_row_start` | 0.45 | Top of scan region (fraction from top) |
| `scan_row_end` | 0.85 | Bottom of scan region (fraction from top) |
| `scan_num_rows` | 10 | Number of horizontal rows to sample |
| `no_lane_timeout` | 1.5 | Seconds without guidance before stopping |

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
- [x] Two-pass lane following (yellow priority, white fallback)
- [x] Curve speed reduction
- [x] cmd_vel mux with gamepad override
- [x] Live track validation and tuning
- [x] Intersection navigation controller (state machine)
- [x] 3-level cmd_vel mux (teleop > intersection > auto)
- [ ] Intersection maneuver field tuning
- [ ] Traffic light response (re-enable after tuning)
- [ ] Nav2 integration with lane constraints
- [ ] Depth-based obstacle avoidance
