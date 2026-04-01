# Autonomous Robot Perception Stack — Dissertation Research

Real-time perception system for autonomous navigation on **NVIDIA Jetson Orin** using **Isaac ROS** and **Intel RealSense D455**.

## System Architecture

```
┌─────────────┐     ┌──────────┐     ┌──────────┐
│  RealSense  │────▶│  cuVSLAM │────▶│  Nav2    │  (future)
│  D455       │     │ (6-DOF   │     │ (Path    │
│             │     │  Pose)   │     │ Planning)│
│ Stereo IR ──┤     └──────────┘     └──────────┘
│             │
│ RGB ────────┤     ┌──────────┐
│             │────▶│ YOLOv10  │────▶ Detection2DArray
│             │     │ (Object  │
│ Depth ──────┤     │  Det.)   │
│             │     └──────────┘
└─────────────┘
```

## Hardware

| Component | Details |
|---|---|
| **Compute** | NVIDIA Jetson AGX Orin 64GB |
| **Camera** | Intel RealSense D455 (USB 3.2) |
| **Storage** | External NVMe SSD (`/mnt/ssd/`) |
| **Power Mode** | MAXN (`nvpmodel -m 0` + `jetson_clocks`) |

## Software Stack

| Component | Version |
|---|---|
| **ROS 2** | Humble |
| **Isaac ROS** | 3.2 |
| **cuVSLAM** | 12.6 |
| **TensorRT** | 10.3 |
| **Container** | `isaac_ros_dev` (aarch64) |

## Files

| File | Description |
|---|---|
| `vslam_rtdetr_realsense.launch.py` | Combined VSLAM + object detection launch |
| `isaac_ros_visual_slam_realsense_no_imu.launch.py` | VSLAM-only launch (stereo, no IMU) |
| `rtdetr_visualizer_fixed.py` | Detection overlay visualizer for rqt_image_view |
| `setup_packages.sh` | Installs required ROS packages inside container |

## Quick Start

```bash
# 1. Set Jetson to MAX performance (on host)
sudo nvpmodel -m 0 && sudo jetson_clocks

# 2. Launch Isaac ROS container
cd /mnt/ssd/ws/robot_ws
isaac_dev

# 3. Inside container — install packages (first time)
./setup_packages.sh

# 4. Launch the perception pipeline
source /opt/ros/humble/setup.bash
ros2 launch /workspaces/isaac_ros-dev/vslam_rtdetr_realsense.launch.py

# 5. (Optional) Visualize detections — in another container terminal
python3 /workspaces/isaac_ros-dev/rtdetr_visualizer_fixed.py --ros-args \
  -r image:=/camera/color/image_raw \
  -r detections_output:=/detections_output
```

## Performance (MAXN Mode)

| Pipeline | Frame Rate |
|---|---|
| cuVSLAM (stereo IR) | 30 FPS |
| Object Detection | ~30 FPS |
| Combined | Both at 30 FPS ✅ |

## Dependencies (installed via setup_packages.sh)

- `ros-humble-isaac-ros-visual-slam`
- `ros-humble-isaac-ros-rtdetr`
- `ros-humble-isaac-ros-tensor-rt`
- `ros-humble-isaac-ros-dnn-image-encoder`
- `ros-humble-isaac-ros-image-proc`
- `ros-humble-realsense2-camera`
- `ros-humble-isaac-ros-tensor-proc`

## Known Issues

- **RealSense IMU**: HID Motion Sensor failure with firmware 5.17.0.10 — IMU fusion disabled
- **USB reconnect**: Camera occasionally needs physical unplug/replug after I/O errors
- **Container packages**: Packages must be reinstalled after container restart (use `setup_packages.sh`)

## Roadmap

- [x] cuVSLAM stereo tracking at 30 FPS
- [x] RT-DETR object detection pipeline
- [x] Combined VSLAM + Detection launch
- [x] Detection visualization
- [ ] Switch to YOLOv10-S (COCO, 80 classes)
- [ ] Enable IMU fusion (firmware update)
- [ ] Nav2 integration
- [ ] Depth-based costmap
- [ ] Motor control integration
