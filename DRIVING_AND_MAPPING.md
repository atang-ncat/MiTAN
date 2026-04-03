# Driving and Mapping Guide

## Overview

This guide walks through driving the Yahboom Ackermann car with a gamepad
while simultaneously building a 2D occupancy grid map of the environment.

### Mapping Stack

| Component | What it does |
|---|---|
| **RPLiDAR A3** | 360° laser scans at 10 Hz (`/scan`) |
| **cuVSLAM** (Isaac ROS) | Visual odometry from RealSense stereo IR → publishes `odom → base_link` TF |
| **slam_toolbox** | Takes `/scan` + odometry, performs scan matching & loop closure, publishes `map → odom` TF and `/map` occupancy grid |

The map is built using **slam_toolbox** in async mapping mode with the
Ceres solver. It consumes 2D lidar scans and odometry to produce a
5 cm resolution occupancy grid. Loop closure is enabled, so driving back
to your starting point will correct accumulated drift.

---

## Step-by-Step Commands

### Step 0 — Upload Arduino firmware (one-time)

> Skip this if you've already uploaded `ackermann_drive.ino` — the firmware
> persists in flash memory across power cycles and reboots.

1. Open `ackermann_drive.ino` in the Arduino IDE
2. Select **Board → Arduino Mega 2560**
3. Select **Port** → `/dev/ttyACM0` (or whichever port shows up)
4. Click **Upload**
5. Open **Serial Monitor** (115200 baud) and confirm you see `ACKERMANN_READY`

---

### Step 1 — Launch the perception stack (in the container)

Open a terminal **inside the Isaac ROS container**:

```bash
source /opt/ros/humble/setup.bash
source /workspaces/isaac_ros-dev/lidar_ws/install/setup.bash
ros2 launch /workspaces/isaac_ros-dev/vslam_yolov10_realsense.launch.py
```

Wait until you see:
- `RealSense Node Is Up!`
- `cuVSLAM tracker was successfully initialized`
- `Registering sensor: [Custom Described Lidar]`

This starts: RealSense camera, RPLiDAR, cuVSLAM (visual odometry),
slam_toolbox (2D mapping), and YOLOv10 (object detection).

---

### Step 2 — Launch teleop (on the host)

Open a **new terminal on the host** (not the container):

```bash
source /opt/ros/humble/setup.bash
cd /mnt/ssd/ws/robot_ws
ros2 launch teleop.launch.py serial_port:=/dev/ttyACM0
```

> If `/dev/ttyACM0` doesn't work, check the actual port:
> `ls /dev/ttyACM*` and use whichever is listed (e.g., `/dev/ttyACM1`).

This starts: joy_node (gamepad), teleop_twist_joy (deadman + scaling),
and serial_bridge (sends motor/servo commands to the Arduino).

---

### Step 3 — Open RViz2 for visualization (on the host)

Open another **host terminal**:

```bash
source /opt/ros/humble/setup.bash
rviz2
```

In RViz2:
1. Set **Fixed Frame** (top-left) to `map`
2. Click **Add** (bottom-left) and add these displays:
   - **Map** → topic: `/map`
   - **LaserScan** → topic: `/scan`
   - **TF** (shows map → odom → base_link → sensors)
   - **Image** → topic: `/camera/color/image_raw` (optional, live camera)

---

### Step 4 — Drive and build the map

1. **Hold LB** (left bumper) — this is the deadman switch
2. **Left stick forward/back** — drive forward/reverse
3. **Right stick left/right** — steer the front wheels
4. **Release LB** — everything stops immediately

**Tips for a good map:**
- Drive slowly (~0.2 m/s) for best scan matching
- Make a full loop back to your starting position for loop closure
- Avoid sudden sharp turns — gentle arcs work best
- Stay at least 0.3 m from walls to avoid lidar minimum range issues

---

### Step 5 — Save the map (on the host)

Once you've covered the area, save the map:

```bash
source /opt/ros/humble/setup.bash
ros2 run nav2_map_server map_saver_cli -f ~/map
```

This creates two files:
- `~/map.pgm` — the occupancy grid image
- `~/map.yaml` — metadata (resolution, origin, thresholds)

> If `nav2_map_server` is not installed:
> `sudo apt-get install -y ros-humble-nav2-map-server`

---

## Quick Reference

| Where | Command |
|---|---|
| Container | `ros2 launch /workspaces/isaac_ros-dev/vslam_yolov10_realsense.launch.py` |
| Host | `ros2 launch teleop.launch.py serial_port:=/dev/ttyACM0` |
| Host | `rviz2` |
| Host | `ros2 run nav2_map_server map_saver_cli -f ~/map` |

## Gamepad Controls

| Control | Action |
|---|---|
| LB (hold) | Deadman switch — must hold to drive |
| Left stick ↑↓ | Forward / reverse |
| Right stick ←→ | Steer left / right |
| Release LB | Emergency stop |
