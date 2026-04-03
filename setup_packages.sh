#!/bin/bash
set -e

sudo apt-get update && sudo apt-get install -y \
    ros-humble-isaac-ros-visual-slam \
    ros-humble-isaac-ros-tensor-rt \
    ros-humble-isaac-ros-dnn-image-encoder \
    ros-humble-isaac-ros-image-proc \
    ros-humble-realsense2-camera \
    ros-humble-isaac-ros-tensor-proc \
    ros-humble-slam-toolbox \
    ros-humble-joy \
    ros-humble-teleop-twist-joy \
    python3-serial

# ── 1. sllidar_ros2 from source (apt rplidar_ros has buffer overflow on aarch64) ──
LIDAR_WS="/workspaces/isaac_ros-dev/lidar_ws"
if [ ! -d "$LIDAR_WS/install" ]; then
    echo "=== Building sllidar_ros2 from source ==="
    mkdir -p "$LIDAR_WS/src"
    cd "$LIDAR_WS/src"
    if [ ! -d sllidar_ros2 ]; then
        git clone --depth 1 https://github.com/Slamtec/sllidar_ros2.git
    fi
    cd "$LIDAR_WS"
    source /opt/ros/humble/setup.bash
    colcon build --symlink-install
    echo "sllidar_ros2 built successfully"
else
    echo "sllidar_ros2 already built at $LIDAR_WS"
fi

echo ""
echo "=== Done. Source these before launching: ==="
echo "  source /opt/ros/humble/setup.bash"
echo "  source /workspaces/isaac_ros-dev/lidar_ws/install/setup.bash"
