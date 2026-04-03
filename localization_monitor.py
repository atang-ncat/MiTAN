#!/usr/bin/env python3
"""
Localization monitor for cuVSLAM + slam_toolbox + RPLiDAR.
Monitors visual-inertial odometry, 2D lidar SLAM map, and the full TF tree
to provide real-time feedback on localization health.

Usage (inside isaac_ros_dev container):
  python3 /workspaces/isaac_ros-dev/localization_monitor.py
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener, TransformException
from rclpy.time import Time


def quat_to_yaw(q):
    """Extract yaw (heading) from a quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class LocalizationMonitor(Node):

    EXPECTED_FRAMES = [
        ('map', 'odom'),          # slam_toolbox
        ('odom', 'base_link'),    # cuVSLAM
        ('base_link', 'camera_link'),
        ('base_link', 'laser_frame'),
        ('camera_link', 'camera_infra1_optical_frame'),
    ]

    def __init__(self):
        super().__init__('localization_monitor')

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self._odom_sub = self.create_subscription(
            Odometry, 'visual_slam/tracking/odometry',
            self._odom_cb, sensor_qos)

        self._scan_sub = self.create_subscription(
            LaserScan, '/scan',
            self._scan_cb, sensor_qos)

        self._map_sub = self.create_subscription(
            OccupancyGrid, '/map',
            self._map_cb, map_qos)

        self._origin = None
        self._prev_pos = None
        self._total_distance = 0.0
        self._odom_count = 0
        self._last_odom_time = None
        self._scan_count = 0
        self._map_info = None

        self._pose_timer = self.create_timer(2.0, self._print_pose)
        self._tf_timer = self.create_timer(5.0, self._check_tf_tree)
        self._latest_odom = None

        self.get_logger().info(
            'Localization monitor started — waiting for cuVSLAM + slam_toolbox + RPLiDAR...')

    def _odom_cb(self, msg: Odometry):
        self._latest_odom = msg
        self._odom_count += 1
        now = self.get_clock().now()

        pos = msg.pose.pose.position

        if self._origin is None:
            self._origin = (pos.x, pos.y, pos.z)
            self.get_logger().info(
                f'Origin set at ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})')

        if self._prev_pos is not None:
            dx = pos.x - self._prev_pos[0]
            dy = pos.y - self._prev_pos[1]
            dz = pos.z - self._prev_pos[2]
            self._total_distance += math.sqrt(dx*dx + dy*dy + dz*dz)

        self._prev_pos = (pos.x, pos.y, pos.z)
        self._last_odom_time = now

    def _scan_cb(self, msg: LaserScan):
        self._scan_count += 1

    def _map_cb(self, msg: OccupancyGrid):
        self._map_info = msg.info
        if self._scan_count == 0:
            return
        w = msg.info.width
        h = msg.info.height
        res = msg.info.resolution
        self.get_logger().info(
            f'MAP received: {w}x{h} cells @ {res:.2f}m/cell '
            f'({w * res:.1f}m x {h * res:.1f}m)')

    def _print_pose(self):
        status_parts = []

        if self._latest_odom is None:
            status_parts.append('cuVSLAM: NO DATA')
        else:
            pos = self._latest_odom.pose.pose.position
            ori = self._latest_odom.pose.pose.orientation
            yaw_deg = math.degrees(quat_to_yaw(ori))

            dx = pos.x - self._origin[0]
            dy = pos.y - self._origin[1]
            dz = pos.z - self._origin[2]
            displacement = math.sqrt(dx*dx + dy*dy + dz*dz)

            cov = self._latest_odom.pose.covariance
            pos_cov_trace = cov[0] + cov[7] + cov[14]

            status_parts.append(
                f'POSE x={pos.x:+.3f} y={pos.y:+.3f} z={pos.z:+.3f} '
                f'yaw={yaw_deg:+.1f} deg  '
                f'disp={displacement:.3f}m travel={self._total_distance:.3f}m '
                f'cov={pos_cov_trace:.6f}')

        scan_status = f'lidar={self._scan_count} scans'
        map_status = 'map=NONE'
        if self._map_info is not None:
            w = self._map_info.width
            h = self._map_info.height
            map_status = f'map={w}x{h}'

        status_parts.append(f'{scan_status}  {map_status}')

        for part in status_parts:
            self.get_logger().info(part)

    def _check_tf_tree(self):
        healthy = True
        for parent, child in self.EXPECTED_FRAMES:
            try:
                self._tf_buffer.lookup_transform(
                    parent, child, Time(seconds=0))
            except TransformException:
                self.get_logger().warn(f'TF missing: {parent} -> {child}')
                healthy = False

        if healthy:
            try:
                t = self._tf_buffer.lookup_transform(
                    'map', 'base_link', Time(seconds=0))
                p = t.transform.translation
                self.get_logger().info(
                    f'TF OK  map->base_link: '
                    f'({p.x:+.3f}, {p.y:+.3f}, {p.z:+.3f})')
            except TransformException:
                self.get_logger().warn('TF missing: map -> base_link (composite)')


def main():
    rclpy.init()
    node = LocalizationMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
