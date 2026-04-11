#!/usr/bin/env python3
"""
Command Velocity Multiplexer for ROS 2

Priority-based mux for /cmd_vel:
  Priority 1 (highest): /cmd_vel_teleop        — gamepad (deadman switch)
  Priority 2:           /cmd_vel_intersection   — intersection maneuver
  Priority 3:           /cmd_vel_auto           — autonomous lane-following

If a teleop command is received within the timeout, it overrides everything.
If an intersection command is received within the timeout, it overrides auto.
Otherwise, autonomous lane-following commands are used.

Output goes to /cmd_vel, which serial_bridge.py reads.

Usage:
    python3 cmd_vel_mux.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time


class CmdVelMux(Node):

    def __init__(self):
        super().__init__('cmd_vel_mux')

        self.declare_parameter('teleop_timeout', 0.5)          # seconds
        self.declare_parameter('intersection_timeout', 0.3)    # seconds

        self.last_teleop_time = 0.0
        self.last_teleop_cmd = Twist()
        self.last_intersection_time = 0.0
        self.last_intersection_cmd = Twist()
        self.last_auto_cmd = Twist()

        # Subscribers
        self.create_subscription(Twist, 'cmd_vel_teleop', self._teleop_cb, 10)
        self.create_subscription(Twist, 'cmd_vel_intersection', self._intersection_cb, 10)
        self.create_subscription(Twist, 'cmd_vel_auto', self._auto_cb, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Timer for output at 20 Hz
        self.create_timer(0.05, self._mux_tick)

        self._active_source = 'auto'  # for logging

        self.get_logger().info('cmd_vel_mux ready (3-level priority)')
        self.get_logger().info('  Priority 1: /cmd_vel_teleop (gamepad)')
        self.get_logger().info('  Priority 2: /cmd_vel_intersection (maneuver)')
        self.get_logger().info('  Priority 3: /cmd_vel_auto (lane following)')
        self.get_logger().info('  Output:     /cmd_vel')

    def _teleop_cb(self, msg: Twist):
        """Teleop always takes priority when active."""
        self.last_teleop_cmd = msg
        self.last_teleop_time = time.time()

    def _intersection_cb(self, msg: Twist):
        """Intersection commands override auto when active."""
        self.last_intersection_cmd = msg
        self.last_intersection_time = time.time()

    def _auto_cb(self, msg: Twist):
        """Autonomous commands are used when nothing else is active."""
        self.last_auto_cmd = msg

    def _mux_tick(self):
        """Publish the highest-priority command."""
        teleop_timeout = self.get_parameter('teleop_timeout').value
        intersection_timeout = self.get_parameter('intersection_timeout').value
        now = time.time()

        if (now - self.last_teleop_time) < teleop_timeout:
            # Teleop is active — use teleop command
            self.cmd_pub.publish(self.last_teleop_cmd)
            if self._active_source != 'teleop':
                self._active_source = 'teleop'
                self.get_logger().info('MUX → TELEOP (gamepad override)')
        elif (now - self.last_intersection_time) < intersection_timeout:
            # Intersection maneuver is active — use intersection command
            self.cmd_pub.publish(self.last_intersection_cmd)
            if self._active_source != 'intersection':
                self._active_source = 'intersection'
                self.get_logger().info('MUX → INTERSECTION (maneuver override)')
        else:
            # No override — use autonomous command
            self.cmd_pub.publish(self.last_auto_cmd)
            if self._active_source != 'auto':
                self._active_source = 'auto'
                self.get_logger().info('MUX → AUTO (lane following)')

    def destroy_node(self):
        # Send stop on shutdown
        try:
            stop = Twist()
            self.cmd_pub.publish(stop)
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = CmdVelMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == '__main__':
    main()
