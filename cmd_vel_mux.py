#!/usr/bin/env python3
"""
Command Velocity Multiplexer for ROS 2

Priority-based mux for /cmd_vel:
  Priority 1 (highest): /cmd_vel_teleop  — gamepad (deadman switch)
  Priority 2:           /cmd_vel_auto    — autonomous lane-following

If a teleop command is received within the timeout, it overrides autonomous.
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

        self.declare_parameter('teleop_timeout', 0.5)  # seconds

        self.last_teleop_time = 0.0
        self.last_teleop_cmd = Twist()
        self.last_auto_cmd = Twist()

        # Subscribers
        self.create_subscription(Twist, 'cmd_vel_teleop', self._teleop_cb, 10)
        self.create_subscription(Twist, 'cmd_vel_auto', self._auto_cb, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Timer for output at 20 Hz
        self.create_timer(0.05, self._mux_tick)

        self.get_logger().info('cmd_vel_mux ready')
        self.get_logger().info('  Priority 1: /cmd_vel_teleop (gamepad)')
        self.get_logger().info('  Priority 2: /cmd_vel_auto (autonomous)')
        self.get_logger().info('  Output:     /cmd_vel')

    def _teleop_cb(self, msg: Twist):
        """Teleop always takes priority when active."""
        self.last_teleop_cmd = msg
        self.last_teleop_time = time.time()

    def _auto_cb(self, msg: Twist):
        """Autonomous commands are stored and used when teleop is inactive."""
        self.last_auto_cmd = msg

    def _mux_tick(self):
        """Publish the highest-priority command."""
        timeout = self.get_parameter('teleop_timeout').value
        now = time.time()

        if (now - self.last_teleop_time) < timeout:
            # Teleop is active — use teleop command
            self.cmd_pub.publish(self.last_teleop_cmd)
        else:
            # No teleop — use autonomous command
            self.cmd_pub.publish(self.last_auto_cmd)

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
