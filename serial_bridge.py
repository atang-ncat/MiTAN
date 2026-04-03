#!/usr/bin/env python3
"""
Serial bridge between ROS 2 /cmd_vel and the Arduino Ackermann drive controller.

Subscribes to geometry_msgs/Twist on /cmd_vel, maps linear.x to motor speed
and angular.z to servo angle, then sends the command over USB serial to the
Arduino using the protocol:  S<speed>,A<angle>\n
"""

import serial
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class SerialBridge(Node):

    def __init__(self):
        super().__init__('serial_bridge')

        self.declare_parameter('serial_port', '/dev/ttyACM0')
        self.declare_parameter('baud_rate', 115200)
        self.declare_parameter('max_speed_pwm', 60)
        self.declare_parameter('max_linear_vel', 0.5)
        self.declare_parameter('max_angular_vel', 1.0)
        self.declare_parameter('servo_center', 92)
        self.declare_parameter('servo_min', 20)
        self.declare_parameter('servo_max', 160)

        port = self.get_parameter('serial_port').value
        baud = self.get_parameter('baud_rate').value

        try:
            self.ser = serial.Serial(port, baud, timeout=0.1)
            self.get_logger().info(f'Opened serial port {port} at {baud} baud')
        except serial.SerialException as e:
            self.get_logger().error(f'Cannot open {port}: {e}')
            self.ser = None

        self.create_subscription(Twist, 'cmd_vel', self._cmd_vel_cb, 10)
        self.get_logger().info('Serial bridge ready, waiting for /cmd_vel...')

    def _cmd_vel_cb(self, msg: Twist):
        if self.ser is None or not self.ser.is_open:
            return

        max_pwm = self.get_parameter('max_speed_pwm').value
        max_lin = self.get_parameter('max_linear_vel').value
        max_ang = self.get_parameter('max_angular_vel').value
        servo_center = self.get_parameter('servo_center').value
        servo_min = self.get_parameter('servo_min').value
        servo_max = self.get_parameter('servo_max').value
        lin = max(-max_lin, min(max_lin, msg.linear.x))
        ang = max(-max_ang, min(max_ang, msg.angular.z))

        speed = int(-(lin / max_lin) * max_pwm) if max_lin != 0 else 0
        speed = max(-255, min(255, speed))

        norm_ang = ang / max_ang if max_ang != 0 else 0.0
        if norm_ang > 0:
            angle = int(servo_center - norm_ang * (servo_center - servo_min))
        else:
            angle = int(servo_center - norm_ang * (servo_max - servo_center))
        angle = max(servo_min, min(servo_max, angle))

        cmd = f'S{speed},A{angle}\n'
        try:
            self.ser.write(cmd.encode('ascii'))
        except serial.SerialException as e:
            self.get_logger().warn(f'Serial write failed: {e}')

    def destroy_node(self):
        if self.ser and self.ser.is_open:
            self.ser.write(b'S0,A92\n')  # center steering on shutdown
            self.ser.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = SerialBridge()
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
