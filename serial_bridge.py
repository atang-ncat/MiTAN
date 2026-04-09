#!/usr/bin/env python3
"""
Serial bridge between ROS 2 /cmd_vel and the Arduino Ackermann drive controller.
DEBUG VERSION — reads and logs all Arduino serial output.

Subscribes to geometry_msgs/Twist on /cmd_vel, maps linear.x to motor speed
and angular.z to servo angle, then sends the command over USB serial to the
Arduino using the protocol:  S<speed>,A<angle>\n
"""

import serial
import threading
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

        self._stop_reader = False

        try:
            import time as _time
            self.ser = serial.Serial()
            self.ser.port = port
            self.ser.baudrate = baud
            self.ser.timeout = 0.1
            self.ser.dtr = False   # Don't reset Arduino on connect
            self.ser.rts = False   # Don't assert RTS (may interfere with motor driver)
            self.ser.open()

            # Log actual DTR/RTS state after open
            self.get_logger().info(
                f'Opened serial port {port} at {baud} baud '
                f'(DTR={self.ser.dtr}, RTS={self.ser.rts}, '
                f'DSR={self.ser.dsr}, CTS={self.ser.cts})')

            _time.sleep(3.0)       # Wait for Arduino boot + self-test

            # Drain any buffered Arduino output from boot
            self.get_logger().info('--- Arduino boot output ---')
            while self.ser.in_waiting:
                line = self.ser.readline().decode('ascii', errors='replace').strip()
                if line:
                    self.get_logger().info(f'[ARDUINO] {line}')
            self.get_logger().info('--- end boot output ---')

        except serial.SerialException as e:
            self.get_logger().error(f'Cannot open {port}: {e}')
            self.ser = None

        # Start background thread to read Arduino debug output
        if self.ser and self.ser.is_open:
            self._reader_thread = threading.Thread(
                target=self._serial_reader, daemon=True)
            self._reader_thread.start()

        self.create_subscription(Twist, 'cmd_vel', self._cmd_vel_cb, 10)
        self.get_logger().info('Serial bridge ready, waiting for /cmd_vel...')

        self._cmd_count = 0

    def _serial_reader(self):
        """Background thread: continuously read and log Arduino serial output."""
        while not self._stop_reader:
            try:
                if self.ser and self.ser.is_open and self.ser.in_waiting:
                    line = self.ser.readline().decode('ascii', errors='replace').strip()
                    if line:
                        self.get_logger().info(f'[ARDUINO] {line}')
            except serial.SerialException as e:
                self.get_logger().warn(f'Serial read error: {e}')
                break
            except Exception:
                pass

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

        # Arduino: positive speed → FORWARD_DIR
        speed = int((lin / max_lin) * max_pwm) if max_lin != 0 else 0
        speed = max(-255, min(255, speed))

        norm_ang = ang / max_ang if max_ang != 0 else 0.0
        if norm_ang > 0:
            angle = int(servo_center - norm_ang * (servo_center - servo_min))
        else:
            angle = int(servo_center - norm_ang * (servo_max - servo_center))
        angle = max(servo_min, min(servo_max, angle))

        cmd = f'S{speed},A{angle}\n'
        cmd_bytes = cmd.encode('ascii')

        self._cmd_count += 1

        # Log every command with full detail
        self.get_logger().info(
            f'[TX #{self._cmd_count}] lin={msg.linear.x:.3f} ang={msg.angular.z:.3f} '
            f'→ {cmd.strip()} (bytes: {cmd_bytes.hex()})',
            throttle_duration_sec=1.0)

        try:
            written = self.ser.write(cmd_bytes)
            if written != len(cmd_bytes):
                self.get_logger().warn(
                    f'Partial write: {written}/{len(cmd_bytes)} bytes')
        except serial.SerialException as e:
            self.get_logger().warn(f'Serial write failed: {e}')

    def destroy_node(self):
        self._stop_reader = True
        if self.ser and self.ser.is_open:
            self.get_logger().info('Shutdown: sending S0,A92 stop command')
            self.ser.write(b'S0,A92\n')
            import time
            time.sleep(0.1)
            # Read any last Arduino output
            while self.ser.in_waiting:
                line = self.ser.readline().decode('ascii', errors='replace').strip()
                if line:
                    self.get_logger().info(f'[ARDUINO] {line}')
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
