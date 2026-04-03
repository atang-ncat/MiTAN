# Teleop launch: joy_node + teleop_twist_joy + serial_bridge
# Drive the Yahboom Ackermann car with an Xbox-style gamepad.
#
# Usage:  ros2 launch teleop.launch.py
# Optional overrides:
#   serial_port:=/dev/ttyACM1  max_speed_pwm:=120

import os
import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')


def generate_launch_description():

    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value='/dev/ttyACM0',
        description='Arduino serial port')
    max_speed_arg = DeclareLaunchArgument(
        'max_speed_pwm', default_value='80',
        description='Maximum motor PWM (0-255)')

    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{
            'dev': '/dev/input/js0',
            'deadzone': 0.05,
            'autorepeat_rate': 20.0,
        }],
        output='screen',
    )

    teleop_twist_joy_node = Node(
        package='teleop_twist_joy',
        executable='teleop_node',
        name='teleop_twist_joy_node',
        parameters=[
            os.path.join(CONFIG_DIR, 'joy_teleop.yaml'),
        ],
        output='screen',
    )

    serial_bridge_proc = launch.actions.ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'serial_bridge.py'),
            '--ros-args',
            '-p', ['serial_port:=', LaunchConfiguration('serial_port')],
            '-p', ['max_speed_pwm:=', LaunchConfiguration('max_speed_pwm')],
        ],
        output='screen',
    )

    return launch.LaunchDescription([
        serial_port_arg,
        max_speed_arg,
        joy_node,
        teleop_twist_joy_node,
        serial_bridge_proc,
    ])
