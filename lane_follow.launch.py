# Lane-Following Launch File
# Launches HybridNets lane follower + RealSense camera + serial bridge + gamepad override
#
# Usage:
#   ros2 launch lane_follow.launch.py
#
# Optional overrides:
#   serial_port:=/dev/ttyACM1
#   cruise_speed:=0.2

import os
import launch
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(SCRIPT_DIR, 'config')
HYBRIDNETS_DIR = os.path.join(SCRIPT_DIR, 'hybridnets_deploy')


def generate_launch_description():

    # ── Launch Arguments ──────────────────────────────────────────
    serial_port_arg = DeclareLaunchArgument(
        'serial_port', default_value='/dev/ttyACM0',
        description='Arduino serial port')
    max_speed_arg = DeclareLaunchArgument(
        'max_speed_pwm', default_value='150',
        description='Maximum motor PWM (0-255). Tuned for lane-following.')
    cruise_speed_arg = DeclareLaunchArgument(
        'cruise_speed', default_value='0.20',
        description='Autonomous cruising speed (m/s)')


    # ── RealSense Camera ──────────────────────────────────────────
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='camera',
        namespace='camera',
        parameters=[{
            'enable_color': True,
            'enable_depth': False,
            'enable_infra1': False,
            'enable_infra2': False,
            'enable_gyro': False,
            'enable_accel': False,
            'rgb_camera.color_profile': '1280x720x30',
            'rgb_camera.color_format': 'RGB8',
        }],
        output='screen',
    )

    # ── HybridNets Lane Follower ─────────────────────────────────────
    lane_follower = launch.actions.ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(SCRIPT_DIR, 'hybridnets_lane_follower.py'),
            '--ros-args',
            '-p', ['cruise_speed:=', LaunchConfiguration('cruise_speed')],
            '-p', 'camera_topic:=/camera/camera/color/image_raw',
        ],
        output='screen',
    )

    # ── Command Velocity Mux ─────────────────────────────────────
    cmd_vel_mux = launch.actions.ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(SCRIPT_DIR, 'cmd_vel_mux.py'),
        ],
        output='screen',
    )

    # ── Gamepad Teleop (override) ─────────────────────────────────
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
        remappings=[
            ('cmd_vel', 'cmd_vel_teleop'),  # publish to mux input, not directly
        ],
        output='screen',
    )

    # ── Serial Bridge ─────────────────────────────────────────────
    serial_bridge = launch.actions.ExecuteProcess(
        cmd=[
            'python3',
            os.path.join(SCRIPT_DIR, 'serial_bridge.py'),
            '--ros-args',
            '-p', ['serial_port:=', LaunchConfiguration('serial_port')],
            '-p', ['max_speed_pwm:=', LaunchConfiguration('max_speed_pwm')],
        ],
        output='screen',
    )

    return launch.LaunchDescription([
        serial_port_arg,
        max_speed_arg,
        cruise_speed_arg,


        LogInfo(msg='\n========================================'),
        LogInfo(msg='  HybridNets Lane Following Stack'),
        LogInfo(msg='  Hold LB for gamepad override'),
        LogInfo(msg='========================================\n'),

        realsense_node,
        lane_follower,
        cmd_vel_mux,
        joy_node,
        teleop_twist_joy_node,
        serial_bridge,
    ])
