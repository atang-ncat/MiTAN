# Combined launch: cuVSLAM + YOLOv10 + RPLiDAR + slam_toolbox
# For Intel RealSense D455 + RPLiDAR A3 on Jetson Orin

import os
import launch
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

MODEL_INPUT_SIZE = 640
MODEL_NUM_CHANNELS = 3

ENGINE_FILE_PATH = '/workspaces/isaac_ros-dev/isaac_ros_assets/models/yolov10m/yolov10m.engine'
SLAM_TOOLBOX_PARAMS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'config', 'slam_toolbox_params.yaml')

# ── Sensor mount positions (base_link -> sensor) ──────────────
# Adjust these to match your actual robot geometry.
CAMERA_X = 0.10   # meters forward from base center
CAMERA_Y = 0.0    # meters left (0 = centered)
CAMERA_Z = 0.15   # meters above base_link origin
CAMERA_ROLL = 0.0
CAMERA_PITCH = 0.0
CAMERA_YAW = 0.0

LIDAR_X = 0.0     # meters forward from base center
LIDAR_Y = 0.0     # meters left (0 = centered)
LIDAR_Z = 0.10    # meters above base_link origin
LIDAR_YAW = 0.0   # radians rotation about Z (0 = forward)

def generate_launch_description():

    # ── RealSense Camera Node ──────────────────────────────────────
    # Streams stereo IR (for VSLAM) + RGB color (for YOLOv10) + IMU (fusion)
    realsense_camera_node = Node(
        name='camera', namespace='',
        package='realsense2_camera', executable='realsense2_camera_node',
        parameters=[{
            'camera_name': 'camera',
            'enable_infra1': True,
            'enable_infra2': True,
            'enable_color': True,
            'enable_depth': True,
            'depth_module.emitter_enabled': 0,
            'depth_module.profile': '848x480x30',
            'rgb_camera.profile': '1280x720x30',
            # IMU disabled — JetPack 6 Tegra kernel lacks HID sensor framework.
            # Add external IMU (e.g. BNO055) if visual-inertial fusion is needed.
            'enable_gyro': False,
            'enable_accel': False,
        }],
    )

    # ── Static TF: base_link -> camera_link ───────────────────────
    # Describes where the camera is physically mounted on the robot.
    # Nav2 expects the TF chain: map -> odom -> base_link -> sensors
    base_to_camera_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        arguments=[
            '--x', str(CAMERA_X),
            '--y', str(CAMERA_Y),
            '--z', str(CAMERA_Z),
            '--roll', str(CAMERA_ROLL),
            '--pitch', str(CAMERA_PITCH),
            '--yaw', str(CAMERA_YAW),
            '--frame-id', 'base_link',
            '--child-frame-id', 'camera_link',
        ],
    )

    # ── Static TF: base_link -> laser_frame ──────────────────────
    base_to_lidar_tf = Node(
        package='tf2_ros', executable='static_transform_publisher',
        arguments=[
            '--x', str(LIDAR_X),
            '--y', str(LIDAR_Y),
            '--z', str(LIDAR_Z),
            '--roll', '0.0',
            '--pitch', '0.0',
            '--yaw', str(LIDAR_YAW),
            '--frame-id', 'base_link',
            '--child-frame-id', 'laser_frame',
        ],
    )

    # ── RPLiDAR A3 (sllidar_ros2 — built from source) ────────────
    rplidar_node = Node(
        package='sllidar_ros2', executable='sllidar_node',
        name='sllidar_node',
        parameters=[{
            'channel_type': 'serial',
            'serial_port': '/dev/ttyUSB0',
            'serial_baudrate': 256000,
            'frame_id': 'laser_frame',
            'angle_compensate': True,
            'scan_mode': 'Sensitivity',
        }],
        output='screen',
    )

    # ── cuVSLAM Node (stereo IR + IMU) ────────────────────────────
    visual_slam_node = ComposableNode(
        name='visual_slam_node', package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            'enable_image_denoising': False,
            'rectified_images': True,
            'enable_imu_fusion': False,
            'gyro_noise_density': 0.000244,
            'gyro_random_walk': 0.000019393,
            'accel_noise_density': 0.001862,
            'accel_random_walk': 0.003,
            'image_jitter_threshold_ms': 200.00,
            'base_frame': 'base_link',
            'map_frame': 'map',
            'odom_frame': 'odom',
            'publish_map_to_odom_tf': False,
            'publish_odom_to_base_tf': True,
            'enable_slam_visualization': True,
            'enable_landmarks_view': True,
            'enable_observations_view': True,
            'camera_optical_frames': [
                'camera_infra1_optical_frame',
                'camera_infra2_optical_frame',
            ],
        }],
        remappings=[
            ('visual_slam/image_0', 'camera/infra1/image_rect_raw'),
            ('visual_slam/camera_info_0', 'camera/infra1/camera_info'),
            ('visual_slam/image_1', 'camera/infra2/image_rect_raw'),
            ('visual_slam/camera_info_1', 'camera/infra2/camera_info'),
            ('visual_slam/imu', 'camera/imu'),
        ],
    )

    # ── YOLOv10 Detection Pipeline (RGB) ──────────────────────────
    resize_node = ComposableNode(
        name='resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{'input_width': 1280, 'input_height': 720,
            'output_width': MODEL_INPUT_SIZE,
            'output_height': MODEL_INPUT_SIZE,
            'keep_aspect_ratio': True,
            'encoding_desired': 'rgb8',
            'disable_padding': True,
        }],
        remappings=[
            ('image', 'camera/color/image_raw'),
            ('camera_info', 'camera/color/camera_info'),
        ],
    )

    pad_node = ComposableNode(
        name='pad_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::PadNode',
        parameters=[{
            'output_image_width': MODEL_INPUT_SIZE,
            'output_image_height': MODEL_INPUT_SIZE,
            'padding_type': 'BOTTOM_RIGHT',
        }],
        remappings=[('image', 'resize/image')],
    )

    image_format_node = ComposableNode(
        name='image_format_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ImageFormatConverterNode',
        parameters=[{
            'encoding_desired': 'rgb8',
            'image_width': MODEL_INPUT_SIZE,
            'image_height': MODEL_INPUT_SIZE,
        }],
        remappings=[
            ('image_raw', 'padded_image'),
            ('image', 'image_rgb'),
        ],
    )

    image_to_tensor_node = ComposableNode(
        name='image_to_tensor_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
        parameters=[{
            'scale': True,      # YOLOv10 expects 0-1 values usually. Check training configs.
            'tensor_name': 'image',
        }],
        remappings=[
            ('image', 'image_rgb'),
            ('tensor', 'normalized_tensor'),
        ],
    )

    interleave_to_planar_node = ComposableNode(
        name='interleaved_to_planar_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
        parameters=[{
            'input_tensor_shape': [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, MODEL_NUM_CHANNELS],
        }],
        remappings=[('interleaved_tensor', 'normalized_tensor')],
    )

    reshape_node = ComposableNode(
        name='reshape_node',
        package='isaac_ros_tensor_proc',
        plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
        parameters=[{
            'output_tensor_name': 'images',
            'input_tensor_shape': [MODEL_NUM_CHANNELS, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            'output_tensor_shape': [1, MODEL_NUM_CHANNELS, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        }],
        remappings=[('tensor', 'planar_tensor'),
                    ('reshaped_tensor', 'tensor_pub')],
    )

    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': '',
            'engine_file_path': ENGINE_FILE_PATH,
            'output_binding_names': ['output0'],
            'output_tensor_names': ['output_tensor'],
            'input_tensor_names': ['images'],
            'input_binding_names': ['images'],
            'verbose': False,
            'force_engine_update': False,
        }],
    )

    # ── slam_toolbox (2D lidar SLAM → map->odom TF + /map) ───────
    slam_toolbox_node = Node(
        package='slam_toolbox', executable='async_slam_toolbox_node',
        name='slam_toolbox',
        parameters=[SLAM_TOOLBOX_PARAMS],
        output='screen',
    )

    # ── Container: VSLAM + YOLOv10 run in same process ────────────
    combined_container = ComposableNodeContainer(
        name='perception_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            # VSLAM
            visual_slam_node,
            # YOLOv10 pipeline
            resize_node,
            pad_node,
            image_format_node,
            image_to_tensor_node,
            interleave_to_planar_node,
            reshape_node,
            tensor_rt_node,
        ],
        output='screen',
    )

    return launch.LaunchDescription([
        base_to_camera_tf,
        base_to_lidar_tf,
        rplidar_node,
        slam_toolbox_node,
        combined_container,
        realsense_camera_node,
    ])
