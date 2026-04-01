# Combined launch: cuVSLAM (stereo IR) + RT-DETR object detection (RGB)
# For Intel RealSense D455 on Jetson Orin

import launch
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

MODEL_INPUT_SIZE = 640
MODEL_NUM_CHANNELS = 3
ENGINE_FILE_PATH = '/workspaces/isaac_ros-dev/isaac_ros_assets/models/synthetica_detr/sdetr_grasp.plan'


def generate_launch_description():

    # ── RealSense Camera Node ──────────────────────────────────────
    # Streams stereo IR (for VSLAM) + RGB color (for RT-DETR)
    realsense_camera_node = Node(
        name='camera', namespace='',
        package='realsense2_camera', executable='realsense2_camera_node',
        parameters=[{
            'camera_name': 'camera',
            'enable_infra1': True,
            'enable_infra2': True,
            'enable_color': True,
            'enable_depth': False,
            'depth_module.emitter_enabled': 0,
            'depth_module.profile': '640x480x15',
            'rgb_camera.profile': '640x480x15',
            'enable_gyro': False,
            'enable_accel': False,
        }],
    )

    # ── cuVSLAM Node (stereo IR) ──────────────────────────────────
    visual_slam_node = ComposableNode(
        name='visual_slam_node', package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            'enable_image_denoising': False,
            'rectified_images': True,
            'enable_imu_fusion': False,
            'image_jitter_threshold_ms': 200.00,
            'base_frame': 'camera_link',
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
        ],
    )

    # ── RT-DETR Detection Pipeline (RGB) ──────────────────────────
    resize_node = ComposableNode(
        name='resize_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{'input_width': 640, 'input_height': 480,
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
            'scale': False,
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
            'output_tensor_name': 'input_tensor',
            'input_tensor_shape': [MODEL_NUM_CHANNELS, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
            'output_tensor_shape': [1, MODEL_NUM_CHANNELS, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE],
        }],
        remappings=[('tensor', 'planar_tensor')],
    )

    rtdetr_preprocessor_node = ComposableNode(
        name='rtdetr_preprocessor',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
        remappings=[('encoded_tensor', 'reshaped_tensor')],
    )

    tensor_rt_node = ComposableNode(
        name='tensor_rt',
        package='isaac_ros_tensor_rt',
        plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
        parameters=[{
            'model_file_path': '',
            'engine_file_path': ENGINE_FILE_PATH,
            'output_binding_names': ['labels', 'boxes', 'scores'],
            'output_tensor_names': ['labels', 'boxes', 'scores'],
            'input_tensor_names': ['images', 'orig_target_sizes'],
            'input_binding_names': ['images', 'orig_target_sizes'],
            'verbose': False,
            'force_engine_update': False,
        }],
    )

    rtdetr_decoder_node = ComposableNode(
        name='rtdetr_decoder',
        package='isaac_ros_rtdetr',
        plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
        parameters=[{
            'confidence_threshold': 0.3,
        }],
    )

    # ── Container: VSLAM + RT-DETR run in same process ────────────
    combined_container = ComposableNodeContainer(
        name='perception_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            # VSLAM
            visual_slam_node,
            # RT-DETR pipeline
            resize_node,
            pad_node,
            image_format_node,
            image_to_tensor_node,
            interleave_to_planar_node,
            reshape_node,
            rtdetr_preprocessor_node,
            tensor_rt_node,
            rtdetr_decoder_node,
        ],
        output='screen',
    )

    return launch.LaunchDescription([
        combined_container,
        realsense_camera_node,
    ])
