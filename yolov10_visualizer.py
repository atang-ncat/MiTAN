#!/usr/bin/env python3
# YOLOv10 TensorRT output visualizer
# Subscribes to TensorList from Isaac ROS TensorRT node and overlays bounding boxes

import cv2
import cv_bridge
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from isaac_ros_tensor_list_interfaces.msg import TensorList
import numpy as np
import threading


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]


class Yolov10Visualizer(Node):
    DEFAULT_COLOR = (0, 255, 0)
    bbox_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    line_type = 2

    def __init__(self):
        super().__init__('yolov10_visualizer')
        self._bridge = cv_bridge.CvBridge()
        self._lock = threading.Lock()
        self._latest_tensor = None
        self._confidence_threshold = 0.3
        self._model_size = 640

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self._processed_image_pub = self.create_publisher(
            Image, 'yolov10_processed_image', 10)

        # Subscribe to TensorList from TensorRT node 
        self._tensor_sub = self.create_subscription(
            TensorList, 'tensor_sub',
            self._tensor_callback, qos)

        # Subscribe to camera image
        self._img_sub = self.create_subscription(
            Image, '/camera/color/image_raw',
            self._img_callback, qos)

        self._frame_count = 0
        self.get_logger().info('YOLOv10 Visualizer started, waiting for data...')

    def _tensor_callback(self, msg):
        with self._lock:
            self._latest_tensor = msg

    def _img_callback(self, img_msg):
        with self._lock:
            tensor_msg = self._latest_tensor

        cv2_img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        img_h, img_w = cv2_img.shape[:2]

        # The preprocessing pipeline uses keep_aspect_ratio + BOTTOM_RIGHT pad
        # to letterbox the image into model_size x model_size. We need to undo
        # that same transform: find the scale factor used, then map coords back.
        resize_scale = min(
            self._model_size / img_w, self._model_size / img_h)
        inv_scale = 1.0 / resize_scale

        num_det = 0
        if tensor_msg is not None and len(tensor_msg.tensors) > 0:
            output_tensor = tensor_msg.tensors[0]
            try:
                data = np.frombuffer(output_tensor.data, dtype=np.float32)
                dets = data.reshape(-1, 6)

                for det in dets:
                    x1, y1, x2, y2, score, class_id = det

                    if score < self._confidence_threshold:
                        continue

                    class_id = int(class_id)

                    min_x = max(0, int(x1 * inv_scale))
                    min_y = max(0, int(y1 * inv_scale))
                    max_x = min(img_w, int(x2 * inv_scale))
                    max_y = min(img_h, int(y2 * inv_scale))

                    color_idx = class_id * 50
                    color = (
                        (color_idx + 90) % 256,
                        (color_idx + 140) % 256,
                        (color_idx + 200) % 256,
                    )

                    cv2.rectangle(
                        cv2_img, (min_x, min_y), (max_x, max_y),
                        color, self.bbox_thickness)

                    cls_name = (COCO_CLASSES[class_id]
                                if class_id < len(COCO_CLASSES)
                                else str(class_id))
                    label = f'{cls_name} {score:.2f}'
                    (tw, th), _ = cv2.getTextSize(
                        label, self.font, self.font_scale, 1)
                    cv2.rectangle(
                        cv2_img,
                        (min_x, min_y - th - 8),
                        (min_x + tw + 4, min_y),
                        color, -1)
                    cv2.putText(
                        cv2_img, label,
                        (min_x + 2, min_y - 5),
                        self.font, self.font_scale, (0, 0, 0), 1)
                    num_det += 1
            except Exception as e:
                self.get_logger().warn(f'Failed to parse YOLOv10 tensor: {e}')

        processed_img = self._bridge.cv2_to_imgmsg(cv2_img, encoding='bgr8')
        self._processed_image_pub.publish(processed_img)

        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self.get_logger().info(
                f'Frame {self._frame_count}: {num_det} detections drawn')


def main():
    rclpy.init()
    node = Yolov10Visualizer()
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
