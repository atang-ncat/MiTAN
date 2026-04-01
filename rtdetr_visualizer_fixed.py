#!/usr/bin/env python3
# Optimized RT-DETR visualizer
# Uses independent subscribers + latest-message caching instead of TimeSynchronizer
# This ensures every camera frame gets annotated with the most recent detections

import cv2
import cv_bridge
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray
import threading


class RtDetrVisualizer(Node):
    # Colors for different class IDs
    COLORS = {
        '21': (255, 165, 0),   # orange
        '24': (255, 0, 255),   # magenta
        '25': (0, 255, 0),     # green
        '26': (0, 255, 255),   # cyan
    }
    DEFAULT_COLOR = (0, 255, 0)
    bbox_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    line_type = 2

    def __init__(self):
        super().__init__('rtdetr_visualizer')
        self._bridge = cv_bridge.CvBridge()
        self._lock = threading.Lock()
        self._latest_detections = None

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1)

        self._processed_image_pub = self.create_publisher(
            Image, 'rtdetr_processed_image', 10)

        # Subscribe to detections — just cache the latest
        self._det_sub = self.create_subscription(
            Detection2DArray, 'detections_output',
            self._det_callback, qos)

        # Subscribe to camera image — draw on every frame
        self._img_sub = self.create_subscription(
            Image, 'image',
            self._img_callback, qos)

        self._frame_count = 0
        self.get_logger().info('RT-DETR Visualizer started, waiting for data...')

    def _det_callback(self, msg):
        with self._lock:
            self._latest_detections = msg

    def _img_callback(self, img_msg):
        # Get latest detections (thread-safe)
        with self._lock:
            detections_msg = self._latest_detections

        cv2_img = self._bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        img_h, img_w = cv2_img.shape[:2]

        # Detection bbox coords are in model space (640x640)
        model_size = 640.0
        scale_x = img_w / model_size
        scale_y = img_h / model_size

        num_det = 0
        if detections_msg is not None:
            for detection in detections_msg.detections:
                center_x = detection.bbox.center.position.x * scale_x
                center_y = detection.bbox.center.position.y * scale_y
                width = detection.bbox.size_x * scale_x
                height = detection.bbox.size_y * scale_y
                class_id = detection.results[0].hypothesis.class_id
                score = detection.results[0].hypothesis.score

                color = self.COLORS.get(class_id, self.DEFAULT_COLOR)

                try:
                    min_pt = (round(center_x - width / 2.0),
                              round(center_y - height / 2.0))
                    max_pt = (round(center_x + width / 2.0),
                              round(center_y + height / 2.0))

                    cv2.rectangle(cv2_img, min_pt, max_pt, color, self.bbox_thickness)

                    label = f'cls:{class_id} {score:.2f}'
                    # Background for text
                    (tw, th), _ = cv2.getTextSize(label, self.font, self.font_scale, 1)
                    cv2.rectangle(cv2_img,
                                  (min_pt[0], min_pt[1] - th - 8),
                                  (min_pt[0] + tw + 4, min_pt[1]),
                                  color, -1)
                    cv2.putText(cv2_img, label,
                                (min_pt[0] + 2, min_pt[1] - 5),
                                self.font, self.font_scale, (0, 0, 0), 1)
                    num_det += 1
                except (ValueError, IndexError):
                    pass

        processed_img = self._bridge.cv2_to_imgmsg(cv2_img, encoding='bgr8')
        self._processed_image_pub.publish(processed_img)

        self._frame_count += 1
        if self._frame_count % 30 == 0:
            self.get_logger().info(
                f'Frame {self._frame_count}: {num_det} detections drawn')


def main():
    rclpy.init()
    node = RtDetrVisualizer()
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
