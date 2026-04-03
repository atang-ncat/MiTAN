#!/usr/bin/env python3
"""
HybridNets Lane-Following Node for ROS 2

Subscribes to the RealSense RGB camera, runs HybridNets inference (via
ONNX Runtime with TensorRT EP), extracts lane center from segmentation,
and publishes steering commands on /cmd_vel_auto.

Usage (standalone test):
    python3 hybridnets_lane_follower.py --ros-args -p use_tensorrt:=true

Usage (via launch file):
    ros2 launch lane_follow.launch.py
"""

import os
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

# Add the hybridnets_deploy directory to the path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HYBRIDNETS_DIR = os.path.join(SCRIPT_DIR, 'hybridnets_deploy')
sys.path.insert(0, HYBRIDNETS_DIR)

from hybridnets_trt_inference import HybridNetsTRTInference, INPUT_H, INPUT_W

# Default paths
DEFAULT_ENGINE = os.path.join(HYBRIDNETS_DIR, 'weights', 'hybridnets_384x640.engine')
DEFAULT_ONNX = os.path.join(HYBRIDNETS_DIR, 'weights', 'hybridnets_384x640.onnx')
DEFAULT_CONFIG = os.path.join(HYBRIDNETS_DIR, 'config', 'our_dataset.yml')


class LaneFollowerNode(Node):
    """ROS 2 node for HybridNets-based lane following."""

    def __init__(self):
        super().__init__('hybridnets_lane_follower')

        # ── Declare Parameters ───────────────────────────────────────
        self.declare_parameter('engine_path', DEFAULT_ENGINE)
        self.declare_parameter('model_path', DEFAULT_ONNX)
        self.declare_parameter('config_path', DEFAULT_CONFIG)
        self.declare_parameter('use_tensorrt', True)
        self.declare_parameter('conf_thresh', 0.30)

        # Driving parameters
        self.declare_parameter('cruise_speed', 0.15)       # m/s forward speed
        self.declare_parameter('min_curve_speed', 0.05)    # m/s minimum speed on tight curves
        self.declare_parameter('curve_slowdown_factor', 0.6)  # how much to slow on curves (0=no slowdown, 1=full)
        self.declare_parameter('kp', 0.005)                # proportional gain
        self.declare_parameter('ki', 0.0)                  # integral gain
        self.declare_parameter('kd', 0.002)                # derivative gain
        self.declare_parameter('max_angular_z', 0.8)       # max steering clamp
        self.declare_parameter('lane_width_px', 200)       # assumed lane width for single-line fallback
        self.declare_parameter('scan_row_start', 0.55)     # fraction from top — start of scan region
        self.declare_parameter('scan_row_end', 0.85)       # fraction from top — end of scan region
        self.declare_parameter('scan_num_rows', 8)         # number of rows to sample in scan region
        self.declare_parameter('no_lane_timeout', 1.5)     # seconds w/o lane → stop
        self.declare_parameter('road_fallback_speed', 0.08)  # slower speed when using road centroid only
        self.declare_parameter('enabled', True)            # master enable/disable

        # Traffic light parameters
        self.declare_parameter('traffic_light_enabled', True)   # respond to traffic lights
        self.declare_parameter('red_light_hold_time', 2.0)     # seconds to remain stopped after red

        # ── Initialize HybridNets ────────────────────────────────────
        engine_path = self.get_parameter('engine_path').value
        config_path = self.get_parameter('config_path').value
        conf_thresh = self.get_parameter('conf_thresh').value

        # Always use TensorRT engine (pre-built with trtexec)
        if os.path.exists(engine_path):
            self.get_logger().info(f'Loading TensorRT engine: {engine_path}')
            self.engine = HybridNetsTRTInference(
                engine_path=engine_path,
                config_path=config_path,
                conf_thresh=conf_thresh,
            )
            self.get_logger().info('HybridNets TensorRT engine loaded successfully')
        else:
            self.get_logger().error(
                f'TensorRT engine not found: {engine_path}\n'
                f'Run: bash hybridnets_deploy/convert_hybridnets_trt.sh')
            raise FileNotFoundError(f'Engine not found: {engine_path}')

        # ── State ────────────────────────────────────────────────────
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.last_lane_time = time.time()
        self.last_inference_time = time.time()
        self.frame_count = 0
        self.fps = 0.0

        # Traffic light state
        self.traffic_light_state = 'none'  # 'none', 'red', 'green', 'off'
        self.last_red_light_time = 0.0     # when was red last seen
        self.stopped_for_red = False

        # ── Publishers ───────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_auto', 10)
        self.vis_pub = self.create_publisher(Image, 'hybridnets/visualization', 1)
        self.lane_detected_pub = self.create_publisher(Bool, 'hybridnets/lane_detected', 10)

        # ── Subscribers ──────────────────────────────────────────────
        self.create_subscription(
            Image, 'camera/color/image_raw',
            self._image_cb, 1)  # queue=1: drop old frames

        self.get_logger().info('Lane follower node ready. Waiting for camera frames...')

    def _image_cb(self, msg: Image):
        """Process each camera frame."""
        enabled = self.get_parameter('enabled').value
        if not enabled:
            return

        # Convert ROS Image → OpenCV BGR
        img_bgr = self._ros_to_cv2(msg)
        if img_bgr is None:
            return

        # Run HybridNets inference
        detections, road_mask, lane_mask, ratio, dw, dh, dt = self.engine.run(img_bgr)

        # Update FPS
        self.frame_count += 1
        now = time.time()
        elapsed = now - self.last_inference_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_inference_time = now

        # Check traffic lights
        self._update_traffic_light_state(detections, now)

        # Extract lane center and compute steering (with road mask fallback)
        h_orig, w_orig = img_bgr.shape[:2]
        lane_info = self._extract_lane_center(
            lane_mask, road_mask, ratio, dw, dh, w_orig, h_orig)
        error, guidance_found, guidance_source, center_points = lane_info
        self._last_guidance_source = guidance_source

        # Publish lane detection status
        lane_msg = Bool()
        lane_msg.data = guidance_found
        self.lane_detected_pub.publish(lane_msg)

        # Compute and publish steering command
        cmd = Twist()

        # Traffic light override: stop on red
        if self.stopped_for_red:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info('Stopped for RED light', throttle_duration_sec=2.0)
        elif guidance_found:
            self.last_lane_time = now
            if guidance_source == 'road_only':
                # Road centroid fallback — use slower speed
                cmd = self._compute_steering(error)
                cmd.linear.x = self.get_parameter('road_fallback_speed').value
                self.get_logger().info(
                    'Using road centroid (no lane lines)', throttle_duration_sec=2.0)
            else:
                cmd = self._compute_steering(error)
        else:
            # Check timeout
            if (now - self.last_lane_time) > self.get_parameter('no_lane_timeout').value:
                # No guidance at all → stop
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
                self.get_logger().warn('No lane or road detected — stopping',
                                       throttle_duration_sec=2.0)
            else:
                # Brief loss — continue with last steering
                cmd = self._compute_steering(self.prev_error)

        self.cmd_pub.publish(cmd)

        # Publish visualization (throttled to ~10 Hz)
        try:
            if self.vis_pub.get_subscription_count() > 0:
                vis_img = self._draw_visualization(
                    img_bgr, road_mask, lane_mask, ratio, dw, dh,
                    detections, center_points, error, lane_found, cmd)
                self.vis_pub.publish(self._cv2_to_ros(vis_img, msg.header))
        except Exception as e:
            self.get_logger().warn(f'Visualization error: {e}', throttle_duration_sec=5.0)

    def _mask_to_original(self, mask, dw, dh, w_orig, h_orig):
        """Resize a model-output mask back to original image dimensions."""
        resized = cv2.resize(mask, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)
        top = int(round(dh))
        left = int(round(dw))
        bottom = INPUT_H - int(round(dh))
        right = INPUT_W - int(round(dw))
        cropped = resized[top:bottom, left:right]
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            return None
        return cv2.resize(cropped, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    def _extract_lane_center(self, lane_mask, road_mask, ratio, dw, dh, w_orig, h_orig):
        """Extract lane center using a tiered approach:

        Tier 1 (best):   Two lane lines visible → use midpoint between them
        Tier 2 (good):   One lane line visible  → offset by assumed lane width
        Tier 3 (fallback): No lane lines but drivable area visible → use road centroid
        Tier 4 (fail):   Nothing visible → return not found

        Returns:
            error: lateral error in [-1, 1] (positive = car is right of center)
            guidance_found: True if any guidance (lane or road) was found
            guidance_source: 'two_lanes', 'one_lane', 'road_only', or 'none'
            center_points: list of (x, y) center points for visualization
        """
        scan_start = self.get_parameter('scan_row_start').value
        scan_end = self.get_parameter('scan_row_end').value
        num_rows = self.get_parameter('scan_num_rows').value
        lane_width_fallback = self.get_parameter('lane_width_px').value

        image_center_x = w_orig / 2.0

        # Convert masks to original image dimensions
        lane_orig = self._mask_to_original(lane_mask, dw, dh, w_orig, h_orig)
        road_orig = self._mask_to_original(road_mask, dw, dh, w_orig, h_orig)

        # ── Tier 1 & 2: Scan for lane lines ──────────────────────────
        center_points = []
        has_two_lanes = False

        if lane_orig is not None:
            row_start = int(h_orig * scan_start)
            row_end = int(h_orig * scan_end)
            row_step = max(1, (row_end - row_start) // num_rows)

            for y in range(row_start, row_end, row_step):
                row = lane_orig[y, :]
                lane_pixels = np.where(row > 0)[0]

                if len(lane_pixels) < 3:
                    continue

                # Split by gap to identify left vs right lane line
                gaps = np.diff(lane_pixels)
                gap_threshold = w_orig * 0.15
                large_gaps = np.where(gaps > gap_threshold)[0]

                if len(large_gaps) >= 1:
                    # Tier 1: Two lane lines found
                    split_idx = large_gaps[np.argmax(gaps[large_gaps])]
                    left_pixels = lane_pixels[:split_idx + 1]
                    right_pixels = lane_pixels[split_idx + 1:]
                    left_edge = np.median(left_pixels)
                    right_edge = np.median(right_pixels)
                    center_x = (left_edge + right_edge) / 2.0
                    has_two_lanes = True
                else:
                    # Tier 2: Single lane line — offset by lane width
                    cluster_center = np.median(lane_pixels)
                    if cluster_center < image_center_x:
                        center_x = cluster_center + lane_width_fallback / 2.0
                    else:
                        center_x = cluster_center - lane_width_fallback / 2.0

                center_points.append((int(center_x), y))

        if len(center_points) > 0:
            # Got lane-based guidance — compute weighted average
            weights = np.array([i + 1 for i in range(len(center_points))], dtype=np.float32)
            cx_values = np.array([p[0] for p in center_points], dtype=np.float32)
            avg_center_x = np.average(cx_values, weights=weights)

            # Validate against road mask: clamp center to drivable area
            if road_orig is not None:
                scan_y = center_points[-1][1]  # bottom-most scan row
                road_row = road_orig[scan_y, :]
                road_pixels = np.where(road_row > 0)[0]
                if len(road_pixels) > 10:
                    road_left = road_pixels[0]
                    road_right = road_pixels[-1]
                    avg_center_x = np.clip(avg_center_x, road_left + 10, road_right - 10)

            error = (avg_center_x - image_center_x) / (w_orig / 2.0)
            error = float(np.clip(error, -1.0, 1.0))
            source = 'two_lanes' if has_two_lanes else 'one_lane'
            return error, True, source, center_points

        # ── Tier 3: Road centroid fallback ────────────────────────────
        if road_orig is not None:
            row_start = int(h_orig * scan_start)
            row_end = int(h_orig * scan_end)
            row_step = max(1, (row_end - row_start) // max(1, num_rows // 2))

            road_centers = []
            for y in range(row_start, row_end, row_step):
                road_row = road_orig[y, :]
                road_pixels = np.where(road_row > 0)[0]
                if len(road_pixels) > 20:
                    # Road centroid for this row
                    road_cx = (road_pixels[0] + road_pixels[-1]) / 2.0
                    road_centers.append((int(road_cx), y))

            if len(road_centers) > 0:
                weights = np.array([i + 1 for i in range(len(road_centers))], dtype=np.float32)
                cx_values = np.array([p[0] for p in road_centers], dtype=np.float32)
                avg_center_x = np.average(cx_values, weights=weights)

                error = (avg_center_x - image_center_x) / (w_orig / 2.0)
                error = float(np.clip(error, -1.0, 1.0))
                return error, True, 'road_only', road_centers

        # ── Tier 4: Nothing found ────────────────────────────────────
        return 0.0, False, 'none', []

    def _update_traffic_light_state(self, detections, now):
        """Check detections for traffic lights and update state."""
        if not self.get_parameter('traffic_light_enabled').value:
            self.stopped_for_red = False
            return

        red_hold = self.get_parameter('red_light_hold_time').value

        # Find the highest-confidence traffic light detection
        best_light = None
        best_score = 0.0
        for det in detections:
            if det['class'].startswith('traffic_light_'):
                if det['score'] > best_score:
                    best_score = det['score']
                    best_light = det['class']

        if best_light == 'traffic_light_red':
            self.traffic_light_state = 'red'
            self.last_red_light_time = now
            self.stopped_for_red = True
            self.get_logger().info(
                f'RED light detected ({best_score:.0%})', throttle_duration_sec=1.0)

        elif best_light == 'traffic_light_green':
            self.traffic_light_state = 'green'
            self.stopped_for_red = False
            self.get_logger().info(
                f'GREEN light — go! ({best_score:.0%})', throttle_duration_sec=1.0)

        elif best_light == 'traffic_light_off':
            self.traffic_light_state = 'off'
            # Treat off as caution — don't change stopped state

        else:
            # No traffic light detected
            self.traffic_light_state = 'none'
            # If we were stopped for red, stay stopped for hold_time
            if self.stopped_for_red:
                if (now - self.last_red_light_time) > red_hold:
                    # Red light no longer visible and hold time expired → cautiously proceed
                    self.stopped_for_red = False
                    self.get_logger().info('Red light cleared — proceeding')

    def _compute_steering(self, error):
        """PID controller: lateral error → Twist command with curve speed reduction."""
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        max_ang = self.get_parameter('max_angular_z').value
        cruise_speed = self.get_parameter('cruise_speed').value
        min_curve_speed = self.get_parameter('min_curve_speed').value
        curve_slowdown = self.get_parameter('curve_slowdown_factor').value

        # PID
        d_error = error - self.prev_error
        self.integral_error += error
        self.integral_error = np.clip(self.integral_error, -100.0, 100.0)

        steering = -(kp * error + ki * self.integral_error + kd * d_error)
        steering = np.clip(steering, -max_ang, max_ang)

        self.prev_error = error

        # Curve speed reduction: slow down proportionally to steering angle
        # steering_ratio = 0 when straight, 1 when at max steering
        steering_ratio = abs(float(steering)) / max_ang
        speed = cruise_speed * (1.0 - curve_slowdown * steering_ratio)
        speed = max(min_curve_speed, speed)

        cmd = Twist()
        cmd.linear.x = speed
        cmd.angular.z = float(steering)
        return cmd

    def _draw_visualization(self, img_bgr, road_mask, lane_mask, ratio, dw, dh,
                            detections, center_points, error, lane_found, cmd):
        """Draw annotated visualization image."""
        SEG_COLORS = {'road': (255, 0, 255), 'lane': (0, 255, 0)}

        h, w = img_bgr.shape[:2]
        result = img_bgr.copy()

        # Draw segmentation overlay (semi-transparent)
        road_resized = cv2.resize(road_mask, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)
        lane_resized = cv2.resize(lane_mask, (INPUT_W, INPUT_H), interpolation=cv2.INTER_NEAREST)

        top = int(round(dh))
        left = int(round(dw))
        bottom = INPUT_H - int(round(dh))
        right = INPUT_W - int(round(dw))

        if bottom > top and right > left:
            road_cropped = road_resized[top:bottom, left:right]
            lane_cropped = lane_resized[top:bottom, left:right]
            road_orig = cv2.resize(road_cropped, (w, h), interpolation=cv2.INTER_NEAREST)
            lane_orig = cv2.resize(lane_cropped, (w, h), interpolation=cv2.INTER_NEAREST)

            overlay = result.copy()
            overlay[road_orig == 1] = SEG_COLORS['road']
            overlay[lane_orig == 1] = SEG_COLORS['lane']
            result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)

        # Draw lane center points
        for i, (cx, cy) in enumerate(center_points):
            cv2.circle(result, (cx, cy), 6, (0, 255, 255), -1)  # yellow dots
            if i > 0:
                px, py = center_points[i - 1]
                cv2.line(result, (px, py), (cx, cy), (0, 255, 255), 2)

        # Draw image center line
        cv2.line(result, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)

        # Draw lane center target line
        if center_points:
            avg_cx = int(np.mean([p[0] for p in center_points]))
            cv2.line(result, (avg_cx, int(h * 0.5)), (avg_cx, int(h * 0.9)),
                     (0, 255, 255), 3)

        # Status text with guidance source
        source_info = getattr(self, '_last_guidance_source', 'none')
        if lane_found:
            if source_info == 'two_lanes':
                status_text = 'LANE TRACKING (2 lines)'
                status_color = (0, 255, 0)
            elif source_info == 'one_lane':
                status_text = 'LANE TRACKING (1 line)'
                status_color = (0, 200, 255)
            elif source_info == 'road_only':
                status_text = 'ROAD CENTROID (fallback)'
                status_color = (255, 0, 255)
            else:
                status_text = 'LANE TRACKING'
                status_color = (0, 255, 0)
        else:
            status_text = 'NO GUIDANCE'
            status_color = (0, 0, 255)
        cv2.putText(result, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Traffic light status
        tl_colors = {
            'red': (0, 0, 255), 'green': (0, 255, 0),
            'off': (128, 128, 128), 'none': (200, 200, 200)
        }
        tl_color = tl_colors.get(self.traffic_light_state, (200, 200, 200))
        tl_text = f'TL: {self.traffic_light_state.upper()}'
        if self.stopped_for_red:
            tl_text += ' [STOPPED]'
        cv2.putText(result, tl_text, (w - 300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, tl_color, 2)

        cv2.putText(result, f'FPS: {self.fps:.1f}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f'Error: {error:.3f}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(result, f'Steer: {cmd.angular.z:.3f}', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(result, f'Speed: {cmd.linear.x:.2f} m/s', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw detection boxes for traffic lights
        for det in detections:
            if det['class'].startswith('traffic_light_'):
                box = det['box']
                x1 = int((box[0] - dw) / ratio)
                y1 = int((box[1] - dh) / ratio)
                x2 = int((box[2] - dw) / ratio)
                y2 = int((box[3] - dh) / ratio)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                det_color = (0, 0, 255) if 'red' in det['class'] else \
                            (0, 255, 0) if 'green' in det['class'] else (128, 128, 128)
                cv2.rectangle(result, (x1, y1), (x2, y2), det_color, 3)
                cv2.putText(result, f"{det['class'].replace('traffic_light_', 'TL:')}: {det['score']:.0%}",
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, det_color, 2)

        # Steering direction indicator
        bar_cx = w // 2
        bar_y = h - 40
        bar_w = int(error * (w // 4))
        cv2.rectangle(result, (bar_cx, bar_y - 10), (bar_cx + bar_w, bar_y + 10),
                      (0, 165, 255), -1)
        cv2.rectangle(result, (bar_cx - w // 4, bar_y - 12),
                      (bar_cx + w // 4, bar_y + 12), (255, 255, 255), 1)

        return result

    # ── ROS ↔ OpenCV conversion ──────────────────────────────────────

    def _ros_to_cv2(self, msg: Image):
        """Convert sensor_msgs/Image to OpenCV BGR array."""
        try:
            if msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                return np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3)
            elif msg.encoding == 'mono8':
                return np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width)
            else:
                self.get_logger().warn(
                    f'Unsupported encoding: {msg.encoding}', throttle_duration_sec=5.0)
                return None
        except Exception as e:
            self.get_logger().warn(f'Image conversion error: {e}', throttle_duration_sec=5.0)
            return None

    def _cv2_to_ros(self, img_bgr, header):
        """Convert OpenCV BGR array to sensor_msgs/Image."""
        msg = Image()
        msg.header = header
        msg.height, msg.width = img_bgr.shape[:2]
        msg.encoding = 'bgr8'
        msg.step = msg.width * 3
        msg.data = img_bgr.tobytes()
        return msg

    def destroy_node(self):
        """Cleanup: publish stop command."""
        try:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)
            self.get_logger().info('Lane follower shutting down — stop command sent')
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = LaneFollowerNode()
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
