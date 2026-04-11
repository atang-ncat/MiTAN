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
from std_msgs.msg import Bool, String, Float32

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
        self.declare_parameter('cruise_speed', 0.08)       # m/s forward speed
        self.declare_parameter('min_curve_speed', 0.16)    # m/s minimum speed on tight curves (PWM ~48 at max_pwm=150)
        self.declare_parameter('curve_slowdown_factor', 0.6)  # how much to slow on curves (0=no slowdown, 1=full)
        self.declare_parameter('kp', 0.8)                  # proportional gain
        self.declare_parameter('ki', 0.0)                  # integral gain
        self.declare_parameter('kd', 0.20)                 # derivative gain
        self.declare_parameter('max_angular_z', 0.8)       # max steering clamp
        self.declare_parameter('max_steer_rate', 0.30)     # max steering change per frame (rad/s) — prevents oscillation
        self.declare_parameter('curvature_gain', 0.4)      # how much road curvature biases the steering error
        self.declare_parameter('lane_width_px', 200)       # assumed lane width for single-line fallback
        self.declare_parameter('max_road_width_px', 350)   # max expected road width in pixels — caps road mask to reject floor
        self.declare_parameter('max_dashed_line_width_px', 20)  # cluster extent above this → solid bold line, not dashed
        self.declare_parameter('scan_row_start', 0.45)     # fraction from top — start of scan region (look further ahead)
        self.declare_parameter('scan_row_end', 0.85)       # fraction from top — end of scan region
        self.declare_parameter('scan_num_rows', 10)        # number of rows to sample in scan region
        self.declare_parameter('no_lane_timeout', 1.5)     # seconds w/o lane → stop
        self.declare_parameter('road_fallback_speed', 0.06)  # slower speed when using road-only fallback
        self.declare_parameter('road_lane_bias', 0.65)       # 0.5=road center, 0.65=right lane bias
        self.declare_parameter('enabled', True)            # master enable/disable
        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')

        # Traffic light parameters (disabled for now — focus on lane following)
        self.declare_parameter('traffic_light_enabled', False)
        self.declare_parameter('red_light_hold_time', 2.0)

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
        self.smoothed_error = 0.0          # EMA-smoothed lateral error
        self.error_smooth_alpha = 0.35     # base EMA factor (adaptive: goes higher on curves)
        self.prev_steering = 0.0           # for steering rate limiter
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
        self.guidance_source_pub = self.create_publisher(String, 'hybridnets/guidance_source', 10)
        self.road_width_pub = self.create_publisher(Float32, 'hybridnets/road_width', 10)

        # ── Subscribers ──────────────────────────────────────────────
        camera_topic = self.get_parameter('camera_topic').value
        self.create_subscription(
            Image, camera_topic,
            self._image_cb, 1)  # queue=1: drop old frames
        self.get_logger().info(f'Subscribed to camera topic: {camera_topic}')

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

        # Downscale to 640x360 before inference (halves preprocess cost)
        h_full, w_full = img_bgr.shape[:2]
        if w_full > 800:
            img_small = cv2.resize(img_bgr, (640, 360), interpolation=cv2.INTER_LINEAR)
        else:
            img_small = img_bgr

        # Run HybridNets inference
        detections, road_mask, lane_mask, ratio, dw, dh, dt = self.engine.run(img_small)

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
        h_orig, w_orig = img_small.shape[:2]
        lane_info = self._extract_lane_center(
            img_small, lane_mask, road_mask, ratio, dw, dh, w_orig, h_orig)
        raw_error, guidance_found, guidance_source, center_points = lane_info
        self._last_guidance_source = guidance_source

        # Adaptive EMA: respond fast when error changes a lot (curves),
        # stay smooth when error is stable (straights).
        if guidance_found:
            change = abs(raw_error - self.smoothed_error)
            alpha = min(0.85, self.error_smooth_alpha + change * 3.0)
            self.smoothed_error = alpha * raw_error + (1.0 - alpha) * self.smoothed_error
        error = self.smoothed_error

        # Publish lane detection status
        lane_msg = Bool()
        lane_msg.data = guidance_found
        self.lane_detected_pub.publish(lane_msg)

        # Publish guidance source for intersection controller
        src_msg = String()
        src_msg.data = guidance_source
        self.guidance_source_pub.publish(src_msg)

        # Compute and publish steering command
        cmd = Twist()

        # Traffic light override: stop on red
        if self.stopped_for_red:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info('Stopped for RED light', throttle_duration_sec=2.0)
        elif guidance_found:
            self.last_lane_time = now
            if guidance_source == 'road_right_lane':
                cmd = self._compute_steering(error)
                cmd.linear.x = self.get_parameter('road_fallback_speed').value
                self.get_logger().info(
                    'Road right-lane fallback (no lane lines)', throttle_duration_sec=2.0)
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

        # Publish visualization (throttled to ~4 Hz to save CPU)
        try:
            self._vis_counter = getattr(self, '_vis_counter', 0) + 1
            if self.vis_pub.get_subscription_count() > 0 and self._vis_counter % 5 == 0:
                vis_img = self._draw_visualization(
                    img_small, road_mask, lane_mask, ratio, dw, dh,
                    detections, center_points, error, guidance_found, cmd)
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

    def _classify_lane_color(self, img_hsv, pixel_xs, y,
                             max_line_width=0):
        """Classify a cluster of lane pixels by colour and line type.

        Returns one of:
            'yellow'      — center line (solid yellow)
            'white'       — dashed/thin white lane marking (true boundary)
            'white_bold'  — solid wide white line/curb (NOT a lane boundary;
                            following it at corners leads off-track)
            'unknown'     — can't determine colour

        When *max_line_width* > 0, a white cluster whose spatial extent
        exceeds the threshold is classified as 'white_bold' instead of
        'white'.  This prevents the car from steering towards thick curb
        edges or crosswalk markings that go straight at corners while the
        actual dashed lane line curves with the road.
        """
        if img_hsv is None or len(pixel_xs) == 0:
            return 'unknown'

        step = max(1, len(pixel_xs) // 10)
        samples = pixel_xs[::step]

        h_vals = img_hsv[y, samples, 0].astype(np.float32)
        s_vals = img_hsv[y, samples, 1].astype(np.float32)
        v_vals = img_hsv[y, samples, 2].astype(np.float32)

        avg_h = np.mean(h_vals)
        avg_s = np.mean(s_vals)
        avg_v = np.mean(v_vals)

        if avg_s > 80 and 15 < avg_h < 40 and avg_v > 80:
            return 'yellow'
        if avg_s < 60 and avg_v > 140:
            extent = int(pixel_xs[-1]) - int(pixel_xs[0])
            if max_line_width > 0 and extent > max_line_width:
                return 'white_bold'
            return 'white'
        return 'unknown'

    def _is_dashed_line(self, lane_orig, cluster_x_center, y, window=40):
        """Check if a lane line at (cluster_x_center, y) is dashed.

        Scans a vertical strip around the x-position and counts how
        many rows have lane-mask pixels.  Dashed lines have physical
        gaps → intermittent pixels; solid/bold lines are continuous.

        Returns True if dashed (fill_ratio < 0.5), False if solid.
        """
        if lane_orig is None:
            return False

        h, w = lane_orig.shape[:2]
        x_c = int(cluster_x_center)
        x_left = max(0, x_c - 5)
        x_right = min(w, x_c + 6)

        y_start = max(0, y - window)
        y_end = min(h, y + window + 1)

        total_rows = y_end - y_start
        if total_rows < 10:
            return False

        strip = lane_orig[y_start:y_end, x_left:x_right]
        rows_with_pixels = int(np.any(strip > 0, axis=1).sum())
        fill_ratio = rows_with_pixels / total_rows
        return fill_ratio < 0.5

    def _extract_lane_center(self, img_bgr, lane_mask, road_mask,
                             ratio, dw, dh, w_orig, h_orig):
        """Two-pass lane following: yellow line ONLY, white line ONLY as fallback.

        Pass 1: Scan all rows, collect yellow and white line positions separately.
        Pass 2: Decide GLOBALLY which source to use:
            - If yellow was found in ANY row → use ONLY yellow-based points
            - If yellow was found in ZERO rows → use right white line
            - If neither → road mask fallback

        This prevents white-line points from contaminating the path when
        yellow is visible but intermittent (e.g. at corners).

        Returns (error, guidance_found, guidance_source, center_points).
        """
        scan_start = self.get_parameter('scan_row_start').value
        scan_end = self.get_parameter('scan_row_end').value
        num_rows = self.get_parameter('scan_num_rows').value
        lane_half_w = self.get_parameter('lane_width_px').value / 2.0
        max_road_w = self.get_parameter('max_road_width_px').value
        road_bias = self.get_parameter('road_lane_bias').value

        image_cx = w_orig / 2.0

        # ── Map masks to original image space ────────────────────
        lane_orig = self._mask_to_original(lane_mask, dw, dh, w_orig, h_orig)
        road_orig = self._mask_to_original(road_mask, dw, dh, w_orig, h_orig)

        # ── HSV for yellow detection ─────────────────────────────
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # Direct HSV yellow mask (catches yellow even if lane mask misses it)
        yellow_hsv_mask = cv2.inRange(
            img_hsv,
            np.array([15, 80, 80], dtype=np.uint8),
            np.array([40, 255, 255], dtype=np.uint8))
        yellow_hsv_mask = cv2.morphologyEx(
            yellow_hsv_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # ── Viz storage ──────────────────────────────────────────
        self._viz_yellow_pts = []
        self._viz_dashed_white_pts = []
        self._viz_solid_white_pts = []

        row_start = int(h_orig * scan_start)
        row_end = int(h_orig * scan_end)
        row_step = max(1, (row_end - row_start) // num_rows)

        # ══════════════════════════════════════════════════════════
        # PASS 1: Scan all rows — collect yellow and white separately
        # ══════════════════════════════════════════════════════════
        yellow_rows = []   # list of (yellow_x, y)
        white_rows = []    # list of (rightmost_white_x, y)

        for y in range(row_start, row_end, row_step):
            # ── Gather lane-mask pixels at this row ──────────
            if lane_orig is not None:
                lane_px = np.where(lane_orig[y, :] > 0)[0]
            else:
                lane_px = np.array([], dtype=int)

            # Also gather HSV-yellow pixels (union with lane mask)
            yellow_px = np.where(yellow_hsv_mask[y, :] > 0)[0]
            if len(yellow_px) >= 3 and len(lane_px) > 0:
                combined = np.union1d(lane_px, yellow_px)
            elif len(yellow_px) >= 3:
                combined = yellow_px
            else:
                combined = lane_px

            # ── Cluster into separate line segments ──────────
            clusters = []
            if len(combined) >= 2:
                gaps = np.diff(combined)
                splits = np.where(gaps > 15)[0]
                raw_cls = (np.split(combined, splits + 1)
                           if len(splits) else [combined])
                clusters = [cl for cl in raw_cls if len(cl) >= 2]

            # ── Classify each cluster: yellow or white ───────
            yellow_x = None
            white_xs = []

            for cl in clusters:
                cl_center = float(np.median(cl))
                color = self._classify_lane_color(img_hsv, cl, y)

                if color == 'yellow':
                    yellow_x = cl_center
                    self._viz_yellow_pts.append((int(cl_center), y))
                else:
                    white_xs.append(cl_center)
                    self._viz_solid_white_pts.append((int(cl_center), y))

            # Store results per row
            if yellow_x is not None:
                yellow_rows.append((yellow_x, y))

            if white_xs:
                right_white_x = max(white_xs)
                white_rows.append((right_white_x, y))

        # ══════════════════════════════════════════════════════════
        # PASS 2: Decide GLOBALLY — yellow or white (never mix)
        # ══════════════════════════════════════════════════════════
        center_points = []
        dominant_src = 'none'

        if len(yellow_rows) > 0:
            # ── YELLOW FOUND → use ONLY yellow, ignore white entirely ──
            dominant_src = 'yellow_line'
            for (yx, y) in yellow_rows:
                depth_ratio = (y - row_start) / max(1, row_end - row_start)
                # Offset: drive to the right of yellow, centered in lane
                offset = lane_half_w * (0.40 + 0.30 * depth_ratio)
                center_x = yx + offset
                center_points.append((int(center_x), y))

        elif len(white_rows) > 0:
            # ── NO YELLOW AT ALL → use right white line as boundary ──
            dominant_src = 'white_line'
            for (wx, y) in white_rows:
                depth_ratio = (y - row_start) / max(1, row_end - row_start)
                # Larger offset: stay well to the left of the white edge
                offset = lane_half_w * (0.3 + 0.4 * depth_ratio)
                center_x = wx - offset
                center_points.append((int(center_x), y))

        else:
            # ── NEITHER → road mask fallback ──
            for y in range(row_start, row_end, row_step):
                if road_orig is not None:
                    rp = np.where(road_orig[y, :] > 0)[0]
                    if len(rp) > 20:
                        rl, rr = float(rp[0]), float(rp[-1])
                        road_w = rr - rl
                        if road_w > max_road_w:
                            rcx = (rl + rr) / 2.0
                            rl = rcx - max_road_w / 2.0
                            rr = rcx + max_road_w / 2.0
                        center_x = rl + (road_w * road_bias)
                        center_points.append((int(center_x), y))
            if center_points:
                dominant_src = 'road_right_lane'

        # ── Measure and publish road width (for intersection detection) ──
        road_widths = []
        for y in range(row_start, row_end, row_step):
            if road_orig is not None:
                rp = np.where(road_orig[y, :] > 0)[0]
                if len(rp) > 20:
                    road_widths.append(float(rp[-1]) - float(rp[0]))
        avg_road_width = float(np.mean(road_widths)) if road_widths else 0.0
        rw_msg = Float32()
        rw_msg.data = avg_road_width
        self.road_width_pub.publish(rw_msg)

        # ── Smooth center path (quadratic fit to preserve curves) ───
        if len(center_points) >= 3:
            raw_ys = np.array([p[1] for p in center_points], dtype=np.float32)
            raw_xs = np.array([p[0] for p in center_points], dtype=np.float32)
            deg = min(2, len(center_points) - 1)
            coeffs = np.polyfit(raw_ys, raw_xs, deg)
            smooth_xs = np.polyval(coeffs, raw_ys)
            center_points = [(int(sx), int(sy))
                             for sx, sy in zip(smooth_xs, raw_ys)]

        # ── Compute lateral error (look-ahead weighted) ──────────
        if len(center_points) > 0:
            # Weight earlier (top) rows more → anticipate curves
            weights = np.arange(len(center_points), 0, -1,
                                dtype=np.float32)
            cx_vals = np.array([p[0] for p in center_points],
                               dtype=np.float32)
            avg_cx = float(np.average(cx_vals, weights=weights))

            error = float(np.clip(
                (avg_cx - image_cx) / (w_orig / 2.0), -1.0, 1.0))
            return error, True, dominant_src, center_points

        # ── Nothing found ────────────────────────────────────────
        return 0.0, False, 'none', []


    def _validate_center_line(self, med_x, road_orig, y,
                              max_dist_ratio=0.6):
        """Return True if a cluster looks like a genuine center line.

        The yellow center tape should be near the MIDDLE of the road, not
        at the edge.  Under warm lighting the beige curb surface can have
        HSV values that fall in the yellow range, so a 'yellow' cluster
        near the road edge is almost certainly a curb edge — not the tape.
        """
        if road_orig is None:
            return True
        rp = np.where(road_orig[y, :] > 0)[0]
        if len(rp) <= 20:
            return True
        rl, rr = float(rp[0]), float(rp[-1])
        road_hw = (rr - rl) / 2.0
        if road_hw <= 0:
            return True
        road_cx = (rl + rr) / 2.0
        return abs(med_x - road_cx) / road_hw <= max_dist_ratio

    def _offset_unknown_line(self, cluster_x, y, road_orig, image_cx, offset):
        """When lane colour can't be determined, use road edges to decide
        which side the line is on and offset accordingly."""
        if road_orig is not None:
            rp = np.where(road_orig[y, :] > 0)[0]
            if len(rp) > 20:
                road_center = (float(rp[0]) + float(rp[-1])) / 2.0
                if cluster_x < road_center:
                    return cluster_x + offset   # left side → offset right
                return cluster_x - offset       # right side → offset left
        # Last resort: use image center
        if cluster_x < image_cx:
            return cluster_x + offset
        return cluster_x - offset

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

        # Steering rate limiter: prevent wild oscillation
        max_steer_rate = self.get_parameter('max_steer_rate').value
        delta = steering - self.prev_steering
        if abs(delta) > max_steer_rate:
            steering = self.prev_steering + np.sign(delta) * max_steer_rate
        self.prev_steering = steering

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

        # Draw image center reference (thin dashed-style)
        for dy in range(0, h, 12):
            cv2.line(result, (w // 2, dy), (w // 2, min(dy + 6, h)),
                     (180, 180, 180), 1, cv2.LINE_AA)

        # Draw lane-center path and target marker
        if len(center_points) >= 2:
            pts = np.array(center_points, dtype=np.int32)
            # Smooth polyline through scan points (glow + core)
            cv2.polylines(result, [pts], False, (0, 140, 140), 5, cv2.LINE_AA)
            cv2.polylines(result, [pts], False, (0, 255, 255), 2, cv2.LINE_AA)
            # Small dots at each scan point
            for (cx, cy) in center_points:
                cv2.circle(result, (cx, cy), 3, (0, 255, 255), -1, cv2.LINE_AA)

        # Draw yellow line detections only (skip white/unknown noise dots)
        for (cx, cy) in getattr(self, '_viz_yellow_pts', []):
            cv2.circle(result, (cx, cy), 4, (0, 255, 255), -1, cv2.LINE_AA)

        # Target crosshair at weighted aim point
        if center_points:
            weights = np.arange(len(center_points), 0, -1, dtype=np.float32)
            aim_x = int(np.average(
                [p[0] for p in center_points], weights=weights))
            aim_y = center_points[-1][1]   # bottom scan row
            sz = 10
            cv2.line(result, (aim_x - sz, aim_y), (aim_x + sz, aim_y),
                     (0, 220, 255), 2, cv2.LINE_AA)
            cv2.line(result, (aim_x, aim_y - sz), (aim_x, aim_y + sz),
                     (0, 220, 255), 2, cv2.LINE_AA)
            cv2.circle(result, (aim_x, aim_y), sz,
                       (0, 220, 255), 2, cv2.LINE_AA)

        # Status text with guidance source
        source_info = getattr(self, '_last_guidance_source', 'none')
        if lane_found:
            if source_info == 'two_lanes':
                status_text = 'TWO LANES'
                status_color = (0, 255, 0)
            elif source_info == 'yellow_line':
                status_text = 'YELLOW LINE (center)'
                status_color = (0, 255, 255)
            elif source_info == 'white_line':
                status_text = 'WHITE LINE (edge)'
                status_color = (255, 255, 255)
            elif source_info == 'one_lane':
                status_text = 'SINGLE LINE'
                status_color = (0, 200, 255)
            elif source_info == 'road_right_lane':
                status_text = 'ROAD FALLBACK (right lane)'
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
