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
        self.declare_parameter('cruise_speed', 0.08)       # m/s forward speed
        self.declare_parameter('min_curve_speed', 0.16)    # m/s minimum speed on tight curves (PWM ~48 at max_pwm=150)
        self.declare_parameter('curve_slowdown_factor', 0.6)  # how much to slow on curves (0=no slowdown, 1=full)
        self.declare_parameter('kp', 0.8)                  # proportional gain
        self.declare_parameter('ki', 0.0)                  # integral gain
        self.declare_parameter('kd', 0.05)                 # derivative gain
        self.declare_parameter('max_angular_z', 0.8)       # max steering clamp
        self.declare_parameter('max_steer_rate', 0.15)     # max steering change per frame (rad/s) — prevents oscillation
        self.declare_parameter('curvature_gain', 0.4)      # how much road curvature biases the steering error
        self.declare_parameter('lane_width_px', 200)       # assumed lane width for single-line fallback
        self.declare_parameter('max_road_width_px', 350)   # max expected road width in pixels — caps road mask to reject floor
        self.declare_parameter('scan_row_start', 0.45)     # fraction from top — start of scan region (look further ahead)
        self.declare_parameter('scan_row_end', 0.85)       # fraction from top — end of scan region
        self.declare_parameter('scan_num_rows', 10)        # number of rows to sample in scan region
        self.declare_parameter('no_lane_timeout', 1.5)     # seconds w/o lane → stop
        self.declare_parameter('road_fallback_speed', 0.06)  # slower speed when using road-only fallback
        self.declare_parameter('road_lane_bias', 0.65)       # 0.5=road center, 0.65=right lane bias
        self.declare_parameter('enabled', True)            # master enable/disable

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
        self.error_smooth_alpha = 0.15     # EMA factor (0=full smooth, 1=no smooth)
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

        # Smooth the error with EMA to prevent jitter / drift
        if guidance_found:
            self.smoothed_error = (self.error_smooth_alpha * raw_error +
                                  (1.0 - self.error_smooth_alpha) * self.smoothed_error)
        error = self.smoothed_error

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

    def _classify_lane_color(self, img_hsv, pixel_xs, y):
        """Classify a cluster of lane pixels as 'yellow' (center line) or 'white' (edge).

        Samples HSV values from the original image at lane mask locations.
        On the black road surface the two colors separate cleanly:
            Yellow  →  H ∈ [15, 40], S > 80, V > 80
            White   →  S < 60, V > 140
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
            return 'white'
        return 'unknown'

    def _extract_lane_center(self, img_bgr, lane_mask, road_mask,
                             ratio, dw, dh, w_orig, h_orig):
        """Color-identified lane following.

        Tier 1:  Two lane-line clusters visible → midpoint between them.
        Tier 2:  Single cluster, colour known →
                     yellow (center line) → drive to its RIGHT
                     white  (edge line)   → drive to its LEFT
        Tier 2b: Single cluster, colour unknown → use road-edge context to
                 decide which side the line is on, then offset accordingly.
        Tier 3:  No lane lines, road mask visible → right-lane-biased position.
        Tier 4:  Nothing → not found.

        Returns (error, guidance_found, guidance_source, center_points).
        """
        scan_start = self.get_parameter('scan_row_start').value
        scan_end = self.get_parameter('scan_row_end').value
        num_rows = self.get_parameter('scan_num_rows').value
        lane_half_w = self.get_parameter('lane_width_px').value / 2.0
        max_road_w = self.get_parameter('max_road_width_px').value
        road_bias = self.get_parameter('road_lane_bias').value

        image_cx = w_orig / 2.0

        lane_orig = self._mask_to_original(lane_mask, dw, dh, w_orig, h_orig)
        road_orig = self._mask_to_original(road_mask, dw, dh, w_orig, h_orig)

        # ── Cap road mask width to reject floor outside the track ────
        # The Quanser lab floor has similar colour/texture to the track,
        # so HybridNets often segments it as "drivable."  Capping the
        # mask width discards everything beyond the expected track width.
        if road_orig is not None:
            # Fully vectorised road width cap (no Python loop)
            road_any = (road_orig > 0)
            row_has_road = road_any.any(axis=1)
            if row_has_road.any():
                # For each row, find leftmost and rightmost road pixel
                col_indices = np.arange(w_orig)
                # Broadcast: road_any[y, x] * col_indices[x] → 0 where no road
                # Use masked operations for left/right edge
                masked = np.where(road_any, col_indices[np.newaxis, :], w_orig)
                left_edges = masked.min(axis=1)  # leftmost road pixel per row
                masked_r = np.where(road_any, col_indices[np.newaxis, :], -1)
                right_edges = masked_r.max(axis=1)  # rightmost road pixel per row
                widths = right_edges - left_edges
                too_wide = row_has_road & (widths > max_road_w)
                if too_wide.any():
                    mids = (left_edges[too_wide] + right_edges[too_wide]) // 2
                    new_left = np.maximum(0, mids - max_road_w // 2)
                    new_right = np.minimum(w_orig - 1, mids + max_road_w // 2)
                    wide_rows = np.where(too_wide)[0]
                    for i, y_cap in enumerate(wide_rows):
                        road_orig[y_cap, :new_left[i]] = 0
                        road_orig[y_cap, new_right[i]:] = 0

        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV) \
            if lane_orig is not None else None

        center_points = []
        dominant_src = 'none'

        # ── Tier 1 & 2: Scan for lane lines ──────────────────────────
        if lane_orig is not None:
            row_start = int(h_orig * scan_start)
            row_end = int(h_orig * scan_end)
            row_step = max(1, (row_end - row_start) // num_rows)

            for y in range(row_start, row_end, row_step):
                lane_pixels = np.where(lane_orig[y, :] > 0)[0]
                if len(lane_pixels) < 3:
                    continue

                # ── Layer 1: Filter lane pixels outside drivable area ─
                # At corners the bold straight white line diverges from
                # the road; removing pixels outside the road mask keeps
                # only the dotted curve lines that follow the actual turn.
                if road_orig is not None:
                    road_row = road_orig[y, :]
                    inside_road = road_row[lane_pixels] > 0
                    # Allow a small margin (8 px) outside the road edge
                    # so lines right at the boundary are kept.
                    if np.any(inside_road):
                        rp = np.where(road_row > 0)[0]
                        if len(rp) > 0:
                            road_left = rp[0] - 8
                            road_right = rp[-1] + 8
                            lane_pixels = lane_pixels[
                                (lane_pixels >= road_left) &
                                (lane_pixels <= road_right)]
                    if len(lane_pixels) < 3:
                        continue

                gaps = np.diff(lane_pixels)
                gap_threshold = w_orig * 0.12
                large_gaps = np.where(gaps > gap_threshold)[0]

                if len(large_gaps) >= 1:
                    # ── Two clusters ──────────────────────────────
                    split = large_gaps[np.argmax(gaps[large_gaps])]
                    left_px = lane_pixels[:split + 1]
                    right_px = lane_pixels[split + 1:]
                    left_med = float(np.median(left_px))
                    right_med = float(np.median(right_px))

                    # ── Layer 2: Lane width sanity check ──────────
                    # If the gap between clusters is much wider than
                    # expected lane width, we're probably spanning
                    # across the dotted curve line AND the bold
                    # straight line.  In that case, pick the pair
                    # whose right cluster is closest to the road
                    # mask right edge (the actual lane boundary).
                    cluster_gap = right_med - left_med
                    max_expected = lane_half_w * 2.0 * 1.5  # 1.5× lane width

                    if cluster_gap > max_expected and road_orig is not None:
                        rp = np.where(road_orig[y, :] > 0)[0]
                        if len(rp) > 20:
                            road_right_edge = float(rp[-1])
                            # If right cluster is far from road edge,
                            # it might be the bold outer line.  Check
                            # if there's a third cluster in between.
                            all_splits = large_gaps
                            if len(all_splits) >= 2:
                                # 3+ clusters: pick the two innermost
                                # (i.e. closest pair straddling the road)
                                segments = []
                                prev = 0
                                for si in all_splits:
                                    segments.append(lane_pixels[prev:si + 1])
                                    prev = si + 1
                                segments.append(lane_pixels[prev:])

                                # Find best pair: smallest gap that
                                # still straddles the expected lane center
                                best_pair = None
                                best_gap = float('inf')
                                for i in range(len(segments) - 1):
                                    lm = float(np.median(segments[i]))
                                    rm = float(np.median(segments[i + 1]))
                                    g = rm - lm
                                    if g < best_gap:
                                        best_gap = g
                                        best_pair = (lm, rm)
                                if best_pair is not None:
                                    left_med, right_med = best_pair
                            else:
                                # Only 2 clusters but too wide:
                                # Clamp right boundary to road edge
                                right_med = min(right_med,
                                                road_right_edge)

                    center_x = (left_med + right_med) / 2.0
                    dominant_src = 'two_lanes'
                else:
                    # ── Single cluster — identify by colour ───────
                    cluster_med = float(np.median(lane_pixels))
                    color = self._classify_lane_color(img_hsv, lane_pixels, y)

                    if color == 'yellow':
                        center_x = cluster_med + lane_half_w
                        if dominant_src not in ('two_lanes',):
                            dominant_src = 'yellow_line'
                    elif color == 'white':
                        center_x = cluster_med - lane_half_w
                        if dominant_src not in ('two_lanes',):
                            dominant_src = 'white_line'
                    else:
                        # Unknown colour — use road edges for context
                        center_x = self._offset_unknown_line(
                            cluster_med, y, road_orig, image_cx, lane_half_w)
                        if dominant_src == 'none':
                            dominant_src = 'one_lane'

                center_points.append((int(center_x), y))

        if len(center_points) > 0:
            # Weight top (further ahead) rows MORE so the vehicle
            # anticipates curves instead of only reacting when close.
            weights = np.arange(len(center_points), 0, -1, dtype=np.float32)
            cx_vals = np.array([p[0] for p in center_points], dtype=np.float32)
            avg_cx = float(np.average(cx_vals, weights=weights))

            # ── Curvature detection ──────────────────────────────
            # Measure how the lane center shifts from bottom (near)
            # to top (far).  If top points are left of bottom, the
            # road curves left and we should steer earlier.
            curvature_gain = self.get_parameter('curvature_gain').value
            if len(center_points) >= 4:
                n = len(center_points)
                top_cx = np.mean(cx_vals[:n // 3])       # far ahead
                bot_cx = np.mean(cx_vals[2 * n // 3:])   # close
                # Positive curvature = road curves left (top is left of bottom)
                curvature = (bot_cx - top_cx) / (w_orig / 2.0)
            else:
                curvature = 0.0

            # Clamp inside the drivable area (with road width cap)
            if road_orig is not None:
                scan_y = center_points[-1][1]
                rp = np.where(road_orig[scan_y, :] > 0)[0]
                if len(rp) > 10:
                    rl, rr = float(rp[0]), float(rp[-1])
                    road_w = rr - rl
                    if road_w > max_road_w:
                        road_cx_here = (rl + rr) / 2.0
                        rl = road_cx_here - max_road_w / 2.0
                        rr = road_cx_here + max_road_w / 2.0
                    avg_cx = np.clip(avg_cx, rl + 10, rr - 10)

            error = float(np.clip((avg_cx - image_cx) / (w_orig / 2.0),
                                  -1.0, 1.0))
            # Add curvature bias: steer into the curve earlier
            error = float(np.clip(error - curvature * curvature_gain,
                                  -1.0, 1.0))
            return error, True, dominant_src, center_points

        # ── Tier 3: Road mask — right-lane biased ────────────────────
        if road_orig is not None:
            row_start = int(h_orig * scan_start)
            row_end = int(h_orig * scan_end)
            row_step = max(1, (row_end - row_start) // max(1, num_rows // 2))

            road_centers = []
            for y in range(row_start, row_end, row_step):
                rp = np.where(road_orig[y, :] > 0)[0]
                if len(rp) > 20:
                    rl, rr = float(rp[0]), float(rp[-1])
                    # Cap road width to reject floor areas outside track
                    road_w = rr - rl
                    if road_w > max_road_w:
                        road_cx_here = (rl + rr) / 2.0
                        rl = road_cx_here - max_road_w / 2.0
                        rr = road_cx_here + max_road_w / 2.0
                    road_cx = rl + (rr - rl) * road_bias
                    road_centers.append((int(road_cx), y))

            if len(road_centers) > 0:
                weights = np.arange(1, len(road_centers) + 1, dtype=np.float32)
                cx_vals = np.array([p[0] for p in road_centers], dtype=np.float32)
                avg_cx = float(np.average(cx_vals, weights=weights))
                error = float(np.clip((avg_cx - image_cx) / (w_orig / 2.0),
                                      -1.0, 1.0))
                return error, True, 'road_right_lane', road_centers

        # ── Tier 4: Nothing found ────────────────────────────────────
        return 0.0, False, 'none', []

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
