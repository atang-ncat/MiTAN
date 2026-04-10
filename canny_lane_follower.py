#!/usr/bin/env python3
"""
Canny + HSV Lane Follower for ROS 2

Pure classical computer-vision lane follower — no neural network.
Uses HSV colour filtering to isolate yellow centre-line tape and
white dashed lane marks on the black Quanser road, combined with
Canny edge detection for clean, thin features.

Publishes steering on /cmd_vel_auto (same as HybridNets node) so
cmd_vel_mux + serial_bridge work without changes.

Usage:
    python3 canny_lane_follower.py
    python3 canny_lane_follower.py --ros-args -p cruise_speed:=0.15
"""

import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class CannyLaneFollower(Node):

    def __init__(self):
        super().__init__('canny_lane_follower')

        # ── Parameters ────────────────────────────────────────────────
        # Driving
        self.declare_parameter('cruise_speed', 0.20)
        self.declare_parameter('min_curve_speed', 0.14)
        self.declare_parameter('kp', 1.2)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.25)
        self.declare_parameter('max_angular_z', 1.0)
        self.declare_parameter('max_steer_rate', 0.50)
        self.declare_parameter('curvature_gain', 0.6)

        # Detection
        self.declare_parameter('scan_row_start', 0.45)
        self.declare_parameter('scan_row_end', 0.85)
        self.declare_parameter('scan_num_rows', 12)
        self.declare_parameter('lane_width_px', 120)
        self.declare_parameter('max_dashed_line_width_px', 25)
        self.declare_parameter('use_canny', False)
        self.declare_parameter('canny_low', 50)
        self.declare_parameter('canny_high', 150)

        # HSV thresholds — yellow centre tape
        self.declare_parameter('yellow_h_low', 15)
        self.declare_parameter('yellow_h_high', 40)
        self.declare_parameter('yellow_s_low', 80)
        self.declare_parameter('yellow_v_low', 80)

        # HSV thresholds — white lane marks
        self.declare_parameter('white_s_high', 55)
        self.declare_parameter('white_v_low', 170)

        self.declare_parameter('no_lane_timeout', 1.5)
        self.declare_parameter('enabled', True)

        # ── State ─────────────────────────────────────────────────────
        self.prev_error = 0.0
        self.integral_error = 0.0
        self.smoothed_error = 0.0
        self.error_alpha = 0.35
        self.prev_steering = 0.0
        self.last_lane_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self._fps_time = time.time()

        # ── Cached parameters (refreshed periodically) ────────────
        self._param_cache = {}
        self._param_cache_time = 0.0
        self._param_cache_interval = 2.0  # seconds
        self._refresh_param_cache()

        # ── Pub / Sub ─────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_auto', 10)
        self.vis_pub = self.create_publisher(Image, 'canny/visualization', 1)
        self.det_pub = self.create_publisher(Bool, 'canny/lane_detected', 10)

        self.declare_parameter('camera_topic', '/camera/camera/color/image_raw')
        cam_topic = self.get_parameter('camera_topic').value
        self.create_subscription(Image, cam_topic, self._image_cb,
                                 qos_profile_sensor_data)

        self.get_logger().info('Canny lane follower ready — waiting for camera …')

    # ──────────────────────────────────────────────────────────────────
    #  Parameter cache — avoids costly get_parameter() calls every frame
    # ──────────────────────────────────────────────────────────────────
    def _refresh_param_cache(self):
        p = self.get_parameter
        self._param_cache = {
            'cruise_speed': p('cruise_speed').value,
            'min_curve_speed': p('min_curve_speed').value,
            'kp': p('kp').value,
            'ki': p('ki').value,
            'kd': p('kd').value,
            'max_angular_z': p('max_angular_z').value,
            'max_steer_rate': p('max_steer_rate').value,
            'curvature_gain': p('curvature_gain').value,
            'scan_row_start': p('scan_row_start').value,
            'scan_row_end': p('scan_row_end').value,
            'scan_num_rows': p('scan_num_rows').value,
            'lane_width_px': p('lane_width_px').value,
            'max_dashed_line_width_px': p('max_dashed_line_width_px').value,
            'use_canny': p('use_canny').value,
            'canny_low': p('canny_low').value,
            'canny_high': p('canny_high').value,
            'yellow_h_low': p('yellow_h_low').value,
            'yellow_h_high': p('yellow_h_high').value,
            'yellow_s_low': p('yellow_s_low').value,
            'yellow_v_low': p('yellow_v_low').value,
            'white_s_high': p('white_s_high').value,
            'white_v_low': p('white_v_low').value,
            'no_lane_timeout': p('no_lane_timeout').value,
            'enabled': p('enabled').value,
        }
        self._param_cache_time = time.time()

    def _p(self, name):
        """Fast cached parameter lookup."""
        return self._param_cache[name]

    # ──────────────────────────────────────────────────────────────────
    #  Image callback
    # ──────────────────────────────────────────────────────────────────
    def _image_cb(self, msg: Image):
        if not self._p('enabled'):
            return

        # Refresh parameter cache periodically (not every frame)
        now = time.time()
        if now - self._param_cache_time > self._param_cache_interval:
            self._refresh_param_cache()

        img = self._ros_to_cv2(msg)
        if img is None:
            self.get_logger().warn(
                f'Unsupported encoding: {msg.encoding}',
                throttle_duration_sec=5.0)
            return

        h0, w0 = img.shape[:2]
        if w0 > 640:
            img = cv2.resize(img, (480, 270), interpolation=cv2.INTER_AREA)

        self.frame_count += 1
        if now - self._fps_time > 1.0:
            self.fps = self.frame_count / (now - self._fps_time)
            self.frame_count = 0
            self._fps_time = now

        # ── Detect lanes ──────────────────────────────────────────
        result = self._detect_lanes(img)
        raw_err, found, src, pts, y_mask, w_mask, edges = result

        # Adaptive EMA smoothing
        if found:
            self.last_lane_time = now
            change = abs(raw_err - self.smoothed_error)
            alpha = min(0.85, self.error_alpha + change * 3.0)
            self.smoothed_error = (alpha * raw_err
                                   + (1.0 - alpha) * self.smoothed_error)
        error = self.smoothed_error

        # ── Steering ──────────────────────────────────────────────
        timeout = self._p('no_lane_timeout')
        if (now - self.last_lane_time) > timeout:
            cmd = Twist()
        else:
            cmd = self._compute_steering(error)
        self.cmd_pub.publish(cmd)
        self.det_pub.publish(Bool(data=found))

        # ── Publish visualisation only if someone is subscribed ───
        if self.vis_pub.get_subscription_count() > 0:
            vis = self._draw_vis(img, pts, error, cmd, src,
                                 y_mask, w_mask, edges)
            out = Image()
            out.header = msg.header
            out.height, out.width = vis.shape[:2]
            out.encoding = 'bgr8'
            out.step = out.width * 3
            out.data = vis.tobytes()
            self.vis_pub.publish(out)

    # ──────────────────────────────────────────────────────────────────
    #  Lane detection pipeline
    # ──────────────────────────────────────────────────────────────────
    def _detect_lanes(self, img):
        h, w = img.shape[:2]
        image_cx = w / 2.0

        scan_start = self._p('scan_row_start')
        scan_end = self._p('scan_row_end')
        num_rows = self._p('scan_num_rows')
        lane_half_w = self._p('lane_width_px') / 2.0
        max_dash_w = self._p('max_dashed_line_width_px')
        use_canny = self._p('use_canny')

        # ── Crop to ROI before any processing ────────────────────
        roi_top = int(h * scan_start)
        roi_bot = int(h * scan_end)
        roi = img[roi_top:roi_bot, :]
        roi_h = roi_bot - roi_top

        # ── 1. HSV colour masks (on ROI only) ───────────────────
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        yhl = self._p('yellow_h_low')
        yhh = self._p('yellow_h_high')
        ysl = self._p('yellow_s_low')
        yvl = self._p('yellow_v_low')
        wsh = self._p('white_s_high')
        wvl = self._p('white_v_low')

        yellow_mask = cv2.inRange(
            hsv, np.array([yhl, ysl, yvl], np.uint8),
            np.array([yhh, 255, 255], np.uint8))
        white_mask = cv2.inRange(
            hsv, np.array([0, 0, wvl], np.uint8),
            np.array([180, wsh, 255], np.uint8))

        # Morphological cleanup
        kern_s = np.ones((3, 3), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kern_s)

        # ── 2. Optional Canny refinement ─────────────────────────
        if use_canny:
            canny_lo = self._p('canny_low')
            canny_hi = self._p('canny_high')
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, canny_lo, canny_hi)
            kern = np.ones((5, 5), np.uint8)
            y_edges = cv2.bitwise_and(edges, cv2.dilate(yellow_mask, kern))
            w_edges = cv2.bitwise_and(edges, cv2.dilate(white_mask, kern))
            yellow_mask = cv2.bitwise_or(yellow_mask, y_edges)
            white_mask = cv2.bitwise_or(white_mask, w_edges)
        else:
            edges = np.zeros((roi_h, w), dtype=np.uint8)

        # ── 3. Row scanning — two-pass approach ───────────────────
        #  Pass 1: collect raw yellow line positions (no offset)
        #          to measure the actual road curvature.
        #  Pass 2: apply a curvature-scaled offset so the car
        #          hugs the line on tight curves.
        row_step = max(1, roi_h // num_rows)

        raw_yellow_xs = []   # raw median positions (no offset)
        raw_yellow_ys = []

        for ry in range(0, roi_h, row_step):
            ypx = np.where(yellow_mask[ry, :] > 0)[0]
            if len(ypx) >= 3:
                raw_yellow_xs.append(float(np.median(ypx)))
                raw_yellow_ys.append(ry)

        # Measure curvature from raw positions
        raw_curvature = 0.0
        if len(raw_yellow_xs) >= 4:
            n = len(raw_yellow_xs)
            far_x = float(np.mean(raw_yellow_xs[:n // 3]))
            near_x = float(np.mean(raw_yellow_xs[2 * n // 3:]))
            raw_curvature = abs(near_x - far_x) / (w / 2.0)

        # Scale offset: full offset on straights, reduced on curves
        # curvature 0 → scale 1.0,  curvature 0.5+ → scale 0.2
        offset_scale = float(np.clip(1.0 - raw_curvature * 1.6,
                                     0.2, 1.0))

        center_points = []
        dominant_src = 'none'

        for ry in range(0, roi_h, row_step):
            depth_ratio = ry / max(1, roi_h)
            offset = lane_half_w * (0.45 + 0.55 * depth_ratio) \
                * offset_scale

            ypx = np.where(yellow_mask[ry, :] > 0)[0]
            wpx = np.where(white_mask[ry, :] > 0)[0]
            y = ry + roi_top  # full-image y for center_points

            if len(ypx) >= 3:
                # Yellow centre line → drive to its right
                y_med = float(np.median(ypx))
                cx = y_med + offset
                dominant_src = 'yellow_line'

                # Hard boundary: thin white dashed marks to the right
                if len(wpx) >= 2:
                    rw = wpx[wpx > y_med + 20]
                    if len(rw) >= 2:
                        for cl in self._thin_clusters(rw, max_dash_w):
                            boundary = float(cl[0]) - 10
                            cx = min(cx, boundary)
                            break

            elif len(wpx) >= 3:
                cl = self._first_thin_cluster(wpx, max_dash_w)
                if cl is not None:
                    w_med = float(np.median(cl))
                    cx = (w_med - offset) if w_med > image_cx \
                        else (w_med + offset)
                    if dominant_src != 'yellow_line':
                        dominant_src = 'white_line'
                else:
                    continue
            else:
                continue

            center_points.append((int(cx), y))

        # ── 4. Polynomial smoothing ──────────────────────────────
        if len(center_points) >= 3:
            ys = np.array([p[1] for p in center_points], np.float32)
            xs = np.array([p[0] for p in center_points], np.float32)
            c1 = np.polyfit(ys, xs, 1)
            r1 = float(np.sum((xs - np.polyval(c1, ys)) ** 2))
            if len(center_points) >= 4:
                c2 = np.polyfit(ys, xs, 2)
                r2 = float(np.sum((xs - np.polyval(c2, ys)) ** 2))
                coeffs = c2 if r2 < r1 * 0.6 else c1
            else:
                coeffs = c1
            sxs = np.polyval(coeffs, ys)
            center_points = [(int(sx), int(sy))
                             for sx, sy in zip(sxs, ys)]

        # ── 5. Compute lateral error ─────────────────────────────
        if len(center_points) > 0:
            wts = np.arange(len(center_points), 0, -1, np.float32)
            cxv = np.array([p[0] for p in center_points], np.float32)
            avg_cx = float(np.average(cxv, weights=wts))

            curv_gain = self._p('curvature_gain')
            curv = 0.0
            if len(center_points) >= 4:
                n = len(center_points)
                top = float(np.mean(cxv[:n // 3]))
                bot = float(np.mean(cxv[2 * n // 3:]))
                curv = (bot - top) / (w / 2.0)

            err = float(np.clip((avg_cx - image_cx) / (w / 2.0),
                                -1.0, 1.0))
            err = float(np.clip(err - curv * curv_gain, -1.0, 1.0))
            return err, True, dominant_src, center_points, \
                yellow_mask, white_mask, edges

        return 0.0, False, 'none', [], yellow_mask, white_mask, edges

    # ── Cluster helpers ───────────────────────────────────────────────
    @staticmethod
    def _thin_clusters(px, max_w):
        """Yield contiguous clusters whose extent is <= max_w."""
        gaps = np.diff(px)
        splits = np.where(gaps > 15)[0]
        clusters = np.split(px, splits + 1) if len(splits) else [px]
        for cl in clusters:
            if len(cl) >= 2 and int(cl[-1]) - int(cl[0]) <= max_w:
                yield cl

    @staticmethod
    def _first_thin_cluster(px, max_w):
        gaps = np.diff(px)
        splits = np.where(gaps > 15)[0]
        clusters = np.split(px, splits + 1) if len(splits) else [px]
        for cl in clusters:
            if len(cl) >= 2 and int(cl[-1]) - int(cl[0]) <= max_w:
                return cl
        return None

    # ──────────────────────────────────────────────────────────────────
    #  PID steering
    # ──────────────────────────────────────────────────────────────────
    def _compute_steering(self, error):
        kp = self._p('kp')
        ki = self._p('ki')
        kd = self._p('kd')
        max_ang = self._p('max_angular_z')
        cruise = self._p('cruise_speed')
        min_speed = self._p('min_curve_speed')
        max_rate = self._p('max_steer_rate')

        d_err = error - self.prev_error
        self.integral_error += error
        self.integral_error = float(np.clip(self.integral_error,
                                            -100.0, 100.0))

        steer = -(kp * error + ki * self.integral_error + kd * d_err)
        steer = float(np.clip(steer, -max_ang, max_ang))

        delta = steer - self.prev_steering
        if abs(delta) > max_rate:
            steer = self.prev_steering + max_rate * np.sign(delta)

        self.prev_error = error
        self.prev_steering = steer

        speed = max(min_speed, cruise * (1.0 - abs(steer) * 0.6))

        cmd = Twist()
        cmd.linear.x = float(speed)
        cmd.angular.z = float(steer)
        return cmd

    # ──────────────────────────────────────────────────────────────────
    #  Visualisation
    # ──────────────────────────────────────────────────────────────────
    def _draw_vis(self, img, pts, error, cmd, source,
                  y_mask, w_mask, edges):
        vis = img.copy()
        h, w = vis.shape[:2]

        # Masks are ROI-sized — overlay only in the ROI region
        roi_top = int(h * self._p('scan_row_start'))
        roi_bot = roi_top + y_mask.shape[0]
        roi_slice = vis[roi_top:roi_bot, :]
        roi_slice[y_mask > 0] = (
            (roi_slice[y_mask > 0].astype(np.int16) + (0, 90, 90)) // 2
        ).clip(0, 255).astype(np.uint8)
        roi_slice[w_mask > 0] = (
            (roi_slice[w_mask > 0].astype(np.int16) + (90, 90, 90)) // 2
        ).clip(0, 255).astype(np.uint8)

        # Centre reference (dashed grey)
        for dy in range(0, h, 12):
            cv2.line(vis, (w // 2, dy), (w // 2, min(dy + 6, h)),
                     (160, 160, 160), 1, cv2.LINE_AA)

        # Lane centre path
        if len(pts) >= 2:
            cv2.polylines(vis, [np.array(pts)], False,
                          (0, 255, 255), 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(vis, pt, 3, (0, 200, 200), -1, cv2.LINE_AA)

        # Target crosshair
        if pts:
            tx, ty = pts[len(pts) // 2]
            cv2.circle(vis, (tx, ty), 12, (0, 165, 255), 2, cv2.LINE_AA)
            cv2.line(vis, (tx - 8, ty), (tx + 8, ty),
                     (0, 165, 255), 1, cv2.LINE_AA)
            cv2.line(vis, (tx, ty - 8), (tx, ty + 8),
                     (0, 165, 255), 1, cv2.LINE_AA)

        # ── Debug strip: small masks at the bottom ────────────────
        strip_h = 50
        strip_w = w // 3
        y0 = h - strip_h
        ym = cv2.resize(y_mask, (strip_w, strip_h))
        wm = cv2.resize(w_mask, (strip_w, strip_h))
        em = cv2.resize(edges, (strip_w, strip_h))
        vis[y0:, :strip_w] = cv2.cvtColor(ym, cv2.COLOR_GRAY2BGR)
        vis[y0:, strip_w:strip_w * 2] = cv2.cvtColor(wm, cv2.COLOR_GRAY2BGR)
        vis[y0:, strip_w * 2:strip_w * 3] = cv2.cvtColor(em, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, 'YELLOW', (4, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
        cv2.putText(vis, 'WHITE', (strip_w + 4, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(vis, 'EDGES', (strip_w * 2 + 4, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)

        # ── HUD text ─────────────────────────────────────────────
        src_txt = source.upper().replace('_', ' ')
        color_map = {
            'YELLOW LINE': (0, 255, 255),
            'WHITE LINE': (255, 255, 255),
            'NONE': (0, 0, 255),
        }
        src_col = color_map.get(src_txt, (0, 255, 255))
        cv2.putText(vis, f'{src_txt}', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, src_col, 2, cv2.LINE_AA)

        info = [
            (f'FPS: {self.fps:.1f}', (0, 255, 0)),
            (f'Error: {error:.3f}', (0, 255, 0)),
            (f'Steer: {cmd.angular.z:.3f}', (0, 255, 0)),
            (f'Speed: {cmd.linear.x:.2f} m/s', (0, 255, 0)),
        ]
        for i, (txt, col) in enumerate(info):
            cv2.putText(vis, txt, (10, 55 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)

        return vis

    # ──────────────────────────────────────────────────────────────────
    #  Utility
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _ros_to_cv2(msg: Image):
        try:
            if msg.encoding == 'rgb8':
                img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3)
                return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                return np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width, 3).copy()
            elif msg.encoding == 'mono8':
                gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                    msg.height, msg.width)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return None
        except Exception:
            return None


def main(args=None):
    rclpy.init(args=args)
    node = CannyLaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.cmd_pub.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
