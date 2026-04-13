#!/usr/bin/env python3
"""
Intersection Controller for ROS 2

State-machine node that detects intersections (yellow-line disappearance +
road widening) and executes timed turn maneuvers (left, right, straight).

States:
    LANE_FOLLOWING        — normal driving, monitors for intersection
    INTERSECTION_DETECTED — yellow lost + wide road, debouncing
    EXECUTING_MANEUVER    — timed steering override
    RECOVERING            — waiting for yellow line re-acquisition

Communication:
    Subscribes to:
        hybridnets/guidance_source  (String)  — current lane guidance source
        hybridnets/road_width       (Float32) — measured road width in pixels
    Publishes to:
        cmd_vel_intersection  (Twist)  — maneuver steering commands
        intersection/state    (String) — current state for debugging

Usage:
    python3 intersection_controller.py --ros-args -p maneuver_sequence:="right,straight,left"
"""

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32


class IntersectionState:
    """State enum for the intersection FSM."""
    LANE_FOLLOWING = 'LANE_FOLLOWING'
    INTERSECTION_DETECTED = 'INTERSECTION_DETECTED'
    EXECUTING_MANEUVER = 'EXECUTING_MANEUVER'
    RECOVERING = 'RECOVERING'


class IntersectionController(Node):
    """ROS 2 node for autonomous intersection navigation."""

    def __init__(self):
        super().__init__('intersection_controller')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('enabled', True)

        # Maneuver selection
        self.declare_parameter('intersection_maneuver', 'right')        # single maneuver: right, straight, left
        self.declare_parameter('maneuver_sequence', 'right,straight,left')  # default: outer loop (3 intersections)

        # Detection thresholds
        self.declare_parameter('yellow_loss_debounce', 0.4)             # seconds yellow must be lost
        self.declare_parameter('min_road_width_trigger', 300.0)         # pixels — road wider than this → intersection
        self.declare_parameter('normal_road_width', 200.0)              # pixels — typical lane road width

        # Maneuver parameters
        # Converted from Arduino servo angles via serial_bridge mapping:
        #   servo_center=92, servo_min=20, servo_max=160
        #   Right: servo 121 → angular.z = (92-121)/68 = -0.43
        #   Straight: servo 92 → angular.z = 0.0
        #   Left: servo 74 → angular.z = (92-74)/72 = +0.25
        #   Left sharp: servo 68 → angular.z = (92-68)/72 = +0.33
        # Speed: Arduino PWM 80 with max_speed_pwm=150 → ~0.20 m/s (conservative)
        self.declare_parameter('maneuver_speed', 0.20)                  # m/s forward speed during maneuver

        self.declare_parameter('right_steering', -0.43)                 # angular.z for right turn (servo 121)
        self.declare_parameter('right_duration', 2.5)                   # seconds for right turn

        self.declare_parameter('straight_steering', 0.0)                # angular.z for straight (servo 92)
        self.declare_parameter('straight_duration', 2.5)                # seconds — push through entire confusing zone

        self.declare_parameter('left_steering', 0.25)                   # angular.z for left turn (servo 74)
        self.declare_parameter('left_duration', 3.5)                    # seconds (from Arduino LEFT_ACTION_DURATION)

        # Recovery and cooldown
        self.declare_parameter('recovery_timeout', 5.0)                 # max seconds in RECOVERING
        self.declare_parameter('recovery_speed', 0.10)                  # forward speed while recovering
        self.declare_parameter('sustained_yellow_time', 0.3)            # seconds yellow must be seen continuously before recovery
        self.declare_parameter('cooldown_time', 10.0)                   # min seconds between intersection triggers
        self.declare_parameter('lead_in_time', 1.0)                     # seconds to drive straight before turning

        # ── State ────────────────────────────────────────────────────
        self.state = IntersectionState.LANE_FOLLOWING
        self.guidance_source = 'none'
        self.road_width = 0.0

        # Timing
        self.yellow_lost_time = 0.0            # when yellow was first lost
        self.yellow_was_seen = False            # have we ever seen yellow?
        self.last_yellow_time = 0.0            # when yellow was last seen
        self.maneuver_start_time = 0.0         # when maneuver started
        self.recovery_start_time = 0.0         # when recovery started
        self.last_intersection_time = 0.0      # when last intersection was handled
        self._yellow_sustained_start = 0.0     # when yellow re-appeared in recovery

        # Sequence tracking
        self.sequence_index = 0                # current position in maneuver_sequence

        # ── Publishers ───────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_intersection', 10)
        self.state_pub = self.create_publisher(String, 'intersection/state', 10)

        # ── Subscribers ──────────────────────────────────────────────
        self.create_subscription(
            String, 'hybridnets/guidance_source',
            self._guidance_source_cb, 10)
        self.create_subscription(
            Float32, 'hybridnets/road_width',
            self._road_width_cb, 10)

        # ── Control loop at 20 Hz ────────────────────────────────────
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info('Intersection controller ready')
        self._log_config()

    def _log_config(self):
        """Log current configuration."""
        seq = self.get_parameter('maneuver_sequence').value
        single = self.get_parameter('intersection_maneuver').value
        if seq:
            self.get_logger().info(f'  Maneuver sequence: {seq}')
        else:
            self.get_logger().info(f'  Single maneuver: {single}')
        self.get_logger().info(
            f'  Detection: yellow_loss>{self.get_parameter("yellow_loss_debounce").value}s '
            f'+ road_width>{self.get_parameter("min_road_width_trigger").value}px')

    # ── Subscriber Callbacks ─────────────────────────────────────────

    def _guidance_source_cb(self, msg: String):
        """Track guidance source from lane follower."""
        self.guidance_source = msg.data
        now = time.time()
        if msg.data == 'yellow_line':
            self.yellow_was_seen = True
            self.last_yellow_time = now

    def _road_width_cb(self, msg: Float32):
        """Track road width from lane follower."""
        self.road_width = msg.data

    # ── State Machine ────────────────────────────────────────────────

    def _control_loop(self):
        """Main state machine tick — runs at 20 Hz."""
        if not self.get_parameter('enabled').value:
            return

        now = time.time()

        # Publish current state
        state_msg = String()
        state_msg.data = self.state
        self.state_pub.publish(state_msg)

        if self.state == IntersectionState.LANE_FOLLOWING:
            self._tick_lane_following(now)
        elif self.state == IntersectionState.INTERSECTION_DETECTED:
            self._tick_intersection_detected(now)
        elif self.state == IntersectionState.EXECUTING_MANEUVER:
            self._tick_executing_maneuver(now)
        elif self.state == IntersectionState.RECOVERING:
            self._tick_recovering(now)

    def _tick_lane_following(self, now):
        """Monitor for intersection entry conditions."""
        # Don't publish any commands — let lane follower drive

        # Check cooldown
        cooldown = self.get_parameter('cooldown_time').value
        if (now - self.last_intersection_time) < cooldown:
            return

        # Check: was yellow recently visible, but now it's gone?
        if not self.yellow_was_seen:
            return  # Haven't seen yellow yet — don't trigger

        debounce = self.get_parameter('yellow_loss_debounce').value
        yellow_gone = self.guidance_source != 'yellow_line'
        yellow_gone_long = (now - self.last_yellow_time) > debounce

        # Check: is the road unusually wide?
        min_road_w = self.get_parameter('min_road_width_trigger').value
        road_wide = self.road_width > min_road_w

        # PRIMARY trigger: yellow lost + road wide
        if yellow_gone and yellow_gone_long and road_wide:
            self._transition_to(IntersectionState.INTERSECTION_DETECTED)
            self.yellow_lost_time = now
            self.get_logger().info(
                f'⚠ Yellow lost for {debounce:.1f}s + road width '
                f'{self.road_width:.0f}px > {min_road_w:.0f}px — '
                f'intersection detected!')

    def _tick_intersection_detected(self, now):
        """Brief confirmation state — then execute maneuver."""
        # Double-check: if yellow reappears, abort
        if self.guidance_source == 'yellow_line':
            self.get_logger().info('Yellow re-appeared — false alarm, resuming lane following')
            self._transition_to(IntersectionState.LANE_FOLLOWING)
            return

        # Determine next maneuver
        maneuver = self._get_next_maneuver()


        # Start the maneuver immediately (debounce already happened in LANE_FOLLOWING)
        self.get_logger().info(f'🚗 Executing intersection maneuver: {maneuver.upper()}')
        self.maneuver_start_time = now
        self._current_maneuver = maneuver
        self._transition_to(IntersectionState.EXECUTING_MANEUVER)

    def _tick_executing_maneuver(self, now):
        """Drive with fixed steering for the maneuver duration."""
        maneuver = self._current_maneuver
        speed = self.get_parameter('maneuver_speed').value
        lead_in = self.get_parameter('lead_in_time').value

        # Get maneuver-specific parameters
        steering = self.get_parameter(f'{maneuver}_steering').value
        duration = self.get_parameter(f'{maneuver}_duration').value

        elapsed = now - self.maneuver_start_time
        total_duration = lead_in + duration

        if elapsed < total_duration:
            # Publish maneuver command
            cmd = Twist()
            cmd.linear.x = speed

            if elapsed < lead_in:
                # Lead-in phase: drive straight into intersection
                cmd.angular.z = 0.0
                remaining = total_duration - elapsed
                self.get_logger().info(
                    f'  [{maneuver.upper()}] LEAD-IN (straight) '
                    f'remaining={remaining:.1f}s',
                    throttle_duration_sec=0.5)
            else:
                # Actual turn phase
                cmd.angular.z = float(steering)
                remaining = total_duration - elapsed
                self.get_logger().info(
                    f'  [{maneuver.upper()}] steer={steering:.2f} '
                    f'remaining={remaining:.1f}s',
                    throttle_duration_sec=0.5)

            self.cmd_pub.publish(cmd)
        else:
            # Maneuver complete
            self.last_intersection_time = now  # Start cooldown
            self._advance_sequence()

            # Straight maneuver: car is already aligned, skip recovery
            if maneuver == 'straight':
                self.get_logger().info(
                    f'✅ Maneuver STRAIGHT complete '
                    f'({elapsed:.1f}s) — resuming lane following (no recovery needed)')
                self._transition_to(IntersectionState.LANE_FOLLOWING)
            else:
                # Turns need recovery to re-find yellow line
                self.get_logger().info(
                    f'✅ Maneuver {maneuver.upper()} complete '
                    f'({elapsed:.1f}s) — entering recovery')
                self.recovery_start_time = now
                self._yellow_sustained_start = 0.0
                self._transition_to(IntersectionState.RECOVERING)

    def _tick_recovering(self, now):
        """Drive forward slowly, looking for sustained yellow line re-acquisition."""
        recovery_timeout = self.get_parameter('recovery_timeout').value
        recovery_speed = self.get_parameter('recovery_speed').value
        sustained_time = self.get_parameter('sustained_yellow_time').value
        elapsed = now - self.recovery_start_time

        # Check for SUSTAINED yellow re-acquisition
        if self.guidance_source == 'yellow_line':
            if self._yellow_sustained_start == 0.0:
                self._yellow_sustained_start = now
                self.get_logger().info(
                    f'  [RECOVERING] yellow spotted, need {sustained_time:.1f}s sustained...',
                    throttle_duration_sec=1.0)

            yellow_duration = now - self._yellow_sustained_start
            if yellow_duration >= sustained_time:
                self.get_logger().info(
                    f'🟢 Yellow line sustained for {yellow_duration:.1f}s — '
                    f'resuming lane following')
                self._transition_to(IntersectionState.LANE_FOLLOWING)
                return
        else:
            # Yellow disappeared — reset sustained counter
            if self._yellow_sustained_start > 0.0:
                self.get_logger().info(
                    '  [RECOVERING] yellow lost again, resetting sustained counter',
                    throttle_duration_sec=1.0)
            self._yellow_sustained_start = 0.0

        # Check timeout
        if elapsed > recovery_timeout:
            self.get_logger().warn(
                f'⏱ Recovery timeout ({recovery_timeout:.1f}s) — '
                f'forcing lane following resume')
            self._transition_to(IntersectionState.LANE_FOLLOWING)
            return

        # Drive forward slowly while searching
        cmd = Twist()
        cmd.linear.x = recovery_speed
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

        self.get_logger().info(
            f'  [RECOVERING] searching for yellow... '
            f'{elapsed:.1f}/{recovery_timeout:.1f}s  '
            f'(source: {self.guidance_source})',
            throttle_duration_sec=1.0)

    # ── Helpers ──────────────────────────────────────────────────────

    def _transition_to(self, new_state):
        """Transition to a new state."""
        old_state = self.state
        self.state = new_state
        self.get_logger().info(f'State: {old_state} → {new_state}')

        # When leaving EXECUTING_MANEUVER or RECOVERING, stop publishing
        # maneuver commands (send one final stop so mux falls through).
        # But NOT when cancelling a false-alarm from INTERSECTION_DETECTED —
        # that was only 1 tick, no need to brake.
        if new_state == IntersectionState.LANE_FOLLOWING and \
           old_state == IntersectionState.EXECUTING_MANEUVER:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)

    def _get_next_maneuver(self):
        """Get the next maneuver to execute."""
        seq_str = self.get_parameter('maneuver_sequence').value
        if seq_str:
            sequence = [s.strip().lower() for s in seq_str.split(',') if s.strip()]
            if sequence:
                idx = self.sequence_index % len(sequence)
                return sequence[idx]
        # Fall back to single maneuver
        return self.get_parameter('intersection_maneuver').value

    def _advance_sequence(self):
        """Move to next maneuver in the sequence."""
        seq_str = self.get_parameter('maneuver_sequence').value
        if seq_str:
            sequence = [s.strip().lower() for s in seq_str.split(',') if s.strip()]
            if sequence:
                self.sequence_index = (self.sequence_index + 1) % len(sequence)
                next_maneuver = sequence[self.sequence_index]
                self.get_logger().info(
                    f'📋 Next maneuver in sequence: {next_maneuver.upper()} '
                    f'(index {self.sequence_index}/{len(sequence)})')

    def destroy_node(self):
        """Cleanup: stop command."""
        try:
            cmd = Twist()
            self.cmd_pub.publish(cmd)
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = IntersectionController()
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
