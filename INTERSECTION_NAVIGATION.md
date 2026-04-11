# Intersection Navigation System

## Overview

The intersection navigation system extends the MiTAN lane-following stack with
autonomous intersection handling. It uses a **state-machine controller** that
detects when the vehicle enters an intersection (by observing the yellow center
line disappearance and road widening) and executes pre-defined timed maneuvers
(left turn, right turn, or straight-through).

## Architecture

```
┌──────────────────────┐     guidance_source    ┌───────────────────────┐
│                      │─────────────────────►  │                       │
│  HybridNets Lane     │     road_width         │  Intersection         │
│  Follower            │─────────────────────►  │  Controller           │
│                      │                        │                       │
│  publishes:          │                        │  publishes:           │
│   cmd_vel_auto       │                        │   cmd_vel_intersection│
│   guidance_source    │                        │   intersection/state  │
│   road_width         │                        │                       │
└──────────────────────┘                        └───────────────────────┘
         │                                               │
         │ cmd_vel_auto                                  │ cmd_vel_intersection
         ▼                                               ▼
┌────────────────────────────────────────────────────────────────────┐
│                     Command Velocity Mux                           │
│                                                                    │
│  Priority 1: cmd_vel_teleop        (gamepad — always wins)         │
│  Priority 2: cmd_vel_intersection  (maneuver — during turns)       │
│  Priority 3: cmd_vel_auto          (lane following — default)      │
│                                                                    │
│  Output: cmd_vel → serial_bridge → Arduino                         │
└────────────────────────────────────────────────────────────────────┘
```

## State Machine

The intersection controller runs a 4-state FSM:

| State | Behavior |
|-------|----------|
| **LANE_FOLLOWING** | Passive monitoring. No commands published. Lane follower drives normally. |
| **INTERSECTION_DETECTED** | Yellow line lost + road wide for >0.4s. Brief confirmation — aborts if yellow reappears. |
| **EXECUTING_MANEUVER** | Publishes fixed steering + speed for a timed duration. Overrides lane follower via mux priority. |
| **RECOVERING** | Drives forward slowly, searching for yellow line. Resumes lane following on re-acquisition or timeout. |

### Detection Logic

An intersection is detected when **both** conditions are true:
1. **Yellow center line was visible** → then **disappeared** for ≥0.4 seconds
2. **Road mask width** exceeds 300 pixels (normal lane ≈ 200px)

This double-condition prevents false triggers from:
- Momentary yellow dropout at sharp corners (road stays narrow)
- Random road width fluctuation (yellow stays visible)

### Cooldown

After completing a maneuver, a **5-second cooldown** prevents re-triggering at the same intersection.

## Parameters

### Intersection Controller (`intersection_controller.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `true` | Master enable/disable |
| `intersection_maneuver` | `"right"` | Single maneuver type |
| `maneuver_sequence` | `""` | Comma-separated loop (overrides single) |
| `yellow_loss_debounce` | `0.4` | Seconds yellow must be gone |
| `min_road_width_trigger` | `300.0` | Pixels — road width threshold |
| `maneuver_speed` | `0.08` | m/s during maneuver |
| `right_steering` | `-0.50` | angular.z for right turn |
| `right_duration` | `3.0` | Seconds for right turn |
| `straight_steering` | `0.0` | angular.z for straight |
| `straight_duration` | `2.5` | Seconds for straight |
| `left_steering` | `0.50` | angular.z for left turn |
| `left_duration` | `3.5` | Seconds for left turn |
| `recovery_timeout` | `3.0` | Max seconds searching for yellow |
| `recovery_speed` | `0.06` | m/s while recovering |
| `cooldown_time` | `5.0` | Min seconds between triggers |

### New Lane Follower Topics

| Topic | Type | Description |
|-------|------|-------------|
| `hybridnets/guidance_source` | `String` | `yellow_line`, `white_line`, `road_right_lane`, or `none` |
| `hybridnets/road_width` | `Float32` | Average road mask width in pixels |

## Usage

### Outer Loop (right → straight → left)

```bash
ros2 launch lane_follow.launch.py maneuver_sequence:="right,straight,left"
```

### Single Intersection Test

```bash
ros2 launch lane_follow.launch.py
# Then set maneuver at runtime:
ros2 param set /intersection_controller intersection_maneuver right
```

### Disable Intersection Controller

```bash
ros2 launch lane_follow.launch.py intersection_enabled:=false
```

### Monitor State

```bash
ros2 topic echo /intersection/state
```

### Tune Maneuver at Runtime

```bash
# Adjust right turn steering angle
ros2 param set /intersection_controller right_steering -0.45

# Adjust right turn duration
ros2 param set /intersection_controller right_duration 3.2
```

## Tuning Guide

### Step 1: Verify Detection

1. Drive the car to an intersection approach
2. Monitor `ros2 topic echo /hybridnets/guidance_source` — should be `yellow_line`
3. Monitor `ros2 topic echo /hybridnets/road_width` — note the value
4. Drive into the intersection — source should change to `none` or `road_right_lane`
5. Road width should increase significantly
6. Adjust `min_road_width_trigger` if needed

### Step 2: Tune Right Turn

1. Set `maneuver_sequence:="right"`
2. Position car before the first intersection
3. Let it trigger — observe the turn
4. Adjust `right_steering` (more negative = tighter turn) and `right_duration`
5. Iterate until the car exits the intersection into the correct lane

### Step 3: Tune Straight-Through

Same process with `maneuver_sequence:="straight"`

### Step 4: Tune Left Turn

Same process with `maneuver_sequence:="left"`

### Step 5: Full Loop

Set `maneuver_sequence:="right,straight,left"` and run the outer loop.

## Legacy Comparison

| Feature | Arduino Code | MiTAN Intersection Controller |
|---------|-------------|-------------------------------|
| Trigger | Global coordinates (`y == 3.0`) | Vision: yellow loss + road width |
| Execution | Servo angle + PWM for duration | angular.z + speed via cmd_vel |
| Control | Arduino-side blocking loop | ROS 2 state machine (non-blocking) |
| Lane resume | mode_control topic | Mux priority fallthrough |
| Sequence | Hardcoded `light` counter | Configurable `maneuver_sequence` |
| Localization | Required | Not required |
