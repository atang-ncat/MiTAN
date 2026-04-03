# HybridNets Lane-Following Logic

## Overview

The MiTAN vehicle uses a fine-tuned HybridNets model running on TensorRT to perform
real-time lane-following on the Quanser studio track. The system processes camera frames,
extracts steering guidance, and outputs Ackermann drive commands.

## Architecture

```
Camera (1280x720 RGB @ 30 FPS)
        │
        ▼
┌─────────────────┐
│   HybridNets    │  TensorRT FP16 engine
│   (384×640)     │  Outputs: detection boxes, road mask, lane mask
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│  Lane Center    │  Tiered guidance extraction (see below)
│  Extraction     │
└─────┬───────────┘
      │  lateral error [-1, 1]
      ▼
┌─────────────────┐
│  PD Controller  │  Kp=0.005, Kd=0.002
│  + Curve Speed  │  Slows on curves (60% reduction at max steer)
└─────┬───────────┘
      │  Twist (speed, steering)
      ▼
┌─────────────────┐
│  cmd_vel_mux    │  Gamepad (hold LB) always overrides autonomous
└─────┬───────────┘
      │  /cmd_vel
      ▼
┌─────────────────┐
│  serial_bridge  │  Twist → S<speed>,A<angle> serial protocol
└─────┬───────────┘
      │  USB serial
      ▼
┌─────────────────┐
│  Arduino Mega   │  Servo (steering) + DC motors (rear wheels)
└─────────────────┘
```

## Tiered Guidance Extraction

The lane center extraction uses a **4-tier priority system** that combines both
lane line segmentation AND drivable area segmentation from HybridNets:

### Tier 1 — Two Lane Lines (Highest Confidence)
- **Condition**: Both left and right lane lines visible in a scan row
- **Method**: Find the two clusters of lane pixels (separated by a gap > 15% image width),
  take their medians, compute the midpoint
- **Speed**: Full cruise speed (0.15 m/s)
- **Status**: `LANE TRACKING (2 lines)` (green)

### Tier 2 — One Lane Line (Medium Confidence)
- **Condition**: Only one lane line cluster visible (left or right)
- **Method**: Determine if it's the left or right line by comparing to image center,
  then offset by `lane_width_px` (200 pixels) toward the lane center
- **Speed**: Full cruise speed (0.15 m/s)
- **Status**: `LANE TRACKING (1 line)` (yellow)

### Tier 3 — Road Centroid Only (Fallback)
- **Condition**: No lane lines visible, but drivable area (road mask) is detected
- **Method**: Scan the same region for road mask pixels, compute the centroid
  (midpoint of leftmost and rightmost road pixels per row)
- **Speed**: Reduced speed (0.08 m/s) for safety
- **Status**: `ROAD CENTROID (fallback)` (magenta)

### Tier 4 — No Guidance (Fail)
- **Condition**: Neither lane lines nor road mask detected
- **Action**: Continue with last steering for `no_lane_timeout` (1.5s), then stop
- **Status**: `NO GUIDANCE` (red)

### Road Mask as Safety Constraint (All Tiers)
When lane lines ARE detected (Tier 1 or 2), the road mask acts as a **safety clamp**:
the computed lane center is constrained to stay within the drivable area boundaries.
This prevents the steering target from going off-road even if lane detection is noisy.

## Scan Region

The system scans **8 horizontal rows** in the bottom 30% of the image
(rows 55%–85% from top). This region is chosen because:
- Lane lines are largest and most visible closest to the vehicle
- The upper portion has perspective distortion and less reliable detections
- Bottom rows are weighted more heavily in the average (closer = more relevant)

```
┌─────────────────────────┐
│                         │  0%
│     (not scanned)       │
│                         │
├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤  55% ← scan_row_start
│   ● ─ ─ ─ ─ ─ ─ ● row1│      (8 rows sampled)
│   ● ─ ─ ─ ─ ─ ─ ● row2│
│   ● ─ ─ ─ ─ ─ ─ ● ...│
│   ● ─ ─ ─ ─ ─ ─ ● row8│
├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤  85% ← scan_row_end
│     (not scanned)       │
└─────────────────────────┘ 100%
```

## PD Steering Controller

```
error = (lane_center_x - image_center_x) / (image_width / 2)

steering = -(Kp × error + Kd × d_error/dt)

speed = cruise_speed × (1 - curve_slowdown × |steering| / max_steering)
speed = max(min_curve_speed, speed)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kp` | 0.005 | Proportional gain. Increase for sharper corrections. |
| `kd` | 0.002 | Derivative gain. Increase to dampen oscillation. |
| `cruise_speed` | 0.15 m/s | Forward speed on straights |
| `min_curve_speed` | 0.05 m/s | Minimum speed on tight curves |
| `curve_slowdown_factor` | 0.6 | 0=no slowdown, 1=full stop on tight curves |
| `road_fallback_speed` | 0.08 m/s | Speed when using road centroid only |
| `lane_width_px` | 200 | Assumed lane width for single-line offset |
| `max_angular_z` | 0.8 | Maximum steering output (rad/s) |

## Traffic Light Response

HybridNets also detects traffic lights (red, green, off). The system:
1. **Red light** → Immediately stops. Stays stopped for `red_light_hold_time` (2s)
   after the red light disappears (in case of brief occlusion)
2. **Green light** → Clears the stop, resumes lane following
3. **Off light** → Treats as caution, holds current state
4. **No light** → No effect on driving

## Safety Systems

1. **Gamepad override**: Hold LB on the gamepad to take manual control at any time.
   The `cmd_vel_mux` ensures teleop always has priority over autonomous commands.
2. **No-guidance timeout**: If no lane or road is detected for 1.5 seconds, the vehicle stops.
3. **Shutdown stop**: On Ctrl+C or node crash, a stop command is sent.
4. **Arduino watchdog**: The Arduino firmware stops motors if no command received for 500ms.
5. **Speed limits**: Maximum PWM is capped at 60 (conservative for lane-following).
