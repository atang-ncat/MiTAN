# HybridNets Lane-Following Logic

## Overview

The MiTAN vehicle uses a fine-tuned HybridNets model running on TensorRT to perform
real-time lane-following on the Quanser studio track. The system processes camera frames,
extracts steering guidance using a **two-pass global priority system**, and outputs
Ackermann drive commands via PID control.

## Architecture

```
Camera (1280x720 RGB @ 30 FPS)
        │
        ▼ (downscaled to 640×360)
┌─────────────────┐
│   HybridNets    │  TensorRT FP16 engine
│   (384×640)     │  Outputs: detection boxes, road mask, lane mask
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│  Lane Center    │  Two-pass: yellow priority, white fallback
│  Extraction     │  + HSV color classification
└─────┬───────────┘
      │  lateral error [-1, 1]
      ▼
┌─────────────────┐
│  PID Controller │  Kp=0.8, Kd=0.20 (normalized error)
│  + Curve Speed  │  Slows on curves (60% reduction at max steer)
│  + Rate Limiter │  Max 0.30 rad/s change per frame
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

## Two-Pass Lane Guidance

The lane center extraction uses a **two-pass global priority system** that ensures
the yellow center line is always the primary guidance source when visible, with no
contamination from the white edge line.

### Pass 1 — Scan & Classify

The system scans **10 horizontal rows** between 45%–85% of the frame height.
At each row:

1. **Gather pixels** from HybridNets lane segmentation mask
2. **Union with HSV yellow mask** — catches yellow even if the neural network misses it
3. **Cluster** pixels into separate line segments (gap > 15px = separate cluster)
4. **Classify** each cluster by HSV color:
   - **Yellow** (H: 15–40, S > 80, V > 80) → center lane line
   - **White** (S < 60, V > 140) → edge lane line
   - **Unknown** → treated as white

Yellow and white positions are stored **separately** across all rows.

### Pass 2 — Decide (Global, Never Mixed)

After scanning all rows, the system makes a **single global decision**:

| Priority | Condition | Steering Target | Speed |
|----------|-----------|-----------------|-------|
| **1 (best)** | Yellow found in **any** row | Drive to the right of yellow line (offset ≈ 35–70 px) | cruise (0.20 m/s) |
| **2 (fallback)** | **No yellow at all** | Stay left of rightmost white line (offset ≈ 30–70 px) | cruise (0.20 m/s) |
| **3 (emergency)** | No lane lines at all | Road-mask drivable area, right-biased (65%) | reduced (0.06 m/s) |
| **4 (fail)** | Nothing visible for 1.5 s | Full stop | 0 m/s |

> **Critical design rule:** Yellow and white are **never mixed** in the same
> frame.  If yellow is visible in even one scan row, **all white detections
> are completely ignored**.  This prevents the white edge line from
> pulling the vehicle off-track at corners where yellow curves but white
> goes straight.

### Why Two-Pass?

The earlier per-row approach would decide independently at each scan row:
"see yellow → use it, don't see yellow → fall back to white." At corners,
HybridNets detects yellow intermittently — some rows catch it, others don't.
The rows that missed yellow would fall through to the white line, injecting
rightward-biased points that contaminated the steering path. The two-pass
approach eliminates this by deciding **once** for the entire frame.

## Scan Region

```
┌─────────────────────────┐
│                         │  0%
│     (not scanned)       │
│                         │
├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤  45% ← scan_row_start
│   ●                row1 │      (10 rows sampled)
│   ●                row2 │
│   ●                ...  │
│   ●                row10│
├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┤  85% ← scan_row_end
│     (not scanned)       │
└─────────────────────────┘ 100%
```

The scan region starts at 45% to provide **look-ahead** for curves — earlier
rows (further ahead on the road) are weighted more heavily in the final error
calculation, allowing the vehicle to anticipate turns.

## Path Smoothing

After collecting center points, a **quadratic polynomial fit** (degree 2) is
applied to smooth the path while preserving curve shape. This is important
because a linear fit would straighten curved paths, causing the vehicle to
cut corners.

## PID Steering Controller

```
error = (weighted_lane_center_x - image_center_x) / (image_width / 2)

steering = -(Kp × error + Ki × ∫error + Kd × d_error/dt)
steering = clamp(steering, -max_angular_z, max_angular_z)

speed = cruise_speed × (1 - curve_slowdown × |steering| / max_steering)
speed = max(min_curve_speed, speed)
```

The error uses **look-ahead weighting**: top scan rows (further ahead) have
higher weights, so the vehicle steers toward where the road is going rather
than where it currently is.

A **steering rate limiter** (max 0.30 rad/s change per frame) prevents
wild oscillations. An **adaptive EMA** smooths the lateral error, with
the smoothing factor increasing on curves for faster response.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kp` | 0.8 | Proportional gain (on normalized error ±1) |
| `ki` | 0.0 | Integral gain (currently disabled) |
| `kd` | 0.20 | Derivative gain (dampens oscillation) |
| `cruise_speed` | 0.08 m/s | Base forward speed (overridden to 0.20 via launch) |
| `min_curve_speed` | 0.16 m/s | Minimum speed on tight curves |
| `curve_slowdown_factor` | 0.6 | 0=no slowdown, 1=full stop on tight curves |
| `road_fallback_speed` | 0.06 m/s | Speed when using road mask only |
| `lane_width_px` | 200 | Lane width used for offset calculation |
| `max_angular_z` | 0.8 | Maximum steering output (rad/s) |
| `max_steer_rate` | 0.30 | Maximum steering change per frame |

## Traffic Light Response

HybridNets also detects traffic lights (red, green, off). Currently **disabled**
to focus on lane following. When enabled:

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
5. **Speed limits**: Maximum PWM is capped at 150 for lane-following.
