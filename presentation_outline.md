# Autonomous Ackermann Vehicle: Progress & Future Roadmap

## Slide 1: System Architecture & Setup
* **Core Compute:** Jetson AGX Orin with a high-speed 512GB NVMe SSD for fast model loading and data logging.
* **Hardware Chassis:** Yahboom Ackermann steering car.
* **Sensors:** 
  * Intel RealSense D455 (Stereo IR, RGB, Depth)
  * RPLiDAR A3
* **Software Environment:** ROS 2 Humble operating within an Isaac ROS Docker container.
* **Low-Level Control:** Custom V3 Arduino Mega firmware communicating via a robust Python `serial_bridge` handling speed inversion, scaling, and DTR/RTS signal management for stable continuous operation.

## Slide 2: Perception & Mapping Foundation
* **Compute Migration:** Full Isaac ROS perception stack migrated to the SSD to eliminate I/O bottlenecks.
* **Visual Odometry:** Configured Isaac ROS cuVSLAM utilizing RealSense stereo IR.
* **Object Detection:** Integrated state-of-the-art YOLOv10 for real-time bounding boxes.
* **Mapping:** Setup 2D occupancy grid mapping utilizing `slam_toolbox` combining odometry and `RPLiDAR` scans for loop-closure and drift correction.

## Slide 3: Autonomous Lane Following via HybridNets (Current Core Focus)
* **Model Deployment:** Successfully deployed TensorRT-optimized HybridNets for high-accuracy, real-time lane segmentation.
* **Pipeline Optimization:** Reduced compute overhead and increased FPS by downscaling input frames (640x360) and dynamically throttling visualizers.
* **Lane Keeping Strategy (Pending Real-World Validation):** We have developed robust logic to translate model outputs into steering, though actual physical track tuning is pending. The logic uses a tiered fallback system:
  * **Tier 1:** Two lane lines visible → Navigate using the midpoint between the bounds.
  * **Tier 2:** One lane line visible → Navigate via an assumed constant lane width offset.
  * **Tier 3:** No lines, but road visible → Navigate using the centroid of the drivable road mask.
  * **Traffic Lights:** Detection of a Red Light functions as a hard override, zeroing the throttle immediately.
* **Steering Control & Smoothing:** Detected lane centers generate a lateral error which is smoothed using an **Exponential Moving Average (EMA)** filter to completely eliminate jitter. This clean error feeds into a custom customizable Proportional-Derivative (PD) controller mapping the bounds to Ackermann steering angles.
* **Hardware Alignment:** Resolved physical motor layout discrepancies via the `serial_bridge`, ensuring reliable foundational hardware responsiveness for when we tune the physical movement.

## Slide 4: Future Work Part 1: Accelerating the Navigation Roadmap
* **Nav2 Ackermann Integration:** Implement the Ackermann controller plugin for Nav2, enabling the vehicle to navigate toward global GPS/map goals while respecting non-holonomic (car-like) turning constraints.
* **Depth-based Obstacle Avoidance:** Process the RealSense D455 depth stream to act as an override "virtual bumper." This will trigger emergency stops or immediate detours if an object is within a critical range, regardless of lane-following directives.
* **LiDAR-based Local Planner:** Integrate 360° RPLiDAR scans into the local costmap, allowing the system to dynamically avoid moving obstacles (like pedestrians) while maintaining the lane.

## Slide 5: Future Work Part 2: Improving Control & Robustness
* **Advanced Steering Logic:** Upgrade the foundational PD controller to industry-standard algorithms like **Pure Pursuit** or the **Stanley Controller**, drastically improving tracking smoothness and high-speed stability.
* **Guidance Smoothing:** Transition from "hard switching" between lane guidance confidence tiers (e.g., from "Two Lines" to "One Line" tracking) to a weighted interpolation blend. This prevents jerky steering adjustments during confidence drops.
* **Formal State Machine:** Refactor conditional logic in the lane follower into a dedicated State Machine (e.g., using SMACH). This provides robust handling of high-level transitions like `CRUISING` -> `STOPPED` (at red lights/signs) -> `RECOVERING`.

## Slide 6: Future Work Part 3: System Optimization & Debugging
* **Latency Profiling:** Conduct deep performance profiling to analyze end-to-end latency between the camera shutter, HybridNets inference, and serial output to squeeze out maximum responsiveness.
* **Automated Data-Driven Testing:** Construct a "simulation harness" utilizing `rosbag2` record/playback features. This allows tuning of steering logic and PID values on recorded physical data without constantly testing on the physical car.
* **Arduino Micro-Optimization:** Review and refine the `ackermann_drive.ino` firmware to ensure the hardware watchdog, PWM cycling, and serial buffer polling are highly cycle-efficient.
