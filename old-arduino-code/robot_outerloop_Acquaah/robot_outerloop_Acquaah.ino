#include <ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Int32.h>
#include <Servo.h>
#include <math.h>  // For isnan()

const int RIGHT_SERVO_ANGLE = 120;
const int LEFT_SERVO_ANGLE = 72;
const int STRAIGHT_SERVO_ANGLE = 93;
const float STOP_THRESHOLD = 0.7; // Distance to stop
const int DRIVE_SPEED = 70;  // Speed when moving

int DIR1 = 8;
int PWM1 = 9;
int DIR2 = 10;
int PWM2 = 11;

float distance = 0.0;
float last_valid_distance = 0.0; // New: track last usable value
unsigned long last_distance_update = 0;
const unsigned long DISTANCE_TIMEOUT = 1000;

int object_detected = 0;
Servo myservo;
ros::NodeHandle nh;

int steering_angle = STRAIGHT_SERVO_ANGLE;

std_msgs::String debug_msg;
ros::Publisher debug_pub("debug_info", &debug_msg);
char debug_buffer[100];

enum RobotState {IDLE = 0, STOPPED = 1, MOVING = 2};
RobotState currentState = IDLE;

void drive(int angle, int speed) {
    myservo.write(angle);
    digitalWrite(DIR1, LOW);
    digitalWrite(DIR2, LOW);
    analogWrite(PWM1, speed);
    analogWrite(PWM2, speed);
}

void stopRobot() {
    myservo.write(STRAIGHT_SERVO_ANGLE);
    digitalWrite(DIR1, LOW);
    digitalWrite(DIR2, LOW);
    analogWrite(PWM1, 0);
    analogWrite(PWM2, 0);
    currentState = STOPPED;

    snprintf(debug_buffer, sizeof(debug_buffer), "Robot stopped due to object at %.2f meters.", distance);
    debug_msg.data = debug_buffer;
    debug_pub.publish(&debug_msg);
}

void resumeMovement() {
    snprintf(debug_buffer, sizeof(debug_buffer), "Moving forward. Distance: %.2f meters.", distance);
    debug_msg.data = debug_buffer;
    debug_pub.publish(&debug_msg);

    drive(steering_angle, DRIVE_SPEED);
    currentState = MOVING;
}

// ✅ Callback: Receives object distance
void distanceCallback(const geometry_msgs::PointStamped& msg) {
    last_distance_update = millis();
    distance = msg.point.x;

    if (distance > 0.05 && !isnan(distance)) {
        last_valid_distance = distance;
    }

    snprintf(debug_buffer, sizeof(debug_buffer), "Received distance from Python: %.2f meters", distance);
    debug_msg.data = debug_buffer;
    debug_pub.publish(&debug_msg);
}

// ✅ Callback: Detection status
void detectionCallback(const std_msgs::Int32& msg) {
    object_detected = msg.data;
    snprintf(debug_buffer, sizeof(debug_buffer), "Object detection status: %d", object_detected);
    debug_msg.data = debug_buffer;
    debug_pub.publish(&debug_msg);
}

// ✅ Callback: Steering angle
void steeringCallback(const std_msgs::Int32& msg) {
    steering_angle = msg.data;
    snprintf(debug_buffer, sizeof(debug_buffer), "Updated Steering Angle: %d", steering_angle);
    debug_msg.data = debug_buffer;
    debug_pub.publish(&debug_msg);

    if (currentState == MOVING) {
        drive(steering_angle, DRIVE_SPEED);
    }
}

void controlRobot() {
    // 🔍 Invalid depth but object detected
    if ((distance <= 0.05 || isnan(distance)) && object_detected > 0) {
        if (last_valid_distance <= STOP_THRESHOLD) {
            snprintf(debug_buffer, sizeof(debug_buffer), "Stopping: invalid distance and last valid was close (%.2f m).", last_valid_distance);
            debug_msg.data = debug_buffer;
            debug_pub.publish(&debug_msg);
            stopRobot();
        } else {
            snprintf(debug_buffer, sizeof(debug_buffer), "Skipping stop: invalid distance but last valid was far (%.2f m).", last_valid_distance);
            debug_msg.data = debug_buffer;
            debug_pub.publish(&debug_msg);
        }
        return;
    }

    // ⏱ Timeout + no detection
    if ((millis() - last_distance_update > DISTANCE_TIMEOUT) && object_detected == 0) {
        snprintf(debug_buffer, sizeof(debug_buffer), "No new distance updates and no object detected. Resuming movement.");
        debug_msg.data = debug_buffer;
        debug_pub.publish(&debug_msg);

        if (currentState == STOPPED) {
            snprintf(debug_buffer, sizeof(debug_buffer), "Resuming movement since no object is detected.");
            debug_msg.data = debug_buffer;
            debug_pub.publish(&debug_msg);
            resumeMovement();
        }
        return;
    }

    // 🚦Normal distance-based logic
    if (distance > 0 && distance <= STOP_THRESHOLD) {
        if (currentState != STOPPED) {
            stopRobot();
        }
    } else if (distance > STOP_THRESHOLD) {
        if (currentState == STOPPED) {
            resumeMovement();
        }
    }
}

// ✅ Subscribers
ros::Subscriber<geometry_msgs::PointStamped> sub_distance("object_distance", distanceCallback);
ros::Subscriber<std_msgs::Int32> sub_detection("/yolov5/detection_status", detectionCallback);
ros::Subscriber<std_msgs::Int32> sub_steering("steering_angle", steeringCallback);

void setup() {
    Serial.begin(57600);
    nh.initNode();
    nh.subscribe(sub_distance);
    nh.subscribe(sub_detection);
    nh.subscribe(sub_steering);
    nh.advertise(debug_pub);

    myservo.attach(6);
    myservo.write(STRAIGHT_SERVO_ANGLE);

    pinMode(DIR1, OUTPUT);
    pinMode(PWM1, OUTPUT);
    pinMode(DIR2, OUTPUT);
    pinMode(PWM2, OUTPUT);

    currentState = IDLE;
    snprintf(debug_buffer, sizeof(debug_buffer), "Setup complete. State: IDLE");
    debug_msg.data = debug_buffer;
    debug_pub.publish(&debug_msg);
}

void loop() {
    nh.spinOnce();
    controlRobot();
}
