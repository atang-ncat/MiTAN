#include <ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Int32.h>
#include <Arduino.h>
#include <Servo.h>

const int RIGHT_SERVO_ANGLE = 121;
const int LEFT_SERVO_ANGLE = 73;
const int STRAIGHT_SERVO_ANGLE = 80;
const unsigned long RIGHT_ACTION_DURATION = 100;
const unsigned long LEFT_ACTION_DURATION = 50;
const unsigned long STRAIGHT_ACTION_DURATION = 50;
const float STOP_THRESHOLD = 1.70;
const int DRIVE_SPEED = 74;

int DIR1 = 8;
int PWM1 = 9;
int DIR2 = 10;
int PWM2 = 11;

float x = 0.0, y = 5.0; // Start with no traffic light detected
Servo myservo;
ros::NodeHandle nh;

int steering_angle = STRAIGHT_SERVO_ANGLE;
int light = 1;

std_msgs::String debug_msg;
ros::Publisher debug_pub("debug_info", &debug_msg);

// New publisher for mode control
std_msgs::String mode_msg;
ros::Publisher mode_pub("mode_control", &mode_msg);

char debug_buffer[100];
char mode_buffer[20];

enum RobotState {IDLE = 0, STOPPED = 1, MOVING = 2};
RobotState currentState = IDLE;

void drive(int angle, int speed) {
  myservo.write(angle);
  digitalWrite(DIR1, HIGH);
  digitalWrite(DIR2, HIGH);
  analogWrite(PWM1, speed);
  analogWrite(PWM2, speed);
}

void stopRobot() {
  myservo.write(90);
  digitalWrite(DIR1, LOW);
  digitalWrite(DIR2, LOW);
  analogWrite(PWM1, 0);
  analogWrite(PWM2, 0);
  currentState = STOPPED;
  snprintf(debug_buffer, sizeof(debug_buffer), "Robot stopped. Current state: STOPPED");
  debug_msg.data = debug_buffer;
  debug_pub.publish(&debug_msg);
}

void performMovement(int servoAngle, unsigned long duration) {
  // Tell the lane-keeping node to pause
  snprintf(mode_buffer, sizeof(mode_buffer), "manual_control");
  mode_msg.data = mode_buffer;
  mode_pub.publish(&mode_msg);

  snprintf(debug_buffer, sizeof(debug_buffer), "Starting movement with servo angle: %d", servoAngle);
  debug_msg.data = debug_buffer;
  debug_pub.publish(&debug_msg);
  currentState = MOVING;
  unsigned long startTime = millis();
  while (millis() - startTime < duration) {
    drive(servoAngle, DRIVE_SPEED);
    nh.spinOnce();
  }

  // Tell the lane-keeping node to resume
  snprintf(mode_buffer, sizeof(mode_buffer), "auto_lane_keeping");
  mode_msg.data = mode_buffer;
  mode_pub.publish(&mode_msg);
  
  currentState = IDLE;
}

void messageCb1(const geometry_msgs::PointStamped& pose_msg) {
  if (currentState == IDLE || currentState == STOPPED) {
    x = pose_msg.point.x;
    y = pose_msg.point.y;
    snprintf(debug_buffer, sizeof(debug_buffer), "Received data: x=%.2f, y=%.2f, light=%d", x, y, light);
    debug_msg.data = debug_buffer;
    debug_pub.publish(&debug_msg);
  }
}

void messageCb(const std_msgs::Int32& angle_msg) {
  steering_angle = angle_msg.data;
  if (currentState == IDLE) {
    drive(steering_angle, DRIVE_SPEED);
  }
}

void controlRobot() {
  if (currentState == IDLE || currentState == STOPPED) {
    if (y == 1.0 && x < STOP_THRESHOLD && x != 0.0) {
      // Manual control for stopping
      snprintf(mode_buffer, sizeof(mode_buffer), "manual_control");
      mode_msg.data = mode_buffer;
      mode_pub.publish(&mode_msg);
      stopRobot();
      snprintf(mode_buffer, sizeof(mode_buffer), "auto_lane_keeping");
      mode_msg.data = mode_buffer;
      mode_pub.publish(&mode_msg);
    } else if (y == 3.0 && x < STOP_THRESHOLD && x != 0.0) {
      if (light == 1) {
        snprintf(debug_buffer, sizeof(debug_buffer), "Turning right at green light, resetting light to 2");
        debug_msg.data = debug_buffer;
        debug_pub.publish(&debug_msg);
        if (x > 0.60) {
          performMovement(RIGHT_SERVO_ANGLE, RIGHT_ACTION_DURATION);
        }
        light = 2;
      } else if (light == 2) {
        snprintf(debug_buffer, sizeof(debug_buffer), "Going straight at green light, resetting light to 3");
        debug_msg.data = debug_buffer;
        debug_pub.publish(&debug_msg);
        performMovement(STRAIGHT_SERVO_ANGLE, STRAIGHT_ACTION_DURATION);
        light = 3;
      } else if (light == 3) {
        snprintf(debug_buffer, sizeof(debug_buffer), "Turning left at green light, resetting light to 1");
        debug_msg.data = debug_buffer;
        debug_pub.publish(&debug_msg);
        if (y == 3 && x < 1.80) {
          performMovement(LEFT_SERVO_ANGLE, LEFT_ACTION_DURATION);
        } else {
          performMovement(LEFT_SERVO_ANGLE, LEFT_ACTION_DURATION);
        }
        light = 1;
      }
      y = 5.0;
    } else if (y == 5) {
      drive(steering_angle, 70);
      snprintf(debug_buffer, sizeof(debug_buffer), "Driving normally, no traffic light, current light=%d", light);
      debug_msg.data = debug_buffer;
      debug_pub.publish(&debug_msg);
    }
  }
}

ros::Subscriber<geometry_msgs::PointStamped> sub1("detect", messageCb1);
ros::Subscriber<std_msgs::Int32> sub2("steering_angle", messageCb);

void setup() {
  Serial.begin(57600);
  nh.initNode();
  nh.subscribe(sub1);
  nh.subscribe(sub2);
  nh.advertise(debug_pub);
  nh.advertise(mode_pub); // Advertise the new topic

  myservo.attach(6);
  myservo.write(90);
  pinMode(DIR1, OUTPUT);
  pinMode(PWM1, OUTPUT);
  pinMode(DIR2, OUTPUT);
  pinMode(PWM2, OUTPUT);

  currentState = IDLE;
  snprintf(debug_buffer, sizeof(debug_buffer), "Setup completed. Initial state: IDLE, light=%d", light);
  debug_msg.data = debug_buffer;
  debug_pub.publish(&debug_msg);
}

void loop() {
  nh.spinOnce();
  controlRobot();
}
