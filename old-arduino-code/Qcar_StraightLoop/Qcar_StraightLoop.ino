#include <ros.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PointStamped.h>
#include <std_msgs/Int32.h>
#include <Servo.h>

const int FIRST_LEFT_SERVO_ANGLE = 68; //adjusted for sharper  turns
const int SECOND_LEFT_SERVO_ANGLE = 73;   // Adjusted for less sharp left turns73
const int STRAIGHT_SERVO_ANGLE = 90;  // Adjusted for better straight alignment (was 92)
const unsigned long FIRST_LEFT_ACTION_DURATION = 3000; // ms
const unsigned long SECOND_LEFT_ACTION_DURATION = 4800;  // ms4800
const unsigned long STRAIGHT_ACTION_DURATION = 2700; // ms
const float STOP_THRESHOLD = 1.90; // Distance threshold for stopping
const int DRIVE_SPEED = 72;

const unsigned long TRAFFIC_LIGHT_DELAY = 5000;  // 5 second delay between light detections

int DIR1 = 8;
int PWM1 = 9;
int DIR2 = 10;
int PWM2 = 11;

float x = 0.0, y = 5.0; // Start with no traffic light detected
Servo myservo;
ros::NodeHandle nh;

int steering_angle = STRAIGHT_SERVO_ANGLE;
int light = 1;  // First green light to be processed
unsigned long lastLightDetectedTime = 0;  // Store the time of the last light detection

std_msgs::String debug_msg;
ros::Publisher debug_pub("debug_info", &debug_msg);

char debug_buffer[100];

enum RobotState {IDLE = 0, STOPPED = 1, MOVING = 2};
RobotState currentState = IDLE; // Start in IDLE state

void drive(int angle, int speed) {
  myservo.write(angle);  // Immediately update servo with the new angle
  digitalWrite(DIR1, HIGH); // was LOW
  digitalWrite(DIR2, HIGH);// was LOW
  analogWrite(PWM1, speed);
  analogWrite(PWM2, speed);
}

void stopRobot() {
  myservo.write(90);  // Center the wheels
  digitalWrite(DIR1,LOW); // was HIGH
  digitalWrite(DIR2, LOW); //was HIGH
  analogWrite(PWM1, 0);
  analogWrite(PWM2, 0);
  currentState = STOPPED;  // Set state to STOPPED
  snprintf(debug_buffer, sizeof(debug_buffer), "Robot stopped. Current state: STOPPED");
  debug_msg.data = debug_buffer;
  debug_pub.publish(&debug_msg);
}

void performMovement(int servoAngle, unsigned long duration) {
  snprintf(debug_buffer, sizeof(debug_buffer), "Starting movement with servo angle: %d", servoAngle);
  debug_msg.data = debug_buffer;
  debug_pub.publish(&debug_msg);
  currentState = MOVING;
  unsigned long startTime = millis();
  while (millis() - startTime < duration) {
    drive(servoAngle, DRIVE_SPEED);  // Maintain the movement angle
    nh.spinOnce();  // Process ROS messages during movement
  }
  currentState = IDLE;  // Ensure robot returns to IDLE after movement
}

void messageCb1(const geometry_msgs::PointStamped& pose_msg) {
  if (currentState == IDLE || currentState == STOPPED) {  // Only process when in IDLE or STOPPED
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
  unsigned long currentTime = millis();

  if (currentState == IDLE || currentState == STOPPED) {
    if (y == 1.0 && x < STOP_THRESHOLD && x != 0.0) {
      stopRobot();
    } else if (y == 3.0 && x < STOP_THRESHOLD && x != 0.0) {
      // Check if enough time has passed since the last traffic light detection
      if (currentTime - lastLightDetectedTime > TRAFFIC_LIGHT_DELAY) {
        // Green light actions: change based on the light number
        if (light == 1) {  // First green light: go straight
          myservo.write(STRAIGHT_SERVO_ANGLE);
          delay(100);  // Brief delay to ensure servo sets to straight
          snprintf(debug_buffer, sizeof(debug_buffer), "Going straight at first green light after delay, resetting light to 2");
          debug_msg.data = debug_buffer;
          debug_pub.publish(&debug_msg);

          // Proceed with the straight movement after delay
          performMovement(STRAIGHT_SERVO_ANGLE, STRAIGHT_ACTION_DURATION);

          light = 2;
          lastLightDetectedTime = currentTime;  // Update the time of detection
        } else if (light == 2) {  // Second green light: turn left
          // Ensure straight angle is correctly set

          snprintf(debug_buffer, sizeof(debug_buffer), "Turning left at second green light, resetting light to 3");
          debug_msg.data = debug_buffer;
          debug_pub.publish(&debug_msg);
          performMovement(FIRST_LEFT_SERVO_ANGLE, FIRST_LEFT_ACTION_DURATION);

          light = 3;
          lastLightDetectedTime = currentTime;  // Update the time of detection
        } else if (light == 3) {  // Third green light: unchanged behavior (left turn)
          snprintf(debug_buffer, sizeof(debug_buffer), "Turning left at third green light, resetting light to 1");
          debug_msg.data = debug_buffer;
          debug_pub.publish(&debug_msg);
          performMovement(SECOND_LEFT_SERVO_ANGLE, SECOND_LEFT_ACTION_DURATION);

          light = 1;  // Reset to the first light
          lastLightDetectedTime = currentTime;  // Update the time of detection
        }
        y = 5.0;  // Reset the light detection
      }
    } else if (y == 5) {  // No traffic light detected
      drive(steering_angle, 72);  // Keep driving with current steering angle
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

  myservo.attach(6);
  myservo.write(90);

  pinMode(DIR1, OUTPUT);
  pinMode(PWM1, OUTPUT);
  pinMode(DIR2, OUTPUT);
  pinMode(PWM2, OUTPUT);

  currentState = IDLE;  // Ensure starting in IDLE state
  snprintf(debug_buffer, sizeof(debug_buffer), "Setup completed. Initial state: IDLE, light=%d", light);
  debug_msg.data = debug_buffer;
  debug_pub.publish(&debug_msg);
}

void loop() {
  nh.spinOnce();
  controlRobot();
}
