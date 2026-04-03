// Ackermann Drive Controller for Yahboom Chassis
// Receives serial commands from ROS 2 serial bridge and drives
// steering servo + two rear DC motors.
//
// Protocol (newline-terminated):  S<speed>,A<angle>\n
//   speed: -255..255   (negative = reverse)
//   angle:   15..165   (servo degrees; 92 = straight)
//
// Pin assignments match the Yahboom Ackermann chassis wiring.

#include <Servo.h>

// ── Pin Definitions (from working Qcar code) ────────────────────
const int SERVO_PIN = 6;
const int DIR1      = 8;   // Motor A direction
const int PWM1      = 9;   // Motor A speed
const int DIR2      = 10;  // Motor B direction
const int PWM2      = 11;  // Motor B speed

// ── Servo Limits ────────────────────────────────────────────────
const int SERVO_CENTER = 92;
const int SERVO_MIN    = 15;   // full right (low angle side)
const int SERVO_MAX    = 165;  // full left  (high angle side)

// ── Safety ──────────────────────────────────────────────────────
const unsigned long CMD_TIMEOUT_MS = 500;

Servo steeringServo;
unsigned long lastCmdTime = 0;
char cmdBuf[32];
int  cmdIdx = 0;

void setMotors(int speed) {
    if (speed > 0) {
        digitalWrite(DIR1, HIGH);
        digitalWrite(DIR2, HIGH);
        analogWrite(PWM1, constrain(speed, 0, 255));
        analogWrite(PWM2, constrain(speed, 0, 255));
    } else if (speed < 0) {
        digitalWrite(DIR1, LOW);
        digitalWrite(DIR2, LOW);
        analogWrite(PWM1, constrain(-speed, 0, 255));
        analogWrite(PWM2, constrain(-speed, 0, 255));
    } else {
        analogWrite(PWM1, 0);
        analogWrite(PWM2, 0);
    }
}

void stopAll() {
    steeringServo.write(SERVO_CENTER);
    analogWrite(PWM1, 0);
    analogWrite(PWM2, 0);
}

void processCommand(const char *cmd) {
    int speed = 0;
    int angle = SERVO_CENTER;

    // Expected: S<int>,A<int>
    if (sscanf(cmd, "S%d,A%d", &speed, &angle) == 2) {
        speed = constrain(speed, -255, 255);
        angle = constrain(angle, SERVO_MIN, SERVO_MAX);

        steeringServo.write(angle);
        setMotors(speed);
        lastCmdTime = millis();
    }
}

void setup() {
    Serial.begin(115200);

    steeringServo.attach(SERVO_PIN);
    steeringServo.write(SERVO_CENTER);

    pinMode(DIR1, OUTPUT);
    pinMode(PWM1, OUTPUT);
    pinMode(DIR2, OUTPUT);
    pinMode(PWM2, OUTPUT);
    stopAll();

    Serial.println("ACKERMANN_READY");
}

void loop() {
    // Read serial commands
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (cmdIdx > 0) {
                cmdBuf[cmdIdx] = '\0';
                processCommand(cmdBuf);
                cmdIdx = 0;
            }
        } else if (cmdIdx < (int)sizeof(cmdBuf) - 1) {
            cmdBuf[cmdIdx++] = c;
        }
    }

    // Watchdog: stop if no command received recently
    if (millis() - lastCmdTime > CMD_TIMEOUT_MS) {
        stopAll();
    }
}
