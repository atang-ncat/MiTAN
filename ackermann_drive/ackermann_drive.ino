// Ackermann Drive Controller — V4-DEBUG
// Heavy diagnostic output for tracking down direction issues.
//
// Protocol:  S<speed>,A<angle>\n
//
// ─── DIRECTION CONFIG ────────────────────────────────────────
#define FORWARD_DIR  LOW    // Confirmed: DIR=LOW = forward on this hardware
#define BACKWARD_DIR HIGH   // Confirmed: DIR=HIGH = backward
// ─────────────────────────────────────────────────────────────

#include <Servo.h>

const int SERVO_PIN = 6;
const int DIR1      = 8;
const int PWM1      = 9;
const int DIR2      = 10;
const int PWM2      = 11;
const int LED_PIN   = 13;

const int SERVO_CENTER = 92;
const int SERVO_MIN    = 15;
const int SERVO_MAX    = 165;
const unsigned long CMD_TIMEOUT_MS = 500;

Servo steeringServo;
unsigned long lastCmdTime = 0;
char cmdBuf[32];
int  cmdIdx = 0;
unsigned long cmdCount = 0;

// ── PIN STATE READBACK ──────────────────────────────────────
void logPinStates(const char* context) {
    Serial.print("[PIN] ");
    Serial.print(context);
    Serial.print(" | DIR1(8)=");
    Serial.print(digitalRead(DIR1));
    Serial.print(" DIR2(10)=");
    Serial.print(digitalRead(DIR2));
    // Read actual PWM register values for Timer2 (pins 9,10)
    Serial.print(" | TCCR2A=0x");
    Serial.print(TCCR2A, HEX);
    Serial.print(" OCR2A=");
    Serial.print(OCR2A);
    Serial.print(" OCR2B=");
    Serial.println(OCR2B);
}

// ── SELF-TEST ───────────────────────────────────────────────
void pinSelfTest() {
    Serial.println("\n=== PIN SELF-TEST ===");

    // Test DIR1 (pin 8)
    digitalWrite(DIR1, HIGH);
    delay(1);
    int d1h = digitalRead(DIR1);
    digitalWrite(DIR1, LOW);
    delay(1);
    int d1l = digitalRead(DIR1);
    Serial.print("DIR1(8): write HIGH→read ");
    Serial.print(d1h);
    Serial.print(", write LOW→read ");
    Serial.print(d1l);
    Serial.println(d1h == HIGH && d1l == LOW ? " ✓ OK" : " ✗ FAIL!");

    // Test DIR2 (pin 10) — shares Timer2 with PWM1 (pin 9)!
    digitalWrite(DIR2, HIGH);
    delay(1);
    int d2h = digitalRead(DIR2);
    digitalWrite(DIR2, LOW);
    delay(1);
    int d2l = digitalRead(DIR2);
    Serial.print("DIR2(10): write HIGH→read ");
    Serial.print(d2h);
    Serial.print(", write LOW→read ");
    Serial.print(d2l);
    Serial.println(d2h == HIGH && d2l == LOW ? " ✓ OK" : " ✗ FAIL!");

    // Test PWM1 (pin 9)
    analogWrite(PWM1, 100);
    delay(1);
    Serial.print("PWM1(9): After analogWrite(100), TCCR2A=0x");
    Serial.print(TCCR2A, HEX);
    Serial.print(" — DIR2(10) now reads: ");
    Serial.println(digitalRead(DIR2));
    analogWrite(PWM1, 0);

    // Test: does analogWrite(9) corrupt DIR2(10)?
    Serial.println("--- Timer2 Conflict Test ---");
    digitalWrite(DIR2, HIGH);
    Serial.print("  Set DIR2=HIGH, read: ");
    Serial.println(digitalRead(DIR2));
    analogWrite(PWM1, 100);   // This uses Timer2!
    Serial.print("  After analogWrite(PWM1,100), DIR2 reads: ");
    Serial.println(digitalRead(DIR2));
    analogWrite(PWM1, 0);
    Serial.print("  After analogWrite(PWM1,0), DIR2 reads: ");
    Serial.println(digitalRead(DIR2));

    // Test with Servo attached
    Serial.println("--- Servo Interference Test ---");
    steeringServo.attach(SERVO_PIN);
    steeringServo.write(SERVO_CENTER);
    delay(20);
    digitalWrite(DIR1, HIGH);
    digitalWrite(DIR2, HIGH);
    delay(1);
    Serial.print("  After servo.write(92): DIR1=");
    Serial.print(digitalRead(DIR1));
    Serial.print(" DIR2=");
    Serial.println(digitalRead(DIR2));

    logPinStates("POST-SELFTEST");
    Serial.println("=== SELF-TEST DONE ===\n");
}

void stopMotors() {
    analogWrite(PWM1, 0);
    analogWrite(PWM2, 0);
}

void setMotors(int speed) {
    int pwm = abs(speed);
    pwm = constrain(pwm, 0, 255);

    if (speed > 0) {
        digitalWrite(DIR1, FORWARD_DIR);
        digitalWrite(DIR2, FORWARD_DIR);
        analogWrite(PWM1, pwm);
        analogWrite(PWM2, pwm);
    } else if (speed < 0) {
        digitalWrite(DIR1, BACKWARD_DIR);
        digitalWrite(DIR2, BACKWARD_DIR);
        analogWrite(PWM1, pwm);
        analogWrite(PWM2, pwm);
    } else {
        stopMotors();
    }

    // Log pin states AFTER setting them
    if (cmdCount % 20 == 0) {  // every 20th command to avoid flooding
        char buf[24];
        snprintf(buf, sizeof(buf), "setMotors(%d)", speed);
        logPinStates(buf);
    }
}

void processCommand(const char *cmd) {
    int speed = 0;
    int angle = SERVO_CENTER;

    if (sscanf(cmd, "S%d,A%d", &speed, &angle) == 2) {
        speed = constrain(speed, -255, 255);
        angle = constrain(angle, SERVO_MIN, SERVO_MAX);

        cmdCount++;

        // Blink LED on valid command
        digitalWrite(LED_PIN, !digitalRead(LED_PIN));

        steeringServo.write(angle);
        setMotors(speed);
        lastCmdTime = millis();

        // Log every 20th command
        if (cmdCount % 20 == 0) {
            Serial.print("[CMD #");
            Serial.print(cmdCount);
            Serial.print("] raw='");
            Serial.print(cmd);
            Serial.print("' → speed=");
            Serial.print(speed);
            Serial.print(" angle=");
            Serial.print(angle);
            Serial.print(" FORWARD_DIR=");
            Serial.println(FORWARD_DIR == HIGH ? "HIGH" : "LOW");
        }
    } else {
        // Parse failed — log the bad input
        Serial.print("[PARSE FAIL] raw='");
        Serial.print(cmd);
        Serial.print("' len=");
        Serial.print(strlen(cmd));
        Serial.print(" hex:");
        for (int i = 0; cmd[i]; i++) {
            Serial.print(" 0x");
            Serial.print((int)cmd[i], HEX);
        }
        Serial.println();
    }
}

void setup() {
    Serial.begin(115200);
    while (!Serial) { ; }  // Wait for serial connection

    Serial.println("\n\n================================");
    Serial.println("  ACKERMANN V4-DEBUG STARTING");
    Serial.println("================================");

    pinMode(LED_PIN, OUTPUT);
    pinMode(DIR1, OUTPUT);
    pinMode(PWM1, OUTPUT);
    pinMode(DIR2, OUTPUT);
    pinMode(PWM2, OUTPUT);

    // Stop motors FIRST
    stopMotors();
    digitalWrite(DIR1, LOW);
    digitalWrite(DIR2, LOW);

    Serial.println("[SETUP] Pins configured, motors stopped");
    Serial.print("[SETUP] Pin assignments: DIR1=");
    Serial.print(DIR1);
    Serial.print(" PWM1=");
    Serial.print(PWM1);
    Serial.print(" DIR2=");
    Serial.print(DIR2);
    Serial.print(" PWM2=");
    Serial.println(PWM2);
    Serial.print("[SETUP] Timer2 note: Pin 9 (PWM1) and Pin 10 (DIR2) share Timer2!\n");

    // Run diagnostics
    pinSelfTest();

    // LED confirmation
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_PIN, HIGH);
        delay(100);
        digitalWrite(LED_PIN, LOW);
        delay(100);
    }

    Serial.println("[SETUP] ACKERMANN_V4_DEBUG_READY");
    Serial.print("[SETUP] FORWARD_DIR=");
    Serial.println(FORWARD_DIR == HIGH ? "HIGH" : "LOW");
    Serial.println("[SETUP] Waiting for serial commands (S<speed>,A<angle>)...\n");

    lastCmdTime = millis();
}

void loop() {
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

    // Watchdog
    if (millis() - lastCmdTime > CMD_TIMEOUT_MS) {
        stopMotors();
    }
}
