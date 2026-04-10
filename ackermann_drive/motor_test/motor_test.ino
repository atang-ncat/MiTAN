// Motor Direction Test
// Drives motors forward for 2 seconds, stops 2 seconds, then reverses 2 seconds.
// Watch which direction is "forward" and which is "reverse".
// No serial commands needed — just upload and observe.

const int DIR1 = 8;
const int PWM1 = 9;
const int DIR2 = 10;
const int PWM2 = 11;

void setup() {
    Serial.begin(115200);
    pinMode(DIR1, OUTPUT);
    pinMode(PWM1, OUTPUT);
    pinMode(DIR2, OUTPUT);
    pinMode(PWM2, OUTPUT);

    // Stop first
    analogWrite(PWM1, 0);
    analogWrite(PWM2, 0);

    delay(1000);
    Serial.println("=== MOTOR DIRECTION TEST ===");

    // ── TEST 1: DIR HIGH + PWM 40 ──
    Serial.println("TEST 1: DIR=HIGH, PWM=40 (should be forward?)");
    digitalWrite(DIR1, HIGH);
    digitalWrite(DIR2, HIGH);
    analogWrite(PWM1, 40);
    analogWrite(PWM2, 40);
    delay(3000);

    // Stop
    analogWrite(PWM1, 0);
    analogWrite(PWM2, 0);
    Serial.println("STOPPED");
    delay(2000);

    // ── TEST 2: DIR LOW + PWM 40 ──
    Serial.println("TEST 2: DIR=LOW, PWM=40 (should be reverse?)");
    digitalWrite(DIR1, LOW);
    digitalWrite(DIR2, LOW);
    analogWrite(PWM1, 40);
    analogWrite(PWM2, 40);
    delay(3000);

    // Stop
    analogWrite(PWM1, 0);
    analogWrite(PWM2, 0);
    Serial.println("STOPPED");
    Serial.println("=== TEST COMPLETE ===");
    Serial.println("Which test drove FORWARD? Type back to me.");
}

void loop() {
    // Nothing — test runs once in setup()
}
