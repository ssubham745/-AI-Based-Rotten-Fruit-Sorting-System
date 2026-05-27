// fruit_sorter.ino


#include <Servo.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ── PIN DEFINITIONS ─────────────────────────────────────
// DC Motor (L298N) pins
const int MOTOR_IN1 = 8;
const int MOTOR_IN2 = 9;
const int MOTOR_IN3 = 10;
const int MOTOR_IN4 = 11;
const int MOTOR_ENA = 6;   // PWM pin (speed control)
const int MOTOR_ENB = 5;   // PWM pin (speed control)

// Servo pin
const int SERVO_PIN = 3;

// Motor speed (0–255)
const int BELT_SPEED = 180;   // Conveyor belt normal speed

// Timing (milliseconds)
// HOW LONG the belt runs when a fruit is detected
const int FRESH_MOVE_TIME  = 2000;  // 2 sec → move fruit to collection
const int ROTTEN_MOVE_TIME = 800;   // 0.8 sec → just bring to ejector arm position

// Servo positions (degrees)
const int SERVO_REST    = 0;    // Arm retracted (out of the way)
const int SERVO_EJECT   = 120;  // Arm extended (pushes rotten fruit off belt)
const int SERVO_HOLD    = 500;  // Hold eject position (ms)
// ────────────────────────────────────────────────────────

// ── OBJECTS ─────────────────────────────────────────────
Servo ejectorArm;
// LCD: I2C address 0x27, 16 columns, 2 rows
LiquidCrystal_I2C lcd(0x27, 16, 2);
// ────────────────────────────────────────────────────────

// ── SETUP ───────────────────────────────────────────────
void setup() {
    Serial.begin(9600);  // Must match Python's BAUD_RATE

    // Motor pins
    pinMode(MOTOR_IN1, OUTPUT);
    pinMode(MOTOR_IN2, OUTPUT);
    pinMode(MOTOR_IN3, OUTPUT);
    pinMode(MOTOR_IN4, OUTPUT);
    pinMode(MOTOR_ENA, OUTPUT);
    pinMode(MOTOR_ENB, OUTPUT);

    // Servo
    ejectorArm.attach(SERVO_PIN);
    ejectorArm.write(SERVO_REST);  // Start with arm retracted

    // LCD init
    lcd.init();
    lcd.backlight();
    lcd.setCursor(0, 0);
    lcd.print("Fruit Sorter v1");
    lcd.setCursor(0, 1);
    lcd.print("Waiting...");

    stopBelt();  // Make sure belt is stopped at startup
    delay(2000);
    lcd.clear();
}

// ── MOTOR CONTROL FUNCTIONS ─────────────────────────────

// Move conveyor belt FORWARD
void moveBeltForward(int speed) {
    analogWrite(MOTOR_ENA, speed);
    analogWrite(MOTOR_ENB, speed);
    // Motor A: forward direction
    digitalWrite(MOTOR_IN1, HIGH);
    digitalWrite(MOTOR_IN2, LOW);
    // Motor B: forward direction
    digitalWrite(MOTOR_IN3, HIGH);
    digitalWrite(MOTOR_IN4, LOW);
}

// Stop the belt completely
void stopBelt() {
    analogWrite(MOTOR_ENA, 0);
    analogWrite(MOTOR_ENB, 0);
    digitalWrite(MOTOR_IN1, LOW);
    digitalWrite(MOTOR_IN2, LOW);
    digitalWrite(MOTOR_IN3, LOW);
    digitalWrite(MOTOR_IN4, LOW);
}

// ── SERVO EJECTION ──────────────────────────────────────

// Extend arm to push rotten fruit off the belt
void ejectFruit() {
    ejectorArm.write(SERVO_EJECT);  // Swing arm out
    delay(SERVO_HOLD);               // Hold position briefly
    ejectorArm.write(SERVO_REST);    // Retract arm back
    delay(400);                      // Wait for retraction
}

// ── LCD HELPERS ─────────────────────────────────────────

void displayFresh() {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("FRESH FRUIT");
    lcd.setCursor(0, 1);
    lcd.print("Moving forward");
}

void displayRotten() {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("ROTTEN FRUIT!");
    lcd.setCursor(0, 1);
    lcd.print("Ejecting...");
}

void displayReady() {
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("System Ready");
    lcd.setCursor(0, 1);
    lcd.print("Waiting...");
}

// ── MAIN LOGIC ──────────────────────────────────────────

// Called when Python sends 'F' → Fresh fruit
void handleFresh() {
    displayFresh();
    moveBeltForward(BELT_SPEED);
    delay(FRESH_MOVE_TIME);   // Belt runs 2 seconds
    stopBelt();
    displayReady();
}

// Called when Python sends 'R' → Rotten fruit
void handleRotten() {
    displayRotten();

    // Step 1: Move belt a small distance to bring
    //         rotten fruit to ejector arm position
    moveBeltForward(BELT_SPEED);
    delay(ROTTEN_MOVE_TIME);  // Belt runs 0.8 seconds
    stopBelt();

    delay(200);  // Brief pause before ejection

    // Step 2: Servo arm swings out and pushes fruit off
    ejectFruit();

    displayReady();
}

// ── LOOP ────────────────────────────────────────────────
void loop() {
    // Check if Python has sent a command via USB Serial
    if (Serial.available() > 0) {
        char command = Serial.read();  // Read one character

        Serial.print("Received: ");
        Serial.println(command);  // Echo back for debugging

        if (command == 'F') {
            handleFresh();
        }
        else if (command == 'R') {
            handleRotten();
        }
    }
}
