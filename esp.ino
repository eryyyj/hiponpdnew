/*
  ESP32 Serial Control: Linear Actuator (L298N) + 2 Servos + 3 Relays + Load Cell (HX711)
  ----------------------------------------------------------------------------------------
  HARDWARE:
    L298N:
      IN1 -> GPIO16
      IN2 -> GPIO17
      ENA -> tied directly to 5V (always enabled, full speed)
    Servo 1 -> GPIO19
    Servo 2 -> GPIO18
    Relay 1 -> GPIO25
    Relay 2 -> GPIO26
    Relay 3 -> GPIO27
    HX711:
      DOUT -> GPIO22
      SCK  -> GPIO23

  SERIAL COMMANDS (type into Serial Monitor, 115200 baud, line ending = Newline):

    --- Actuator ---
    AU:<sec>           -> actuator UP (extend) for <sec> seconds, then auto-stop
    AD:<sec>           -> actuator DOWN (retract) for <sec> seconds, then auto-stop
    AS                 -> stop actuator immediately

    --- Servos ---
    S1:<deg> / S2:<deg>     -> move servo to <deg> (clamped)
    S1:HOME  / S2:HOME      -> move servo to its default position

    --- Relays ---
    R1ON / R1OFF, R2ON / R2OFF, R3ON / R3OFF

    --- Load cell (only 2 commands) ---
    CAL:<grams>        -> calibrate with a known weight, e.g. CAL:10
                          5 tries: remove weight -> tare -> place weight -> read.
                          The average is saved to EEPROM automatically and is
                          used right away by weight detection (and after reboot).
    W                  -> weight detection ON/OFF (toggle).
                          Prints the weight value only, e.g. 10.0

  All load cell work is non-blocking, so the actuator auto-stop, servos and
  relays keep working even during calibration.
*/

#include <ESP32Servo.h>
#include <HX711.h>
#include <EEPROM.h>

// ---------- Pin Definitions ----------
const int ACT_IN1 = 16;
const int ACT_IN2 = 17;

const int SERVO1_PIN = 19;
const int SERVO2_PIN = 18;

const int RELAY1_PIN = 25;
const int RELAY2_PIN = 26;
const int RELAY3_PIN = 27;

const int LOADCELL_DOUT_PIN = 22;
const int LOADCELL_SCK_PIN  = 23;

// If your relay modules are "active LOW" (most common cheap modules),
// set this to true. If active HIGH, set to false.
const bool RELAY_ACTIVE_LOW = true;

// ---------- Servo Objects ----------
Servo servo1;
Servo servo2;

// ---------- Servo Configurable Settings ----------
const int SERVO1_DEFAULT_POS = 160;
const int SERVO1_MAX_DEG     = 100;

const int SERVO2_DEFAULT_POS = 160;
const int SERVO2_MAX_DEG     = 0;

const int SERVO1_CLAMP_MAX = (SERVO1_DEFAULT_POS > SERVO1_MAX_DEG) ? SERVO1_DEFAULT_POS : SERVO1_MAX_DEG;
const int SERVO2_CLAMP_MAX = (SERVO2_DEFAULT_POS > SERVO2_MAX_DEG) ? SERVO2_DEFAULT_POS : SERVO2_MAX_DEG;

// ---------- Actuator State (non-blocking timer) ----------
bool actuatorRunning = false;
unsigned long actuatorStartTime = 0;
unsigned long actuatorDuration = 0;

// ---------- Load Cell Settings ----------
const float DEFAULT_CALIBRATION_FACTOR = 5908.93;  // used until you run CAL
const int   CAL_TRIES          = 5;       // calibration tries to average
const unsigned long CAL_WAIT_MS = 5000;   // time to remove/place the weight
const int   CAL_TARE_SAMPLES   = 15;      // readings averaged per calibration tare
const int   CAL_READ_SAMPLES   = 10;      // readings averaged per calibration reading
const int   BOOT_TARE_SAMPLES  = 20;      // readings averaged for tare at startup
const int   WEIGHT_SAMPLES     = 5;       // readings averaged per weight value
const float ZERO_DEADBAND      = 0.3;     // grams: smaller values print as 0
const int   WEIGHT_DECIMALS    = 1;       // decimal places in weight output
const unsigned long HX711_TIMEOUT_MS = 1500;

// ---------- EEPROM layout ----------
const int      EEPROM_SIZE      = 16;
const int      EEPROM_ADDR_MAGIC = 0;       // uint32_t marker = "factor was saved"
const int      EEPROM_ADDR_CF    = 4;       // float calibration factor
const uint32_t EEPROM_MAGIC      = 0xCA11B8ED;

HX711 scale;
float calibrationFactor = DEFAULT_CALIBRATION_FACTOR;
bool weightDetectionOn = false;

// Non-blocking sampler
enum SamplePurpose { SP_NONE, SP_CAL_TARE, SP_CAL_READ, SP_WEIGHT };
SamplePurpose samplePurpose = SP_NONE;
int64_t sampleSum = 0;
int sampleCount = 0;
int sampleTarget = 0;
unsigned long lastSampleTime = 0;

// Calibration state machine
enum CalState { CAL_IDLE, CAL_WAIT_EMPTY, CAL_TARING, CAL_WAIT_WEIGHT, CAL_READING };
CalState calState = CAL_IDLE;
float calKnownWeight = 0;
int calTry = 0;
long calResults[CAL_TRIES];
long calOffset = 0;
unsigned long calStepStart = 0;

// ---------- Setup ----------
void setup() {
  Serial.begin(115200);
  Serial.println("ESP32 Serial Control Ready.");
  Serial.println("Commands: AU:<sec> | AD:<sec> | AS | S1:<deg> | S2:<deg> | R1ON/R1OFF/R2ON/R2OFF/R3ON/R3OFF");
  Serial.println("Scale:    CAL:<grams> | W");

  pinMode(ACT_IN1, OUTPUT);
  pinMode(ACT_IN2, OUTPUT);
  stopActuator();

  pinMode(RELAY1_PIN, OUTPUT);
  pinMode(RELAY2_PIN, OUTPUT);
  pinMode(RELAY3_PIN, OUTPUT);
  setRelay(RELAY1_PIN, false);
  setRelay(RELAY2_PIN, false);
  setRelay(RELAY3_PIN, false);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  servo1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);
  servo1.attach(SERVO1_PIN, 500, 2400);
  servo2.attach(SERVO2_PIN, 500, 2400);
  servo1.write(SERVO1_DEFAULT_POS);
  servo2.write(SERVO2_DEFAULT_POS);

  // ---- Load cell ----
  EEPROM.begin(EEPROM_SIZE);
  loadCalibrationFactor();

  scale.begin(LOADCELL_DOUT_PIN, LOADCELL_SCK_PIN);
  scale.set_scale(calibrationFactor);

  Serial.print("Calibration factor: ");
  Serial.println(calibrationFactor, 2);

  if (scale.wait_ready_timeout(1000)) {
    Serial.println("Taring scale... keep it empty.");
    scale.tare(BOOT_TARE_SAMPLES);
    Serial.println("Scale ready.");
  } else {
    Serial.println("HX711 not found.");
  }
}

// ---------- Main Loop ----------
void loop() {
  handleSerialInput();
  handleActuatorTimer();
  handleScale();
}

// ---------- Serial Command Handling ----------
void handleSerialInput() {
  if (Serial.available() == 0) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  cmd.toUpperCase();

  if (cmd.length() == 0) return;

  Serial.print("Received: ");
  Serial.println(cmd);

  if (cmd.startsWith("AU:")) {
    long seconds = cmd.substring(3).toInt();
    startActuator(true, (unsigned long)seconds * 1000UL);
  }
  else if (cmd.startsWith("AD:")) {
    long seconds = cmd.substring(3).toInt();
    startActuator(false, (unsigned long)seconds * 1000UL);
  }
  else if (cmd == "AS") {
    stopActuator();
    Serial.println("Actuator stopped.");
  }
  else if (cmd == "S1:HOME") {
    servo1.write(SERVO1_DEFAULT_POS);
    Serial.print("Servo1 -> HOME (");
    Serial.print(SERVO1_DEFAULT_POS);
    Serial.println(")");
  }
  else if (cmd == "S2:HOME") {
    servo2.write(SERVO2_DEFAULT_POS);
    Serial.print("Servo2 -> HOME (");
    Serial.print(SERVO2_DEFAULT_POS);
    Serial.println(")");
  }
  else if (cmd.startsWith("S1:")) {
    int deg = constrain(cmd.substring(3).toInt(), 0, SERVO1_CLAMP_MAX);
    servo1.write(deg);
    Serial.print("Servo1 -> ");
    Serial.println(deg);
  }
  else if (cmd.startsWith("S2:")) {
    int deg = constrain(cmd.substring(3).toInt(), 0, SERVO2_CLAMP_MAX);
    servo2.write(deg);
    Serial.print("Servo2 -> ");
    Serial.println(deg);
  }
  else if (cmd == "R1ON")  { setRelay(RELAY1_PIN, true);  Serial.println("Relay1 ON"); }
  else if (cmd == "R1OFF") { setRelay(RELAY1_PIN, false); Serial.println("Relay1 OFF"); }
  else if (cmd == "R2ON")  { setRelay(RELAY2_PIN, true);  Serial.println("Relay2 ON"); }
  else if (cmd == "R2OFF") { setRelay(RELAY2_PIN, false); Serial.println("Relay2 OFF"); }
  else if (cmd == "R3ON")  { setRelay(RELAY3_PIN, true);  Serial.println("Relay3 ON"); }
  else if (cmd == "R3OFF") { setRelay(RELAY3_PIN, false); Serial.println("Relay3 OFF"); }

  // ---- Load cell: calibration ----
  else if (cmd.startsWith("CAL:")) {
    if (calState != CAL_IDLE) {
      Serial.println("Calibration already running.");
      return;
    }
    float grams = cmd.substring(4).toFloat();
    if (grams <= 0) {
      Serial.println("Invalid weight. Example: CAL:10");
      return;
    }
    startCalibration(grams);
  }

  // ---- Load cell: weight detection toggle ----
  else if (cmd == "W") {
    if (calState != CAL_IDLE) {
      Serial.println("Wait until calibration is done.");
      return;
    }
    weightDetectionOn = !weightDetectionOn;
    Serial.println(weightDetectionOn ? "Weight detection ON." : "Weight detection OFF.");
  }

  else {
    Serial.println("Unknown command.");
  }
}

// ---------- Actuator Control ----------
void startActuator(bool up, unsigned long duration) {
  if (duration <= 0) {
    Serial.println("Invalid duration.");
    return;
  }

  if (up) {
    digitalWrite(ACT_IN1, HIGH);
    digitalWrite(ACT_IN2, LOW);
    Serial.print("Actuator UP for ");
  } else {
    digitalWrite(ACT_IN1, LOW);
    digitalWrite(ACT_IN2, HIGH);
    Serial.print("Actuator DOWN for ");
  }

  Serial.print(duration);
  Serial.println(" ms");

  actuatorRunning = true;
  actuatorStartTime = millis();
  actuatorDuration = duration;
}

void stopActuator() {
  digitalWrite(ACT_IN1, LOW);
  digitalWrite(ACT_IN2, LOW);
  actuatorRunning = false;
}

void handleActuatorTimer() {
  if (actuatorRunning && (millis() - actuatorStartTime >= actuatorDuration)) {
    stopActuator();
    Serial.println("Actuator auto-stopped (duration complete).");
  }
}

// ---------- Relay Control ----------
void setRelay(int pin, bool on) {
  if (RELAY_ACTIVE_LOW) {
    digitalWrite(pin, on ? LOW : HIGH);
  } else {
    digitalWrite(pin, on ? HIGH : LOW);
  }
}

// =====================================================================
// ---------- Load Cell: EEPROM storage ----------
// =====================================================================
void loadCalibrationFactor() {
  uint32_t magic;
  float saved;
  EEPROM.get(EEPROM_ADDR_MAGIC, magic);
  EEPROM.get(EEPROM_ADDR_CF, saved);

  if (magic == EEPROM_MAGIC && !isnan(saved) && fabs(saved) > 0.001) {
    calibrationFactor = saved;
    Serial.println("Loaded calibration factor from EEPROM.");
  } else {
    calibrationFactor = DEFAULT_CALIBRATION_FACTOR;
    Serial.println("No saved calibration. Using default factor.");
  }
}

void saveCalibrationFactor(float value) {
  EEPROM.put(EEPROM_ADDR_MAGIC, EEPROM_MAGIC);
  EEPROM.put(EEPROM_ADDR_CF, value);
  EEPROM.commit();
}

// =====================================================================
// ---------- Load Cell: non-blocking sampler ----------
// =====================================================================
void startSampling(SamplePurpose purpose, int count) {
  samplePurpose = purpose;
  sampleSum = 0;
  sampleCount = 0;
  sampleTarget = count;
  lastSampleTime = millis();
}

void handleScale() {
  // ---- Calibration timed steps ----
  if (calState == CAL_WAIT_EMPTY && millis() - calStepStart >= CAL_WAIT_MS) {
    Serial.println("Taring...");
    calState = CAL_TARING;
    startSampling(SP_CAL_TARE, CAL_TARE_SAMPLES);
  }
  else if (calState == CAL_WAIT_WEIGHT && millis() - calStepStart >= CAL_WAIT_MS) {
    Serial.println("Reading...");
    calState = CAL_READING;
    startSampling(SP_CAL_READ, CAL_READ_SAMPLES);
  }

  // ---- Start a weight measurement when detection is ON ----
  if (samplePurpose == SP_NONE && calState == CAL_IDLE && weightDetectionOn) {
    startSampling(SP_WEIGHT, WEIGHT_SAMPLES);
  }

  if (samplePurpose == SP_NONE) return;

  // ---- Collect one reading when the HX711 has data ----
  if (!scale.is_ready()) {
    if (millis() - lastSampleTime > HX711_TIMEOUT_MS) {
      Serial.println("HX711 not found.");
      samplePurpose = SP_NONE;
      weightDetectionOn = false;
      if (calState != CAL_IDLE) {
        calState = CAL_IDLE;
        Serial.println("Calibration aborted. Factor unchanged.");
      }
    }
    return;
  }

  sampleSum += scale.read();
  sampleCount++;
  lastSampleTime = millis();

  if (sampleCount < sampleTarget) return;

  long average = (long)(sampleSum / sampleCount);
  SamplePurpose finished = samplePurpose;
  samplePurpose = SP_NONE;

  switch (finished) {
    case SP_CAL_TARE:
      calOffset = average;
      Serial.print("Tare done. Place the ");
      Serial.print(calKnownWeight, 2);
      Serial.println(" g weight on the scale...");
      calState = CAL_WAIT_WEIGHT;
      calStepStart = millis();
      break;

    case SP_CAL_READ:
      handleCalibrationReading(average - calOffset);
      break;

    case SP_WEIGHT:
      printWeight(average);
      break;

    default:
      break;
  }
}

// =====================================================================
// ---------- Load Cell: weight detection (value only) ----------
// =====================================================================
void printWeight(long rawAverage) {
  float weight = (float)(rawAverage - scale.get_offset()) / calibrationFactor;
  if (fabs(weight) < ZERO_DEADBAND) {
    weight = 0.0;
  }
  Serial.println(weight, WEIGHT_DECIMALS);
}

// =====================================================================
// ---------- Load Cell: calibration (5 tries, averaged, auto-saved) ----------
// =====================================================================
void startCalibration(float knownGrams) {
  weightDetectionOn = false;
  samplePurpose = SP_NONE;

  calKnownWeight = knownGrams;
  calTry = 0;

  Serial.print("Calibration started with ");
  Serial.print(calKnownWeight, 2);
  Serial.print(" g, ");
  Serial.print(CAL_TRIES);
  Serial.println(" tries.");
  announceTry();
}

void announceTry() {
  Serial.print("--- Try ");
  Serial.print(calTry + 1);
  Serial.print("/");
  Serial.print(CAL_TRIES);
  Serial.println(" ---");
  Serial.println("Remove any weight from the scale...");
  calState = CAL_WAIT_EMPTY;
  calStepStart = millis();
}

void handleCalibrationReading(long reading) {
  calResults[calTry] = reading;
  Serial.print("Result: ");
  Serial.println(reading);
  calTry++;

  if (calTry < CAL_TRIES) {
    announceTry();
  } else {
    finishCalibration();
  }
}

void finishCalibration() {
  calState = CAL_IDLE;

  int64_t total = 0;
  for (int i = 0; i < CAL_TRIES; i++) {
    total += calResults[i];
  }
  float averageReading = (float)total / CAL_TRIES;
  float newFactor = averageReading / calKnownWeight;

  Serial.print("Average reading: ");
  Serial.println(averageReading, 1);

  if (fabs(newFactor) < 0.001) {
    Serial.println("Calibration failed (no change detected). Factor unchanged.");
    return;
  }

  // Apply to weight detection immediately and save to EEPROM
  calibrationFactor = newFactor;
  scale.set_scale(calibrationFactor);
  scale.set_offset(calOffset);   // last empty-scale tare becomes the zero point
  saveCalibrationFactor(calibrationFactor);

  Serial.print("New calibration factor: ");
  Serial.println(calibrationFactor, 2);
  Serial.println("Saved to EEPROM. Remove the weight, then send W for weight detection.");
}
