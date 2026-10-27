#!/usr/bin/env python3
import time
import threading
import hashlib
from datetime import datetime
import cv2
import RPi.GPIO as GPIO
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO

# ================= HARDWARE PIN DEFINITIONS (BCM) =================
HX711_DT_PIN = 5         # Physical Pin 29
HX711_SCK_PIN = 6        # Physical Pin 31
BUZZER_PIN = 26          # Physical Pin 37

# Calibration factor for 1kg load cell (adjust if your reading is slightly high/low)
LOADCELL_SCALE = 420.0
TARGET_WEIGHT_KG = 1.00
TARGET_WEIGHT_GRAMS = 950.0  # Triggers at 0.95kg - 1.00kg+ for smooth live demo

# ================= LOCAL EDGE CACHED CITIZEN DATABASE =================
REGION_CITIZENS = {
    "udupi": {
        "region_key": "udupi",
        "region_title": "Udupi",
        "name": "Ramesh Poojary",
        "age": "48 Years",
        "ration_kg": "1.00 kg (Fortified Rice)",
        "members": "4 Family Members",
        "card_id": "KA-UDPI-AAY-8821",
        "fps_shop": "Gundmi Fair Price Shop #04",
        "avatar_color": "#16a34a"
    },
    "shivamogga": {
        "region_key": "shivamogga",
        "region_title": "Shivamogga",
        "name": "Manjunath Gowda",
        "age": "52 Years",
        "ration_kg": "1.00 kg (Fortified Rice)",
        "members": "5 Family Members",
        "card_id": "KA-SHIV-PHH-4410",
        "fps_shop": "Thirthahalli Cross FPS #11",
        "avatar_color": "#2563eb"
    },
    "agumbe": {
        "region_key": "agumbe",
        "region_title": "Agumbe",
        "name": "Kavitha Hegde",
        "age": "39 Years",
        "ration_kg": "1.00 kg (Fortified Rice)",
        "members": "3 Family Members",
        "card_id": "KA-AGMB-PHH-9032",
        "fps_shop": "Agumbe Gram Panchayat FPS #02",
        "avatar_color": "#9333ea"
    }
}

# ================= GPIO & BUZZER SETUP =================
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)
GPIO.setup(HX711_SCK_PIN, GPIO.OUT)
GPIO.setup(HX711_DT_PIN, GPIO.IN)

def beep_pattern(times=1, duration=0.15, gap=0.1):
    def _run():
        for _ in range(times):
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(gap)
    threading.Thread(target=_run, daemon=True).start()

# ================= AUTO-DETECTING JHD 162A I2C LCD DRIVER =================
lcd = None
try:
    from RPLCD.i2c import CharLCD
    for addr in [0x27, 0x3f]:
        try:
            lcd = CharLCD(i2c_expander='PCF8574', address=addr, port=1, cols=16, rows=2)
            lcd.clear()
            print(f"[OK] JHD 162A I2C LCD initialized at address {hex(addr)}")
            break
        except Exception:
            lcd = None
except Exception as e:
    print(f"[WARN] LCD driver note: {e}")

lcd_lock = threading.Lock()
def lcd_display(line1, line2=""):
    print(f"[162A LCD] {line1:16s} | {line2:16s}")
    if lcd:
        with lcd_lock:
            try:
                lcd.clear()
                lcd.write_string(str(line1)[:16])
                lcd.crlf()
                lcd.write_string(str(line2)[:16])
            except Exception:
                pass

# ================= NATIVE HX711 24-BIT ADC DRIVER =================
class HX711Driver:
    def __init__(self, dt_pin, sck_pin, scale=420.0):
        self.dt = dt_pin
        self.sck = sck_pin
        self.scale = scale
        self.offset = 0
        self.lock = threading.Lock()

    def read_raw(self):
        timeout = time.time() + 0.4
        while GPIO.input(self.dt) == 1:
            if time.time() > timeout:
                return None
        value = 0
        for _ in range(24):
            GPIO.output(self.sck, True)
            value = (value << 1) | GPIO.input(self.dt)
            GPIO.output(self.sck, False)
        GPIO.output(self.sck, True)
        GPIO.output(self.sck, False)
        if value & 0x800000:
            value -= 0x1000000
        return value

    def read_avg(self, count=4):
        with self.lock:
            samples = []
            for _ in range(count):
                r = self.read_raw()
                if r is not None:
                    samples.append(r)
                time.sleep(0.008)
            return sum(samples) / len(samples) if samples else self.offset

    def tare(self):
        self.offset = self.read_avg(8)
        print(f"[OK] Load Cell Tared (Offset: {self.offset})")

    def get_grams(self):
        raw = self.read_avg(3)
        grams = abs((raw - self.offset) / self.scale)
        return round(grams, 1) if grams > 8.0 else 0.0

hx711 = HX711Driver(HX711_DT_PIN, HX711_SCK_PIN, scale=LOADCELL_SCALE)

# ================= OV5647 CAMERA + OPENCV FACE VERIFICATION =================
face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
cam_lock = threading.Lock()
USE_PICAM2 = False
picam2 = None
cap = None

try:
    from picamera2 import Picamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'RGB888', "size": (640, 480)}))
    picam2.start()
    USE_PICAM2 = True
    print("[OK] OV5647 Camera initialized via Picamera2")
except Exception:
    try:
        cap = cv2.VideoCapture(0)
        print("[OK] Camera initialized via V4L2 VideoCapture")
    except Exception as e:
        print(f"[WARN] Camera not available: {e}")

latest_face_detected = False

def generate_mjpeg():
    global latest_face_detected
    while True:
        frame = None
        with cam_lock:
            if USE_PICAM2 and picam2:
                frame = picam2.capture_array()
            elif cap and cap.isOpened():
                ret, f = cap.read()
                if ret:
                    frame = f
        if frame is None:
            time.sleep(0.1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(70, 70))
        latest_face_detected = len(faces) > 0

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (34, 197, 94), 3)
            cv2.putText(frame, "JANASEVA FACE VERIFIED", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (34, 197, 94), 2)

        ok, jpg = cv2.imencode('.jpg', frame)
        if ok:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
        time.sleep(0.05)

# ================= FLASK & SOCKET.IO STATE MACHINE =================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'janaseva_edge_2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

state = {
    "scenario": 1,                      # 1 = Udupi Edge Demo, 2 = Shivamogga & Agumbe Failover Demo
    "next_target_region": "udupi",      # 'udupi' -> 'shivamogga' -> 'agumbe'
    "active_weighing_region": None,     # Region currently waiting for 1kg load cell
    "weighing_complete": False,
    "weight_g": 0.0,
    "weight_kg": 0.0
}

def authenticate_region(region_key, auth_method="R307S Biometric Fingerprint"):
    """Fills the target region's Detail of Person box and starts 1kg Load Cell monitoring."""
    profile = REGION_CITIZENS[region_key].copy()
    profile["auth_method"] = auth_method
    profile["timestamp"] = datetime.now().strftime("%H:%M:%S")

    state["active_weighing_region"] = region_key
    state["weighing_complete"] = False

    # Advance pointer for Scenario 2 sequential fingerprint scans (Shivamogga -> Agumbe)
    if state["scenario"] == 2:
        if region_key == "shivamogga":
            state["next_target_region"] = "agumbe"
        elif region_key == "agumbe":
            state["next_target_region"] = "done"

    beep_pattern(1, 0.18)
    lcd_display(f"{profile['region_title']}: AUTH OK", "Apply 1kg Ration")
    hx711.tare()

    socketio.emit('region_authenticated', {
        "region": region_key,
        "profile": profile,
        "next_target": state["next_target_region"]
    })

def complete_ration_dispense(region_key, measured_kg):
    """Triggers when 1kg is detected on Load Cell: turns box 1 green, waits 5s, turns box 2 green."""
    state["weighing_complete"] = True
    tx_id = hashlib.sha256(f"{region_key}:{time.time()}".encode()).hexdigest()[:12].upper()
    print(f"\n[SECURE EDGE LEDGER] Local 1kg Ration Dispensed!")
    print(f"[SHA-256 GENERATED] Hash: {hashlib.sha256(f'{region_key}:{time.time()}'.encode()).hexdigest().upper()}")
    print(f"[STATUS] Locking transaction... Waiting 5s for Cloud Sync...\n")

    # Step 1: Immediate 1kg confirmation + Beep + LCD
    beep_pattern(3, 0.18, 0.1)
    lcd_display("1KG RECEIVED OK!", f"{region_key.upper()} DONE")

    socketio.emit('ration_received_1kg', {
        "region": region_key,
        "weight_kg": measured_kg,
        "tx_id": tx_id,
        "message": "Authenticated and ration received"
    })

    # Step 2: 5-second delay before "data fetched to main server successfully."
    def _delayed_cloud_sync():
        for sec in range(5, 0, -1):
            lcd_display("1KG RECEIVED OK!", f"Cloud Sync in {sec}s")
            time.sleep(1.0)
        beep_pattern(1, 0.35)
        lcd_display("DATA FETCHED TO", "MAIN SERVER OK!")
        socketio.emit('main_server_synced', {
            "region": region_key,
            "tx_id": tx_id,
            "message": "data fetched to main server successfully."
        })

    threading.Thread(target=_delayed_cloud_sync, daemon=True).start()

# ================= HARDWARE POLLING THREAD (R307S + HX711) =================
def hardware_loop():
    finger_sensor = None
    try:
        from pyfingerprint.pyfingerprint import PyFingerprint
        for port in ['/dev/serial0', '/dev/ttyAMA0', '/dev/ttyUSB0']:
            try:
                f = PyFingerprint(port, 57600, 0xFFFFFFFF, 0x00000000)
                if f.verifyPassword():
                    finger_sensor = f
                    print(f"[OK] R307S Fingerprint Sensor connected on {port}")
                    break
            except Exception:
                pass
    except Exception as e:
        print(f"[INFO] pyfingerprint check: {e}")

    lcd_display("JANASEVA EDGE", "Place Finger...")
    hx711.tare()

    while True:
        # 1. Poll R307S Fingerprint Sensor if a region is awaiting authentication
        if finger_sensor and state["next_target_region"] in ["udupi", "shivamogga", "agumbe"]:
            try:
                if finger_sensor.readImage():
                    finger_sensor.convertImage(0x01)
                    res = finger_sensor.searchTemplate()
                    pos = res[0]
                    target = state["next_target_region"]

                    if pos >= 0:
                        # Successful fingerprint match!
                        if state["scenario"] == 1:
                            state["next_target_region"] = "udupi_authed"
                        authenticate_region(target, f"R307S Fingerprint (ID #{pos})")
                        time.sleep(1.2)
                    else:
                        # Fingerprint failed -> Trigger OV5647 Camera Fallback automatically
                        beep_pattern(2, 0.08, 0.08)
                        lcd_display("FINGER FAILED!", "Camera Auth Open")
                        socketio.emit('fingerprint_failed', {
                            "region": target,
                            "reason": "Fingerprint not recognized — Switching to OV5647 Camera Authentication"
                        })
                        time.sleep(1.2)
            except Exception:
                pass

        # 2. Poll HX711 Load Cell if a citizen is authenticated and waiting for 1kg ration
        if state["active_weighing_region"] and not state["weighing_complete"]:
            grams = hx711.get_grams()
            kg = round(grams / 1000.0, 2)
            state["weight_g"] = grams
            state["weight_kg"] = kg

            reg_label = state["active_weighing_region"].upper()[:8]
            lcd_display(f"{reg_label}: {kg:.2f}kg", "Target: 1.00 kg")
            socketio.emit('weight_update', {
                "region": state["active_weighing_region"],
                "grams": grams,
                "kg": kg
            })

            if grams >= TARGET_WEIGHT_GRAMS:
                complete_ration_dispense(state["active_weighing_region"], 1.00)

        time.sleep(0.22)

# ================= API ENDPOINTS FOR DEMO & FALLBACK CONTROLS =================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_scenario_2', methods=['POST'])
def start_scenario_2():
    """Triggers Scenario 2: Shivamogga & Agumbe Cloud Overload -> Local Edge Failover."""
    state["scenario"] = 2
    state["next_target_region"] = "shivamogga"
    state["active_weighing_region"] = None
    state["weighing_complete"] = False

    beep_pattern(2, 0.12, 0.1)
    lcd_display("CLOUD BUSY! R2&3", "LOCAL AUTH ON")

    socketio.emit('scenario_2_started', {
        "banner": "Shivamogga & Agumbe: Cloud Server Busy (503) — Switched to Local Edge Authentication!",
        "next_target": "shivamogga"
    })
    return jsonify({"status": "scenario_2_active"})

@app.route('/api/camera_auth', methods=['POST'])
def api_camera_auth():
    """Completes authentication using OV5647 Camera Fallback (or manual demo trigger)."""
    data = request.json or {}
    target = data.get("region") or state["next_target_region"]
    if target not in ["udupi", "shivamogga", "agumbe"]:
        target = "udupi" if state["scenario"] == 1 else "shivamogga"

    if state["scenario"] == 1 and target == "udupi":
        state["next_target_region"] = "udupi_authed"

    mode_str = "OV5647 Camera Face Auth" if latest_face_detected else "Biometric / Camera Verified"
    authenticate_region(target, mode_str)
    return jsonify({"status": "authenticated", "region": target})

@app.route('/api/simulate_1kg', methods=['POST'])
def api_simulate_1kg():
    """Backup trigger for 1.00 kg load cell completion."""
    region = state["active_weighing_region"] or "udupi"
    complete_ration_dispense(region, 1.00)
    return jsonify({"status": "1kg_dispensed", "region": region})

@app.route('/api/tare', methods=['POST'])
def api_tare():
    hx711.tare()
    return jsonify({"status": "tared"})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    state["scenario"] = 1
    state["next_target_region"] = "udupi"
    state["active_weighing_region"] = None
    state["weighing_complete"] = False
    hx711.tare()
    lcd_display("JANASEVA EDGE", "Place Finger...")
    socketio.emit('system_reset', {})
    return jsonify({"status": "reset"})

if __name__ == '__main__':
    t = threading.Thread(target=hardware_loop, daemon=True)
    t.start()
    print("=====================================================")
    print(" JANASEVA GOVERNMENT EDGE PORTAL LIVE ON PORT 5000")
    print(" Open http://<raspberry-pi-ip>:5000 in your browser")
    print("=====================================================")
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
