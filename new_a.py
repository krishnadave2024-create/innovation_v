from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import os
import logging
import sqlite3
import speech_recognition as sr
import threading
import time
from datetime import datetime, date
import queue
from flask_socketio import SocketIO, emit
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import math

load_dotenv()

# ✅ FIXED: Safe OpenAI client with fallback
try:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    OPENAI_AVAILABLE = True
    print("✅ OpenAI API available")
except:
    client = None
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI unavailable - using offline mode")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('absl').setLevel(logging.ERROR)

app = Flask(__name__)
app.static_folder = 'static'
socketio = SocketIO(app, cors_allowed_origins="*")

DB_PATH = 'attendance.db'
SPEECH_DB_PATH = 'speech.db'

speech_queue = queue.Queue()
is_listening = False
current_speech_text = ""
recognizer = sr.Recognizer()
microphone = None
present_count = 0
absent_count = 0
LOCKED_ABSENT = "LOCKED ABSENT"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS attendance
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  userid TEXT NOT NULL,
                  date TEXT NOT NULL,
                  status TEXT NOT NULL,
                  last_updated TEXT NOT NULL,
                  is_locked INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()

def init_speech_db():
    conn = sqlite3.connect(SPEECH_DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS speech_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            text_content TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def set_attendance(userid, status):
    today_str = date.today().isoformat()
    now_str = datetime.now().isoformat(timespec='seconds')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM attendance WHERE userid=? AND status="Absent"', (userid,))
    total_absent = c.fetchone()[0]
    c.execute('SELECT id, status, is_locked FROM attendance WHERE userid=? AND date=?', (userid, today_str))
    row = c.fetchone()
    if total_absent >= 3:
        if row:
            c.execute('UPDATE attendance SET status="Absent", is_locked=1, last_updated=? WHERE id=?', (now_str, row[0]))
        else:
            c.execute('INSERT INTO attendance (userid, date, status, last_updated, is_locked) VALUES (?, ?, "Absent", ?, 1)',
                      (userid, today_str, now_str))
        print(f"🔒 LOCKED {userid} - {total_absent} total absences!")
        conn.commit()
        conn.close()
        return False
    if row:
        record_id, _, is_locked = row
        if is_locked == 1:
            conn.close()
            return False
        c.execute('UPDATE attendance SET status=?, last_updated=? WHERE id=?', (status, now_str, record_id))
    else:
        c.execute('INSERT INTO attendance (userid, date, status, last_updated) VALUES (?, ?, ?, ?)',
                  (userid, today_str, status, now_str))
    conn.commit()
    conn.close()
    return True

def save_speech_record(user_id, text_content):
    now = datetime.now()
    conn = sqlite3.connect(SPEECH_DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO speech_records (user_id, date, time, text_content) VALUES (?, ?, ?, ?)",
              (user_id, now.date().isoformat(), now.time().isoformat(timespec='seconds'), text_content))
    conn.commit()
    conn.close()

def get_attendance_counts(userid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM attendance WHERE userid=? AND status="Present"', (userid,))
    present = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM attendance WHERE userid=? AND status="Absent"', (userid,))
    absent = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM attendance WHERE userid=? AND is_locked=1', (userid,))
    locked = c.fetchone()[0]
    conn.close()
    return present, absent, locked

def get_offline_response(user_message, context):
    """✅ NEW: Smart offline responses based on attendance data"""
    user_msg_lower = user_message.lower()
    
    if "present" in user_msg_lower or "attendance" in user_msg_lower:
        return f"Your current status: **{attendance_status}**. Present frames: {present_count}, Absent frames: {absent_count}. {'🔒 LOCKED' if attendance_status == LOCKED_ABSENT else ''}"
    
    if "gesture" in user_msg_lower:
        return f"Current gesture detected: **{gesture}**. Try ✌️ peace, 👍 thumbs up, or 👌 pinch!"
    
    if "speech" in user_msg_lower or "mic" in user_msg_lower:
        status = "ON" if is_listening else "OFF"
        return f"Speech recognition: **{status}**. Last heard: '{current_speech_text[:50]}...' Toggle with /toggle-speech endpoint."
    
    if "help" in user_msg_lower:
        return """**Smart Attendance Commands:**
• 'status' - Current attendance
• 'gesture' - Hand detection  
• 'speech' - Voice recognition
• 'records' - Visit /speech-records or /attendance-all
• 'filter' - Try /filter/bw, /filter/red, etc."""
    
    return f"🤖 Attendance: **{attendance_status}** | Gesture: {gesture} | Mic: {'ON' if is_listening else 'OFF'}. Type 'help' for commands!"

def speech_listener():
    global is_listening, current_speech_text, microphone

    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    try:
        microphone = sr.Microphone()
        print("🎤 Microphone initialized successfully")
    except Exception as e:
        print("❌ Mic error:", e)
        return

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

    while True:
        if is_listening:
            try:
                with microphone as source:
                    print("🎧 Listening...")
                    audio = recognizer.listen(source, timeout=None)

                text = recognizer.recognizer_google(audio)
                current_speech_text = text
                save_speech_record("student1", text)
                print("✅ HEARD:", text)

            except sr.UnknownValueError:
                print("🤷 Couldn't understand")
            except Exception as e:
                print("Speech error:", e)
        time.sleep(0.2)

# Initialize databases
init_db()
init_speech_db()

camera = cv2.VideoCapture(0)
DETECT_WIDTH, DETECT_HEIGHT = 320, 240
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480
camera.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Global application state
face_detected = False
expression = "neutral"
gesture = "none"
current_filter = "normal"
filters = ["normal", "bw", "red", "blur", "cartoon"]
CURRENT_USERID = "student1"
attendance_status = "Absent"

def fingers_up(hand, hand_label):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    if hand_label == "Right":
        fingers.append(hand.landmark[tips[0]].x < hand.landmark[tips[0]-1].x)
    else:
        fingers.append(hand.landmark[tips[0]].x > hand.landmark[tips[0]-1].x)

    for i in range(1, 5):
        fingers.append(hand.landmark[tips[i]].y < hand.landmark[tips[i]-2].y)

    return fingers

def detect_gesture(hand, hand_label):
    f = fingers_up(hand, hand_label)
    if f == [0,0,0,0,0]: return "✊"
    if f == [1,1,1,1,1]: return "🤚"
    if f == [1,0,0,0,0]: return "👍"
    if f == [0,1,1,0,0]: return "✌️"
    if f == [0,1,0,0,0]: return "☝️"
    if f == [0,1,1,1,0]: return "🤟"
    if f[0] == 1 and hand.landmark[4].y > hand.landmark[3].y:
        return "👎"
    thumb = hand.landmark[4]
    index = hand.landmark[8]
    dist = math.hypot(thumb.x-index.x, thumb.y-index.y)
    if dist < 0.04:
        return "👌"
    return "none"

def filter_bw(frame):
    return cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

def filter_red(frame):
    red = frame.copy()
    red[:,:,2] = cv2.add(red[:,:,2], 60)
    return red

def filter_blur(frame):
    return cv2.GaussianBlur(frame, (21,21), 0)

def filter_cartoon(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(frame, 9, 300, 300)
    return cv2.bitwise_and(color, color, mask=edges)

def generate_frames():
    global face_detected, expression, gesture, current_filter, attendance_status, current_speech_text, is_listening
    global present_count, absent_count
    today_str = date.today().isoformat()
    session_start = None
    while True:
        success, frame = camera.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        if session_start != today_str:
            present_count = 0
            absent_count = 0
            session_start = today_str
            print(f"📅 Daily reset: P{present_count} A{absent_count}")
        
        small_frame = cv2.resize(frame, (DETECT_WIDTH, DETECT_HEIGHT))
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        faces = face_cascade.detectMultiScale(gray_small, 1.3, 5)
        face_detected = len(faces) > 0
        expression = "neutral"
        
        for (x, y, w, h) in faces:
            x, y, w, h = int(x * DISPLAY_WIDTH/DETECT_WIDTH), int(y * DISPLAY_HEIGHT/DETECT_HEIGHT), \
                         int(w * DISPLAY_WIDTH/DETECT_WIDTH), int(h * DISPLAY_HEIGHT/DETECT_HEIGHT)
            roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 10)
            
            if len(smiles) > 0: expression = "smile"
            elif len(eyes) == 0: expression = "no_eyes"
            elif len(eyes) == 1: expression = "one_eye"
            elif len(eyes) >= 2 and h > 180: expression = "happy"
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        new_status = "Present" if face_detected else "Absent"
        if new_status == "Present":
            present_count += 1
        else:
            absent_count += 1
        
        if absent_count >= 5000 and attendance_status != LOCKED_ABSENT:
            set_attendance(CURRENT_USERID, "Absent")
            attendance_status = LOCKED_ABSENT
            print(f"🔒 AUTO-LOCKED Absent! count={absent_count}")
        elif new_status != attendance_status and attendance_status != LOCKED_ABSENT:
            updated = set_attendance(CURRENT_USERID, new_status)
            if updated:
                attendance_status = new_status
        
        # Hand gesture detection
        gesture = "none"
        result = hands.process(rgb_small)
        if result.multi_hand_landmarks and result.multi_handedness:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                hand_label = handedness.classification[0].label
                gesture = detect_gesture(hand_landmarks, hand_label)
                gesture = f"{hand_label}:{gesture}"
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                break

        # Apply video filters
        if current_filter == "bw":
            frame = filter_bw(frame)
        elif current_filter == "red":
            frame = filter_red(frame)
        elif current_filter == "blur":
            frame = filter_blur(frame)
        elif current_filter == "cartoon":
            frame = filter_cartoon(frame)
        
        color = (0, 255, 0) if attendance_status == "Present" else (0, 0, 255)
        if attendance_status == LOCKED_ABSENT:
            color = (0, 0, 200)
        
        cv2.putText(frame, f"Mic: {'ON' if is_listening else 'OFF'}", (10, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 1), 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    present, absent, locked = get_attendance_counts(CURRENT_USERID)
    return jsonify({
        'face': face_detected,
        'expression': expression,
        'gesture': gesture,
        'filter': current_filter,
        'attendance': attendance_status,
        'speech': current_speech_text,
        'listening': is_listening,
        'user': CURRENT_USERID,
        'present_count': present_count,
        'absent_count': absent_count,
        'total_present': present,
        'total_absent': absent,
        'locked': locked > 0,
        'openai_available': OPENAI_AVAILABLE
    })

@app.route('/toggle-speech', methods=['POST'])
def toggle_speech():
    global is_listening
    is_listening = not is_listening
    print(f"Speech listening: {'ON' if is_listening else 'OFF'}")
    return jsonify({'listening': is_listening})

@app.route('/filter/<name>')
def set_filter(name):
    global current_filter
    if name in filters:
        current_filter = name
    return jsonify({'filter': current_filter})

@app.route('/speech-records')
def speech_records():
    conn = sqlite3.connect(SPEECH_DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, date, time, text_content FROM speech_records ORDER BY id DESC LIMIT 50')
    rows = c.fetchall()
    conn.close()
    html = '<!DOCTYPE html><html><head><title>Speech Records</title><style>body{font-family:Arial;margin:40px;background:#f5f5f5;}table{width:100%;border-collapse:collapse;background:white;box-shadow:0 4px 12px rgba(0,0,0,0.1);}th,td{padding:12px;text-align:left;border-bottom:1px solid #eee;}th{background:linear-gradient(135deg,#007bff,#0056b3);color:white;}</style></head><body>'
    html += '<h2>🎤 Speech Records (Last 50)</h2>'
    html += '<table><tr><th>ID</th><th>Date</th><th>Time</th><th>Text</th></tr>'
    for r in rows:
        html += f'<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2][11:]}</td><td>{r[3][:80]}...</td></tr>'
    html += '</table></body></html>'
    return html

@app.route('/attendance-all')
def attendance_all():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, userid, date, status, last_updated, is_locked FROM attendance ORDER BY date DESC, id DESC LIMIT 100')
    rows = c.fetchall()
    conn.close()
    html = '<!DOCTYPE html><html><head><title>Attendance Records</title><style>body{font-family:Arial;margin:40px;background:#f5f5f5;}table{width:100%;border-collapse:collapse;background:white;box-shadow:0 4px 12px rgba(0,0,0,0.1);}th,td{padding:12px;text-align:left;border-bottom:1px solid #eee;}th{background:linear-gradient(135deg,#28a745,#20c997);color:white;}.present{background:#d4edda;}.absent{background:#f8d7da;}.locked{background:#fff3cd;}</style></head><body>'
    html += '<h2>📋 All Attendance Records</h2>'
    html += '<table><tr><th>ID</th><th>User</th><th>Date</th><th>Status</th><th>Time</th><th>Lock</th></tr>'
    for r in rows:
        status_class = 'present' if r[3] == 'Present' else 'absent'
        lock = '🔒 LOCKED' if r[5] else ''
        html += f'<tr class="{status_class}"><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4][11:]}</td><td>{lock}</td></tr>'
    html += '</table></body></html>'
    return html

@socketio.on('message')
def handle_message(data):
    """✅ FIXED: Robust AI handler with quota fallback"""
    global attendance_status, present_count, gesture, current_speech_text
    
    user_message = data['message']
    print(f"💬 User: {user_message}")
    
    context = f"Attendance: {attendance_status} | Present: {present_count} | Gesture: {gesture} | Speech: {current_speech_text}"
    
    # Try OpenAI first, fallback to offline
    ai_response = None
    if OPENAI_AVAILABLE and client:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI Attendance Assistant. " + context},
                    {"role": "user", "content": user_message}
                ],
                timeout=10  # Short timeout
            )
            ai_response = response.choices[0].message.content
            print(f"🤖 OpenAI: {ai_response[:50]}...")
        except Exception as e:
            print(f"❌ OpenAI Error: {str(e)}")
    
    # Fallback to offline responses
    if not ai_response:
        ai_response = get_offline_response(user_message, context)
        print(f"📱 Offline: {ai_response[:50]}...")
    
    # Send response to client
    emit('response', {
        'message': ai_response,
        'timestamp': datetime.now().isoformat(),
        'openai': OPENAI_AVAILABLE
    })

if __name__ == '__main__':
    speech_thread = threading.Thread(target=speech_listener, daemon=True)
    speech_thread.start()
    print("🚀 Smart Attendance System Started! (Offline mode ready)")
    print("🌐 Dashboard: http://localhost:5000")
    print("📋 Records: http://localhost:5000/attendance-all")
    print("💬 Chat works with/without OpenAI!")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
