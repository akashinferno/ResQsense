from flask import Flask, render_template, request, redirect, url_for,jsonify,Response
import sqlite3
from datetime import datetime
import cv2
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

IP_CAMERA_URL = "http://192.168.57.199:8080/video"
THERMAL_CAMERA_URL="http://192.168.57.199:8080/video_feed"
location_data = {"latitude": None, "longitude": None}

# Get local time from your device
formatted_time = datetime.now().strftime('%H:%M:%S %d-%m-%Y')

DATABASE = 'sensor_data.db'

def init_db():
    """Initialize the database and create the sensor_data table if it doesn't exist."""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            dht_temperature REAL,
            dht_humidity REAL,
            bmp_temperature REAL,
            bmp_pressure REAL,
            bmp_altitude REAL,
            mq4_analog_value INTEGER,
            mq4_ppm_estimate REAL,
            mq4_digital_output INTEGER,
            adxl345_x REAL,
            adxl345_y REAL,
            adxl345_z REAL,
            vibration INTEGER,
            sound_sensor INTEGER,
            flame_sensor INTEGER,
            sos INTEGER
        )
    ''')
    conn.commit()
    conn.close()
init_db()


def get_sensor_data():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 10")  # Get latest 10 records
    data = cursor.fetchall()
    conn.close()
    return data
#----------------------------------------------------------------------------------------------------
#object detection:


import torch
#  Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Define IP Camera URL (update this with your actual camera URL)

# Enable/Disable Thermal Filter
thermal_filter_enabled = True

def simulate_temperature(intensity):
    """Simulate temperature based on intensity."""
    return round((intensity / 255.0) * 40, 2)

def generate_frames_with_thermal():
    cap = cv2.VideoCapture(IP_CAMERA_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise Exception(f"Unable to open RTSP stream: {IP_CAMERA_URL}")
    
    frame_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Skip frames for performance
        frame_counter += 1
        if frame_counter % 3 != 0:
            continue

        # Resize frame for better performance
        frame = cv2.resize(frame, (640, 480))

        # Perform object detection using YOLOv5
        results = model(frame)
        detections = results.xyxy[0]  # Format: [x1, y1, x2, y2, confidence, class]

        for det in detections:
            x1, y1, x2, y2, conf, cls = det[:6]
            cls = int(cls)  # Convert class index to int
            label = results.names[cls]  # Get class name
            confidence = f"{conf:.2f}"

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Apply thermal filter if enabled
        if thermal_filter_enabled:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thermal_frame = cv2.applyColorMap(gray_frame, cv2.COLORMAP_JET)
            frame = thermal_frame

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    """Serve MJPEG stream"""
    return Response(generate_frames_with_thermal(), mimetype='multipart/x-mixed-replace; boundary=frame')



#----------------------------------------------------------------------------------------------------



@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username =="admin" and password == "1234":
        # Authentication logic can be added here
            return redirect(url_for('dashboard'))
    return render_template('login.html')



@app.route('/dashboard')
def dashboard():
    sensor_data = get_sensor_data()
    return render_template('dashboard.html', sensor_data=sensor_data)

@app.route('/map')
def gmap():
    return render_template('map.html')

@app.route('/update_location', methods=['POST'])
def update_location():
    global location_data
    data = request.get_json()
    location_data["latitude"] = data.get('latitude')
    location_data["longitude"] = data.get('longitude')
    print(f"Received Location: {location_data}")
    return jsonify({'status': 'success'})

@app.route('/get_location')
def get_location():
    return jsonify(location_data)

@app.route("/get_camera_urls")
def get_camera_urls():
    return jsonify({
        "normal_url": IP_CAMERA_URL,
        "thermal_url": THERMAL_CAMERA_URL
    })

#----------------------------------------------------------
sensor_data_list = []
MAX_DATA=20
@app.route('/data', methods=['POST'])
def receive_data():
    global MAX_DATA,sensor_data_list
    data = request.get_json()
    print("Received data:", data)

    # Extract values, setting to None if missing
    dht_temp      = data.get('dht', {}).get('temperature', None)
    dht_humidity  = data.get('dht', {}).get('humidity', None)
    bmp_temp      = data.get('bmp', {}).get('temperature', None)
    bmp_pressure  = data.get('bmp', {}).get('pressure', None)
    bmp_altitude  = data.get('bmp', {}).get('altitude', None)
    mq4_analog    = data.get('mq4', {}).get('analog_value', None)
    mq4_ppm       = data.get('mq4', {}).get('ppm_estimate', None)
    mq4_digital   = data.get('mq4', {}).get('digital_output', None)
    adxl_x        = data.get('adxl345', {}).get('x', None)
    adxl_y        = data.get('adxl345', {}).get('y', None)
    adxl_z        = data.get('adxl345', {}).get('z', None)
    vibration     = data.get('vibration', None)
    sound_sensor  = data.get('sound_sensor', None)
    flame_sensor  = data.get('flame_sensor', None)
    sos           = data.get('sos', None)

    # Insert into database
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO sensor_data (
            dht_temperature,
            dht_humidity,
            bmp_temperature,
            bmp_pressure,
            bmp_altitude,
            mq4_analog_value,
            mq4_ppm_estimate,
            mq4_digital_output,
            adxl345_x,
            adxl345_y,
            adxl345_z,
            vibration,
            sound_sensor,
            flame_sensor,
            sos,
            timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (dht_temp, dht_humidity, bmp_temp, bmp_pressure, bmp_altitude,
          mq4_analog, mq4_ppm, mq4_digital, adxl_x, adxl_y, adxl_z,
          vibration, sound_sensor, flame_sensor, sos,formatted_time))
    conn.commit()
    conn.close()

        # Create a record dictionary for the global list
    sensor_record = {
        "dht_temperature": dht_temp,
        "dht_humidity": dht_humidity,
        "bmp_temperature": bmp_temp,
        "bmp_pressure": bmp_pressure,
        "bmp_altitude": bmp_altitude,
        "mq4_analog_value": mq4_analog,
        "mq4_ppm_estimate": mq4_ppm,
        "mq4_digital_output": mq4_digital,
        "adxl345_x": adxl_x,
        "adxl345_y": adxl_y,
        "adxl345_z": adxl_z,
        "vibration": vibration,
        "sound_sensor": sound_sensor,
        "flame_sensor": flame_sensor,
        "sos": sos,
        "timestamp": formatted_time
    }
    check_sos(sos)


    sensor_data_list.append(sensor_record)
    if len(sensor_data_list) > MAX_DATA:
        sensor_data_list.pop(0)
    socketio.emit('sensor_update', sensor_data_list)

    return jsonify({"message": "Data received and stored successfully"}), 200



# Function to check SOS value and emit event
def check_sos(sos):
    if sos == 1:
        socketio.emit('sos_alert', {'message': 'Emergency Alert!'})


if __name__ == '__main__':
     socketio.run(app, host='0.0.0.0', port=5000, debug=True)