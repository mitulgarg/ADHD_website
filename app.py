from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)
video = cv2.VideoCapture(0)
time_data = []  # Move time_data outside the generate_frames function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/about')
def about():
    return render_template('about.html')


def generate_frames():
    static_back = None
    motion_list = [None, None]

    while True:
        _, frame = video.read()
        motion = 0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if static_back is None:
            static_back = gray
            continue

        diff_frame = cv2.absdiff(static_back, gray)
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        cnts, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            if cv2.contourArea(contour) < 110000:
                continue
            motion = 1
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        motion_list.append(motion)
        motion_list = motion_list[-2:]

        current_time = datetime.now()

        if motion_list[-1] == 1 and motion_list[-2] == 0:
            time_data.append(current_time)

        if motion_list[-1] == 0 and motion_list[-2] == 1:
            time_data.append(current_time)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('generate_graph')
def handle_generate_graph():
    # Emit the time_data for graph generation
    socketio.emit('motion_data', {'timestamps': [t.strftime('%H:%M:%S') for t in time_data]})

if __name__ == "__main__":
    socketio.run(app, debug=True)
