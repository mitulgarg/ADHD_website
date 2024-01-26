from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
from datetime import datetime

app = Flask(__name__)
socketio = SocketIO(app)
video = cv2.VideoCapture(0)
time_list = []  # Store all motion timestamps

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')

def generate_frames():
    global time_list

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

        if motion_list[-1] == 1 and motion_list[-2] == 0:
            current_time = datetime.now()
            time_list.append(current_time)
            formatted_time_list = [time.strftime("%Y-%m-%d %H:%M:%S") for time in time_list]

            # Send the event to the client
            socketio.emit('motion_detected', {'time_list': formatted_time_list})

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    socketio.run(app, debug=True)
