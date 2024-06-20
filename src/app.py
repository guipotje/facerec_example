from flask import Flask, Response, render_template
import numpy as np
import cv2
import time
import threading

from modules.core import FaceRecManager

app = Flask(__name__)

# Global variable to store the frame streams
current_frame = None
last_frame = None

manager = FaceRecManager()

def capture_frames():
    global current_frame
    video_capture = cv2.VideoCapture('/dev/video0')  # Adjust device path if necessary

    if not video_capture.isOpened():
        raise RuntimeError("Could not start camera.")

    while True:
        success, frame = video_capture.read()
        if success:
            current_frame = frame 

    # Release the capture when done
    video_capture.release()

def generate_image():
    global current_frame, last_frame
    while True:
        time.sleep(0.03)
        if current_frame is not None and current_frame is not last_frame:
            last_frame = current_frame

            out = manager.process_frame(current_frame)

            # Encode the frame to JPEG format
            _, jpeg = cv2.imencode('.jpg', out)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_image(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the webcam capture in a separate thread
    capture_thread = threading.Thread(target=capture_frames)
    capture_thread.daemon = True
    capture_thread.start()
    time.sleep(1)
    app.run(host='0.0.0.0', port=8080, debug=False)
