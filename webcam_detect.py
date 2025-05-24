from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
import numpy as np

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Small and fast version

# Global variable for the webcam
cap = None

def get_frame():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access webcam.")
            return None

    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        return None

    # Run YOLOv8 on the frame
    results = model(frame)
    
    # Draw results
    annotated_frame = results[0].plot()
    
    # Convert to jpg format
    ret, buffer = cv2.imencode('.jpg', annotated_frame)
    frame = buffer.tobytes()
    
    return frame

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    while True:
        frame = get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
