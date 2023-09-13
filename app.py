from flask import Flask, render_template, request, jsonify, Response
import cv2
import base64
import numpy as np
import os
import tempfile
from ultralytics import YOLO
import pandas as pd
from sort import Sort

app = Flask(__name__)

# Initialize global variables
uploaded_video_path = None
areas = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    
    global uploaded_video_path  # Declare it as global
    
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        # Save the uploaded file to a temporary file
        temp_fd, temp_filename = tempfile.mkstemp()
        uploaded_file.save(temp_filename)
        
        uploaded_video_path = temp_filename  # Set the video path but don't delete it here
        
        # Read video and get the first frame
        video = cv2.VideoCapture(temp_filename)
        ret, frame = video.read()
        # frame = cv2.resize(frame, (1280, 720))
        video.release()
        
        # Close the temp file descriptor
        os.close(temp_fd)
        
        if not ret:
            return jsonify({'error': 'Cannot read video'}), 400
        
        # Convert frame to JPG and then to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'frame': frame_base64})

@app.route('/set_roi', methods=['POST'])
def set_areas():
    global areas  # Declare it as global so you can modify it
    data = request.json
    raw_areas = data.get('polylines', {})
    
    # Convert list of dictionaries to list of tuples
    for name, points in raw_areas.items():
        areas[name] = [(point['x'], point['y']) for point in points]

    print('Converted areas:', areas)
    return jsonify({'message': 'ROI received'})


def gen():
    global areas, uploaded_video_path  # Declare it as global so you can use it

    if uploaded_video_path is None:
        return "No video uploaded"

    model = YOLO('models/yolov8n.pt')
    cap = cv2.VideoCapture(uploaded_video_path)  # Read from uploaded video

    tracker = Sort()

    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 3 != 0:
            continue

        # frame = cv2.resize(frame, (1280, 720))

        # Make predictions using the YOLO model
        results = model.predict(frame, conf=0.3)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype(float)
        
        bbox_list = []
        for index, row in px.iterrows():
            x1, y1, x2, y2, _, class_id = map(int, row)
            class_name = class_list[class_id]
            # if 'car' in class_name:
            bbox_list.append([x1, y1, x2, y2])
        
        bbox_np_array = np.array(bbox_list)  # Convert list to NumPy array
        bbox_id = tracker.update(bbox_np_array)  # Update the tracker using NumPy array
        
        if bbox_np_array.size == 0:
            print("No bounding boxes found, skipping tracker update.")
        else:
            bbox_id = tracker.update(bbox_np_array)  # Update the tracker using NumPy array

        for bbox in bbox_id:
            x1, y1, x2, y2, id = map(int, bbox)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


            for name, points in areas.items():
                is_inside = cv2.pointPolygonTest(np.array(points, np.int32), (cx, cy), False)
                if is_inside >= 0:
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, str(id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw the polygons (areas)
        print("Drawing ROIs.")  # Debug print
        for name, points in areas.items():
            cv2.putText(frame, name, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.polylines(frame, [np.array(points, np.int32)], True, (0, 0, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    cap.release()
        
    try:
        os.remove(uploaded_video_path)
    except FileNotFoundError:
        print(f"File {uploaded_video_path} not found. Continuing...")

        
@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)