import numpy as np
import cv2
import time
from ultralytics import YOLO
from sort import Sort

def initialize_resources():
    try:
        cap = cv2.VideoCapture("videos/API.mp4")
        model = YOLO("models/yolov8x.pt")
        tracker = Sort()
        ret, first_frame = cap.read()
        if not ret:
            print("Error reading first frame")
            return None

        trajectory_dict = {}
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (int(cap.get(3)), int(cap.get(4)))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter('output_with_trajectory.mp4', fourcc, frame_rate, frame_size)

        return cap, model, tracker, trajectory_dict, out
    except Exception as e:
        print(f"Error initializing resources: {e}")
        return None

def update_trajectory(center_point, trajectory_dict, track_id):
    if track_id not in trajectory_dict:
        trajectory_dict[track_id] = []
    trajectory_dict[track_id].append(center_point)

def draw_trajectory(frame, trajectory_dict):
    for track_id, points in trajectory_dict.items():
        for i in range(1, len(points)):
            cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 1)

def process_frame(frame, model, tracker, trajectory_dict):
    try:
        results = model(frame, stream=True)
        for res in results:
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.3)[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(boxes).astype(int)
            for xmin, ymin, xmax, ymax, track_id in tracks:
                center_point = ((xmin + xmax) // 2, (ymin + ymax) // 2)
                update_trajectory(center_point, trajectory_dict, track_id)

        draw_trajectory(frame, trajectory_dict)
        return frame
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame

if __name__ == '__main__':
    resources = initialize_resources()
    if resources is None:
        print("Failed to initialize resources.")
        exit(1)

    cap, model, tracker, trajectory_dict, out = resources

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            print("No more frames to read.")
            break
        
        frame = process_frame(frame, model, tracker, trajectory_dict)
        
        out.write(frame)
        
        cv2.imshow("Original Frame with Trajectory", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
