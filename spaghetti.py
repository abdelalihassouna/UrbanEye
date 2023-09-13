import numpy as np
import cv2
import time
from ultralytics import YOLO
from sort import Sort
import random
from datetime import datetime
import pandas as pd


# Initialize traffic heatmap
frame_height, frame_width = 720, 1280  # Replace with your video's dimensions
traffic_heatmap = np.zeros((frame_height, frame_width), dtype=np.int32)

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Custom colormap (Green to Red)
custom_colormap = np.zeros((256, 3), dtype=np.uint8)
custom_colormap[:, 1] = np.linspace(255, 0, 256)
custom_colormap[:, 2] = np.linspace(0, 255, 256)

def initialize_resources():
    try:
        cap = cv2.VideoCapture("videos/test1.mp4")
        model = YOLO("models/yolov8m.pt")
        tracker = Sort()
        ret, first_frame = cap.read()
        if not ret:
            print("Error reading first frame")
            return None

        trajectory_dict = {}
        color_dict = {}
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (int(cap.get(3)), int(cap.get(4)))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter('output_with_spaghetti.mp4', fourcc, frame_rate, frame_size)

        return cap, model, tracker, trajectory_dict, color_dict, out
    except Exception as e:
        print(f"Error initializing resources: {e}")
        return None

def update_trajectory(center_point, trajectory_dict, color_dict, track_id):
    if track_id not in trajectory_dict:
        trajectory_dict[track_id] = []
        color_dict[track_id] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    trajectory_dict[track_id].append(center_point)
    
N_PIXELS_PER_METER = 10  # This value needs to be determined for your specific setup

def calculate_speed(point1, point2, time_interval):
    x1, y1 = point1
    x2, y2 = point2
    distance_in_pixels = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    speed_m_per_s = (distance_in_pixels / N_PIXELS_PER_METER) / time_interval  # speed = distance/time
    speed_km_per_h = speed_m_per_s * 3.6  # Convert from m/s to km/h
    return speed_km_per_h

def draw_trajectory(frame, trajectory_dict, color_dict, frame_rate):
    time_interval = 1.0 / frame_rate  # time between frames
    for track_id, points in trajectory_dict.items():
        color = color_dict[track_id]
        
        # Only consider the last 10 points for drawing
        last_10_points = points[-30:]
        
        for i in range(1, len(last_10_points)):
            cv2.line(frame, last_10_points[i - 1], last_10_points[i], color, 2)

        
        # Draw an arrow indicating direction at the last point
        # if len(points) >= 2:
        #     start_point = points[-2]
        #     end_point = points[-1]
        #     cv2.arrowedLine(frame, start_point, end_point, color, 2, tipLength=2)

        # Draw the track ID and speed at the last point
        # if len(points) >= 2:
        #     speed = calculate_speed(points[-2], points[-1], time_interval)
        #     cv2.putText(frame, f"ID: {track_id}, Speed: {speed:.2f} km/h", end_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



def process_frame(frame, model, tracker, trajectory_dict, color_dict, traffic_heatmap, custom_colormap):
    try:
        results = model(frame, stream=True)
        results = model.predict(frame, conf=0.3)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype(float)
        
        for res in results:
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.3)[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            class_labels = res.labels.cpu().numpy()[filtered_indices]  # get class labels here
            tracks = tracker.update(boxes).astype(int)

            for i, (xmin, ymin, xmax, ymax, track_id) in enumerate(tracks):
                center_point = ((xmin + xmax) // 2, (ymin + ymax) // 2)
                x, y = center_point
                class_id = int(class_labels[i])
                class_name = class_list[class_id]  # get class name

                # Draw bounding box
                color = color_dict.get(track_id, (0, 255, 0))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(frame, class_name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Update trajectory
                update_trajectory(center_point, trajectory_dict, color_dict, track_id)

                # Update heatmap
                traffic_heatmap[y][x] += 1

        # Add heatmap overlay if there is data in traffic_heatmap
        if np.max(traffic_heatmap) > 0:  # Avoid division by zero
            normalized_heatmap = (traffic_heatmap / np.max(traffic_heatmap) * 255).astype(np.uint8)
            colored_heatmap = cv2.applyColorMap(normalized_heatmap, custom_colormap)
            frame = cv2.addWeighted(frame, 0.6, colored_heatmap, 0.4, 0)  # Adjust the weights here

        # Additional annotations
        object_count = len(trajectory_dict)
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, f"Time: {current_time}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, f"Object Count: {object_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    except Exception as e:
        print(f"Error processing frame: {e}")

    return frame



if __name__ == '__main__':
    resources = initialize_resources()
    if resources is None:
        print("Failed to initialize resources.")
        exit(1)

    cap, model, tracker, trajectory_dict, color_dict, out = resources

    # Initialize the traffic heatmap and custom colormap
    frame_height, frame_width = 720, 1280  # Replace with your video's dimensions
    traffic_heatmap = np.zeros((frame_height, frame_width), dtype=np.int32)
    
    custom_colormap = np.zeros((256, 3), dtype=np.uint8)
    custom_colormap[:, 1] = np.linspace(255, 0, 256)
    custom_colormap[:, 2] = np.linspace(0, 255, 256)

    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            print("No more frames to read.")
            break


        # First, process the frame to add the heatmap
        frame_with_heatmap = process_frame(frame, model, tracker, trajectory_dict, color_dict, traffic_heatmap, custom_colormap)
        
        # Then, draw the trajectories
        draw_trajectory(frame_with_heatmap, trajectory_dict, color_dict, frame_rate)

        # Write and display the frame
        out.write(frame_with_heatmap)
        cv2.imshow("Spaghetti Chart with Heatmap", frame_with_heatmap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

