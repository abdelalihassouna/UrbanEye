import numpy as np
import cv2
import time
from ultralytics import YOLO
from sort import Sort

N = 10

def initialize_resources():
    try:
        cap = cv2.VideoCapture("videos/test1.mp4")
        model = YOLO("models/yolov8n.pt")
        tracker = Sort()
        ret, first_frame = cap.read()
        if not ret:
            print("Error reading first frame")
            return None

        decay_heatmap = np.zeros((first_frame.shape[0], first_frame.shape[1]), dtype=np.float32)
        summary_heatmap = np.zeros((first_frame.shape[0], first_frame.shape[1]), dtype=np.float32)  # Added
        trajectory_dict = {}
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_size = (int(cap.get(3)), int(cap.get(4)))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter('output_with_heatmapx2.mp4', fourcc, frame_rate, frame_size)

        return cap, model, tracker, decay_heatmap, summary_heatmap, trajectory_dict, out  # Modified
    except Exception as e:
        print(f"Error initializing resources: {e}")
        return None
    
def identify_high_traffic_areas(summary_heatmap, threshold=10):
    high_traffic_coords = np.argwhere(summary_heatmap > threshold)
    return high_traffic_coords

def annotate_high_traffic_areas(summary_heatmap, high_traffic_coords):
    annotated_heatmap = cv2.applyColorMap(normalize_heatmap(summary_heatmap), cv2.COLORMAP_JET)
    for y, x in high_traffic_coords:
        cv2.circle(annotated_heatmap, (x, y), 20, (255, 255, 255), 2)  # White circles for high-traffic areas
    return annotated_heatmap

def update_decay_heatmap(center_point, decay_heatmap, increase_amount=10):
    x, y = center_point
    radius = 5
    cv2.circle(decay_heatmap, (x, y), radius, increase_amount, -1)

def apply_decay(decay_heatmap, decay_factor=0.98):
    decay_heatmap *= decay_factor

def normalize_heatmap(heatmap):
    max_value = np.max(heatmap)
    if max_value == 0:
        return heatmap
    return (heatmap / max_value * 255).astype(np.uint8)

def overlay_heatmap(frame, decay_heatmap):
    normalized_heatmap = normalize_heatmap(decay_heatmap)
    heatmap_color = cv2.applyColorMap(normalized_heatmap, cv2.COLORMAP_JET)
    overlayed_frame = cv2.addWeighted(frame, 0.8, heatmap_color, 0.2, 0)
    return overlayed_frame

# def process_frame(frame, model, tracker, trajectory_dict, decay_heatmap):
#     try:
#         results = model(frame, stream=True)
#         for res in results:
#             filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.3)[0]
#             boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
#             tracks = tracker.update(boxes).astype(int)
#             for xmin, ymin, xmax, ymax, track_id in tracks:
#                 center_point = ((xmin + xmax) // 2, (ymin + ymax) // 2)
#                 update_decay_heatmap(center_point, decay_heatmap)
                
#         frame = overlay_heatmap(frame, decay_heatmap)
#         return frame
#     except Exception as e:
#         print(f"Error processing frame: {e}")
#         return frame

# Initialize the traffic counter outside the function
traffic_counter = 0

# Define a rectangle by the top-left and bottom-right coordinates for the zone of interest
zone_top_left = (200, 200)
zone_bottom_right = (400, 400)

def process_frame(frame, model, tracker, trajectory_dict, decay_heatmap):
    global traffic_counter  # Declare the counter as global to update it
    try:
        results = model(frame, stream=True)
        for res in results:
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.3)[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(boxes).astype(int)
            
            for xmin, ymin, xmax, ymax, track_id in tracks:
                x_center = (xmin + xmax) // 2
                
                # Create points along the bottom edge of the bounding box
                for x in range(x_center - 20, x_center + 20):  # 20 units away from the center
                    if xmin <= x <= xmax:  # Ensure the point is within the bounding box
                        bottom_edge_point = (x, ymax)
                        update_decay_heatmap(bottom_edge_point, decay_heatmap)
                
                # Check if the bounding box intersects with the zone
                if xmin <= zone_bottom_right[0] and xmax >= zone_top_left[0] and ymax >= zone_top_left[1] and ymax <= zone_bottom_right[1]:
                    traffic_counter += 1  # Increment the traffic counter
        
        # Overlay the heatmap on the frame
        frame = overlay_heatmap(frame, decay_heatmap)
        
        # Display the traffic counter on the frame
        cv2.putText(frame, f"Traffic Count: {traffic_counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    except Exception as e:
        print(f"Error processing frame: {e}")
        return frame



def overlay_and_annotate(frame, summary_heatmap, high_traffic_coords):
    overlayed_frame = overlay_heatmap(frame, summary_heatmap)
    for y, x in high_traffic_coords:
        cv2.circle(overlayed_frame, (x, y), 20, (255, 255, 255), 2)  # White circles for high-traffic areas
    return overlayed_frame



if __name__ == '__main__':
    resources = initialize_resources()
    if resources is None:
        print("Failed to initialize resources.")
        exit(1)

    cap, model, tracker, decay_heatmap, summary_heatmap, trajectory_dict, out = resources  # Modified

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            print("No more frames to read.")
            break

        apply_decay(decay_heatmap)
        
        frame = process_frame(frame, model, tracker, trajectory_dict, decay_heatmap)

        # Update the summary heatmap
        summary_heatmap += decay_heatmap  # Added
        
        out.write(frame)
        
        cv2.imshow("Original Frame with Heatmap", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    high_traffic_coords = identify_high_traffic_areas(summary_heatmap)  # Modified
    final_annotated_heatmap = annotate_high_traffic_areas(summary_heatmap, high_traffic_coords)  # Modified

    # To display
    cv2.imshow("Annotated Summary Heatmap", final_annotated_heatmap)
    cv2.waitKey(0)

    # To save
    cv2.imwrite("Annotated_Summary_Heatmap.png", final_annotated_heatmap)

    cap.release()
    out.release()
    cv2.destroyAllWindows()