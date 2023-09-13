import cv2
import numpy as np

polyline_pts = []  # List to store polyline points

# Mouse callback function to draw polyline
def draw_polyline(event, x, y, flags, param):
    global img, polyline_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        polyline_pts.append((x, y))
        if len(polyline_pts) > 1:
            cv2.polylines(img, [np.array(polyline_pts)], isClosed=False, color=(0, 0, 255), thickness=2)

# Load video
video_path = 'videos/test1.mp4'
cap = cv2.VideoCapture(video_path)

# Read first frame from video
ret, first_frame = cap.read()

if not ret:
    print("Failed to grab frame")
    exit()

img = first_frame.copy()

# Create window and set mouse callback
cv2.namedWindow("Draw Window")
cv2.setMouseCallback("Draw Window", draw_polyline)

# Show the first frame and allow drawing
while True:
    cv2.imshow("Draw Window", img)
    key = cv2.waitKey(10)
    if key == ord('p'):  # Press 'p' to play video
        break
    elif key == 27:  # Escape key to exit
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Play video and overlay polyline
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Overlay polyline on each frame
    if len(polyline_pts) > 1:
        cv2.polylines(frame, [np.array(polyline_pts)], isClosed=False, color=(0, 0, 255), thickness=2)

    cv2.imshow("Video with Polyline", frame)

    if cv2.waitKey(30) == 27:  # Press Escape to exit
        break

cap.release()
cv2.destroyAllWindows()
