import torch
import cv2
import numpy as np

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set up the video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to draw bounding boxes and labels on the frame
def draw_boxes(frame, results):
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    n = len(labels)
    for i in range(n):
        row = cords[i]
        if row[4] >= 0.2:  # Confidence threshold
            x1, y1, x2, y2 = int(row[0] * frame.shape[1]), int(row[1] * frame.shape[0]), int(row[2] * frame.shape[1]), int(row[3] * frame.shape[0])
            bgr = (0, 255, 0)  # Color of the box
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, model.names[int(labels[i])], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get predictions
    results = model(frame)

    # Draw bounding boxes
    frame = draw_boxes(frame, results)

    # Display the frame
    cv2.imshow('YOLOv5 Object Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()