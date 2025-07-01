import cv2
import torch
import numpy as np

# Load your custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/apple/Documents/First/module_wise/yolov5s_weapon.pt')  # Replace with your .pt file path

# Set confidence threshold (adjust as needed)
model.conf = 0.5  # Minimum confidence score (0-1)

# Open video file or webcam
video_path = "/Users/apple/Documents/First/Weapon.mp4"  # Replace with 0 for webcam
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to RGB (YOLO expects RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Run detection
    results = model(rgb_frame)
    
    # Process detections
    for *box, conf, cls in results.xyxy[0]:
        if conf > model.conf:  # Only show high-confidence detections
            x1, y1, x2, y2 = map(int, box)
            
            # Get class name
            class_name = model.names[int(cls)]
            
            # Draw bounding box
            color = (0, 0, 255)  # Red for weapons
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label with class and confidence
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Display output
    cv2.imshow("Weapon Detection", frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()