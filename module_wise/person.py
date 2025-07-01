import cv2
from ultralytics import YOLO
from collections import defaultdict

# Initialize YOLOv8 model (will auto-download if not present)
model = YOLO('yolov8n.pt')  # or yolov8s.pt for better accuracy

# Open video source (0 for webcam, or file path)
video_source = "/Users/apple/Documents/First/persons.mp4"  # Change to your video path or 0 for webcam
cap = cv2.VideoCapture(video_source)

# Storage for tracked persons
tracked_persons = {}
current_id = 1
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    frame_count += 1
    
    if not success:
        break
    
    # Run YOLOv8 tracking (persist=True maintains IDs between frames)
    results = model.track(
        frame, 
        persist=True, 
        classes=[0],  # Only track persons (class 0 in COCO)
        verbose=False  # Disable console output for cleaner execution
    )
    
    # Initialize frame-specific variables
    annotated_frame = frame.copy()
    current_frame_ids = set()

    # Process detections if any exist
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy().tolist()
        
        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Assign new ID if this is a new track
            if track_id not in tracked_persons:
                tracked_persons[track_id] = current_id
                current_id += 1
            
            # Store current active IDs
            current_frame_ids.add(tracked_persons[track_id])
            
            # Draw bounding box and ID
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame, 
                f"ID:{tracked_persons[track_id]}", 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 255), 2
            )
    
    # Display frame information
    info_text = f"Frame: {frame_count} | Current Persons: {len(current_frame_ids)} | Total Unique: {len(tracked_persons)}"
    cv2.putText(
        annotated_frame, 
        info_text, 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, (255, 0, 0), 2
    )
    
    # Show the frame
    cv2.imshow("Person Tracking", annotated_frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print summary
print("\nTracking Summary:")
print(f"Total frames processed: {frame_count}")
print(f"Total unique persons detected: {len(tracked_persons)}")
print(f"Person IDs assigned: {sorted(tracked_persons.values())}")