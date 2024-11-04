
from ultralytics import YOLO
import cv2

video_path = '../../data/video/test.mp4'
cap = cv2.VideoCapture(video_path)

# Load YOLO
yolo_model = YOLO('yolo11n.pt')  # Use a YOLO model (e.g., yolov5s.pt for YOLOv5)

# Read frames
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break

    # put frames in yolo
    results = yolo_model(frame)  # Pass the frame to the YOLO model
    print('r : ', results)

    # # Draw bounding boxes around detected objects
    # for box in results[0].boxes:
    #     x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
    #     class_id = int(box.cls[0])  # Detected class ID
    #     confidence = float(box.conf[0])  # Confidence score
    #
    #     # Get label name from class ID
    #     label = results[0].names[class_id]  # Convert class_id to label
    #
    #     # Draw bounding box and label on the frame
    #     # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
    #     # cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
    #     #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label
    #
    # # Display the frame with YOLO detections
    # cv2.imshow('YOLO Object Detection', frame)

    # Exit the loop if 'q' is pressed
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
cap.release()
cv2.destroyAllWindows()