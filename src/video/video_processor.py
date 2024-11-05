from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import sys
import os

sys.path.append(os.path.abspath('../dataset'))
sys.path.append(os.path.abspath('../model'))
sys.path.append(os.path.abspath('../config'))

from preprocessor import Preprocessor
from config import Config

video_path = '../../data/video/test.mp4'
cap = cv2.VideoCapture(video_path)

# Load YOLO
yolo_model = YOLO('yolo11n.pt')  # Use a YOLO model (e.g., yolov5s.pt for YOLOv5)

mask_model_path = '../train/model.pth'
mask_model = torch.load(mask_model_path)

config = Config()
image_size = config.get_image_size()

# Read frames
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break

    # put frames in yolo
    results = yolo_model(frame)  # Pass the frame to the YOLO model

    # Draw bounding boxes around detected objects
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        class_id = int(box.cls[0])  # Detected class ID

        confidence = float(box.conf[0])  # Confidence score

        if class_id == 0:
            cropped_frame = frame[y1:y2, x1:x2]
            # cropped_frame = cropped_frame.transpose((2, 0, 1))
            # tensor = torch.tensor(cropped_frame)

            image = Image.fromarray(cropped_frame)

            transforms = Preprocessor(image_size=image_size).get_transforms()

            tensor = transforms(image)
            tensor = tensor.unsqueeze(0)

            mask_output = mask_model(tensor)
            detection_result = mask_output.tolist()
            # print('mask_output ' , mask_output.tolist())

            prop_not_mask = detection_result[0][0]
            prop_mask = detection_result[0][1]

            label = 'No mask' if prop_not_mask > prop_mask else 'Mask'




        # Get label name from class ID
        # label = results[0].names[class_id]  # Convert class_id to label

        # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label

    # Display the frame with YOLO detections
    cv2.imshow('YOLO Object Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()