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

video_path = '../../data/video/tes1t.mp4'
cap = cv2.VideoCapture(video_path)

# Load YOLO
yolo_model = YOLO('../yolo/runs/train/yolo_train5/weights/best.pt')

mask_model_path = '../train/model.pth'
mask_model = torch.load(mask_model_path)

config = Config()
image_size = config.get_image_size()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mask_model.to(device)

# Read frames
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break

    # Pass the frame to the YOLO model
    results = yolo_model(frame, conf=0.5)

    for box in results[0].boxes:

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        if class_id == 0:
            cropped_frame = frame[y1:y2, x1:x2]
            tensor = torch.tensor(cropped_frame)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"{confidence: .2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)  # Label

            image = Image.fromarray(cropped_frame)


            transforms = Preprocessor(image_size=image_size).get_transforms()

            tensor = transforms(image)
            tensor = tensor.unsqueeze(0)

            # move tensor to gpu
            tensor = tensor.to(device)


            mask_output = mask_model(tensor)
            device = next(mask_model.parameters()).device
            print(f"The model is running on: {device}")

            # if prop_not_mask < prop_mask:
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle
            #     cv2.putText(frame, f"{''} {prop_mask:.2f}", (x1, y1 - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Label


    cv2.imshow('YOLO Object Detection', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()