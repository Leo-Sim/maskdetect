
if __name__ == '__main__':
    from ultralytics import YOLO

    model = YOLO('yolov8n.pt')
    # Train Yolo using configuration in data.yaml file
    model.train(data='data.yaml', epochs=70, batch=32, imgsz=640, project="runs/train", name="yolo_train", val=False)




