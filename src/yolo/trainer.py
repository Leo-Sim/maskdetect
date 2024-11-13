import sys
import os


sys.path.append(os.path.abspath('../config'))
sys.path.append(os.path.abspath('../dataset'))

from torch.utils.data import DataLoader
from face_dataset import FaceDataset
from config import Config
from preprocessor import Preprocessor

if __name__ == '__main__':
    from ultralytics import YOLO

    yolo_config = Config()
    mask_dataset_path = yolo_config.get_dataset_path1()
    face_dataset_path = yolo_config.get_yolo_config_dataset_path()
    image_size = yolo_config.get_image_size()
    batch_size = yolo_config.get_yolo_config_batch_size()
    yolo_format_export_path = yolo_config.get_yolo_config_export_path()


    preprocessor = Preprocessor(image_size=image_size)
    transform = preprocessor.get_transforms()

    # train_dataset = FaceDataset(face_dataset_path, mask_dataset_path, transform, 'train')
    # train_dataset.export_yolo_train_format(yolo_format_export_path)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


    # test_dataset = FaceDataset(face_dataset_path, mask_dataset_path, transform, 'test')
    # test_dataset.export_yolo_train_format(yolo_format_export_path)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = YOLO('yolov8n.pt')
    model.train(data='data.yaml', epochs=20, batch=16, imgsz=640, project="runs/train", name="exp_with_new_label", val=False)




