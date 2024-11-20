import sys
import os

sys.path.append(os.path.abspath('../config'))
sys.path.append(os.path.abspath('../dataset'))

from torch.utils.data import DataLoader
from face_dataset import FaceDataset
from config import Config
from preprocessor import Preprocessor

if __name__ == '__main__':


    yolo_config = Config()
    mask_dataset_path = yolo_config.get_dataset_path1()
    face_dataset_path = yolo_config.get_yolo_config_dataset_path()
    image_size = yolo_config.get_image_size()
    batch_size = yolo_config.get_yolo_config_batch_size()
    yolo_format_export_path = yolo_config.get_yolo_config_export_path()
    face_data_path = yolo_config.get_face_data_path()

    preprocessor = Preprocessor(image_size=image_size)
    transform = preprocessor.get_transforms()

    train_dataset = FaceDataset(face_dataset_path, mask_dataset_path, transform, 'train', image_size)
    train_dataset.export_yolo_train_format(yolo_format_export_path, face_data_path)


    # test_dataset = FaceDataset(face_dataset_path, mask_dataset_path, transform, 'test')
    # test_dataset.export_yolo_train_format(yolo_format_export_path)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


