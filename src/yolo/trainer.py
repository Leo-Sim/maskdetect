import sys
import os

sys.path.append(os.path.abspath('../config'))
sys.path.append(os.path.abspath('../dataset'))

from torch.utils.data import DataLoader
from face_dataset import FaceDataset
from config import Config
from preprocessor import Preprocessor


yolo_config = Config()
mask_dataset_path = yolo_config.get_dataset_path1()
face_dataset_path = yolo_config.get_yolo_config_dataset_path()
image_size = yolo_config.get_image_size()
batch_size = yolo_config.get_yolo_config_batch_size()



print(mask_dataset_path)
print(face_dataset_path)

preprocessor = Preprocessor(image_size=image_size)
transform = preprocessor.get_transforms()

train_dataset = FaceDataset(face_dataset_path, mask_dataset_path, transform, 'train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

i = iter(train_loader)

for a in range(1,100):
    next(i)



