import sys
import os
import lightning as L
import torch

from torch.utils.data import DataLoader

sys.path.append(os.path.abspath('../dataset'))
sys.path.append(os.path.abspath('../config'))
sys.path.append(os.path.abspath('../model'))


from mask_dataset import MaskDataset
from preprocessor import Preprocessor
from mask_detection_model import MaskDetectionModel

from config import Config
from preprocessor import Preprocessor


config = Config()

batch_size = config.get_batch_size()
epoch = config.get_epochs()
learning_rate = config.get_learning_rate()
momentum = config.get_momentum()
image_size = config.get_image_size()
class_num = config.get_class_num()


dataset_path1 = config.get_dataset_path1()




preprocessor = Preprocessor(image_size=image_size)
transform = preprocessor.get_transforms()

train_dataset = MaskDataset(directory_path=dataset_path1, transform=transform, goal='train')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


test_dataset = MaskDataset(directory_path=dataset_path1, transform=transform, goal='test')
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = MaskDetectionModel(num_of_classes=class_num, lr=learning_rate, momentum=momentum, image_size=image_size)
trainer = L.Trainer(max_epochs=epoch)
trainer.fit(model, train_dataloaders=train_loader)

trainer.test(model, dataloaders=test_loader)

model_path = 'model.pth'

torch.save(model, model_path)
















