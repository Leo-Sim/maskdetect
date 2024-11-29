import sys
import os
import lightning as L
import torch

from torch.utils.data import DataLoader, ConcatDataset

sys.path.append(os.path.abspath('../dataset'))
sys.path.append(os.path.abspath('../config'))
sys.path.append(os.path.abspath('../model'))


from mask_dataset import MaskDataset, MaskDataset2
from mask_dataset_high_resolution import HighResolutionMaskDataset
from preprocessor import Preprocessor
from mask_detection_model import MaskDetectionModel

from config import Config
from preprocessor import Preprocessor

#  This file is to train CNN model for detecting mask.

# get configuration from Config()
config = Config()

batch_size = config.get_batch_size()
epoch = config.get_epochs()
learning_rate = config.get_learning_rate()
momentum = config.get_momentum()
image_size = config.get_image_size()
class_num = config.get_class_num()

dataset_path1 = config.get_dataset_path1()
dataset_path2 = config.get_dataset_path2()
dataset_path3 = config.get_dataset_path3()
dataset_path4 = config.get_dataset_path4()

# apply preprocessing to images
preprocessor = Preprocessor(image_size=image_size)
transform = preprocessor.get_transforms()

# get test and train dataset, dataloader
train_dataset4 = MaskDataset2(directory_path=dataset_path4, transform=transform, goal="train")
train_loader4 = DataLoader(train_dataset4, batch_size=batch_size, shuffle=True)

test_dataset4 = MaskDataset2(directory_path=dataset_path4, transform=transform, goal="test")
test_loader4 = DataLoader(test_dataset4, batch_size=batch_size, shuffle=True)

# get CNN model for detecting mask.
model = MaskDetectionModel(num_of_classes=class_num, lr=learning_rate, momentum=momentum, image_size=image_size)
trainer = L.Trainer(max_epochs=epoch)

trainer.fit(model, train_dataloaders=train_loader4)

trainer.test(model, dataloaders=test_loader4)

# path to save output model
model_path = 'complete/model.pth'

torch.save(model, model_path)
















