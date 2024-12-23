from idlelib.pyparse import trans
from typing import Tuple
import numpy as np
from torchvision import transforms


class Preprocessor:
    def __init__(self, image_size: Tuple[int, int]=(64, 64), Normalize: bool=True):
        """
                init preprocessor class

                parameter:
                - image_size : Target size for image:
                - Normalize (bool) : Apply normalization
                - augmentation (transforms): Additional Augmentation
                """
        self.transforms_list = [transforms.RandomResizedCrop(image_size, scale=(0.8, 1.2))]

        self.transforms_list.append(transforms.RandomHorizontalFlip())
        self.transforms_list.append(transforms.RandomRotation(10))
        self.transforms_list.append(transforms.ToTensor())
        self.transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    def get_transforms(self) -> transforms:
        """
        :return:
            - torchvision.transforms object
        """
        return transforms.Compose(self.transforms_list)