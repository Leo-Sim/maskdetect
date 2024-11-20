from torch.utils.data import Dataset
import random
import os
from PIL import Image

class HighResolutionMaskDataset(Dataset):
    def __init__(self, directory_path, transform):
        super().__init__()

        self.dataset_path_list = []

        self.transform = transform
        self.directory = directory_path

        self.with_mask_directory = os.path.join(directory_path, "with_mask")
        self.without_mask_directory = os.path.join(directory_path, "without_mask")

        self.mask_path_list = os.listdir(self.with_mask_directory)
        self.without_mask_path_list = os.listdir(self.without_mask_directory)

        # Combine the lists
        self.dataset_path_list = self.mask_path_list + self.without_mask_path_list

        # Shuffle the combined list
        random.shuffle(self.dataset_path_list)

    def __len__(self):
        return len(self.dataset_path_list)

    def __getitem__(self, idx):
        image_name = self.dataset_path_list[idx]



        if "with-mask" in image_name:
            sub_path = "with_mask"
            label = 1
        else:
            sub_path = "without_mask"
            label = 0

        image_full_path = os.path.join(self.directory, sub_path, image_name)

        image = Image.open(image_full_path)

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            tensor_image = self.transform(image)



        return tensor_image, label