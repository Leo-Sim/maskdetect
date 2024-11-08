import sys
import os
import random
import warnings

from PIL import Image
sys.path.append(os.path.abspath('../dataset'))
from mask_dataset import MaskDataset, LabelInfo


"""
    This dataset combines datasets located in data/face and data/mask
"""
from mask_dataset import MaskDataset

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

# Dataset for training 'face'
# Change all labels MaskDataset to 'Face' and add extra face training data
class FaceDataset(MaskDataset):
    def __init__(self, face_dataset_path, mask_dataset_path, transform, goal):
        super().__init__(mask_dataset_path, transform, goal)

        self.face_data_full_path = os.path.join(face_dataset_path, goal)
        self.mask_data_full_path = os.path.join(mask_dataset_path, goal, self.image_path)


        # dataset information is stored in 'self._label_info_list'

        self.face_data_full_path = face_dataset_path + os.sep + goal
        self.add_face_dataset()

    def add_face_dataset(self) -> None:
        """
        add face datasets in random index of _label_info_list
        :return: None
        """

        face_label_info_list = []

        for filename in os.listdir(self.face_data_full_path):
            label_info = LabelInfo(filename, '', 0, 0, 0, 0)
            face_label_info_list.append(label_info)

        print('face list : ', len(face_label_info_list))
        print('label_info list ', len(self._label_info_list))

        # mix mask dataset and face dataset randomly
        indices = random.sample(range(len(face_label_info_list) + 1), len(self._label_info_list))
        indices.sort()

        for index, value in zip(indices, self._label_info_list):
            face_label_info_list.insert(index, value)

        self._label_info_list = face_label_info_list



    def __len__(self):
        return len(self._label_info_list)

    def __getitem__(self, index):

        label_info = self._label_info_list[index]

        file_name = label_info.path

        if file_name.startswith('mak'):
            full_path = self.mask_data_full_path + os.sep + label_info.path
            print('full path : ',full_path)

            image = Image.open(full_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            image = self.get_cropped_image(image, label_info)

        else:
            full_path = self.face_data_full_path + os.sep + label_info.path
            print('full path : ',full_path)
            image = Image.open(full_path)

            if image.mode != "RGB":
                image = image.convert("RGB")

        if self.transform:
            tensor_image = self.transform(image)



        return tensor_image, '100'







