from cProfile import label

from torch.utils.data import Dataset
from lxml import etree
from typing import List
from PIL import Image

import os


# class that saves label information
class LabelInfo:
    def __init__(self, path: str, label: str, xmin: int, xmax: int, ymin: int, ymax: int):
        self._path = path
        self._label = label
        self._xmin = int(xmin)
        self._xmax = int(xmax)
        self._ymin = int(ymin)
        self._ymax = int(ymax)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, value):
        self._path = value

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        self._label = value

    @property
    def xmin(self):
        return self._xmin

    @xmin.setter
    def xmin(self, value):
        self._xmin = value

    @property
    def xmax(self):
        return self._xmax

    @xmax.setter
    def xmax(self, value):
        self._xmax = value

    @property
    def ymin(self):
        return self._ymin

    @ymin.setter
    def ymin(self, value):
        self._ymin = value

    @property
    def ymax(self):
        return self._ymax

    @ymax.setter
    def ymax(self, value):
        self._ymax = value


# Save all image and label information in '_label_info_list'.
# An image may have multiple mask labels, so this class handle facedataset with 'label_info_list'
class MaskDataset(Dataset):
    def __init__(self, directory_path, transform, goal):
        super().__init__()

        self.transform = transform

        self.directory_path = directory_path + os.sep + goal
        self.label_path = self.directory_path + os.sep + 'annotations'
        self.image_path = self.directory_path + os.sep + 'images'


        # map label information to number
        self._label_mapping = {
            'with_mask': 1,
            'without_mask': 0,
            'mask_weared_incorrect': 0
        }

        self._num_of_classes = len(self._label_mapping)

        # label information. it includes label, coordinate of pictures
        self._label_info_list: List[LabelInfo] = []
        self._get_xml()

    @property
    def label_mapping(self):
        """
        Return Class name - number mapping information
        :return: Dictionary of class name to number
        """
        return self._label_mapping

    @property
    def num_of_classes(self):
        """
        :return: number of classes
        """
        return self._num_of_classes

    def _get_xml(self):
        """
        Read all xml label files for images.
        Retrieve all label information in each image.
        :return: None
        """


        # iterate all files in directory
        for filename in os.listdir(self.label_path):
            file_path = os.path.join(self.label_path, filename)

            # read all xml files
            if os.path.isfile(file_path) and filename.endswith('.xml'):

                try:
                    tree = etree.parse(file_path)
                    root = tree.getroot()

                    for element in root:
                        if element.tag == 'filename':
                            image_path = element.text

                        if element.tag == 'object':

                            label = None
                            ymin = None
                            ymax = None
                            xmin = None
                            xmax = None


                            for sub_element in element:

                                if sub_element.tag == 'name':
                                    label =  sub_element.text
                                elif sub_element.tag == 'bndbox':
                                    for bnd_element in sub_element:
                                        if bnd_element.tag == 'xmin':
                                            xmin = bnd_element.text
                                        if bnd_element.tag == 'xmax':
                                            xmax = bnd_element.text
                                        if bnd_element.tag == 'ymin':
                                            ymin = bnd_element.text
                                        if bnd_element.tag == 'ymax':
                                            ymax = bnd_element.text

                            labe_info = LabelInfo(image_path, label, xmin, ymin, xmax, ymax)
                            self._label_info_list.append(labe_info)

                except etree.XMLSyntaxError as e:
                    print(f"Error parsing {file_path}: {e}")

    def get_cropped_image(self, image, label_info):
        """
        :parameter image
        :parameter label_info: label information
        """

        image_xmin = label_info.xmin
        image_xmax = label_info.xmax
        image_ymin = label_info.ymin
        image_ymax = label_info.ymax

        cropped_image = image.crop((image_xmin, image_xmax, image_ymin, image_ymax))
        return cropped_image

    def __len__(self):
        return len(self._label_info_list)


    def __getitem__(self, index):

        label_info = self._label_info_list[index]

        image = Image.open(self.image_path + os.sep + label_info.path)
        cropped_image = self.get_cropped_image(image, label_info)
        image_label = self._label_mapping[label_info.label]

        # if image is RGBA, convert it into RGB
        if image.mode != "RGB":
            cropped_image = cropped_image.convert("RGB")

        if self.transform:
            tensor_image = self.transform(cropped_image)

        return tensor_image, image_label



class MaskDataset2(Dataset):
    def __init__(self, directory_path, transform, goal):
        super().__init__()
        self.transform = transform
        self.directory_path = directory_path

        self.image_info_list = []

        self.mask_paths = os.path.join(directory_path, goal, "with_mask")

        self.without_mask_path = os.path.join(directory_path, goal, "without_mask")

        self._add_image_path_to_list()

    def _add_image_path_to_list(self):

        # add paths of image with mask to list
        for image_name in os.listdir(self.mask_paths):
            image_path = os.path.join(self.mask_paths, image_name)
            label = "1"
            self.image_info_list.append(LabelInfo(image_path, label, 0, 0, 0, 0))

        for image_name  in os.listdir(self.without_mask_path):
            image_path = os.path.join(self.without_mask_path, image_name)
            label = "0"
            self.image_info_list.append(LabelInfo(image_path, label, 0, 0, 0, 0))

    def __len__(self):
        return len(self.image_info_list)

    def __getitem__(self, index):

        label_info = self.image_info_list[index]
        image = Image.open(label_info.path)
        label = label_info.label

        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, int(label)



# if __name__ == '__main__':
    # facedataset = MaskDataset()
    # facedataset._get_xml()



