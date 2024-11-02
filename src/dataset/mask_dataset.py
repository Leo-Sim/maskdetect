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


# Read all xml files located in ../../data/mask/annotations
class MaskDataset(Dataset):
    def __init__(self):
        super().__init__()

        # map label information to number
        self.label_mapping = {
            'with_mask': 1,
            'without_mask': 0,
            'mask_weared_incorrect': 0
        }

        # label information. it includes label, coordinate of pictures
        self.label_info_list: List[LabelInfo] = []
        self._get_xml()

        for a in self.label_info_list:
            print(a.label)


    def _get_xml(self):
        directory_path = '../../data/mask/annotations'

        # iterate all files in directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            # read all xml files
            if os.path.isfile(file_path) and filename.endswith('.xml'):

                try:
                    tree = etree.parse(file_path)
                    root = tree.getroot()

                    for element in root:

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

                            labe_info = LabelInfo(file_path, label, xmin, ymin, xmax, ymax)
                            self.label_info_list.append(labe_info)

                except etree.XMLSyntaxError as e:
                    print(f"Error parsing {file_path}: {e}")

    def __len__(self):
        return len(self.label_info_list)

    def __getitem__(self, index):

        label_info = self.label_info_list[index]

        image = Image.open(label_info.path)
        image_label = self.label_mapping[label_info.label]

        xmin = label_info.xmin
        xmax = label_info.xmax
        ymin = label_info.ymin
        ymax = label_info.ymax

        cropped_image = image.crop((xmin, ymin, xmax, ymax))

        return cropped_image, image_label


if __name__ == '__main__':
    dataset = MaskDataset()
    # dataset._get_xml()



