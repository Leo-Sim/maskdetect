import sys
import os
import random
import warnings
from lxml import etree
import shutil

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

        self.class_num = 0

        self.face_data_full_path = os.path.join(face_dataset_path, goal)
        self.mask_data_full_path = os.path.join(mask_dataset_path, goal, self.image_path)
        self.goal = goal

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


        # mix mask dataset and face dataset randomly
        indices = random.sample(range(len(face_label_info_list) + 1), len(self._label_info_list))
        indices.sort()

        for index, value in zip(indices, self._label_info_list):
            face_label_info_list.insert(index, value)

        self._label_info_list = face_label_info_list



    def export_yolo_train_format(self, export_path):
        """
        Export all images and labels as yolo train format

        :parameter export_path: path to export images and label
        :return -> None:
        """

        # create directory if not exist
        export_full_path = os.path.join(export_path, self.goal)

        image_path = os.path.join(export_full_path, 'images')
        label_path = os.path.join(export_full_path, 'labels')

        if not os.path.exists(image_path):
            os.makedirs(image_path)
            print('directory created :', image_path)

        if not os.path.exists(label_path):
            os.makedirs(label_path)
            print('directory created :', label_path)

        # if file is exist in image_path and label_path, then return
        if os.path.isfile(image_path) and os.path.isfile(label_path):
            print('image alread exist.')
            return


        # create label first

        # make label with mask dataset for yolo
        self._make_labels(self.label_path, label_path, '')


        # for mask dataset, crop each face in an image and save it as separate image file
        # for face dataset, copy from original directory
        for i, label_info in enumerate(self._label_info_list):

            if(i %100 == 1):
                print(f'......saving image {i},{len(self._label_info_list)}')

            file_name = label_info.path
            image_target_dir = os.path.join(image_path, file_name)

            # save cropped images
            if file_name.startswith('mak'):

                full_path = os.path.join(self.mask_data_full_path, file_name)
                #
                #-------------------------------------주석풀기 --------------
                image = Image.open(full_path)

                if image.mode != "RGB":
                    image.save(image_target_dir)
                    continue

                else:
                    shutil.copy(full_path, image_target_dir)

    def _make_labels(self, yolo_path, target_path, predefined_label_dir):

        pre_defined_label_path = os.path.join(yolo_path, predefined_label_dir)

        for filename in os.listdir(pre_defined_label_path):

            file_path = os.path.join(pre_defined_label_path, filename)

            # read all xml files
            if os.path.isfile(file_path) and filename.endswith('.xml'):
                label_line = ''
                try:
                    tree = etree.parse(file_path)
                    root = tree.getroot()

                    for element in root:

                        img_width = None
                        img_height = None

                        if element.tag == 'filename':
                            file_name = element.text

                            # iterate only png files
                            if not file_name.endswith('.png'):
                                continue

                        if element.tag == 'path':
                            image_path = element.text
                            image = Image.open(image_path)
                            img_width, img_height = image.size



                        if element.tag == 'object':

                            label = None
                            ymin = None
                            ymax = None
                            xmin = None
                            xmax = None

                            for sub_element in element:

                                if sub_element.tag == 'name':
                                    label = sub_element.text
                                elif sub_element.tag == 'bndbox':
                                    for bnd_element in sub_element:
                                        if bnd_element.tag == 'xmin':
                                            xmin = float(bnd_element.text)
                                        if bnd_element.tag == 'xmax':
                                            xmax = float(bnd_element.text)
                                        if bnd_element.tag == 'ymin':
                                            ymin = float(bnd_element.text)
                                        if bnd_element.tag == 'ymax':
                                            ymax = float(bnd_element.text)

                            if img_width is None and img_height is None and file_name.startswith('mak'):
                                mask_image_path = os.path.join(self.mask_data_full_path, file_name)
                                _image = Image.open(mask_image_path)
                                img_width, img_height = _image.size

                            label_line += self.calculate_relative_position(target_path, xmin, ymin, xmax, ymax, img_width, img_height) + '\n'
                            # label_line = label_line.rstrip("\n")

                    label_file_name = file_name.replace('.png', '')
                    with open(target_path + os.sep+ label_file_name + '.txt', 'w') as f:
                        f.write(label_line)
                        # f.write('78 0.5 0.5 1.0 1.0')
                except etree.XMLSyntaxError as e:
                    print(f"Error parsing {file_path}: {e}")

    def calculate_relative_position(self, target_path, xmin, ymin, xmax, ymax, img_width, img_height):
        # caculate center cordinates
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0

        # calculate height, width
        width = xmax - xmin
        height = ymax - ymin

        # change to relative size
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height


        return f'{str(self.class_num)} {x_center} {y_center} {width} {height}'

    def __len__(self):
        return len(self._label_info_list)

    def __getitem__(self, index):

        label_info = self._label_info_list[index]

        file_name = label_info.path

        if file_name.startswith('mak'):
            full_path = os.path.join(self.mask_data_full_path, label_info.path)

            image = Image.open(full_path)
            if image.mode != "RGB":
                image = image.convert("RGB")

            image = self.get_cropped_image(image, label_info)

        else:
            full_path = os.path.join(self.face_data_full_path, label_info.path)
            image = Image.open(full_path)

            if image.mode != "RGB":
                image = image.convert("RGB")

        if self.transform:
            tensor_image = self.transform(image)



        return tensor_image, self.class_num
