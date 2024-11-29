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

# This class extends the custom dataset (MaskDataset) in 'src/dataset/mask_dataset'
# Dataset for training 'face'
# Change all labels MaskDataset to 'Face' and add extra face training data
class FaceDataset(MaskDataset):
    def __init__(self, face_dataset_path, mask_dataset_path, transform, goal, target_image_size):
        super().__init__(mask_dataset_path, transform, goal)

        self.target_image_size = target_image_size[0]
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



    def export_yolo_train_format(self, export_path, face_data_path):
        """
        Export all images and labels as yolo train format

        :parameter export_path: path to export images and label
        :parameter face_data_path: path to labeled face dataset
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

        # store all images' name located in face_data_path in a set
        labeled_image_path = os.path.join(face_data_path, "images")
        labeled_image_label_path = os.path.join(face_data_path, "labels")

        labeled_file_names = set(os.listdir(labeled_image_path))

        # create label first
        # make label with mask dataset for yolo


        if not os.path.isfile(label_path):
            self._make_labels(self.label_path, label_path)
            self._make_labels(labeled_image_label_path, label_path)


        # for mask dataset, crop each face in an image and save it as separate image file
        # for face dataset, copy from original directory

        # add name of labels in a set
        label_name_set = set()
        for label_name in os.listdir(label_path):
            label_name = label_name.replace(".txt", "")
            label_name_set.add(label_name)


        print(f"Copying images in {labeled_image_path} to {image_path}" )
        for image_name in os.listdir(labeled_image_path):

            if(image_name.replace(".png", "").replace(".jpg", "") in label_name_set):
                full_path = os.path.join(labeled_image_path, image_name)
                image_target_dir = os.path.join(image_path, image_name)
                shutil.copy(full_path, image_target_dir)


        print(f"Copying images in {self.mask_data_full_path} to {image_path}" )

        for image_name in os.listdir(self.mask_data_full_path):

            if image_name.replace(".png", "") in label_name_set:

                image_target_dir = os.path.join(image_path, image_name)
                full_path = os.path.join(self.mask_data_full_path, image_name)

                image = Image.open(full_path)

                # Check if the mode is RGB
                if image.mode != "RGB":
                    # Convert to RGB and save
                    image = image.convert("RGB")

                # Save the image
                image.save(image_target_dir)

    def _make_labels(self, source_path, target_path):
        """

        :param source_path: directory path for making face datasets
        :param target_path: path to save the results
        :return: None
        """

        label_list = os.listdir(source_path)

        for filename in label_list:

            file_path = os.path.join(source_path, filename)

            # read all xml files
            if os.path.isfile(file_path) and filename.endswith('.xml'):
                label_line = ''
                try:
                    tree = etree.parse(file_path)
                    root = tree.getroot()

                    img_width = None
                    img_height = None
                    for element in root:

                        if element.tag == 'filename':
                            file_name = element.text

                            # iterate only png files
                            if not file_name.endswith('.png'):
                                continue

                        if element.tag == 'size':

                            for sub_element in element:

                                if sub_element.tag == 'width':
                                    img_width = float(sub_element.text)

                                if sub_element.tag == 'height':
                                    img_height = float(sub_element.text)

                        #  get face coordinates from image
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

                            # filter images not to include low resolution face
                            min_size = 10
                            if (xmax - xmin) < min_size or (ymax - ymin < min_size):
                                continue

                            label_line += self.calculate_relative_position(target_path, xmin, ymin, xmax, ymax, img_width, img_height) + '\n'

                    if label_line == "":
                        continue

                    label_file_name = file_name.replace('.png', '')
                    label_file_name = label_file_name.replace('.jpg', '')
                    with open(target_path + os.sep+ label_file_name + '.txt', 'w') as f:
                        f.write(label_line)
                        # f.write('78 0.5 0.5 1.0 1.0')
                except etree.XMLSyntaxError as e:
                    print(f"Error parsing {file_path}: {e}")

    def calculate_relative_position(self, target_path, xmin, ymin, xmax, ymax, img_width, img_height):
        """

        retrieve square area (face) in images for training

        :param target_path: image path
        :param xmin: starting x coordinate
        :param ymin: starting y coordinate
        :param xmax: end x coordinate
        :param ymax: end y coordinate
        :param img_width: the width of the image
        :param img_height: the height of the image
        :return:
        """
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
        """
        return length of dataset
        :return: int
        """
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
