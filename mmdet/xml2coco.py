import os
import json
import shutil
import random
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split


class XML2COCO:
    def __init__(self, image_dir, xml_dir, saved_coco_path):
        self.image_dir = image_dir
        self.xml_dir = xml_dir
        self.saved_coco_path = saved_coco_path
        self.train_images = []
        self.val_images = []
        self.train_annotations = []
        self.val_annotations = []
        self.categories = []
        self.image_id = 0
        self.annotation_id = 0
        self.class_name_to_id = {
                "crack": 0, #改成自己的类别
                "finger": 1,
                "black_core": 2,
                "thick_line": 3,
                "star_crack": 4,
                "corner": 5,
                "fragment": 6,
                "scratch": 7,
                "horizontal_dislocation": 8,
                "vertical_dislocation": 9,
                "printing_error": 10,
                "short_circuit": 11,
        }  # 根据自己的类别名称和ID进行替换

    def convert(self, split_ratio=0.1):
        xml_files = [f for f in os.listdir(self.xml_dir) if f.endswith('.xml')]

        train_files, val_files = train_test_split(xml_files, test_size=split_ratio, random_state=42)
        # 在这里增加验证数据集

        for train_file in train_files:
            image_id, annotations = self.parse_xml(os.path.join(self.xml_dir, train_file))
            self.train_images.append({"id": image_id, "file_name": train_file[:-4] + ".jpg"})
            self.train_annotations.extend(annotations)

        for val_file in val_files:
            image_id, annotations = self.parse_xml(os.path.join(self.xml_dir, val_file))
            self.val_images.append({"id": image_id, "file_name": val_file[:-4] + ".jpg"})
            self.val_annotations.extend(annotations)

        self.categories = [{"id": v, "name": k} for k, v in self.class_name_to_id.items()]

        train_coco = {
            "images": self.train_images,
            "annotations": self.train_annotations,
            "categories": self.categories
        }

        val_coco = {
            "images": self.val_images,
            "annotations": self.val_annotations,
            "categories": self.categories
        }

        os.makedirs(os.path.join(self.saved_coco_path, 'annotations'), exist_ok=True)
        os.makedirs(os.path.join(self.saved_coco_path, 'images', 'train2017'), exist_ok=True)
        os.makedirs(os.path.join(self.saved_coco_path, 'images', 'val2017'), exist_ok=True)

        with open(os.path.join(self.saved_coco_path, 'annotations', 'instances_train2017.json'), 'w') as f:
            json.dump(train_coco, f)
        with open(os.path.join(self.saved_coco_path, 'annotations', 'instances_val2017.json'), 'w') as f:
            json.dump(val_coco, f)

        for image in self.train_images:
            image_path = os.path.join(self.image_dir, image['file_name'])
            shutil.copy(image_path, os.path.join(self.saved_coco_path, 'images', 'train2017'))

        for image in self.val_images:
            image_path = os.path.join(self.image_dir, image['file_name'])
            shutil.copy(image_path, os.path.join(self.saved_coco_path, 'images', 'val2017'))

    def parse_xml(self, xml_file):
        with open(xml_file, 'r') as f:
            xml_data = f.read()
        root = ET.fromstring(xml_data)
        image_id = self.image_id
        annotations = []
        image = {
            'file_name': root.find('filename').text,
            'height':root.find('size').find('height').text,
            'width': root.find('size').find('width').text,
            'id': image_id
        }
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            category_id = self.class_name_to_id[class_name]
            bbox = obj.find('bndbox')
            x1 = int(bbox.find('xmin').text)
            y1 = int(bbox.find('ymin').text)
            x2 = int(bbox.find('xmax').text)
            y2 = int(bbox.find('ymax').text)
            w = x2 - x1
            h = y2 - y1
            annotation = {
                'id': self.annotation_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [x1, y1, w, h],
                'area': w * h,
                'iscrowd': 0,
                'image': image
            }
            annotations.append(annotation)
            self.annotation_id += 1

        self.image_id += 1
        return image_id, annotations

if __name__ == '__main__':
    # 声明xml_dir, image_dir, saved_coco_path变量
    xml_dir = "/hy-tmp/solar_cell_EL_image/PVELAD/EL2021/trainval/Annotations"
    image_dir = "/hy-tmp/solar_cell_EL_image/PVELAD/EL2021/trainval/JPEGImages"
    saved_coco_path = "/hy-tmp/mmdetection/mmdet/data/solar_cell_EL_image_coco"
    # 实例化XML2COCO对象
    converter = XML2COCO(image_dir, xml_dir, saved_coco_path)
    # 调用convert方法将xml转成coco
    print("开始转换")
    converter.convert(split_ratio=0.1)
    print("转换结束")

