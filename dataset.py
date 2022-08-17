# -*- coding: utf-8 -*-
# @Time    : 2022/8/17 14:08
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : dataset.py
# @Software: PyCharm

import os

import cv2
import torch
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import albumentations as A

transform_fn = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


class Config:

    VOC_BBOX_LABEL_NAMES = (
    "D00",
    "D01",
    "D10",
    "D11",
    "D20",
    "D40",
    "D43",
    "D44",
    "D50",
    "Repair",
    "Block crack",
    "D0w0")


category_id_to_name = {
    0: "D00",
    1: "D01",
    2: "D10",
    3: "D11",
    4: "D20",
    5: "D40",
    6: "D43",
    7: "D44",
    8: "D50",
    9: "Repair",
    10: "Block crack",
    11: "D0w0"}

BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
    """Visualizes a single bounding box on the image"""

    x_min, y_min, x_max, y_max = bbox
    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img


def visualize(image, bboxes, category_ids, category_id_to_name):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis('off')
    plt.imshow(img)
    plt.show()


class VOCDataset(Dataset):

    def __init__(self, data_dir, mode="train", transform=None):
        super(VOCDataset, self).__init__()
        # 获取txt文件
        self.data_dir = data_dir
        if mode == "train" or "val" or "test":
            split = mode
        else:
            raise NotImplementedError
        id_list_file = os.path.join(self.data_dir, 'train.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]

        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        idd = self.ids[item]
        # 解析xml文件得到图片的bbox, label
        anno = ET.parse(os.path.join(self.data_dir, 'annotations', 'xmls', idd + '.xml'))

        bbox = []
        label = []

        for obj in anno.findall('object'):

            bndbox_anno = obj.find('bndbox')
            box = []
            for tag in ('xmin', 'ymin', 'xmax', 'ymax'):
                box.append(int(bndbox_anno.find(tag).text) - 1)
            bbox.append(box)

            name = obj.find('name').text.strip()
            if name not in Config.VOC_BBOX_LABEL_NAMES:
                print("Name Error")
                print(idd)
                print("-----------")
            else:
                label.append(Config.VOC_BBOX_LABEL_NAMES.index(name))
        if len(bbox) == 0:
            print("Bbox Error")
            print(idd)
            print("-----------")
        elif len(label) == 0:
            print("Label Error")
            print(idd)
            print("-----------")
        else:
            bboxes = np.stack(bbox).astype(np.float32)
            class_labels = np.stack(label).astype(np.uint8)

        # 获取对应图片
        image = cv2.imread(self.data_dir + '/images/' + idd + '.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        # transformed_image = transformed['image']
        # transformed_bboxes = transformed['bboxes']
        # transformed_class_labels = transformed['class_labels']
        # return transformed_image, transformed_bboxes, transformed_class_labels
        return 0


train_dataset = VOCDataset(r'L:\RDD2022_all_countries\Norway\train', mode='train', transform=transform_fn)
Train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
for _ in range(10505):
    a = next(iter(Train_loader))

# with open(r'L:\RDD2022_all_countries\Norway\train.txt', 'w') as f:
#     for file in os.listdir(r'L:\RDD2022_all_countries\Norway\train\images/'):
#         f.write(file[:-4] + '\n')
