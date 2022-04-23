#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]



def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class detector(object):
    def __init__(
        self,
        cls_names=COCO_CLASSES,
        device="gpu",
        fp16=False,
        legacy=False,
    ):
        exp = get_exp('yolox\exp\default\yolox_s.py')
        model = exp.get_model()
        if device == "gpu":
            model.cuda()
            if fp16:
                model.half()  # to FP16
        model.eval()
        ckpt = torch.load("./yolox_s.pth", map_location="cpu")
        model.load_state_dict(ckpt["model"])

        self.model = model
        self.cls_names = cls_names
        self.num_classes = exp.num_classes
        self.confthre = 0.5
        self.nmsthre = 0.3
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)


    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
        
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        
        output = outputs[0]
        if output is None:
            return []
        output = output.cpu()
        # select person
        output = output[output[:, 6] == 0]
        if output.shape[0] == 0:
            return []

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio
        bboxes = bboxes[0]
        bboxes[2:] = bboxes[2:] - bboxes[:2]
        bboxes= bboxes.numpy().tolist()
        bboxes = [int(x) for x in bboxes]
        return bboxes




if __name__ == "__main__":
    human_detector = detector()
    box = human_detector.inference('2.jpg')
