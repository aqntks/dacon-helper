import os

import detectron2
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.checkpoint import DetectionCheckpointer
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fastcore.all import *

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.INPUT.MASK_FORMAT='bitmask'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.MODEL.WEIGHTS = "weights/weight1/model_0014999.pth"
cfg.TEST.DETECTIONS_PER_IMAGE = 1000
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda:1"

predictor = DefaultPredictor(cfg)

dir = os.listdir("./test_img")

results = {
        'file_name': [], 'class_id': [], 'confidence': [], 'point1_x': [], 'point1_y': [],
        'point2_x': [], 'point2_y': [], 'point3_x': [], 'point3_y': [], 'point4_x': [], 'point4_y': []
    }

for fn in dir:
    im = cv2.imread("./test_img/" + fn)
    pred = predictor(im)

    pred_class = pred['instances'].pred_classes.cpu().numpy()
    pred_score = pred['instances'].scores.cpu().numpy()
    pred_boxes = pred['instances'].pred_boxes.tensor.cpu().numpy()

    for i, cls in enumerate(pred_class):
        results['file_name'].append(fn.split('.')[0] + '.json')
        class_id = int(cls + 1)
        confidence = float(pred_score[i])
        boxes = pred_boxes[i]
        point1_x = float(boxes[0])
        point1_y = float(boxes[1])
        point2_x = float(boxes[2])
        point2_y = float(boxes[1])
        point3_x = float(boxes[2])
        point3_y = float(boxes[3])
        point4_x = float(boxes[0])
        point4_y = float(boxes[3])
        results['class_id'].append(class_id)
        results['confidence'].append(confidence)
        results['point1_x'].append(point1_x)
        results['point1_y'].append(point1_y)
        results['point2_x'].append(point2_x)
        results['point2_y'].append(point2_y)
        results['point3_x'].append(point3_x)
        results['point3_y'].append(point3_y)

        results['point4_x'].append(point4_x)
        results['point4_y'].append(point4_y)

        print(f"cls: {class_id}")
        print(f"conf: {confidence}")

submission = pd.DataFrame(results)
submission.to_csv('baseline.csv', index=False)