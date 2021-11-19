import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
from pathlib import Path
import base64
import cv2
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import scipy
from joblib import Parallel , delayed


IMG_SIZE = 256
base_path = Path('../dacon')

test_path = list((base_path / 'test').glob('test*'))

label_info = pd.read_csv((base_path /'class_id_info.csv'))
categories = {i[0]:i[1]-1 for i in label_info.to_numpy()}
# label_info


save_path = Path('./')
new_image_path = save_path / 'test_img' # image폴더
new_image_path.mkdir(parents=True,exist_ok=True)



def xyxy2coco(xyxy):
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    return [x1, y1, w, h]


def xyxy2yolo(xyxy):
    x1, y1, x2, y2 = xyxy
    w, h = x2 - x1, y2 - y1
    xc = x1 + int(np.round(w / 2))  # xmin + width/2
    yc = y1 + int(np.round(h / 2))  # ymin + height/2
    return [xc / IMG_SIZE, yc / IMG_SIZE, w / IMG_SIZE, h / IMG_SIZE]


def scale_bbox(img, xyxy):
    # Get scaling factor
    scale_x = IMG_SIZE / img.shape[1]
    scale_y = IMG_SIZE / img.shape[0]

    x1, y1, x2, y2 = xyxy
    x1 = int(np.round(x1 * scale_x, 4))
    y1 = int(np.round(y1 * scale_y, 4))
    x2 = int(np.round(x2 * scale_x, 4))
    y2 = int(np.round(y2 * scale_y, 4))

    return [x1, y1, x2, y2]  # xmin, ymin, xmax, ymax


def save_image_label(json_file, mode):
    with open(json_file, 'r') as f:
        json_file = json.load(f)

    image_id = json_file['file_name'].replace('.json', '')

    # decode image data
    image = np.frombuffer(base64.b64decode(json_file['imageData']), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite(str(new_image_path / (image_id + '.png')), image)

    # extract bbox
    origin_bbox = []
    if mode == 'train':
        with open(new_label_path / (image_id + '.txt'), 'w') as f:
            for i in json_file['shapes']:
                bbox = i['points'][0] + i['points'][2]
                origin_bbox.append(bbox)
                bbox = scale_bbox(image, bbox)
                bbox = xyxy2yolo(bbox)

                labels = [categories[i['label']]] + bbox
                f.writelines([f'{i} ' for i in labels] + ['\n'])
    return origin_bbox

def test_image(json_file, mode):
    with open(json_file, 'r') as f:
        json_file = json.load(f)

    image_id = json_file['file_name'].replace('.json', '')

    # decode image data
    image = np.frombuffer(base64.b64decode(json_file['imageData']), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    cv2.imwrite(str(new_image_path / (image_id + '.png')), image)



import multiprocessing as mp

Parallel(n_jobs=mp.cpu_count(),prefer="threads")(delayed(test_image)(str(test_json),'test') for test_json in tqdm(test_path))

