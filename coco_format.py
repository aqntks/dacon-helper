import os
import cv2
import json
import shutil
import base64
import random
import numpy as np
from tqdm import tqdm
from glob import glob
from typing import Dict, List
import matplotlib.pyplot as plt
from collections import defaultdict


def convert_to_coco(
        root_path: os.PathLike,
        save_path: os.PathLike
) -> None:
    """
        only for train dataset
    """
    res = defaultdict(list)
    json_paths = glob(os.path.join(root_path, 'train', '*.json'))

    categories = {
        '01_ulcer': 1,
        '02_mass': 2,
        '04_lymph': 3,
        '05_bleeding': 4
    }

    n_id = 0
    for json_path in tqdm(json_paths):
        with open(json_path, 'r') as f:
            tmp = json.load(f)

        image_id = int(tmp['file_name'].split('_')[-1][:6])
        res['images'].append({
            'id': image_id,
            'width': tmp['imageWidth'],
            'height': tmp['imageHeight'],
            'file_name': tmp['file_name'],
        })

        for shape in tmp['shapes']:
            box = shape['points']

            x1, y1, x2, y2 = \
                box[0][0], box[0][1], box[2][0], box[2][1]

            w, h = x2 - x1, y2 - y1

            res['annotations'].append({
                'id': n_id,
                'image_id': image_id,
                'category_id': categories[shape['label']],
                'area': w * h,
                'bbox': [x1, y1, w, h],
                'iscrowd': 0,
            })
            n_id += 1

    for name, id in categories.items():
        res['categories'].append({
            'id': id,
            'name': name,
        })

    with open(save_path, 'w') as f:
        json.dump(res, f)


def get_colors(classes: List) -> Dict[str, tuple]:
    return {c: tuple(map(int, np.random.randint(0, 255, 3))) for c in classes}


def draw_bbox(
        json_path: os.PathLike,
        coco_path: os.PathLike,
        save_path: os.PathLike,
        n_images: int = 10,
) -> None:
    '''
        visualization based on COCO format annotation
    '''
    with open(coco_path, 'r') as f:
        ann_json = json.load(f)

    images = [{v['id']: v['file_name']} for v in ann_json['images']]
    categories = {v['id']: v['name'] for v in ann_json['categories']}

    ann = defaultdict(list)
    for a in ann_json['annotations']:
        bbox = list(map(round, a['bbox']))
        ann[a['image_id']].append(
            {
                'category_id': categories.get(a['category_id']),
                'bbox': bbox,
            }
        )

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)

    colors = get_colors(categories.values())
    for v in tqdm(images[:n_images]):
        image_id, file_name = list(v.items())[0]
        file_path = os.path.join(json_path, file_name)
        with open(file_path, 'r') as f:
            json_file = json.load(f)

        image = np.frombuffer(base64.b64decode(json_file['imageData']), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        annots = ann[image_id]

        for a in annots:
            label = a['category_id']
            x1, y1, w, h = a['bbox']
            x2, y2 = x1 + w, y1 + h

            cv2.rectangle(image, (x1, y1), (x2, y2), colors[label], 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - 20), (x1 + tw, y1), colors[label], -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        file_name = file_name.split('.')[0] + '.jpg'
        cv2.imwrite(os.path.join(save_path, file_name), image)


convert_to_coco('.', './train_annotations.json')
draw_bbox('./train/', './train_annotations.json', './examples/before_train/')