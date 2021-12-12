import json
import random
import numpy as np
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, Normalize, Resize, Compose

DEFAULT_PARAMS_PATH = 'FaceSegmentation/config/params.json'


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def set_device(cuda: bool) -> torch.device:
    device = torch.device("cuda:1") if cuda else torch.device("cpu")
    return device


def load_params(filepath: str) -> Dict:
    with open(filepath) as fp:
        d = json.load(fp)
    return d


def save_dict(d: Dict, filepath: str) -> None:
    with open(filepath, "w") as fp:
        json.dump(d, fp=fp)


def mIOU(label, pred, num_classes=19):
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)

    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else:
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)


def get_data_transform(model_mode: str, image_resize: Tuple[int, int],
                       mean: Tuple[float, float, float],
                       std: Tuple[float, float, float]) -> Compose:
    if model_mode == "train":
        transform_train = Compose([Resize(image_resize[0], image_resize[1]),
                                   Normalize(mean=mean, std=std, p=1),
                                   HorizontalFlip(p=0.5),
                                   ToTensorV2()])
        return transform_train

    elif model_mode == "eval":
        transform_eval = Compose([Resize(image_resize[0], image_resize[1]),
                                  Normalize(mean=mean, std=std, p=1),
                                  ToTensorV2()])
        return transform_eval


def colorize(gray_image, n=19):
    cmap = np.array([(0, 0, 0), (204, 0, 0), (76, 153, 0),
                     (204, 204, 0), (51, 51, 255), (204, 0, 204), (0, 255, 255),
                     (51, 255, 255), (102, 51, 0), (255, 0, 0), (102, 204, 0),
                     (255, 255, 0), (0, 0, 153), (0, 0, 204), (255, 51, 153),
                     (0, 204, 204), (0, 51, 0), (255, 153, 51), (0, 204, 0)],
                    dtype=np.uint8)

    cmap = torch.from_numpy(cmap[:n])

    hight, weight = gray_image.shape
    color_image = torch.ByteTensor(3, hight, weight).fill_(0)
    for label in range(0, len(cmap)):
        mask = (label == gray_image)
        color_image[0][mask] = cmap[label][0]
        color_image[1][mask] = cmap[label][1]
        color_image[2][mask] = cmap[label][2]
    return color_image
