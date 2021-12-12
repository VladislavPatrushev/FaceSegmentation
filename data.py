import os
import cv2
import glob
import numpy as np
import pandas as pd
from tqdm import trange
from typing import Tuple
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from utils import get_data_transform


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms=None, image_resize: Tuple[int, int] = (224, 224)):
        self.df = df
        self.image_resize = image_resize
        self.transforms = transforms
        self.gb = df.groupby('index')
        self.image_ids = df.index.unique().tolist()

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        df = self.gb.get_group(image_id)
        image = cv2.imread(df['image'].values[0])
        mask = cv2.imread(df['mask'].values[0], cv2.IMREAD_GRAYSCALE)

        if self.transforms is not None:
            augmented_data = self.transforms(image=image, mask=mask)
            image, mask = augmented_data['image'], augmented_data['mask']

        return image, mask.reshape(1, self.image_resize[0], self.image_resize[1]).long().squeeze(0)

    def __len__(self):
        return len(self.image_ids)


def create_mask(folder_base: str, folder_save: str) -> None:
    img_num = 30000
    label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye',
                  'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip',
                  'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

    for k in trange(img_num):
        folder_num = k // 2000
        im_base = np.zeros((512, 512))
        for idx, label in enumerate(label_list):
            filename = os.path.join(folder_base,
                                    str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
            print(filename)
            if os.path.exists(filename):
                im = cv2.imread(filename)
                im = im[:, :, 0]
                im_base[im != 0] = (idx + 1)
        filename_save = os.path.join(folder_save, str(k) + '.png')
        cv2.imwrite(filename_save, im_base)


def create_df(path_to_imgs: str, path_to_masks: str) -> pd.DataFrame:
    list_of_imgpath = glob.glob(path_to_imgs + "*")
    image_to_mask = {''.join((path_to_imgs, str(i), '.jpg')): ''.join((path_to_masks, str(i), '.png'))
                     for i in range(len(list_of_imgpath))}

    dict_imgs = dict()
    dict_imgs['image'] = list(image_to_mask.keys())
    dict_imgs['mask'] = list(image_to_mask.values())

    return pd.DataFrame.from_dict(dict_imgs)


def _split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x_train, x_val = train_test_split(df, train_size=0.7, random_state=42)
    x_val, x_test = train_test_split(x_val, train_size=0.7, random_state=42)

    x_train.index = pd.RangeIndex(len(x_train.index))
    x_val.index = pd.RangeIndex(len(x_val.index))
    x_test.index = pd.RangeIndex(len(x_test.index))

    x_train = x_train.reset_index()
    x_val = x_val.reset_index()
    x_test = x_test.reset_index()

    return x_train, x_val, x_test


def create_splited_dataset(df: pd.DataFrame, config) -> Tuple[Dataset, Dataset, Dataset]:
    transform_train = get_data_transform("train", config.image_resize, config.mean, config.std)
    transform_eval = get_data_transform("eval", config.image_resize, config.mean, config.std)

    x_train, x_val, x_test = _split(df)

    ds_train = ImageDataset(x_train, transform_train)
    ds_val = ImageDataset(x_val, transform_eval)
    ds_test = ImageDataset(x_test, transform_eval)

    return ds_train, ds_val, ds_test
