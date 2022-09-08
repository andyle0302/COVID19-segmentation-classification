# import lib
import torch
import numpy as np

# for seed
import os
import random

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# for preprocessing
import cv2
import shutil

import ssl
ssl._create_default_https_context = ssl._create_unverified_context


transfms_M_STD = A.Compose([
    A.Resize(256, 256),
    # scale pixel values from [0,255] to [0,1]
    A.Normalize(mean=[0.], std=[1.]),
    ToTensorV2()
])


def seed_everything(seed_value):

    # set python seed
    random.seed(seed_value)

    # seed the global NumPy RNG
    np.random.seed(seed_value)

    # seed the RNG for all devices (both CPU and CUDA):
    torch.manual_seed(seed_value)

    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_mean_std(dataloader):
    mean1 = 0.
    std1 = 0.
    nb_samples = 0.
    for data, _, _, _ in dataloader:
        batch_samples = data.size(0)

        # Rearrange data to be the shape of [B, C, W * H]
        # torch.Size([32, 1, 65536])
        data = data.view(batch_samples, data.size(1), -1)

        # Compute mean and std here
        # tinh mean theo width*height va lay tong theo batch_size
        mean1 += data.mean(2).sum(0)
        # print(data.mean(2).size())
        std1 += data.std(2).sum(0)

        # Update total number of images
        nb_samples += batch_samples
    mean1 /= nb_samples
    std1 /= nb_samples

    return mean1, std1


def filter(file_name, source_path, des_path):
    mask = cv2.imread(source_path + file_name, 0)
    _, thresh = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, contours, _ = cv2.findContours(
        thresh.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)

    print("Number of Contours found = " + str(len(contours)))

    # tinh dien tich vung phoi phan doan
    areas = []
    for i in range(len(contours)):
        areas.append(cv2.contourArea(contours[i]))
    print(areas)

    # loc anh
    if areas < (256*256)/2:
        shutil.copy(source_path + file_name, des_path)
