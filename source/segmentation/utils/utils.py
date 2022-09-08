import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# import module
from script.custom_dataset import ImageDataset


#import lib
import torch
import numpy as np
from sklearn.metrics import accuracy_score, jaccard_score, f1_score, recall_score, precision_score

#loss
import torch.nn as nn
import torch.nn.functional as F

# for seed
import os
import random

from torch.utils.data import DataLoader
import pandas as pd

from torch.optim import Adam, lr_scheduler

import albumentations  as A
from albumentations.pytorch.transforms import ToTensorV2

from tqdm import tqdm

# for preprocessing
import cv2
import shutil

class ComboLoss(nn.Module): #Dice + BCE + focal
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, alpha = 0.8, gamma = 2,smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha*(1-BCE_EXP)**gamma*BCE
        
        Dice_BCE = BCE + dice_loss +focal_loss

        return Dice_BCE

def calculate_metrics(y_pred, y_true):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = y_pred > 0.5 # True False False True 
    y_pred = y_pred.astype(np.uint8) # 0 1 1 0
    y_pred = y_pred.reshape(-1) # flatten

    jaccard = jaccard_score(y_true, y_pred) #(IoU)
    f1 = f1_score(y_true, y_pred) # Dice
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return [jaccard, acc, f1, recall, precision]

def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    # print(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    train_list = checkpoint['train_list']
    val_list = checkpoint['val_list']
 
    return model, optimizer, start_epoch, train_list, val_list


def seed_everything(seed):        

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed) # set python seed

    np.random.seed(seed) # seed the global NumPy RNG

    torch.manual_seed(seed) # seed the RNG for all devices (both CPU and CUDA):
    torch.cuda.manual_seed_all(seed)

def set_model(device, lr, model):

    optimizer = Adam(model.parameters(), lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loss_fn = ComboLoss()
    return optimizer, scheduler, loss_fn

def set_dataloader(batch_size, augs, transfms):
    lung_data = '/mnt/ca39154e-4274-4299-9720-afdef386cdc3/covid19_resnet152_python-main/archive_14gb/COVID_QU_Ex/Lung Segmentation Data/Lung Segmentation Data/'

    train_csv = pd.read_csv(lung_data+'train.csv') 
    train_dataset = ImageDataset(train_csv, lung_data + 'Train', augs)

    val_csv = pd.read_csv(lung_data+'val.csv')
    val_dataset = ImageDataset(val_csv, lung_data+'Val', transfms)

    test_csv = pd.read_csv(lung_data+'test.csv')
    test_dataset = ImageDataset(test_csv, lung_data+'Test', transfms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True
    )

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader


transfms_M_STD = A.Compose([
    A.Resize(256,256),
    A.Normalize(mean = [0.], std =[1.]), # scale pixel values from [0,255] to [0,1]
    ToTensorV2() 
])

# def compute_mean_std(dataloader):
#     mean1 = 0.
#     std1 = 0.
#     nb_samples = 0.
#     for data, _, _, _ in dataloader:
#         batch_samples = data.size(0)

#         # Rearrange data to be the shape of [B, C, W * H]
#         data = data.view(batch_samples, data.size(1), -1) # torch.Size([32, 1, 65536])
        
#         # Compute mean and std here
#         mean1 += data.mean(2).sum(0) #tinh mean theo width*height va lay tong theo batch_size
#         # print(data.mean(2).size())
#         std1 += data.std(2).sum(0)

#         # Update total number of images
#         nb_samples += batch_samples
#     mean1 /= nb_samples
#     std1 /= nb_samples

#     return mean1, std1



# def filter(file_name, source_path, des_path):
#     mask = cv2.imread(source_path + file_name,0)
#     _, thresh = cv2.threshold(mask, 0,1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     _,contours, _= cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     print("Number of Contours found = "+ str(len(contours)))

#     #tinh dien tich vung phoi phan doan
#     areas = []
#     for i in range(len(contours)):
#         areas.append(cv2.contourArea(contours[i]))
#     print(areas)

#     # loc anh
#     if areas < (256*256)/2:
#         shutil.copy(source_path + file_name, des_path)


