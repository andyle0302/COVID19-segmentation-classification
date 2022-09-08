# import lib for metric
import torch
import numpy as np
from sklearn.metrics import accuracy_score, jaccard_score
from sklearn.metrics import f1_score, recall_score, precision_score

# loss
import torch.nn as nn
import torch.nn.functional as F


class ComboLoss(nn.Module):  # Dice + BCE + focal
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (
            inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP) ** gamma * BCE

        Dice_BCE = BCE + dice_loss + focal_loss

        return Dice_BCE


def calculate_metrics(y_pred, y_true):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = y_pred > 0.5  # True False False True
    y_pred = y_pred.astype(np.uint8)  # 0 1 1 0
    y_pred = y_pred.reshape(-1)  # flatten

    """ calculate metrics """
    jaccard = jaccard_score(y_true, y_pred)  # (IoU)
    f1 = f1_score(y_true, y_pred)  # Dice
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    return {
        'jaccard': jaccard,
        'acc': acc,
        'f1': f1,
        'recall': recall,
        'precision': precision}
