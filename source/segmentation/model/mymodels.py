# https://github.com/qubvel/segmentation_models.pytorch
import segmentation_models_pytorch as smp

# for load_checkpoint
import torch

# for set_model
from torch.optim import Adam, lr_scheduler

# import module
from evaluation.mymetric import ComboLoss


list_encoder = [
    'resnet18', 'resnet34', 'resnet152', 'efficientnet-b0',
    'efficientnet-b7', 'densenet121', 'vgg11_bn']
list_modelUNet, list_model_UNetPP, list_model_FPN = [], [], []

for encoder_name in range(len(list_encoder)):
    list_modelUNet.append(
        smp.Unet(
            list_encoder[encoder_name],
            encoder_weights='imagenet',
            in_channels=1,
            classes=1))

# smp.UnetPlusPlus
for encoder_name in range(len(list_encoder)):
    list_model_UNetPP.append(
        smp.UnetPlusPlus(
            list_encoder[encoder_name],
            encoder_weights='imagenet',
            in_channels=1, classes=1))

# FPN
for encoder_name in range(len(list_encoder)):
    list_model_FPN.append(
        smp.FPN(
            list_encoder[encoder_name],
            encoder_weights='imagenet',
            in_channels=1,
            classes=1))


def load_checkpoint(checkpoint_path, hist_model):
    checkpoint = torch.load(checkpoint_path)
    # print(checkpoint)
    hist_model['model'].load_state_dict(checkpoint['model_state_dict'])
    hist_model['optimizer'].load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    train_dict = checkpoint['train_list']
    val_dict = checkpoint['val_list']

    return {
        'model': hist_model['model'],
        'optimizer': hist_model['optimizer'],
        'start_epoch': start_epoch,
        'train_dict': train_dict,
        'val_dict': val_dict}


def set_model(lr, model):
    optimizer = Adam(model.parameters(), lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loss_fn = ComboLoss()
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'loss_fn': loss_fn}
