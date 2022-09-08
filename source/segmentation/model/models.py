# https://github.com/qubvel/segmentation_models.pytorch
import segmentation_models_pytorch as smp
from torchsummary import summary

list_encoder = ['resnet18', 'resnet34', 'resnet152', 'efficientnet-b0', 'efficientnet-b7', 'densenet121', 'vgg11_bn' ]
list_modelUNet, list_model_UNetPP, list_model_FPN = [], [], []


for encoder_name in range(len(list_encoder)):
    list_modelUNet.append(smp.Unet(list_encoder[encoder_name],encoder_weights='imagenet',in_channels=1, classes=1))

# smp.UnetPlusPlus
for encoder_name in range(len(list_encoder)):
    list_model_UNetPP.append(smp.UnetPlusPlus(list_encoder[encoder_name],encoder_weights='imagenet',in_channels=1, classes=1))

# FPN
for encoder_name in range(len(list_encoder)):
    list_model_FPN.append(smp.FPN(list_encoder[encoder_name],encoder_weights='imagenet',in_channels=1, classes=1))