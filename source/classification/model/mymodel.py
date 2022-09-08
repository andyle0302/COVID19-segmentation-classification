from torchvision import models
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(num_classes, feature_extract):
    model_list = []
    modelvgg19_bn = models.vgg19_bn(weights='VGG19_BN_Weights.DEFAULT')

    set_parameter_requires_grad(modelvgg19_bn, feature_extract)

    num_ftrs = modelvgg19_bn.classifier[6].in_features
    modelvgg19_bn.classifier[6] = nn.Linear(num_ftrs, num_classes)

    modelResNet152 = models.resnet152(weights='ResNet152_Weights.DEFAULT')
    set_parameter_requires_grad(modelResNet152, feature_extract)
    num_ftrs = modelResNet152.fc.in_features
    modelResNet152.fc = nn.Linear(num_ftrs, num_classes)

    modelResNet18 = models.resnet18(weights='ResNet18_Weights.DEFAULT')
    set_parameter_requires_grad(modelResNet18, feature_extract)
    num_ftrs = modelResNet18.fc.in_features
    modelResNet18.fc = nn.Linear(num_ftrs, num_classes)

    modelResNet50 = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    set_parameter_requires_grad(modelResNet50, feature_extract)
    num_ftrs = modelResNet50.fc.in_features
    modelResNet50.fc = nn.Linear(num_ftrs, num_classes)

    modelEff = models.efficientnet_b7(
        weights='EfficientNet_B7_Weights.DEFAULT')
    set_parameter_requires_grad(modelEff, feature_extract)
    num_ftrs = modelEff.classifier[1].in_features
    modelEff.classifier[1] = nn.Linear(num_ftrs, num_classes)

    modelResNet101 = models.resnet101(weights='ResNet101_Weights.DEFAULT')
    set_parameter_requires_grad(modelResNet101, feature_extract)
    num_ftrs = modelResNet101.fc.in_features
    modelResNet101.fc = nn.Linear(num_ftrs, num_classes)
    # model = CovidNet('small', n_classes=num_classes).cuda()
    model_list.append(modelResNet152)
    model_list.append(modelvgg19_bn)
    model_list.append(modelResNet18)
    model_list.append(modelResNet50)
    model_list.append(modelResNet101)
    model_list.append(modelEff)

    return model_list


def load_model(model_path):
    model = torch.load(model_path)
    # model.load_state_dict(checkpoint['model_state_dict'])
    return model


def optimi(model_ft, device, feature_extract, lr, num_epochs):
    # Send the model to GPU
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    # print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for _, param in model_ft.named_parameters():
            if param.requires_grad is True:
                params_to_update.append(param)
                # print("\t",name)
    # else:
    #     for name,param in model_ft.named_parameters():
    #         if param.requires_grad == True:
                # print("\t",name)

    # Observe that all parameters are being optimized
    optimizer = Adam(params_to_update, lr=lr, weight_decay=lr/num_epochs)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[5, 10],
        gamma=0.1,
        last_epoch=-1,
        verbose=False)

    return optimizer, scheduler


def load_chekpoint(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss_list = checkpoint['loss_list']
    acc_list = checkpoint['train_acc']

    return model, epoch, loss_list, acc_list
