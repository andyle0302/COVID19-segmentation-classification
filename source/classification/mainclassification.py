# import module

from configs.myconfigs import get_opt
from dataloader.mydataloader import custom_dataloader
from evaluation.mymetric import confusion, report
from executor.train import training_loop
from executor.test import test_loop
from model.mymodel import optimi, initialize_model, load_model, load_chekpoint
from utils.myutils import seed_everything
from utils.visualize import visualize_acc, visualize_loss
from transform import augs, transfms

# importing the libraries
import time

# Pytorch libraries and modules
import torch
from torch.nn import CrossEntropyLoss

# for evaluating model
from sklearn.metrics import accuracy_score


def main(opt):

    # setting variable
    checkpoint_path = opt.project_path
    + opt.checkpoint_path + 'withoutLung/' + 'cp'

    model_path = opt.project_path
    + opt.checkpoint_path + 'withoutLung/' + 'model'

    result_path = opt.project_path + opt.result_path
    feature_extract = opt.feature_extract
    img_path = opt.img_path

    bs = opt.batch_size
    myloader = custom_dataloader(img_path, augs, transfms, bs)
    lr = opt.lr
    num_epochs = opt.num_epochs
    num_classes = opt.num_classes
    lsmodel_name = opt.model_name

    model_list = initialize_model(num_classes, feature_extract)
    model = model_list[5]
    model_name = lsmodel_name[5]
    print(model_name)
    # print(model)
    model_path = model_path + model_name + '.pt'
    checkpoint_path = checkpoint_path + model_name + '.pt'

    optimizer, scheduler = optimi(
        model, device, feature_extract,
        lr, num_epochs)

    # # # TRAIN
    since = time.time()
    loss_list, acc_list = training_loop(
        model, optimizer, criterion, scheduler, device,
        num_epochs, myloader, checkpoint_path, model_path)
    time_elapsed = time.time() - since
    print(
        'Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60,
            time_elapsed % 60))

    # visualize loss and acc
    model, _, loss_list, acc_list = load_chekpoint(checkpoint_path, model)

    # print("Model's state_dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor,"\t", model.state_dict()[param_tensor].size())
    # print()

    visualize_loss(
        loss_list,
        result_path + 'withoutLung/' +
        model_name + '/lossFT_CXR.png',
        model_name)
    visualize_acc(
        acc_list,
        result_path + 'withoutLung/' + model_name + '/ACCFT_CXR.png',
        model_name)

# TEST
    model = load_model(model_path)
    y_true, y_pred = test_loop(model, device, myloader['test'])
    accuracy = accuracy_score(y_true, y_pred)
    print(accuracy)

    confusion(
        y_true, y_pred, opt.classes,
        result_path + 'withoutLung/' +
        model_name + '/confusionmatrix_CXR.png')

    report(
        y_true, y_pred, opt.classes,
        result_path + 'withoutLung/' +
        model_name + '/classification_report_CXR.txt')

    # pred_str = str('')

    # path_image = './pred/covid.jpg'

    # img = Image.open(path_image)
    # plt.imshow(img)

    # predict(path_image,resnet)
    # plt.title('predict:{}'.format(pred_str))
    # plt.text(5,45,'top {}:{}'.format(1,pred_str), bbox = dict(fc='yellow'))
    # plt.show()


if __name__ == '__main__':
    seed = 262022  # random
    seed_everything(seed)
    opt = get_opt()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = CrossEntropyLoss()
    # resnet = initialize_model(opt.num_classes, opt.feature_extract)

    main(opt)
