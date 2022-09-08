# import modules
from dataloader.transform import augs, transfms
from dataloader.mydataloader import set_dataloader
from executor.train import fit
from utils.visualize import visualize_train, plot_acc_loss, plot_IoU_DSC
from utils.myutils import seed_everything
from executor.test import test
from executor.predict import dataloaderPre, predict
from executor.predict import save_filename, save_lungmask
from model.mymodels import load_checkpoint, set_model
from model.mymodels import list_model_FPN
# from model.mymodels import list_model_UNetPP
# from model.mymodels import list_modelUNet
from configs.myconfigs import get_opt
from evaluation.mymetric import calculate_metrics

# import lib
import torch
import numpy as np
from torch.optim import lr_scheduler


def train(hist_model, opt, first=True):

    if first is False:
        hist_model = load_checkpoint(opt.checkpoint_path, hist_model)
        hist_model['scheduler'] = lr_scheduler.CosineAnnealingLR(
            hist_model['optimizer'], T_max=10)

    train_dict, val_dict = fit(
        hist_model, set_dataloader,  calculate_metrics, opt)

    # plot result
    visualize_train(train_dict, val_dict)

    return train_dict, val_dict


def inference(
    img_path, mask_path, img_np, predict_np,
        filename, model, opt, train=True, neg=True):

    # load model
    model.load_state_dict(torch.load(opt.model_path))

    # predict
    images, y_predict = predict(
        dataloaderPre(img_path, opt.batch_size, transfms, train, neg),
        model, device)

    # luu ket qua inference
    np.save(img_np, images)
    np.save(predict_np, y_predict)

    # save file name
    save_filename(
        dataloaderPre(img_path, opt.batch_size, transfms, train, neg),
        filename, opt)

    # save lung mask
    save_lungmask(img_path, train, neg, filename, mask_path, predict_np, opt)


if __name__ == '__main__':
    seed = 262022
    seed_everything(seed)

    opt = get_opt()
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')

    model = list_model_FPN[5].to(device)
    hist_model = set_model(opt.lr, model)
    hist_model['start_epoch'] = opt.start_epoch
    hist_model['train_dict'], hist_model['val_dict'] = [], []

    # TRAIN
    # train_dict, val_dict = train(hist_model,opt)

    # LOAD CHECKPOINT
    # hist_model = load_checkpoint (opt.checkpoint_path, hist_model)

    # visualize train

    plot_acc_loss(
        hist_model['train_dict']['train_loss'],
        hist_model['val_loss'],
        hist_model['train_acc'],
        hist_model['val_acc'],
        opt.project_path + 'result/segmentation/result/Loss_Acc.png')

    plot_IoU_DSC(
        hist_model['train_iou'],
        hist_model['val_iou'],
        hist_model['train_dice'],
        hist_model['val_dice'],
        opt.project_path + 'result/segmentation/result/IoU_DSC.png')

    # plot_loss(train_loss, val_loss)

    # visualize_train(hist_model)

    # TEST
    model.load_state_dict(torch.load(opt.model_path))
    test_dataloader = set_dataloader(
        opt.data_path, opt.batch_size, augs, transfms)['test']

    image, y_true, y_pred, test_dict = test(
        test_dataloader, device, model, calculate_metrics)
    print(test_dict)

    # ghi ket qua
    with open(
        opt.project_path +
            r'result/segmentation/result/test_result.txt', 'w') as wf:
        wf.writelines(str(test_dict.items()))

    # INFERENCE

    img_path = opt.project_path + 'dataset/COVIDxCXR3/'
    mask_path = opt.project_path + 'result/segmentation/lung_mask/'

    # EDATest_Neg
    inference(
        img_path,
        mask_path + 'EDA_Test/Negative/',
        mask_path + 'EDA_Test/img_npN.npy',
        mask_path + 'EDA_Test/y_predictN.npy',
        mask_path + 'EDA_Test/filenameN.txt',
        model, opt, train=False, neg=True)

    # # EDATest_Pos
    # inference(
    #     img_path,
    #     mask_path + 'EDA_Test/Positive/',
    #     mask_path + 'EDA_Test/img_npP.npy',
    #     mask_path + 'EDA_Test/y_predictP.npy',
    #     mask_path + 'EDA_Test/filenameP.txt',
    #     model, opt, train=False, neg=False)

    # # EDATrain_Neg
    # inference(
    #     img_path,
    #     mask_path + 'EDA_Train/Negative/',
    #     mask_path + 'EDA_Train/img_npN.npy',
    #     mask_path + 'EDA_Train/y_predictN.npy',
    #     mask_path + 'EDA_Train/filenameN.txt',
    #     model, opt)

    # # EDATrain_Pos
    # inference(
    #     img_path,
    #     mask_path + 'EDA_Train/Positive/',
    #     mask_path + 'EDA_Train/img_npP.npy',
    #     mask_path + 'EDA_Train/y_predictP.npy',
    #     mask_path + 'EDA_Train/filenameP.txt',
    #     model, opt, neg=False)

    # # EDATrain_Pos (non pos processing)
    # inference(
    #     img_path,
    #     mask_path + 'EDA_Train/Positivenon/',
    #     mask_path + 'EDA_Train/img_npPnon.npy',
    #     mask_path + 'EDA_Train/y_predictPnon.npy',
    #     mask_path + 'EDA_Train/filenamePnon.txt',
    #     model, opt, train=True, neg=False)
