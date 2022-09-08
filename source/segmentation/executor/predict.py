# link dataset https://www.kaggle.com/datasets/andyczhao/covidx-cxr2

# import module
from dataloader.custom_dataset import DatasetPredict
from dataloader.transform import transfms

# import lib
import cv2
from skimage import morphology
from tqdm import tqdm

from torch.utils.data import DataLoader

import torch
import numpy as np
import matplotlib.pyplot as plt


# data COVIDx
def dataloaderPre(data_dir, batch_size, transfms, train=True, Neg=True):

    if train is True:
        data_dir += 'EDA_Train/'
    else:
        data_dir += 'EDA_Test/'

    if Neg is True:
        data_dir += 'Negative/'
    else:
        data_dir += 'Positive/'

    print('datadir: ', data_dir)

    data = DatasetPredict(data_dir, transform=transfms)

    loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False
            )

    return loader


# dataset COVIDx
def predict(dataloader, model, device):
    model.eval()
    with torch.no_grad():
        image, y_predict = [], []
        for x, _ in tqdm(dataloader):
            x = x.to(device)
            y_pred = model(x)

            # mask output
            pred = y_pred.cpu().numpy()

            pred = pred.reshape(len(pred), 256, 256)
            # threshold
            pred = pred > 0.3
            pred = np.array(pred, dtype=np.uint8)
            y_predict.append(pred)

            x = x.cpu().numpy()
            x = x.reshape(len(x), 256, 256)
            x = x * 0.3 + 0.59
            x = np.squeeze(x)

            x = np.clip(x, 0, 1)
            image.append(x)
    return image, y_predict


def postprocess(img):

    areas = []
    img = img.astype("uint8")
    # làm mờ ảnh
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    # nhị phân hóa ảnh
    _, thresh = cv2.threshold(
        blur, 0, 1,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # tính contours
    contours, _ = cv2.findContours(
        thresh.copy(),
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        areas.append(cv2.contourArea(contours[i]))
    areas = np.sort(areas)[::-1]

    thresh = thresh.astype(bool)
    if len(contours) > 1:
        thresh = morphology.remove_small_objects(thresh.copy(), areas[1])
    if len(contours) > 2:
        thresh = morphology.remove_small_holes(thresh, areas[2])

    return thresh


def save_filename(data, path, opt):
    #################################
    # TAO FILE TXT LUU TEN ANH #####
    ################################
    list_name = []
    # for _, name in tqdm(dataloaderPre(opt.batch_size, transfms)['Positive']):
    for _, name in tqdm(data):
        list_name.append(name)

    res = []
    for i in tqdm(range(len(list_name) - 1)):
        for idx in range(opt.batch_size):
            # print(i, idx, end= ' ')
            res.append(list_name[i][idx])
    # xu li phan le trong batch cuoi
    for i in range(len(list_name[len(list_name) - 1])):
        res.append(list_name[len(list_name) - 1][i])

    np.savetxt(path, res, fmt='%s')


# save lung mask
def save_lungmask(imgpath, train, neg, file_name, path, path_np, opt):
    y = np.load(path_np, allow_pickle=True)
    list_name = np.loadtxt(file_name, dtype=list)
    bs = len(dataloaderPre(imgpath, opt.batch_size, transfms, train, neg))

    for i in tqdm(range(bs - 1)):

        for idx in range(opt.batch_size):
            # ret = postprocess(y[i][idx])
            ret = y[i][idx]
            plt.imsave(path + list_name[i * opt.batch_size + idx], ret)

    # xu li phan le trong batch cuoi
    for idx in tqdm(range(len(list_name) - (bs - 1) * opt.batch_size)):
        # ret = postprocess(y[bs - 1][idx])
        ret = y[bs - 1][idx]
        plt.imsave(path + list_name[(bs - 1) * opt.batch_size + idx], ret)
    plt.close('all')
