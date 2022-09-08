import torch
from PIL import Image
import cv2
import numpy as np

# visualize
import matplotlib.pyplot as plt

# for evaluating model
from sklearn.metrics import accuracy_score


def test_loop(model_ft, device, test_dataloader):
    with torch.no_grad():
        y_true = []
        y_pred = []
        model_ft.to(device)
        model_ft.eval()
        # for data, target in test_dataloader:  # cxr
        # for _, _, _, data, _, target, _  in test_dataloader:  # lung
        for _, _, _, _, data, target, _ in test_dataloader:  # cxrnotlung
            data = data.to(device)
            target = target.to(device)
            output = model_ft(data)
            _, pred = torch.max(output, 1)
            y_true += target.tolist()
            y_pred += pred.tolist()
    return y_true, y_pred


def img_transform(path_img, test_transform):
    img = cv2.imread(path_img, 0)
    # img = Image.open(path_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, dtype=np.float32)
    img = cv2.resize(img, (256, 256))
    aug = test_transform(image=img)
    res = aug['image']
    res = res.float().cuda()
    return res


def testreport(model, confusion, report, dataloader, classes, predict, device):

    y_true, y_pred = test_loop(model, device, dataloader['test'])
    accuracy = accuracy_score(y_true, y_pred)
    print(accuracy)

    confusion(y_true, y_pred, classes, './report/CXR/confusionmatrix_CXR.png')
    report(y_true, y_pred, classes, './report/CXR/report152.txt')

    pred_str = str('')

    path_image = './pred/covid.jpg'

    img = Image.open(path_image)
    plt.imshow(img)

    predict(path_image, model)
    plt.title('predict:{}'.format(pred_str))
    plt.text(5, 45, 'top {}:{}'.format(1, pred_str), bbox=dict(fc='yellow'))
    plt.show()
