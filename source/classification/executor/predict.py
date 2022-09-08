import torch
import warnings
import os


def predict(path_img, model_ft, img_transform, transfms, verbose=False):
    if not verbose:
        warnings.filterwarnings('ignore')

    model_ft.eval()
    if verbose:
        print('Model loader ...')
    image = img_transform(path_img, transfms)
    image1 = image[None, :, :, :]

    with torch.no_grad():
        outputs = model_ft(image1)

        _, pred_int = torch.max(outputs.data, 1)
        _, top1_idx = torch.topk(outputs.data, 1, dim=1)
        pred_idx = int(pred_int.cpu().numpy())
        if pred_idx == 0:
            pred_str = str('Negative')
            print(
                'img: {} is: {}'.format(os.path.basename(path_img), pred_str))
        else:
            pred_str = str('Positive')
            print(
                'img: {} is: {}'.format(os.path.basename(path_img), pred_str))
