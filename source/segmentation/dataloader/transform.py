import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

from configs.myconfigs import get_opt


opt = get_opt()
size = opt.img_size
mean = opt.mean
std = opt.std

augs = A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),

        A.OneOf([
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                alpha_affine=120 * 0.03),  # phép biến đổi co giãn

            A.Rotate(limit=15),

            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=20)  # tuong tu RandomAffine cua PyTorch
        ], p=0.1),

        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])


transfms = A.Compose([
    A.Resize(size, size),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])


def img_de_normalize(img, mask, mean, std):

    img = np.squeeze(img)
    img = img * std + mean
    mask = mask * std + mean
    img = np.clip(img, 0, 1)
    mask = np.clip(mask, 0, 1)

    return img, mask
