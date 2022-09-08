# https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# train tren CXR
class ImageDataset(Dataset):

    # 'Initialization'
    def __init__(self, csv, img_folder, transform):
        self.csv = csv
        self.transform = transform
        self.img_folder = img_folder

        #  [:] lấy hết số cột số hàng của bảng
        self.image_names = self.csv[:]['file_name']
        self.labels = np.array(self.csv[:]['label'])

    # 'Denotes the total number of samples'
    def __len__(self):
        return len(self.image_names)

    # 'Generates one sample of data'
    def __getitem__(self, index):
        # print(self.img_folder+ self.image_names.iloc[index])

        # default BGR
        image = cv2.imread(self.img_folder + self.image_names.iloc[index], 1)
        # convert BGR => RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = np.array(image, dtype=np.float32)

        if self.transform is not None:
            aug = self.transform(image=image)
            image = aug['image']

        targets = self.labels[index]
        targets = int(targets)
        # đọc từng phần tử của mảng, chuyển từ array -> tensor;
        # kiểu int64 tương ứng với long trong pytorch
        targets = torch.tensor(targets, dtype=torch.long)

        return image, targets  # chua 1 cap


# train tren Lung + Img
class LungImageDataset(Dataset):
    # 'Initialization'
    def __init__(
        self, csv, img_folder, mask_folder,
            img_size, train=True, transform=None):

        self.csv = csv
        self.transform = transform
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.train = train
        self.img_size = img_size

        # [:] lấy hết số cột số hàng của bảng
        self.image_names = self.csv[:]['file_name']
        self.labels = np.array(self.csv[:]['label'])

    def __len__(self):
        return len(self.image_names)

    # 'Generates one sample of data'
    def __getitem__(self, index):

        # 0: grayscale
        image = cv2.imread(self.img_folder + self.image_names.iloc[index], 0)

        if self.train is True:
            flag = 'EDA_Train'
        else:
            flag = 'EDA_Test'

        if self.labels[index] == 0:
            lab = '/Negative/'
        else:
            lab = '/Positive/'

        mask = cv2.imread(
            self.mask_folder + flag + lab + self.image_names.iloc[index], 0)

        # nhị phân hóa ảnh
        _, maskthres = cv2.threshold(
            mask, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # dao nguoc anh nhi phan
        _, maskinv = cv2.threshold(
            mask, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # print(image.shape)

        image = cv2.resize(image, (self.img_size, self.img_size))
        maskthres = cv2.resize(maskthres, (self.img_size, self.img_size))
        maskinv = cv2.resize(maskinv, (self.img_size, self.img_size))

        res = cv2.bitwise_and(image, maskthres)
        res = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
        lung = cv2.resize(res, (self.img_size, self.img_size))

        resnotlung = cv2.bitwise_and(image, maskinv)
        resnotlung = cv2.cvtColor(resnotlung, cv2.COLOR_GRAY2RGB)
        cxrnotlung = cv2.resize(resnotlung, (self.img_size, self.img_size))

        # print(lung.shape) # (256, 256, 3)
        if self.transform is not None:
            aug = self.transform(image=lung)
            lung = aug['image']
            lung = lung.float()

        # print(lung.shape) # torch.Size([3, 256, 256])
        # print(cxrnotlung.shape) # (256, 256)
        if self.transform is not None:
            aug = self.transform(image=cxrnotlung)
            cxrnotlung = aug['image']
            cxrnotlung = cxrnotlung.float()

        name = self.image_names[index]
        targets = self.labels[index]

        # đọc từng phần tử của mảng, chuyển từ array -> tensor;
        # kiểu int64 tương ứng với long trong pytorch
        targets = torch.tensor(int(targets), dtype=torch.long)

        return image, mask, maskthres, lung, cxrnotlung, targets, name
