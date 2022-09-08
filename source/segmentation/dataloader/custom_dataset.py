from torch.utils.data import Dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class COVID_QU_ExDataset(Dataset):
    def __init__(self, csv, train_path, transform=None):  # 'Initialization'
        self.csv = csv
        self.transform = transform

        # [:] lấy hết số cột số hàng của bảng
        self.image_names = self.csv[:]['file_name']
        self.labels = self.csv[:]['label']
        self.train_path = train_path

    def __len__(self):  # 'Denotes the total number of samples'
        return len(self.image_names)

    def __getitem__(self, index):  # 'Generates one sample of data'
        ''' repr: xu li duong dan co dau cach
            [1:-1]: bo ' ' dau va cuoi duong dan '''
        img_path = repr(
            self.train_path + '/' + self.labels[index] + '/images/'
            + self.image_names.iloc[index])[1:-1]

        mask_path = repr(
            self.train_path + '/' + self.labels[index] + '/lung masks/'
            + self.image_names.iloc[index])[1:-1]

        # https://viblo.asia/p/series-pandas-dataframe-phan-tich-du-lieu-cung-pandas-phan-3-WAyK8AMEZxX
        image = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, 0)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # print(image.shape)
        # print(mask.shape)
        image = np.array(image, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)

        mask[mask == 255] = 1
        image = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
        # print(image.shape)
        image = np.expand_dims(image, 0).transpose(1, 2, 0)  # (256, 256, 1)
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        label = self.labels[index]
        file_name = self.image_names[index]

        return image, mask, label, file_name


def show_img(img, mask, fn):
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(fn)

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title(fn)
    plt.show()


class DatasetPredict(Dataset):
    def __init__(self, img_folder,  transform=None):  # 'Initialization'

        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):   # 'Denotes the total number of samples'
        return len(os.listdir(self.img_folder))

    def __getitem__(self, index):  # 'Generates one sample of data'

        images_list = os.listdir(self.img_folder)
        images_name = images_list[index]
        images = cv2.imread(self.img_folder + images_name, 0)  # grey

        # đoi qua numpy array kiểu float 32
        images = np.asanyarray(images, dtype=np.float32)
        images = cv2.GaussianBlur(images, (3, 3), cv2.BORDER_DEFAULT)
        images = np.expand_dims(images, 0).transpose(1, 2, 0)  # (256,256,1)

        if self.transform is not None:
            aug = self.transform(image=images)
            images = aug['image']

        return images, images_name
