# import module
from .custom_dataset import COVID_QU_ExDataset

# import lib
from torch.utils.data import DataLoader
import pandas as pd


def set_dataloader(lung_data, batch_size, augs, transfms):

    train_csv = pd.read_csv(lung_data+'train.csv')
    train_dataset = COVID_QU_ExDataset(train_csv, lung_data + 'Train', augs)

    val_csv = pd.read_csv(lung_data+'val.csv')
    val_dataset = COVID_QU_ExDataset(val_csv, lung_data+'Val', transfms)

    test_csv = pd.read_csv(lung_data+'test.csv')
    test_dataset = COVID_QU_ExDataset(test_csv, lung_data+'Test', transfms)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader}
