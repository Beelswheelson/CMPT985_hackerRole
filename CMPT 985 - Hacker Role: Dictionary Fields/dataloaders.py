import os
from random import sample
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import nibabel as nib


class SliceDataset(Dataset):
    def __init__(self):
        super(SliceDataset, self).__init__()
        path = 'data/'
        self.BrainMask = nib.load(f'{path}sub-OAS30003_MR_d1631_T1w_brain_mask.nii.gz').get_fdata()[:, :, 128].astype(bool)
        self.T1w_slice = nib.load(f'{path}sub-OAS30003_MR_d1631_T1w_brain.nii.gz').get_fdata()[:, :, 128]
        self.T2w_slice = nib.load(f'{path}sub-OAS30003_MR_d1631_T2w_brain.nii.gz').get_fdata()[:, :, 128]
        self.SWI_slice = nib.load(f'{path}sub-OAS30003_MR_d1631_swi_brain.nii.gz').get_fdata()[:, :, 128]
        y, x = torch.meshgrid(torch.arange(0, 256), torch.arange(0, 256), indexing='ij')
        self.coordinate = torch.stack((x, y), -1)/float() + 0.5
        self.coordinate = self.coordinate[torch.from_numpy(self.BrainMask)]
        self.T1w_slice = self.T1w_slice[torch.from_numpy(self.BrainMask)]

    def __len__(self):
        return np.shape(self.T1w_slice)[0] ** 2

    def __getitem__(self, idx):
        return self.coordinate, self.T1w_slice


class SliceDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, num_workers=8):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = SliceDataset()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        pass

    def val_dataloader(self):
        pass
