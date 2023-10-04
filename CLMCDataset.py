import random

from pydicom import dcmread
import torch
import os
import pandas as pd
import numpy as np
import glob

from torch.utils.data import Dataset

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class CLMCDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform_img=None, transform_data=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.clinical_data = pd.read_csv(csv_file).sample(frac=1)
        # self.root_dir = root_dir
        self.transform_img = transform_img
        self.transform_data = transform_data

    def __len__(self):
        return 50  #len(self.clinical_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.clinical_data.iloc[idx, 0])
        dicom = dcmread(img_path)
        image = np.array(dicom.pixel_array, dtype=np.float32)
        label = self.clinical_data.iloc[idx, -2]
        clinical = self.clinical_data.iloc[idx, 2:-2]
        mask_path = glob.glob('/'.join(img_path.split('/')[:-2]) + '/*egmentation*/*')[0]
        mask = np.array(dcmread(mask_path).pixel_array, dtype=np.float32)
        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_data:
            clinical = torch.tensor(clinical.astype(float).values)
        data = {'CT': image, 'mask': mask, 'Clinical': clinical}
        return data, label


