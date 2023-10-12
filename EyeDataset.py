import torch
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
import numpy as np
import glob

from torch.utils.data import Dataset

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


class EyeDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, isize=128, transform_img=None, transform_data=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.clinical_data = pd.read_csv(csv_file).sample(frac=1)
        self.root_dir = root_dir
        self.transform_img = transform_img
        self.transform_data = transform_data
        self.isize = isize

    def __len__(self):
        return len(self.clinical_data)

    def __getitem__(self, index):
        if type(index) is tuple:
            index = index[0]
        idx_img = self.clinical_data.iloc[index, 0]
        p_sx = "{}/{}_left.jpg".format(self.root_dir, idx_img)
        p_dx = "{}/{}_right.jpg".format(self.root_dir, idx_img)
        label = 0 if np.argmax(self.clinical_data.iloc[idx_img, -8:]) == 0 else 1
        clinical = self.clinical_data.iloc[idx_img, 1:-8]
        with open(p_sx, "rb") as sx, open(p_dx, "rb") as dx:
            isx = Image.open(sx)
            idx = Image.open(dx)
            if self.transform_img is not None:
                idx = self.transform_img(idx)
                isx = self.transform_img(isx)
        clinical = torch.tensor(clinical.astype(np.float32).values)
        data = {'SX': isx, 'DX': idx, 'Clinical': clinical}
        return data, label

