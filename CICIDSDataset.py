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


class CICIDSDataset(Dataset):
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
        return 10 #len(self.clinical_data)

    def __getitem__(self, index):
        if type(index) is tuple:
            index = index[0]
        idx_img = self.clinical_data.iloc[index, 0]
        path_img = "{}/{}.png".format(self.root_dir, idx_img)
        label = 0 if np.argmax(self.clinical_data.iloc[idx_img, -8:]) == 0 else 1
        clinical = self.clinical_data.iloc[idx_img, 1:-8]
        with open(path_img, "rb") as i_file:
            img = Image.open(i_file)
            img = img.resize((self.isize, self.isize))
            if self.transform_img is not None:
                img = self.transform_img(img)
        clinical = torch.tensor(clinical.astype(np.float32).values)
        data = {'CT': img, 'Clinical': clinical}
        return data, label

