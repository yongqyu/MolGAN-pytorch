from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random

from data.sparse_molecular_dataset import SparseMolecularDataset

class SparseMoleCular(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, data_dir):
        """Initialize and preprocess the CelebA dataset."""
        self.data = SparseMolecularDataset()
        self.data.load(data_dir)

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""

        return index, self.data.data[index], self.data.smiles[index],\
               self.data.data_S[index], self.data.data_A[index],\
               self.data.data_X[index], self.data.data_D[index],\
               self.data.data_F[index], self.data.data_Le[index],\
               self.data.data_Lv[index]

    def __len__(self):
        """Return the number of images."""
        return len(self.data)


def get_loader(image_dir, batch_size, mode, num_workers=1):
    """Build and return a data loader."""

    dataset = SparseMoleCular(image_dir)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
