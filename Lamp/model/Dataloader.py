import torch
from torch.utils import data
import torchvision.transforms as tf
import PIL.Image as Image
import PIL.ImageOps as ImageOps

import numpy as np

class Padding(object):
    def __init__(self, out_shape: tuple or int):
        if type(out_shape) is int:
            self.width, self.height = out_shape, out_shape
        if type(out_shape) is tuple:
            self.width, self.height = out_shape[0], out_shape[1]

    def __call__(self, image):
        im = ImageOps.pad(image,(self.width,self.height),method=Image.BICUBIC)
        return im

class MinMaxNormalization(object):
    """
    Normalized (Min-Max) the image.
    """
    def __init__(self, vmin=0, vmax=1):
        """
        Constructor of the grayscale transform.
        ----------
        INPUT
            |---- vmin (float / int) the desired minimum value.
            |---- vmax (float / int) the desired maximum value.
        OUTPUT
            |---- None
        """
        self.vmin = vmin
        self.vmax = vmax

    def __call__(self, image):
        """
        Apply a Min-Max Normalization to the image.
        ----------
        INPUT
            |---- image (PIL.Image) the image to normalize.
        OUTPUT
            |---- image (np.array) the normalized image.
        """
        arr = np.array(image).astype('float32')
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        arr = (self.vmax - self.vmin) * arr + self.vmin
        return arr

class Dataset(data.Dataset):
    def __init__(self,dataframe,transforms):
        self.dataframe = dataframe
    
        self.transform = transforms
    
    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self,idx):

        im = Image.open(self.dataframe.loc[idx,'Paths'])
        im = self.transform(im)

        label = torch.tensor(self.dataframe.loc[idx,'Label'])

        return im, label