import torch
import torchvision.transforms as tf
from torch.utils import data 
import PIL.Image as Image
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random
import time
import glob 

from baseline import *

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

    def __call__(self, image, mask=None):
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

class Cuttings_Dataset(data.Dataset):
    def __init__(self, sample_df, data_path, mean, std, resize,data_augmentation=True):
        """
        Constructor of the dataset.
        """
        data.Dataset.__init__(self)
        self.sample_df = sample_df
        self.data_path = data_path
        self.data_augmentation = data_augmentation
        
        self.transform =  tf.Compose([tf.Grayscale(num_output_channels=3),
                                        MinMaxNormalization(),
                                        tf.ToTensor(),
                                        tf.Normalize((mean,mean,mean),(std,std,std))])
        
        if data_augmentation:
            self.transform = tf.Compose([tf.Grayscale(num_output_channels=3),
                                            tf.RandomVerticalFlip(p=0.5),
                                            tf.RandomHorizontalFlip(p=0.5),
                                            tf.RandomRotation([-90,90],resample=False, expand=False, center=None, fill=None),
                                            MinMaxNormalization(),
                                            tf.ToTensor(),
                                            tf.Normalize((mean,mean,mean),(std,std,std))])
        
        if resize:
            self.transform = tf.Compose([tf.Grayscale(num_output_channels=3),
                                            tf.Resize(224),
                                            tf.RandomVerticalFlip(p=0.5),
                                            tf.RandomHorizontalFlip(p=0.5),
                                            tf.RandomRotation([-90,90],resample=False, expand=False, center=None, fill=None),
                                            MinMaxNormalization(),
                                            tf.ToTensor(),
                                            tf.Normalize((mean,mean,mean),(std,std,std))])
    def __len__(self):
        """
        Get the number of samples in the dataset.
        """
        return self.sample_df.shape[0]


    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        """
        # load image
        im = Image.open(self.data_path + self.sample_df.loc[idx,'path'])
        # load label
        label = torch.tensor(self.sample_df.loc[idx,'rock_type'])
        
        im = self.transform(im)
        return im,label

# Load model
PATH_to_repo =  "C:/Users/nilso/Documents/EPFL/MA4/PDS Turberg/"
PATH = PATH_to_repo + "Rock_Cuttings_Characterisation/Models/Results/Baseline_resized/model_*"

for model_number, path_model in enumerate(glob.glob(PATH)):
    print(path_model)
    model = Net()
    model = torch.load(path_model)
    model.eval()

    # Load data
    df_path = pd.read_csv('../PostProcessing/Model_Comparison/comparison_data.csv', index_col = 0)

    dataset = Cuttings_Dataset(sample_df=df_path,
                        data_path= PATH_to_repo + 'Rock_Cuttings_Characterisation/Data', 
                        mean = 0.5157003,
                        std = 0.32948261, 
                        resize = True,
                        data_augmentation=False)

    # Prediction
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    preds_vec = []
    time_vec = []
    for i in range(dataset.__len__()):
        inputs, _ = dataset.__getitem__(i)

        time1 = time.time()
        
        outputs = model(inputs.view(1,3,224,224).to(device))
        _, preds = torch.max(outputs, 1)

        time2 = time.time()
        
        time_vec.append(round(time2-time1,5))
        preds_vec+=preds.tolist()

    df_path['baseline_'+str(model_number)] = preds_vec
    df_path['baseline_time_'+str(model_number)] = time_vec
    df_path.to_csv('../PostProcessing/Model_Comparison/comparison_data.csv')