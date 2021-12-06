import torch
from torch.utils import data
import torchvision.transforms as tf

from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image, ImageOps

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random, time, datetime, os, sys, getopt

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../../')))
from Lamp.AttrDict.AttrDict import *


class Transforms():
    def __init__(self, *args, **kwargs):

        self.transforms_list = args[0]
        self.transforms_list.extend([MinMaxNormalization()])

    def get_transforms(self):
        transforms = tf.Compose(
                self.transforms_list
                )
        return transforms

class Padding(object):
    def __init__(self, out_shape: tuple or int):
        if type(out_shape) is int:
            self.width, self.height = out_shape, out_shape
        if type(out_shape) is tuple:
            self.width, self.height = out_shape[0], out_shape[1]

    def __call__(self, image):
        im = ImageOps.pad(image,(self.width,self.height),method=Image.BILINEAR)
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

        return im

def load_config(cfg_path):
    """  """
    if os.path.splitext(cfg_path)[-1] == '.json':
        return AttrDict.from_json_path(cfg_path)
    elif os.path.splitext(cfg_path)[-1] in ['.yaml', '.yml']:
        return AttrDict.from_yaml_path(cfg_path)
    else:
        raise ValueError(f"Unsupported config file format. Only '.json', '.yaml' and '.yml' files are supported.")


class MainWindow():

    #----------------

    def __init__(self, main):

        # canvas for image
        self.canvas = tk.Canvas(main, width=500, height=500,bg='white')
        self.canvas.grid(row=0, column=0,columnspan = 5, rowspan = 1)

        # images
        dict_transform = {
            "Padding":Padding,
            "VerticalFlip":tf.RandomVerticalFlip,
            "HorizontalFlip":tf.RandomHorizontalFlip,
            "Rotation":tf.RandomRotation,
            "CenterCrop":tf.CenterCrop,
            "Resize":tf.Resize,
        }

        transforms_test = Transforms(
            [dict_transform[key]([k for k in item.values()] if len(item.values()) > 1 else [k for k in item.values()][0]) for key, item in inputs.TransformTest.items()] 
        )

        testDataset = Dataset(
            df_data.reset_index(drop=True),
            transforms=transforms_test.get_transforms()
        )

        self.random_list = random.sample(range(0,testDataset.__len__()), testDataset.__len__())

        self.my_images = []
        for im in self.random_list:
            self.my_images.append(ImageTk.PhotoImage(Image.fromarray(testDataset.__getitem__(im)*255).resize((500,500),resample=Image.BILINEAR)))

        self.my_image_number = 0

        # initiate time
        self.time_str = str(datetime.datetime.now().strftime('%d%m%H%M'))

        # set first image on canvas
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor = NW, image = self.my_images[self.my_image_number])
        
        self.time1 = time.time()
        
        # button to change image
        self.button_1 = tk.Button(text="Rock 1",font=('Helvetica',20), command=self.onButton)
        self.button_2 = tk.Button(text="Rock 2",font=('Helvetica',20), command=self.onButton)
        self.button_3 = tk.Button(text="Rock 3",font=('Helvetica',20), command=self.onButton)
        self.button_4 = tk.Button(text="Rock 4",font=('Helvetica',20), command=self.onButton)
        self.button_5 = tk.Button(text="Rock 5",font=('Helvetica',20), command=self.onButton)

        self.button_1.grid(row=1, column=0,sticky = E)
        self.button_2.grid(row=1, column=1,sticky = E)
        self.button_3.grid(row=1, column=2,sticky = E)
        self.button_4.grid(row=1, column=3,sticky = E)
        self.button_5.grid(row=1, column=4,sticky = E)
        
        self.button_1.bind("<Button-1>", self.onClick)
        self.button_2.bind("<Button-1>", self.onClick)
        self.button_3.bind("<Button-1>", self.onClick)
        self.button_4.bind("<Button-1>", self.onClick)
        self.button_5.bind("<Button-1>", self.onClick)
        
        self.count = tk.Label(text=f"1/{df_data.shape[0]}", bg='white',font=('Helvetica',20))
        self.count.grid(row=2,column=0,columnspan=5)

    #----------------
    def onClick(self,event):
        
        btn = event.widget # event.widget is the widget that called the event
        if str(btn) =='.!button' :
            df_data.loc[self.random_list[self.my_image_number],f'{inputs.Name}_{self.time_str}'] = 0
        elif str(btn) =='.!button2' :
            df_data.loc[self.random_list[self.my_image_number],f'{inputs.Name}_{self.time_str}'] = 1 
        elif str(btn) =='.!button3' :
            df_data.loc[self.random_list[self.my_image_number],f'{inputs.Name}_{self.time_str}'] = 2 
        elif str(btn) =='.!button4' :
            df_data.loc[self.random_list[self.my_image_number],f'{inputs.Name}_{self.time_str}'] = 3
        elif str(btn) =='.!button5' :
            df_data.loc[self.random_list[self.my_image_number],f'{inputs.Name}_{self.time_str}'] = 4

        #print(self.random_list[self.my_image_number]) # print corresponding number in the random list

        self.time2 = time.time()
        df_data.loc[self.random_list[self.my_image_number],f'{inputs.Name}_time_{self.time_str}'] = round(self.time2-self.time1,2)
        self.time1 = self.time2
        
    def onButton(self):
        # next image
        self.my_image_number += 1

        # display predictions
        if self.my_image_number == len(self.my_images):

            print('It is finished')
            
            our_model_vec = []
            baseline_vec = []
            
            our_model_vec.append(df_data[df_data['Label'] == df_data['best_model']].groupby('Label')['best_model'].count()/df_data['Label'].value_counts()[0])
            baseline_vec.append(df_data[df_data['Label'] == df_data['baseline_model']].groupby('Label')['baseline_model'].count()/df_data['Label'].value_counts()[0])

            user = df_data[df_data['Label'] == df_data[f'{inputs.Name}_{self.time_str}']].groupby('Label')[f'{inputs.Name}_{self.time_str}'].count()/df_data['Label'].value_counts()[0]
            acc_user = [0,0,0,0,0]
            for i,ind in enumerate(user.index) :
                acc_user[ind] = user.values[i]
                
            n = 5
            fig, ax = plt.subplots()
            DPI = fig.get_dpi()
            fig.set_size_inches(500.0/float(DPI),500.0/float(DPI))
            index = np.arange(n)
            bar_width = 0.25
            ax.bar(index+bar_width, np.array(our_model_vec).mean(axis=0), bar_width, color='g',
                label='Our Model')
            ax.bar(index, np.array(baseline_vec).mean(axis=0), bar_width, color='r',
                label='Baseline')
            ax.bar(index-bar_width, acc_user, bar_width, color='b',
                label='Expert')
            ax.set_xlabel('Rock Type')
            ax.set_ylabel('Accuracy')
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(('Rock 1','Rock 2','Rock 3','Rock 4','Rock 5'))
            ax.legend()

            plt.savefig(f'results_{self.time_str}.png')
            print('Image saved')
            plt.close(fig)
            
            self.results = tk.PhotoImage(file=f'results_{self.time_str}.png')
            self.canvas.itemconfig(self.image_on_canvas, image = self.results)

            df_data.to_csv(inputs.CSVPath)
            print('Data Saved')
            return
            
        # change image
        self.canvas.itemconfig(self.image_on_canvas, image = self.my_images[self.my_image_number])
        # change count 
        self.count.config(text=f'{self.my_image_number+1}/{df_data.shape[0]}',font=('Helvetica',20))   

#----------------------------------------------------------------------

myopts, args = getopt.getopt(sys.argv[1:],"i:o:")

ifile=''
ofile='None'

for o, a in myopts:
    if o == '-i':
        ifile=a
    elif o == '-o':
        ofile=a
    else:
        print("Usage: %s -i input -o output" % sys.argv[0])

inputs = load_config(ifile)

df_data = pd.read_csv(inputs.CSVPath, index_col = 0)
root = tk.Tk()
root.title('Rock Classification App - Nils Olsen')
root.config(bg='white')
root.wm_geometry("550x600")
root.minsize(550,600)

MainWindow(root)
root.mainloop()