import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

PATH_to_repo =  "C:/Users/nilso/Documents/EPFL/MA4/PDS Turberg/"
PATH_data = PATH_to_repo + 'Rock_Cuttings_Characterisation/Data/'
PATH_save = PATH_to_repo + 'Rock_Cuttings_Characterisation/PostProcessing/Model_Comparison/Application/'

df_train_csv = pd.read_csv(PATH_data + 'train/csv_resized_128/train_resized_128_final.csv',index_col=0)

list_im_train = []
random.seed(0)
for i in range(5):
    list_im_train+=random.sample(list(df_train_csv[df_train_csv['rock_type'] ==i].index.values),k = 7)

im_to_load = df_train_csv.loc[list_im_train].sort_index().reset_index(drop=True)[['path','rock_type']]

fig, axs = plt.subplots(7,5,figsize=(7,10))

for col,_ in enumerate(axs[0]):
    for row,_ in enumerate(axs[:,0]):
        axs[row,col].imshow(cv2.imread(PATH_data + im_to_load['path'][col*len(axs[:,0])+row][1:]),cmap='gray')
        axs[row,col].set_title("Rock {}".format(im_to_load['rock_type'][col*len(axs[:,0])+row]+1))
        axs[row,col].axis(False)
plt.tight_layout()
plt.savefig(PATH_save+'examples_application.png',dpi=600)