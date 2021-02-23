from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import datetime

class MainWindow():

    #----------------

    def __init__(self, main):

        # canvas for image
        self.canvas = tk.Canvas(main, width=500, height=500,bg='white')
        self.canvas.grid(row=0, column=0,columnspan = 5, rowspan = 1)

        # images
        self.my_images = []
        self.random_list = random.sample(range(0,df_data.shape[0]), df_data.shape[0])

        print(self.random_list)
        for im in self.random_list:
            self.my_images.append(tk.PhotoImage(file = PATH_to_repo + 'Rock_Cuttings_Characterisation/Data/'+ df_data.loc[im,'path'][1:]).zoom(4))
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
            df_data.loc[self.random_list[self.my_image_number],'pred_app_'+self.time_str] = 0
        elif str(btn) =='.!button2' :
            df_data.loc[self.random_list[self.my_image_number],'pred_app_'+self.time_str] = 1 
        elif str(btn) =='.!button3' :
            df_data.loc[self.random_list[self.my_image_number],'pred_app_'+self.time_str] = 2 
        elif str(btn) =='.!button4' :
            df_data.loc[self.random_list[self.my_image_number],'pred_app_'+self.time_str] = 3
        elif str(btn) =='.!button5' :
            df_data.loc[self.random_list[self.my_image_number],'pred_app_'+self.time_str] = 4

        #print(self.random_list[self.my_image_number]) # print corresponding number in the random list

        self.time2 = time.time()
        df_data.loc[self.random_list[self.my_image_number],'time_app_'+self.time_str] = round(self.time2-self.time1,2)
        self.time1 = self.time2
        
    def onButton(self):
        # next image
        self.my_image_number += 1

        # display predictions
        if self.my_image_number == len(self.my_images):

            print('It is finished')
            
            our_model_vec = []
            baseline_vec = []
            for i in range(5):
                our_model_vec.append(df_data[df_data['rock_type'] == df_data['best_model_'+str(i)]].groupby('rock_type')['best_model_'+str(i)].count()/df_data['rock_type'].value_counts()[0])
                baseline_vec.append(df_data[df_data['rock_type'] == df_data['baseline_'+str(i)]].groupby('rock_type')['baseline_'+str(i)].count()/df_data['rock_type'].value_counts()[0])

            user = df_data[df_data['rock_type'] == df_data['pred_app_'+self.time_str]].groupby('rock_type')['pred_app_'+self.time_str].count()/df_data['rock_type'].value_counts()[0]
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

            plt.savefig(PATH + 'Application/results_'+self.time_str+'.png')
            print('Image saved')
            plt.close(fig)
            
            self.results = tk.PhotoImage(file=PATH + 'Application/results_'+self.time_str+'.png')
            self.canvas.itemconfig(self.image_on_canvas, image = self.results)

            df_data.to_csv(PATH + 'comparison_data.csv')
            print('Data Saved')
            return
            
        # change image
        self.canvas.itemconfig(self.image_on_canvas, image = self.my_images[self.my_image_number])
        # change count 
        self.count.config(text=f'{self.my_image_number+1}/{df_data.shape[0]}',font=('Helvetica',20))   

#----------------------------------------------------------------------

PATH_to_repo = "C:/Users/nilso/Documents/EPFL/MA4/PDS Turberg/"
PATH = PATH_to_repo + 'Rock_Cuttings_Characterisation/PostProcessing/Model_Comparison/'
df_data = pd.read_csv(PATH+'comparison_data.csv', index_col = 0)
root = tk.Tk()
root.title('Rock Classification App - Nils Olsen')
root.config(bg='white')
root.wm_geometry("550x600")
root.minsize(550,600)

MainWindow(root)
root.mainloop()