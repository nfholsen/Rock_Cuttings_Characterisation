import pandas as pd
import numpy as np

import pickle 
import os.path
import time
import copy
import PIL.Image as Image
import random

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils import data 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf
from torchvision import models # torchvision for pre-trained models

from sklearn.model_selection import train_test_split

from configparser import ConfigParser

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

def DataLoader(root,train_path,test_path,batch_size,validation_size=0.2,**kwargs):
    df = pd.read_csv(train_path,index_col=0)

    train, val = train_test_split(df,test_size=validation_size,random_state=0,stratify=df['rock_type'])

    cuttings_datasets = {}

    cuttings_datasets['train'] = Cuttings_Dataset(train.reset_index(drop=True),root,
                                                    mean=kwargs.get('mean'),
                                                    std=kwargs.get('std'),
                                                    resize=kwargs.get('resize'),
                                                    data_augmentation=True)
    
    cuttings_datasets['val'] = Cuttings_Dataset(val.reset_index(drop=True),root,
                                                    mean=kwargs.get('mean'),
                                                    std=kwargs.get('std'),
                                                    resize=kwargs.get('resize'),
                                                    data_augmentation=False)
    
    cuttings_datasets['test'] = Cuttings_Dataset(pd.read_csv(test_path,index_col=0),root,
                                                    mean=kwargs.get('mean'),
                                                    std=kwargs.get('std'),
                                                    resize=kwargs.get('resize'),
                                                    data_augmentation=False)

    dataloaders = {x: torch.utils.data.DataLoader(cuttings_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=0)
                    for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(cuttings_datasets[x]) for x in ['train','val','test']}
    return dataloaders, cuttings_datasets,dataset_sizes

def initialize_model(model,num_classes, layers_to_unfreeze, use_pretrained=True):
    
    # By default, when we load a pretrained model all of the parameters have .requires_grad=True
    if model =='resnet34':
        model = models.resnet34(pretrained=use_pretrained)
    if model =='resnet18':
        model = models.resnet18(pretrained=use_pretrained)
    
    if len(layers_to_unfreeze)>0:
        set_parameter_requires_grad(model, layers_to_unfreeze)
        
    num_ftrs = model.fc.in_features
    
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def set_parameter_requires_grad(model,layers_to_unfreeze):
    
    for name, child in model.named_children():
        if name in layers_to_unfreeze:
            print(name + ' is unfrozen')
            for param in child.parameters():
                param.requires_grad = True
        else:
            print(name + ' is frozen')
            for param in child.parameters():
                param.requires_grad = False
                
def train_model(model,dataloaders, cuttings_datasets,dataset_sizes, criterion, optimizer, scheduler, checkpoint, PATH,i, device,num_epochs=25):
    
    # First time training the model
    if checkpoint == None:
        best_model_wts = copy.deepcopy(model.state_dict())
        
        last_epoch = 1
        train_loss = []
        train_accuracy = []
        val_loss = []
        val_accuracy = []
        
        best_acc = 0.0

    # We are already training the model
    else :
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler = checkpoint['scheduler']
        best_model_wts = checkpoint['best_model_state_dict']
        last_epoch = checkpoint['epoch']
        train_loss = checkpoint['loss']
        train_accuracy = checkpoint['accuracy']
        val_loss = checkpoint['val_loss']
        val_accuracy = checkpoint['val_accuracy']
        
        best_acc = max(val_accuracy)
    
    since = time.time()
    
    
    for epoch in range(last_epoch,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        print('LR:', scheduler.get_lr())

        running_loss_train = 0.0
        running_corrects_train = 0
            
        model.train()# Set model to training mode
        
        # Iterate over data Training
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            inputs.require_grad = True
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            preds = torch.argmax(outputs, 1)
            loss = criterion(outputs, labels)

            # backward + optimize
            loss.backward()
            optimizer.step()

            # statistics
            running_loss_train += loss.item() * inputs.size(0)
            running_corrects_train += torch.sum(preds == labels.data)
            
        epoch_loss_train = running_loss_train / dataset_sizes["train"]
        epoch_acc_train = running_corrects_train.double() / dataset_sizes["train"]
        
        train_loss.append(epoch_loss_train)
        train_accuracy.append(epoch_acc_train)
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                "Train", epoch_loss_train, epoch_acc_train))
        
        running_loss_val = 0.0
        running_corrects_val = 0
            
        model.eval() # Set model to evaluate mode
        
        # Iterate over data Evaluation
        for inputs, labels in dataloaders["val"]:
            inputs = inputs.to(device)
            labels = labels.to(device)
                
            with torch.no_grad():
                # forward
                outputs = model(inputs)
                preds = torch.argmax(outputs, 1)
                loss = criterion(outputs, labels)   

            # statistics
            running_loss_val += loss.item() * inputs.size(0)
            running_corrects_val += torch.sum(preds == labels.data)

        epoch_loss_val = running_loss_val / dataset_sizes["val"]
        epoch_acc_val = running_corrects_val.double() / dataset_sizes["val"]
        
        val_loss.append(epoch_loss_val)
        val_accuracy.append(epoch_acc_val)
        
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                "Val", epoch_loss_val, epoch_acc_val))
            
        # deep copy the model
        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            best_model_wts = copy.deepcopy(model.state_dict())
        
        # step scheduler
        scheduler.step()
        
        print()
        torch.save({
            'epoch': epoch+last_epoch,
            'model_state_dict': model.state_dict(),
            'best_model_state_dict': best_model_wts,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'accuracy': train_accuracy,
            'val_loss':val_loss,
            'val_accuracy':val_accuracy,
            'scheduler': scheduler,
            'model_number':i,
            }, PATH)
        print('Model Saved')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss, train_accuracy, val_loss, val_accuracy

def read_config_file(ini_file):
    """
    
    """
    # Initialize the parser
    parser = ConfigParser()
    parser.read(os.path.abspath(ini_file))
    
    return dict(parser.items( "inputs" ))

def prediciton(model,dataloaders,device):
    preds_vec = []
    true_vec = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            preds_vec+=preds.tolist()
            true_vec+=labels.tolist()
    return preds_vec, true_vec

def main(INPUT_PATH = 'inputs.ini'):

    # # Check input file
    if os.path.isfile(INPUT_PATH):
        print ("Config file exists")
        inputs = read_config_file(INPUT_PATH)
    else:
        print ("Config file does not exist")
    
    PATH_model_partial = inputs.get('model_root')+\
            inputs.get('model_name')+\
            inputs.get('path')
    
    if os.path.isfile(PATH_model_partial):
        print ("Checkpoint file exists")
        checkpoint = torch.load(PATH_model_partial)
        print('Restart at epoch :',checkpoint.get('epoch'))
        model_number = int(checkpoint.get('model_number'))
    else:
        print ("Checkpoint file does not exist")
        model_number = 0
        checkpoint = None
    
    # # Assign inputs parameters 
    root = inputs.get('root')
    train_path = root + inputs.get('train_path')
    test_path = root + inputs.get('test_path')
    
    batch_size = int(inputs.get('batch_size'))
    epoch_tot = int(inputs.get('epoch_tot'))
    
    
    if inputs.get('resize') == 'False':
        resize = False
        print(resize)
    elif inputs.get('resize') == 'True':
        resize = True
        print(resize)
    else :
        print('Wrong bool format, resize set to False')
        resize = False
    
    config ={\
            'mean':float(inputs.get('mean')),
            'std':float(inputs.get('std')),
            'resize':resize\
            }
    
    layers_to_unfreeze = list(inputs.get('layers_to_unfreeeze').split(','))
    
    # # Create dataloaders
    dataloaders, cuttings_datasets, dataset_sizes = DataLoader(root,train_path,test_path,batch_size,validation_size=0.2,**config)
    
    print('Dataloader ok !')
    print()
    
    model_type = inputs.get('model')
    
    classes = np.array(['ML', 'MS','BL','GN','OL'])
    
    # Loop start here
    for i in range(model_number,3) : 
        
        # # Create model 
        model = initialize_model(model_type,
                                 len(classes),
                                 layers_to_unfreeze)
        
        print('\nLayers to be trained :')
        # Visualize layers to be trained
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
        
        print()
        # Pass model to gpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print('\nDevice :', device)
        
        print('--------------')
        print('Start training')
        print('--------------')
        print()
        
        # Create loss, optimizer and scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    
        
        # # Train model
        model, loss_train, acc_train, loss_val, acc_val = train_model(model,dataloaders,cuttings_datasets,dataset_sizes,criterion,optimizer,scheduler,checkpoint,PATH_model_partial,i,device,num_epochs=epoch_tot)
        
        preds_vec, true_vec = prediciton(model,dataloaders,device)
        
        # # Save results
        csv_path = inputs.get('model_root')+inputs.get('model_results')

        labels = pd.DataFrame(data=np.array([preds_vec,true_vec]).T,columns=['preds_vec','true_vec'])

        epoch_data = pd.DataFrame(data=np.array([loss_train,acc_train,loss_val,acc_val]).T,columns=['loss_train','acc_train','loss_val','acc_val'])

        labels.to_csv(csv_path+'labels_logs_'+str(i)+'.csv')
        epoch_data.to_csv(csv_path+'epoch_data_logs_'+str(i)+'.csv')
        
        # # Save model for the training i 
        torch.save(model, inputs.get('model_root')+inputs.get('model_results')+'model_'+str(i))
        
        # We reach the end of the training, reset chepoint to None
        checkpoint = None