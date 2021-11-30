from os import lseek
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from pathlib import Path
import numpy as np
import pandas as pd
import random
import getopt

import sys
sys.path.append('C:/Users/nilso/Documents/EPFL/MA4/PDS Turberg/Rock_Cuttings_Characterisation/')

from Lamp.AttrDict.AttrDict import *
from Lamp.Model.Dataloader import *
from Lamp.Model.BaseModel import *
from Lamp.Model.Resnet import *

# TODO's
# - Add model and params for model

# - Change forward pass to return loss and compute accuracy for train 
# - Adjust validation to be the same as train and adjust loss to match the two
# - Prepare config files for all models to train

def load_config(cfg_path):
    """  """
    if os.path.splitext(cfg_path)[-1] == '.json':
        return AttrDict.from_json_path(cfg_path)
    elif os.path.splitext(cfg_path)[-1] in ['.yaml', '.yml']:
        return AttrDict.from_yaml_path(cfg_path)
    else:
        raise ValueError(f"Unsupported config file format. Only '.json', '.yaml' and '.yml' files are supported.")

def set_seed(seed):
    """ Set the random seed """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def set_parameter_requires_grad(model):
    
    for _, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True

def resnet34(channels=3, num_classes=1000):
    layers=[3, 4, 6, 3]
    model = ResNet(BasicBlock,layers,channels=channels,num_classes=num_classes)
    return model

class Transforms():
    def __init__(self, *args, **kwargs):

        self.transforms_list = args[0]
        self.transforms_list.extend([MinMaxNormalization(),tf.ToTensor()])

    def get_transforms(self):
        transforms = tf.Compose(
                self.transforms_list
                )
        return transforms
    

class Classifier(BaseModelSingle):
    def __init__(self, net: nn.Module, opt: Optimizer = None, sched: _LRScheduler = None, 
        logger: Logger = None, print_progress: bool = True, device: str = 'cuda:0', **kwargs):
        super().__init__(net, opt=opt, sched=sched, logger=logger, print_progress=print_progress, device=device, **kwargs)
        
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")


    def forward_loss(self, data: Tuple[Tensor]) -> Tensor:
        """  """
        input, label = data
        input = input.to(self.device)
        label = label.to(self.device).long()

        output = self.net(input)
        loss = self.loss_fn(output, label)
        return loss

    def validate(self, loader, *args, **kwargs):
        """  """
        valid_loss = 0.0
        pred_label = []
        with torch.no_grad():
            for b, data in enumerate(loader):
                input, label = data
                input = input.to(self.device)
                label = label.to(self.device).long()

                output = self.net(input)
                valid_loss += self.loss_fn(output, label).item()
                pred = torch.argmax(output, dim=1)
                pred_label += list(zip(pred.cpu().data.tolist(), label.cpu().data.tolist()))

        pred, label = zip(*pred_label)
        acc = accuracy_score(np.array(label), np.array(pred))

        return {"Valid Loss": f"{valid_loss:.5f}", "Valid Accuracy": f"{acc:.3f}"}

def main(): 
    
    # Read args from terminal
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

    # Load config_file
    inputs = load_config(ifile)

    # Handle config_file inputs
    n_epochs = int(inputs.NEpochs)
    batch_size = int(inputs.BatchSize)
    k = int(inputs.KFold)
    seed = int(inputs.Seed)
    n_samples = int(inputs.NSamples)

    # Handle file paths
    root_path = os.path.abspath(os.path.join(os.getcwd(), '..')) # Workspace path to Cuttings_Characterisation 
    path_model = f"{root_path}/{inputs.PathSave}/{inputs.ModelName}"

    if os.path.isdir(path_model) is False: # Create new folder for the model
        os.mkdir(path_model)

    path_load_data = f"{root_path}/{inputs.LoadPath}" # Path for the .csv file
    path_checkpoint = f"{path_model}/{inputs.CheckpointName}"
    
    # Seed
    set_seed(seed)

    # Pytorch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(device))

    # Read dataset
    dataframe = pd.read_csv(path_load_data,index_col=0)
    dataframe = dataframe.groupby('Label').sample(n_samples,replace=True,random_state=seed).reset_index(drop=True) # Sample training data from the csv with replacement

    X = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]

    # Stratified KFold
    kf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)

    # Transforms (other than MinMaxNorm and ToTensor)
    dict_transform = {
        "VerticalFlip":tf.RandomVerticalFlip,
        "HorizontalFlip":tf.RandomHorizontalFlip,
        "Rotation":tf.RandomRotation,
        "Resize":tf.Resize,
        "Padding":tf.Pad
    }

    transforms_train = Transforms(
        [dict_transform[key]([k for k in item.values()] if len(item.values()) > 1 else [k for k in item.values()][0]) for key, item in inputs.TransformTrain.items()] 
        )

    transforms_test = Transforms(
        [dict_transform[key]([k for k in item.values()] if len(item.values()) > 1 else [k for k in item.values()][0]) for key, item in inputs.TransformTest.items()] 
        )

    for i_, (train_index, test_index) in enumerate(kf.split(X,y)):

        model_name = f"model_{i_}.pt"
        save_model_path = f"{path_model}/{model_name}"

        trainDataset = Dataset(
            dataframe.loc[train_index,:].reset_index(drop=True),
            transforms=transforms_train.get_transforms()
            )
        testDataset = Dataset(
            dataframe.loc[test_index,:].reset_index(drop=True),
            transforms=transforms_test.get_transforms()
            )

        train_dataloader = torch.utils.data.DataLoader(
            trainDataset, 
            batch_size=batch_size,
            shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(
            testDataset, 
            batch_size=batch_size,
            shuffle=True)

        net = resnet34(channels=1,num_classes=5)

        optimizer = optim.Adam(
            net.parameters(), 
            lr=inputs.Optimizer.lr, 
            weight_decay=inputs.Optimizer.weight_decay
            )

        sched = optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=inputs.Scheduler.gamma
            )

        classifier = Classifier(
            net=net, 
            opt=optimizer, 
            sched=sched, 
            device=device
            )

        classifier.train(
            n_epochs=n_epochs,
            train_loader=train_dataloader,
            valid_loader=test_dataloader,
            checkpoint_path = path_checkpoint,
            checkpoint_freq= inputs.CheckpointFreq
            )

        # Save model weights etc. 
        classifier.save(save_model_path)

        # End of training remove checkpoint file
        if os.path.isfile(path_checkpoint): # Remove checkpoint file at the end of the training
            os.remove(path_checkpoint)

if __name__ == "__main__":
    main()