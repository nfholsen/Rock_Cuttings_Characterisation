import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

from pathlib import Path
import numpy as np
import pandas as pd
import random, getopt, os, sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

from Lamp.AttrDict.AttrDict import *
from Lamp.Model.Dataloader import *
from Lamp.Model.BaseModel import *
from Lamp.Model.Resnet import *

def set_seed(seed):
    """ Set the random seed """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_config(cfg_path):
    """  """
    if os.path.splitext(cfg_path)[-1] == '.json':
        return AttrDict.from_json_path(cfg_path)
    elif os.path.splitext(cfg_path)[-1] in ['.yaml', '.yml']:
        return AttrDict.from_yaml_path(cfg_path)
    else:
        raise ValueError(f"Unsupported config file format. Only '.json', '.yaml' and '.yml' files are supported.")

def resnet(layers=[3, 4, 6, 3],channels=3, num_classes=1000):
    model = ResNet(BasicBlock,layers,channels=channels,num_classes=num_classes)
    return model

class Classifier(BaseModelSingle):
    def __init__(self, net: nn.Module, opt: Optimizer = None, sched: _LRScheduler = None, 
        logger: Logger = None, print_progress: bool = True, device: str = 'cuda:0', **kwargs):
        super().__init__(net, opt=opt, sched=sched, logger=logger, print_progress=print_progress, device=device, **kwargs)

    def forward_loss(self, data: Tuple[Tensor]) -> Tensor:
        """  """
        pass

    def predict(self, loader):
        """  """
        labels = []
        self.net.eval()
        with torch.no_grad():
            for b, data in enumerate(loader):
                input, label = data
                input = input.to(self.device)
                label = label.to(self.device).long()

                output = self.net(input)
                pred = torch.argmax(output, dim=1)

                labels += list(zip(pred.cpu().data.tolist(), label.cpu().data.tolist()))

            pred, label = zip(*labels)
        #acc = accuracy_score(np.array(label), np.array(pred))

        return pred, label

# Load config_file
ifile = '../Models/config/MAR_RESNET34_PADDED_256_ALL.yaml'

inputs = load_config(ifile)

# Handle config_file inputs
k = int(inputs.KFold)
seed = int(inputs.Seed)
n_samples = int(inputs.NSamples)
layers = inputs.Model.Layers # [3, 4, 6, 3] for ResNet34 and [2, 2, 2, 2] for ResNet18
classes = inputs.Model.OutClasses
channels = inputs.Model.Channels
batch_size = inputs.BatchSize

# Handle file paths
root_path = os.path.abspath(os.path.join(os.getcwd(), '..')) # Workspace path to Cuttings_Characterisation 
path_model = f"{root_path}/{inputs.PathSave}/{inputs.ModelName}"
path_load_data = f"{root_path}/{inputs.LoadPath}" # Path for the .csv file

# Seed
set_seed(seed)

# Pytorch device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(device))

# Read dataset
dataframe = pd.read_csv(path_load_data,index_col=0)

# Train Test Split
train_dataframe, _ = train_test_split(dataframe, test_size=(1 - inputs.TrainTestSplit),stratify=dataframe['Label'], random_state=inputs.Seed)

# Reset Index
train_dataframe = train_dataframe.reset_index(drop=True)

# Samples
train_dataframe = train_dataframe.groupby('Label').sample(n_samples,replace=True,random_state=inputs.Seed).reset_index(drop=True)

X = train_dataframe.iloc[:,:-1]
y = train_dataframe.iloc[:,-1]

# Stratified KFold
kf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)

# Transforms (other than MinMaxNorm and ToTensor)
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

accuracy_scores = []
for i_, (train_index, test_index) in enumerate(kf.split(X,y)):

    model_name = f"model_{i_}.pt"
    save_model_path = f"{path_model}/{model_name}"

    print('Model :',save_model_path)

    testDataset = Dataset(
        train_dataframe.loc[test_index,:].reset_index(drop=True),
        transforms=transforms_test.get_transforms()
        )

    test_dataloader = torch.utils.data.DataLoader(
        testDataset, 
        shuffle=True
        )

    net = resnet(layers=layers,channels=channels,num_classes=classes)

    classifier = Classifier(
                net=net, 
                device=device
                )

    classifier.load(save_model_path)

    pred, label = classifier.predict(test_dataloader)

    accuracy_scores.append(accuracy_score(np.array(label), np.array(pred)))

print(accuracy_scores)