from os import lseek
import torch
import torch.nn as nn
from torchvision.models import resnet34
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

    n_epochs = int(inputs.NEpochs)
    batch_size = int(inputs.BatchSize)
    k = int(inputs.KFold)
    path_save = f"{inputs.PathSave}"
    seed = int(inputs.Seed)
    
    set_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(device))

    dataframe = pd.read_csv(inputs.LoadPath,index_col=0)
    dataframe = dataframe.groupby('Label').sample(1000,replace=True,random_state=seed).reset_index(drop=True)

    X = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]

    kf = StratifiedKFold(n_splits=k, random_state=seed)

    transforms = tf.Compose(
        [     
            tf.Grayscale(num_output_channels=3),  
            #tf.ColorJitter(brightness=0.5),
            tf.RandomHorizontalFlip(p=0.5),
            tf.RandomVerticalFlip(p=0.5),
            tf.RandomRotation([-90,90],resample=False, expand=False, center=None, fill=None),
            tf.Resize((256,256)),
            MinMaxNormalization(),
            tf.ToTensor(),
        ])

    for i_, (train_index, test_index) in enumerate(kf.split(X,y)):

        trainDataset = Dataset(
            dataframe.loc[train_index,:].reset_index(drop=True),
            transforms=transforms
            )
        testDataset = Dataset(
            dataframe.loc[test_index,:].reset_index(drop=True),
            transforms=transforms
            )

        train_dataloader = torch.utils.data.DataLoader(
            trainDataset, 
            batch_size=batch_size,
            shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(
            testDataset, 
            batch_size=batch_size,
            shuffle=True)

        net = resnet34(pretrained=True)

        set_parameter_requires_grad(net)
        
        optimizer = optim.Adam(
            net.parameters(), 
            lr=0.001, 
            weight_decay=1e-6
            )

        sched = optim.lr_scheduler.ExponentialLR(
            optimizer, 
            gamma=0.95
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
            checkpoint_path = inputs.CheckpointPath
            )

        if not os.path.isdir(os.path.dirname(path_save)):
            path = os.path.dirname(path_save)
            path.mkdir(parents=True)

        classifier.save(f"{path_save}_{i_}.pt")

if __name__ == "__main__":
    main()