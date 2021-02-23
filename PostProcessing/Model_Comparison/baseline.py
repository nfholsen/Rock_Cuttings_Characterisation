# PyTorch libraries and modules
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from spp_layer import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 48, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(48, 32, 3)

        self.spp = SpatialPyramidPooling()

        self.fc1 = nn.Linear(32*21, 200)
        self.drop1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(200, 5)

    def forward(self, x):

        x = self.pool1(F.relu(self.conv1(x)))

        x = self.pool2(F.relu(self.conv2(x)))

        x = self.spp(F.relu(self.conv3(x)))

        x = x.view(-1,21*32)

        x = self.drop1(F.relu(self.fc1(x)))

        x = F.softmax(self.fc2(x))
    
        return x