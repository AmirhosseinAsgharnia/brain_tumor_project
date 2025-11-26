#%% Import Libraries
import os
import numpy
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

#%% NN Class
class CNN_Class(nn.Module):
    def __init__(self, num_of_classes = 4):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, groups=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        return x
    
model = CNN_Class()