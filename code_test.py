import os

import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm

import cv2
import numpy as np

# %%

batch_size = 32

train_path = os.path.join(".","Training")
test_path  = os.path.join(".","Testing")

device = ["cuda" if torch.cuda.is_available() else "cpu"]
print(f"In this training {device} is used to train the model.")

train_transforms = transforms.Compose ([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5,std=0.5)
])

test_transforms = transforms.Compose ([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5,std=0.5)
])

train_dataset = datasets.ImageFolder(root=train_path , transform=train_transforms)
test_dataset  = datasets.ImageFolder(root=test_path, transform=test_transforms)

# %%