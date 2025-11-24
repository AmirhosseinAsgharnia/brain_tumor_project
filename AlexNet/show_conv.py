import os

import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from tqdm import tqdm

import cv2
import numpy as np

import matplotlib.pyplot as plt

class CNN_Network(nn.Module):
    
    def __init__(self, num_classes = 4):
        super().__init__()

        self.features = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),

        nn.Conv2d(64, 192, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),  

        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),

        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2),            
        )

    def forward(self, x):
        x = self.features(x)
        return x



BATCH_SIZE = 32
NUM_WORKERS = 0
PIN_MEMORY = False

train_path = os.path.join(".","Training")
test_path  = os.path.join(".","Testing")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

model = CNN_Network()
print("I am here")
img, _ = test_dataset[0]
x = model(img.unsqueeze(0)).squeeze()

x = x.detach().cpu()

num_maps = 16  # plot first 16 channels
plt.figure(figsize=(12, 12))

for i in range(num_maps):
    plt.subplot(4, 4, i+1)
    plt.imshow(x[i], cmap="gray")
    plt.axis("off")

plt.show()