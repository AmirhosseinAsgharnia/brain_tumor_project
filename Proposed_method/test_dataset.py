#%% Import Libraries
import os
import numpy
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

#%% Path
abspath = os.path.abspath(__file__)
dname   = os.path.dirname(abspath)
os.chdir(dname)

# training_path = "./training_phase_1"
training_path = "./Image_testing"
#%% Transforms

train_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

#%% Data loader

train_dataset = datasets.ImageFolder(root=training_path, transform=train_transform)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 6, figsize=(18, 3))

for i in range(6):
    img_vis, label = train_dataset[i]
    img_vis = img_vis * 0.5 + 0.5   # bring [-1,1] → [0,1]
    # plt.imshow(img_vis.permute(1,2,0))
    axes[i].imshow(img_vis.permute(1, 2, 0))
    axes[i].axis("off")
    axes[i].set_title(f"Label: {label}")

plt.tight_layout()
plt.show()