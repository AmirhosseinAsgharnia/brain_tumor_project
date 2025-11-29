#%% Import Libraries
import os
import numpy
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
import pyro
from pyro.nn import PyroModule, PyroSample 
import pyro.distributions as dist
#%% Hyper parameters setting

BATCH_SIZE = 32
NUM_WORKERS= 0
PIN_MEMORY = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("This PC does not have NVIDIA GPU, switch to CPU.")
else:
    print("This PC have NVIDIA GPU and it is used for training.")
#%% Path
abspath = os.path.abspath(__file__)
dname   = os.path.dirname(abspath)
os.chdir(dname)

training_path = "./training_phase_1"
testing_path = "./original_dataset/Testing"
#%% Transforms

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

#%% Data loader

train_dataset = datasets.ImageFolder(training_path, transform=train_transform)
test_dataset  = datasets.ImageFolder(testing_path, transform=test_transform)

print("Classes:", train_dataset.classes)
print("Class to index mapping:", train_dataset.class_to_idx)

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

#%% NN Class

class CNN_Class(PyroModule):
    def __init__(self, num_of_classes = 4):
        super().__init__()

        self.conv_11 = PyroModule[nn.Conv2d](1, 1, kernel_size=3, stride=2, padding=1, groups=1 ) #type: ignore
        self.conv_12 = PyroModule[nn.Conv2d](1, 32, kernel_size=3, stride=1, padding=0, groups=1 ) #type: ignore

        self.conv_11.weights =  PyroSample(dist.Normal(0., prior_scale).expand([hid_dim, in_dim]).to_event(2))

        self.conv_21 = PyroModule[nn.Conv2d](32, 32, kernel_size=3, stride=1, padding=1, groups=32 ) #type: ignore
        self.conv_22 = PyroModule[nn.Conv2d](32, 32, kernel_size=1, stride=1, padding=0, groups=1 ) #type: ignore

        self.conv_31 = PyroModule[nn.Conv2d](32, 32, kernel_size=3, stride=1, padding=1, groups=32 ) #type: ignore
        self.conv_32 = PyroModule[nn.Conv2d](32, 64, kernel_size=1, stride=1, padding=0, groups=1 ) #type: ignore
 
        self.conv_41 = PyroModule[nn.Conv2d](64, 64, kernel_size=3, stride=2, padding=1, groups=64 ) #type: ignore
        self.conv_42 = PyroModule[nn.Conv2d](64, 128, kernel_size=1, stride=1, padding=0, groups=1 ) #type: ignore

        self.conv_51 = PyroModule[nn.Conv2d](128, 128, kernel_size=3, stride=2, padding=1, groups=128 ) #type: ignore
        self.conv_52 = PyroModule[nn.Conv2d](128, 256, kernel_size=1, stride=1, padding=0, groups=1 ) #type: ignore

        self.conv_61 = PyroModule[nn.Conv2d](256, 256, kernel_size=3, stride=2, padding=1, groups=256 ) #type: ignore
        self.conv_62 = PyroModule[nn.Conv2d](256, 512, kernel_size=1, stride=1, padding=0, groups=1 ) #type: ignore

        self.conv_71 = PyroModule[nn.Conv2d](512, 512, kernel_size=3, stride=2, padding=1, groups=512 ) #type: ignore
        self.conv_72 = PyroModule[nn.Conv2d](512, 1024, kernel_size=1, stride=1, padding=0, groups=1 ) #type: ignore

        self.avepooling = nn.AdaptiveAvgPool2d((7, 7))

        self.FC_1 = PyroModule[nn.Linear](7 * 7 * 1024 , 1024) #type: ignore
        self.FC_2 = PyroModule[nn.Linear](1024 , 512) #type: ignore
        self.FC_3 = PyroModule[nn.Linear](512 , num_of_classes) #type: ignore

    def forward(self, x):

        x = self.conv_11(x)
        x = self.conv_12(x)
        x = torch.relu(x)

        x = self.conv_21(x)
        x = self.conv_22(x) 
        x = torch.relu(x)

        x = self.conv_31(x)
        x = self.conv_32(x)
        x = torch.relu(x)

        x = self.conv_41(x)
        x = self.conv_42(x)
        x = torch.relu(x)

        x = self.conv_51(x)
        x = self.conv_52(x)
        x = torch.relu(x)

        x = self.conv_61(x)
        x = self.conv_62(x)
        x = torch.relu(x)

        x = self.conv_71(x)
        x = self.conv_72(x)
        x = torch.relu(x)

        x = self.avepooling(x)
        x = torch.flatten(x, 1)

        x = self.FC_1(x)
        x = torch.relu(x)

        x = self.FC_2(x)
        x = torch.relu(x)

        x = self.FC_3(x)

        return x

model = CNN_Class(num_of_classes=4)