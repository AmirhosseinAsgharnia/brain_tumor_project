#%% Import Libraries
import os
import numpy
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

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

class CNN_Class(nn.Module):
    def __init__(self, num_of_classes = 4):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(384, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.avepooling = nn.AdaptiveAvgPool2d((7,7))

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 32 , 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_of_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = self.avepooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
number_of_classes = len(train_dataset.classes)
model = CNN_Class (num_of_classes=number_of_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-4, weight_decay=1e-4)

