#%% Import Libraries
import os
import numpy
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
import tqdm
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
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=0.5, std=0.5)
])

test_transform = transforms.Compose([
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
            nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, groups=3 ),
            nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0 ),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32 ),
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0 ),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32 ),
            nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0 ),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64 ),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0 ),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128 ),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0 ),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256 ),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0 ),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512 ),
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0 ),
            nn.ReLU(inplace=True),
        )

        self.avepooling = nn.AdaptiveAvgPool2d((7,7))

        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 1024 , 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
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

EPOCHS = 20

def train_one_epoch(model, dataloader, criterion, optimizer):
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

best_val_acc = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = validate_one_epoch(model, test_loader, criterion)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # Save best model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_brain_tumor_model.pth")
        print("Saved new best model")

print("\nTraining complete. Best validation accuracy:", best_val_acc)