# stage_2.py
# Train the PRIOR (deterministic) model using prior_data/ and save weights.
#
# Expected folders:
#   prior_data/<class_name>/*.png|*.jpg|...
#
# Output file:
#   prior_weights.pth
#
# Run:
#   python stage_2.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Change this import to match your project
from prior_model import PriorNet


# --------------------
# Simple settings
# --------------------
PRIOR_DIR = "prior_data"
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 2
SAVE_PATH = "prior_weights.pth"
IMAGE_SIZE = 224  # keep consistent with your pipeline


def main():
    if not os.path.isdir(PRIOR_DIR):
        raise RuntimeError(f"Can't find '{PRIOR_DIR}' folder.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1-channel input (MRI grayscale)
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        # Simple normalization for 1-channel images
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = datasets.ImageFolder(PRIOR_DIR, transform=tfm)
    if len(dataset) == 0:
        raise RuntimeError(f"No images found inside '{PRIOR_DIR}' (check your folders).")

    num_classes = len(dataset.classes)
    print("Classes:", dataset.classes)
    print("Num classes:", num_classes)
    print("Total prior samples:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    # Build prior model
    model = PriorNet(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        acc = correct / total
        print(f"Epoch {epoch:02d}/{EPOCHS}  loss={avg_loss:.4f}  acc={acc:.4f}")

    # Save weights
    torch.save({
        "state_dict": model.state_dict(),
        "classes": dataset.classes,   # helps stage_3 sanity-check class ordering
        "image_size": IMAGE_SIZE,
    }, SAVE_PATH)

    print(f"\nSaved prior weights to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
