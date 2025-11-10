import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

class BrainTumorCNN(nn.Module):
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

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# === Config ===
MODEL_PATH = "best_brain_tumor_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your class labels — must match training order
CLASSES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

# === Load model ===
model = BrainTumorCNN(num_classes=len(CLASSES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === Transform (must match test transforms from training) ===
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)  # [1, 1, 224, 224]
    with torch.no_grad():
        output = model(tensor)
        _, pred = torch.max(output, 1)
        predicted_class = CLASSES[pred.item()]
    return predicted_class

# # === Example usage ===
def predict_from_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith((".jpg", ".png", ".jpeg")):
            img_path = os.path.join(folder_path, file)
            result = predict_image(img_path)
            print(f"{file:30s} → {result}")

if __name__ == "__main__":
    # change this to your test folder
    IMG_DIR = "Testing/no_tumor"
    predict_from_folder(IMG_DIR)
