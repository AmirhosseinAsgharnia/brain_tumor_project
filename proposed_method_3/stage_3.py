# stage_3.py
# Train the POSTERIOR Bayesian CNN using ELBO:
#   loss = CE_MC + beta * KL / N_train
#
# Uses:
#   training_data/  (the remaining 2/3 after stage_1)
#   testing_data/   (evaluation only)
#   prior_weights.pth (from stage_2) only to sanity-check class order
#
# Run:
#   python stage_3.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from posterior_model import PosteriorNet


TRAIN_DIR = "training_data"
TEST_DIR = "testing_data"
PRIOR_CKPT = "prior_weights.pth"

BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-3
NUM_WORKERS = 2

IMAGE_SIZE = 224
MC_SAMPLES = 2        # 1-4 typical
BETA = 1.0            # KL weight (you can anneal later if you want)
PRIOR_MU = 0.0
PRIOR_SIGMA = 0.1     # simple fixed prior scale

SAVE_PATH = "posterior_weights.pth"


@torch.no_grad()
def evaluate(model, loader, device, mc_samples=5):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # MC average predictive logits
        logits_acc = None
        for _ in range(mc_samples):
            logits = model(images, sample=True)
            logits_acc = logits if logits_acc is None else (logits_acc + logits)
        logits_mean = logits_acc / float(mc_samples)

        loss = nn.functional.cross_entropy(logits_mean, labels)

        preds = torch.argmax(logits_mean, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        loss_sum += loss.item() * labels.size(0)

    return loss_sum / total, correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not os.path.isdir(TRAIN_DIR):
        raise RuntimeError(f"Can't find '{TRAIN_DIR}' folder.")
    if not os.path.isdir(TEST_DIR):
        raise RuntimeError(f"Can't find '{TEST_DIR}' folder.")
    if not os.path.isfile(PRIOR_CKPT):
        print(f"Warning: '{PRIOR_CKPT}' not found. Will proceed without prior class-order sanity check.")

    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_set = datasets.ImageFolder(TRAIN_DIR, transform=tfm)
    test_set = datasets.ImageFolder(TEST_DIR, transform=tfm)

    if len(train_set) == 0:
        raise RuntimeError(f"No images found inside '{TRAIN_DIR}'.")
    if len(test_set) == 0:
        raise RuntimeError(f"No images found inside '{TEST_DIR}'.")

    print("Train classes:", train_set.classes)
    print("Test classes: ", test_set.classes)

    # Sanity check class ordering vs stage_2
    if os.path.isfile(PRIOR_CKPT):
        ckpt = torch.load(PRIOR_CKPT, map_location="cpu")
        prior_classes = ckpt.get("classes", None)
        if prior_classes is not None and prior_classes != train_set.classes:
            raise RuntimeError(
                "Class ordering mismatch!\n"
                f"prior_weights classes: {prior_classes}\n"
                f"training_data classes: {train_set.classes}\n"
                "Fix your folder names or splits so ImageFolder produces identical class order."
            )
        print("Class-order sanity check: OK")

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    num_classes = len(train_set.classes)

    model = PosteriorNet(num_classes=num_classes, prior_mu=PRIOR_MU, prior_sigma=PRIOR_SIGMA).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    n_train = len(train_set)

    print("\nTraining posterior with ELBO:")
    print(f"  MC_SAMPLES={MC_SAMPLES}, BETA={BETA}, PRIOR_SIGMA={PRIOR_SIGMA}")
    print(f"  N_train={n_train}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_kl = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # MC estimate of expected NLL (cross-entropy)
            ce_acc = 0.0
            logits_mean = None
            for _ in range(MC_SAMPLES):
                logits = model(images, sample=True)
                ce_acc = ce_acc + nn.functional.cross_entropy(logits, labels)
                logits_mean = logits if logits_mean is None else (logits_mean + logits)

            ce = ce_acc / float(MC_SAMPLES)
            logits_mean = logits_mean / float(MC_SAMPLES)

            # KL term (analytic, summed over all Bayesian params)
            kl = model.kl_loss()

            # ELBO loss: CE + beta * KL/N
            loss = ce + (BETA * kl / float(n_train))

            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            total += bs
            total_loss += loss.item() * bs
            total_ce += ce.item() * bs
            total_kl += kl.item() * bs  # scaled for reporting only

            preds = torch.argmax(logits_mean, dim=1)
            correct += (preds == labels).sum().item()

        train_loss = total_loss / total
        train_ce = total_ce / total
        train_acc = correct / total
        # report average KL per sample (raw sum/bs is fine as a number to watch)
        avg_kl_per_sample = total_kl / total

        test_loss, test_acc = evaluate(model, test_loader, device, mc_samples=5)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f}  CE={train_ce:.4f}  KL(sum)~{avg_kl_per_sample:.1f}  acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}"
        )

    # Save posterior parameters
    torch.save({
        "state_dict": model.state_dict(),
        "classes": train_set.classes,
        "image_size": IMAGE_SIZE,
        "prior_mu": PRIOR_MU,
        "prior_sigma": PRIOR_SIGMA,
        "mc_samples_train": MC_SAMPLES,
        "beta": BETA,
    }, SAVE_PATH)

    print(f"\nSaved posterior weights to: {SAVE_PATH}")


if __name__ == "__main__":
    main()
