# stage_2.py
# Stage 2: sample ONE prior CNN (frozen convs), train ONLY the FC layer on prior_data.
# Evaluate on testing_data, save model + recipe + class mapping.

import os
import json
import time
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from prior_model import PriorCNN


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_dataloaders(
    prior_root: str = "./prior_data",
    test_root: str = "./testing_data",
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        # Keep it simple; you can add normalization later if you want.
        # transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_ds = datasets.ImageFolder(root=prior_root, transform=tfm)
    test_ds = datasets.ImageFolder(root=test_root, transform=tfm)

    # class_to_idx is identical ordering for both if folder names match
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )

    return train_loader, test_loader, idx_to_class


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    ce = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        running_loss += float(loss.item()) * x.size(0)

        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_stage2(
    prior_root: str = "./prior_data",
    test_root: str = "./testing_data",
    artifacts_dir: str = "./artifacts",
    prior_pt_path: str = "./artifacts/priors_diag_gauss.pt",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 0,
):
    ensure_dir(artifacts_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_loader, test_loader, idx_to_class = get_dataloaders(
        prior_root=prior_root,
        test_root=test_root,
        img_size=224,
        batch_size=batch_size,
        num_workers=4,
    )

    num_classes = len(idx_to_class)

    # Build ONE sampled prior CNN. Convs are frozen. Only FC trains.
    model = PriorCNN(
        num_classes=num_classes,
        prior_pt_path=prior_pt_path,
        layer1_prior="deriv1_k7",
        layer2_prior="deriv2_k5",
        layer3_prior="gabor_k3",
        layer4_prior="gabor_k3",
        device=device,
        img_size=224,
        final_spatial=28,
        channels=60,
    )

    # Freeze everything except FC (paranoia check)
    for name, p in model.named_parameters():
        if name.startswith("fc."):
            p.requires_grad = True
        else:
            p.requires_grad = False

    optimizer = optim.Adam(model.trainable_parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_path = os.path.join(artifacts_dir, "stage2_prior_model_best.pt")
    last_path = os.path.join(artifacts_dir, "stage2_prior_model_last.pt")

    print(f"[stage_2] device={device}, classes={num_classes}, train={len(train_loader.dataset)}, test={len(test_loader.dataset)}")
    print(f"[stage_2] training on prior_root={prior_root}")
    print(f"[stage_2] priors loaded from {prior_pt_path}")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * x.size(0)
            n += int(x.size(0))

        train_loss = running / max(n, 1)
        test_loss, test_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0

        print(f"[stage_2][{epoch:02d}/{epochs}] train_loss={train_loss:.4f}  test_loss={test_loss:.4f}  test_acc={test_acc:.4f}  time={dt:.1f}s")

        # Save best
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "state_dict": model.state_dict(),
                "prior_recipe": model.prior_recipe,
                "idx_to_class": idx_to_class,
                "seed": seed,
            }, best_path)

    # Save last
    torch.save({
        "state_dict": model.state_dict(),
        "prior_recipe": model.prior_recipe,
        "idx_to_class": idx_to_class,
        "seed": seed,
    }, last_path)

    # Also dump recipe as readable text (optional, handy)
    recipe_txt = os.path.join(artifacts_dir, "stage2_prior_recipe.json")
    with open(recipe_txt, "w", encoding="utf-8") as f:
        json.dump({
            "prior_recipe": model.prior_recipe,
            "idx_to_class": idx_to_class,
            "best_acc": best_acc,
        }, f, indent=2)

    print(f"[stage_2] saved best: {best_path}")
    print(f"[stage_2] saved last: {last_path}")
    print(f"[stage_2] saved recipe: {recipe_txt}")
    print(f"[stage_2] best_acc={best_acc:.4f}")


if __name__ == "__main__":
    train_stage2(
        prior_root="./prior_data",
        test_root="./testing_data",
        artifacts_dir="./artifacts",
        prior_pt_path="./artifacts/priors_diag_gauss.pt",
        epochs=10,
        batch_size=32,
        lr=1e-3,
        seed=0,
    )
