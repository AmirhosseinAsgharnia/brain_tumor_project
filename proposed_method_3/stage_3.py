# stage_3.py
# Stage 3: train posterior with ELBO using MC likelihood + closed-form KL(q||p).
# Train on ./training_data, evaluate on ./testing_data.

import os
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from posterior_model import PosteriorCNN


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_dataloaders(
    train_root: str = "./training_data",
    test_root: str = "./testing_data",
    img_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
    tfm = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.ImageFolder(root=train_root, transform=tfm)
    test_ds = datasets.ImageFolder(root=test_root, transform=tfm)

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


def mc_cross_entropy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, mc_samples: int) -> torch.Tensor:
    """
    Monte Carlo estimate of expected cross-entropy:
      E_q [ CE(logits, y) ] approx average over samples.
    """
    ce = 0.0
    for _ in range(mc_samples):
        logits = model(x, sample=True)
        ce = ce + nn.functional.cross_entropy(logits, y, reduction="mean")
    return ce / float(mc_samples)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, mc_samples: int = 10) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    running_nll = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # MC predictive probs
        probs = 0.0
        for _ in range(mc_samples):
            logits = model(x, sample=True)
            probs = probs + torch.softmax(logits, dim=1)
        probs = probs / float(mc_samples)

        # NLL
        nll = nn.functional.nll_loss(torch.log(probs + 1e-12), y, reduction="mean")
        running_nll += float(nll.item()) * x.size(0)

        pred = torch.argmax(probs, dim=1)
        correct += int((pred == y).sum().item())
        total += int(x.size(0))

    avg_nll = running_nll / max(total, 1)
    acc = correct / max(total, 1)
    return avg_nll, acc


def train_stage3(
    stage2_ckpt_path: str = "./artifacts/stage2_prior_model_best.pt",
    priors_pt_path: str = "./artifacts/priors_diag_gauss.pt",
    train_root: str = "./training_data",
    test_root: str = "./testing_data",
    artifacts_dir: str = "./artifacts",
    epochs: int = 15,
    batch_size: int = 16,
    lr: float = 1e-3,
    beta: float = 1.0,
    mc_train: int = 5,
    mc_eval: int = 10,
    seed: int = 0,
):
    ensure_dir(artifacts_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load Stage 2 checkpoint (recipe + label mapping + state dict for init)
    s2 = torch.load(stage2_ckpt_path, map_location=device)
    prior_recipe = s2["prior_recipe"]
    idx_to_class_s2 = s2.get("idx_to_class", None)
    s2_state = s2.get("state_dict", None)

    train_loader, test_loader, idx_to_class = get_dataloaders(
        train_root=train_root,
        test_root=test_root,
        img_size=224,
        batch_size=batch_size,
        num_workers=4,
    )

    num_classes = len(idx_to_class)

    # Sanity: class count must match Stage 2
    if idx_to_class_s2 is not None and len(idx_to_class_s2) != num_classes:
        raise RuntimeError(f"Class mismatch: stage2={len(idx_to_class_s2)} vs stage3(train)={num_classes}")

    # Build posterior
    model = PosteriorCNN(
        num_classes=num_classes,
        prior_recipe=prior_recipe,
        priors_pt_path=priors_pt_path,
        init_from_stage2_state=s2_state,
        device=device,
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_train = len(train_loader.dataset)
    best_acc = -1.0
    best_path = os.path.join(artifacts_dir, "stage3_posterior_best.pt")
    last_path = os.path.join(artifacts_dir, "stage3_posterior_last.pt")

    print(f"[stage_3] device={device}")
    print(f"[stage_3] train={n_train} from {train_root}, test={len(test_loader.dataset)} from {test_root}")
    print(f"[stage_3] mc_train={mc_train}, mc_eval={mc_eval}, beta={beta}")
    print(f"[stage_3] loaded recipe from {stage2_ckpt_path}")
    print(f"[stage_3] priors from {priors_pt_path}")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()

        running = 0.0
        running_ce = 0.0
        running_kl = 0.0
        n_seen = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            ce = mc_cross_entropy(model, x, y, mc_samples=mc_train)
            kl = model.kl_loss()

            # ELBO-style objective (minimize negative ELBO)
            loss = ce + beta * kl / float(n_train)

            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running += float(loss.item()) * bs
            running_ce += float(ce.item()) * bs
            running_kl += float(kl.item()) * bs
            n_seen += int(bs)

        train_loss = running / max(n_seen, 1)
        train_ce = running_ce / max(n_seen, 1)
        train_kl = running_kl / max(n_seen, 1)

        test_nll, test_acc = evaluate(model, test_loader, device=device, mc_samples=mc_eval)
        dt = time.time() - t0

        print(
            f"[stage_3][{epoch:02d}/{epochs}] "
            f"train_loss={train_loss:.4f}  ce={train_ce:.4f}  (avg_kl*bs)={train_kl:.2f}  "
            f"test_nll={test_nll:.4f}  test_acc={test_acc:.4f}  time={dt:.1f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "state_dict": model.state_dict(),
                "prior_recipe": prior_recipe,
                "idx_to_class": idx_to_class,
                "stage2_ckpt": stage2_ckpt_path,
                "priors_pt": priors_pt_path,
                "beta": beta,
                "mc_train": mc_train,
                "mc_eval": mc_eval,
                "seed": seed,
            }, best_path)

    torch.save({
        "state_dict": model.state_dict(),
        "prior_recipe": prior_recipe,
        "idx_to_class": idx_to_class,
        "stage2_ckpt": stage2_ckpt_path,
        "priors_pt": priors_pt_path,
        "beta": beta,
        "mc_train": mc_train,
        "mc_eval": mc_eval,
        "seed": seed,
    }, last_path)

    print(f"[stage_3] saved best: {best_path}")
    print(f"[stage_3] saved last: {last_path}")
    print(f"[stage_3] best_acc={best_acc:.4f}")


if __name__ == "__main__":
    train_stage3(
        stage2_ckpt_path="./artifacts/stage2_prior_model_best.pt",
        priors_pt_path="./artifacts/priors_diag_gauss.pt",
        train_root="./training_data",
        test_root="./testing_data",
        artifacts_dir="./artifacts",
        epochs=15,
        batch_size=16,
        lr=1e-3,
        beta=1.0,
        mc_train=5,
        mc_eval=10,
        seed=0,
    )
