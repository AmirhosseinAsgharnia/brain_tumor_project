import os
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    train_dir: str = "training_data"
    test_dir: str = "testing_data"

    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 1

    epochs: int = 20
    lr: float = 1e-4
    weight_decay: float = 0.0

    # Bayesian bits
    prior_sigma: float = 1.0          # Gaussian prior std
    beta_kl: float = 1.0              # scale for KL term (you can anneal it)
    mc_train_samples: int = 1         # typically 1 is fine for training
    mc_eval_samples: int = 20         # more samples -> better uncertainty estimate

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def count_params(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# -----------------------------
# Bayesian layers (mean-field Gaussian)
# q(w) = N(mu, sigma^2), sigma = softplus(rho)
# Prior p(w) = N(0, prior_sigma^2)
# KL has closed form for Gaussians.
# -----------------------------
class BayesianParameter(nn.Module):
    def __init__(self, shape, init_mu_std=0.02, init_rho=-5.0):
        super().__init__()
        self.mu = nn.Parameter(torch.empty(shape).normal_(0.0, init_mu_std))
        self.rho = nn.Parameter(torch.empty(shape).fill_(init_rho))

    def sigma(self):
        # softplus to ensure positivity
        return F.softplus(self.rho)

    def sample(self):
        eps = torch.randn_like(self.mu)
        return self.mu + self.sigma() * eps

    def kl_to_standard_normal(self, prior_sigma: float):
        # KL( N(mu, sigma^2) || N(0, prior_sigma^2) )
        # = log(prior_sigma/sigma) + (sigma^2 + mu^2)/(2 prior_sigma^2) - 1/2
        sigma_q = self.sigma()
        prior_var = prior_sigma ** 2
        kl = torch.log(prior_sigma / sigma_q) + (sigma_q**2 + self.mu**2) / (2.0 * prior_var) - 0.5
        return kl.sum()


class BayesConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, groups=1, bias=True, prior_sigma=1.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.use_bias = bias
        self.prior_sigma = prior_sigma

        w_shape = (out_ch, in_ch // groups, self.kernel_size[0], self.kernel_size[1])
        self.w = BayesianParameter(w_shape)

        if self.use_bias:
            self.b = BayesianParameter((out_ch,))
        else:
            self.b = None

    def forward(self, x, sample=True):
        w = self.w.sample() if sample else self.w.mu
        b = self.b.sample() if (self.use_bias and sample) else (self.b.mu if self.use_bias else None)
        return F.conv2d(x, w, b, stride=self.stride, padding=self.padding, groups=self.groups)

    def kl(self):
        kl = self.w.kl_to_standard_normal(self.prior_sigma)
        if self.use_bias:
            kl = kl + self.b.kl_to_standard_normal(self.prior_sigma)
        return kl


class BayesLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.prior_sigma = prior_sigma

        self.w = BayesianParameter((out_features, in_features))
        if self.use_bias:
            self.b = BayesianParameter((out_features,))
        else:
            self.b = None

    def forward(self, x, sample=True):
        w = self.w.sample() if sample else self.w.mu
        b = self.b.sample() if (self.use_bias and sample) else (self.b.mu if self.use_bias else None)
        return F.linear(x, w, b)

    def kl(self):
        kl = self.w.kl_to_standard_normal(self.prior_sigma)
        if self.use_bias:
            kl = kl + self.b.kl_to_standard_normal(self.prior_sigma)
        return kl


# -----------------------------
# Bayesian Depth-wise Separable Block
# depthwise: groups=in_ch
# pointwise: 1x1 conv
# -----------------------------
class BayesDWSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, prior_sigma=1.0):
        super().__init__()
        self.dw = BayesConv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False, prior_sigma=prior_sigma
        )
        self.pw = BayesConv2d(
            in_ch, out_ch, kernel_size=1, stride=1, padding=0, groups=1, bias=False, prior_sigma=prior_sigma
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, sample=True):
        x = self.dw(x, sample=sample)
        x = self.pw(x, sample=sample)
        x = self.bn(x)
        x = self.act(x)
        return x

    def kl(self):
        return self.dw.kl() + self.pw.kl()


# -----------------------------
# Model
# -----------------------------
class BayesianDWNet(nn.Module):
    def __init__(self, num_classes: int, prior_sigma: float = 1.0):
        super().__init__()
        self.prior_sigma = prior_sigma

        # A lightweight stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Bayes depth-wise separable stack
        self.blocks = nn.ModuleList([
            BayesDWSeparable(32, 64, stride=1, prior_sigma=prior_sigma),
            BayesDWSeparable(64, 128, stride=2, prior_sigma=prior_sigma),
            BayesDWSeparable(128, 128, stride=1, prior_sigma=prior_sigma),
            BayesDWSeparable(128, 256, stride=2, prior_sigma=prior_sigma),
            BayesDWSeparable(256, 256, stride=1, prior_sigma=prior_sigma),
            BayesDWSeparable(256, 512, stride=2, prior_sigma=prior_sigma),
        ])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.5)

        self.fc = BayesLinear(512, num_classes, bias=True, prior_sigma=prior_sigma)

    def forward(self, x, sample=True):
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x, sample=sample)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        logits = self.fc(x, sample=sample)
        return logits

    def kl(self):
        total = 0.0
        for blk in self.blocks:
            total = total + blk.kl()
        total = total + self.fc.kl()
        return total


# -----------------------------
# Data
# -----------------------------
def make_loaders(cfg: Config):
    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(cfg.train_dir, transform=train_tf)
    test_ds = datasets.ImageFolder(cfg.test_dir, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=False
    )

    return train_loader, test_loader, train_ds.classes


# -----------------------------
# ELBO training
# -----------------------------
def elbo_step(model, x, y, cfg: Config, optimizer=None, train=True):
    if train:
        model.train()
    else:
        model.eval()

    # Monte Carlo estimate of expected NLL
    ce_accum = 0.0
    logits_accum = 0.0

    S = cfg.mc_train_samples if train else cfg.mc_eval_samples

    with torch.set_grad_enabled(train):
        for _ in range(S):
            logits = model(x, sample=True)  # sample weights
            ce = F.cross_entropy(logits, y)
            ce_accum = ce_accum + ce
            logits_accum = logits_accum + logits

        ce_mean = ce_accum / S

        # KL term: single KL per batch (doesn't depend on sampled eps explicitly)
        kl = model.kl()

        # Normalize KL roughly per data point for stability
        # (Classic trick in BBB/VI training)
        kl_per_example = kl / x.size(0)

        loss = ce_mean + cfg.beta_kl * kl_per_example

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    # For metrics, use the mean logits across samples
    logits_mean = logits_accum / S
    preds = torch.argmax(logits_mean, dim=1)
    acc = (preds == y).float().mean().item()

    return loss.item(), ce_mean.item(), kl_per_example.item(), acc, logits_mean


@torch.no_grad()
def evaluate_with_uncertainty(model, loader, cfg: Config, num_classes: int):
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    entropy_sum = 0.0

    for x, y in loader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        # MC predictive distribution
        probs = torch.zeros((x.size(0), num_classes), device=cfg.device)

        for _ in range(cfg.mc_eval_samples):
            logits = model(x, sample=True)
            probs = probs + F.softmax(logits, dim=1)

        probs = probs / cfg.mc_eval_samples

        # predictive entropy: -sum p log p
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).mean()
        entropy_sum += entropy.item() * x.size(0)

        # use mean probs for prediction
        preds = probs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

        # "loss" as NLL of mean probs (not a true ELBO, but useful)
        nll = F.nll_loss(torch.log(probs.clamp_min(1e-8)), y)
        loss_sum += nll.item() * x.size(0)

    return loss_sum / total, correct / total, entropy_sum / total


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = Config()
    set_seed(cfg.seed)

    train_loader, test_loader, class_names = make_loaders(cfg)
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Device:", cfg.device)

    model = BayesianDWNet(num_classes=num_classes, prior_sigma=cfg.prior_sigma).to(cfg.device)
    print("Trainable params:", count_params(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        running_acc = 0.0
        running_ce = 0.0
        running_kl = 0.0
        n_batches = 0

        # Optional: KL annealing (often helps)
        # ramp beta from 0 -> cfg.beta_kl over first 30% epochs
        ramp = min(1.0, epoch / max(1.0, 0.3 * cfg.epochs))
        cfg.beta_kl = cfg.beta_kl * ramp

        for x, y in train_loader:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            loss, ce, kl, acc, _ = elbo_step(model, x, y, cfg, optimizer=optimizer, train=True)

            running_loss += loss
            running_ce += ce
            running_kl += kl
            running_acc += acc
            n_batches += 1

        train_loss = running_loss / n_batches
        train_acc = running_acc / n_batches
        train_ce = running_ce / n_batches
        train_kl = running_kl / n_batches

        test_nll, test_acc, test_entropy = evaluate_with_uncertainty(model, test_loader, cfg, num_classes)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"train loss {train_loss:.4f} (CE {train_ce:.4f} + beta*KL {train_kl:.4f}) | "
            f"train acc {train_acc:.4f} | "
            f"test nll {test_nll:.4f} | test acc {test_acc:.4f} | "
            f"pred entropy {test_entropy:.4f} | "
            f"time {dt:.1f}s"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": class_names,
                    "cfg": cfg.__dict__,
                },
                "bayesian_dwcnn_best.pt",
            )
            print(f"Saved best checkpoint: bayesian_dwcnn_best.pt (acc={best_acc:.4f})")

    print("Done. Best test acc:", best_acc)


if __name__ == "__main__":
    main()
