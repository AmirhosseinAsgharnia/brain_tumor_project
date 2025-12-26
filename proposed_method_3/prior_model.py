# prior_model.py
# Frozen separable CNN with depthwise spatial filters sampled from stage_1.5 diagonal Gaussians.
# Only the final FC layer is trainable in Stage 2.

import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_prior_fits(prior_pt_path: str, device: torch.device) -> Dict[str, Any]:
    payload = torch.load(prior_pt_path, map_location=device)
    if "fits" not in payload:
        raise RuntimeError(f"Invalid prior file: missing 'fits' in {prior_pt_path}")
    return payload


def _get_fit(payload: Dict[str, Any], name: str) -> Dict[str, torch.Tensor]:
    fits = payload["fits"]
    if name not in fits:
        raise KeyError(f"Prior fit '{name}' not found. Available: {list(fits.keys())}")
    return fits[name]


def _sample_spatial_kernel(mu_flat: torch.Tensor, sigma_flat: torch.Tensor, out_shape: torch.Size) -> torch.Tensor:
    """
    mu_flat, sigma_flat: [k*k]
    out_shape: (C, 1, k, k) or (C, 1, k, k)
    Returns sampled tensor with out_shape.
    """
    k2 = mu_flat.numel()
    k = int(math.isqrt(k2))
    if k * k != k2:
        raise RuntimeError(f"mu length {k2} is not a perfect square")

    mu = mu_flat.view(1, 1, k, k)
    sigma = sigma_flat.view(1, 1, k, k)

    # Broadcast to channels
    C = out_shape[0]
    mu = mu.expand(C, 1, k, k)
    sigma = sigma.expand(C, 1, k, k)

    eps = torch.randn((C, 1, k, k), device=mu.device, dtype=mu.dtype)
    w = mu + sigma * eps
    return w


class FrozenSeparableBlock(nn.Module):
    """
    Depthwise spatial conv (sampled from a specified prior fit) + frozen pointwise mixing conv.
    Stride is applied on depthwise conv to handle downsampling.

    - Depthwise: groups=in_ch, out_ch=in_ch
    - Pointwise: 1x1, out_ch=out_ch
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        ksize: int,
        stride: int,
        prior_mu: torch.Tensor,     # [k*k]
        prior_sigma: torch.Tensor,  # [k*k]
        device: torch.device,
    ):
        super().__init__()

        pad = ksize // 2

        # Depthwise conv
        self.dw = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=in_ch,
            bias=False,
        )

        # Sample and freeze depthwise weights
        with torch.no_grad():
            w = _sample_spatial_kernel(prior_mu, prior_sigma, self.dw.weight.shape)
            self.dw.weight.copy_(w)

        for p in self.dw.parameters():
            p.requires_grad = False

        # Pointwise conv (frozen mixing)
        self.pw = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False,
        )

        # Fixed random mixing (Kaiming init) and freeze
        with torch.no_grad():
            nn.init.kaiming_normal_(self.pw.weight, mode="fan_out", nonlinearity="relu")
        for p in self.pw.parameters():
            p.requires_grad = False

        self.act = nn.ReLU(inplace=True)

        # Move to device explicitly (safety)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.act(x)
        return x


class PriorCNN(nn.Module):
    """
    Stage-2 prior model:
      - 3 frozen separable conv blocks
      - flatten
      - trainable FC classifier only

    Spatial schedule: 224 -> 112 -> 56 -> 28 (stride=2 each block)
    Channel schedule: 1 -> 60 -> 60 -> 60
    Kernel sizes: 7, 5, 3

    Priors (from stage_1.5):
      - layer1 depthwise uses 'deriv1_k7' (or you can switch to 'gauss_k7')
      - layer2 depthwise uses 'deriv2_k5'
      - layer3 depthwise uses 'gabor_k3'
    """
    def __init__(
        self,
        num_classes: int,
        prior_pt_path: str = "./artifacts/priors_diag_gauss.pt",
        layer1_prior: str = "deriv1_k7",
        layer2_prior: str = "deriv2_k5",
        layer3_prior: str = "gabor_k3",
        device: Optional[torch.device] = None,
        img_size: int = 224,
        final_spatial: int = 28,
        channels: int = 60,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        payload = _load_prior_fits(prior_pt_path, device=device)

        f1 = _get_fit(payload, layer1_prior)
        f2 = _get_fit(payload, layer2_prior)
        f3 = _get_fit(payload, layer3_prior)

        # Validate kernel sizes match intended structure
        k1 = int(f1["ksize"].item()) if isinstance(f1["ksize"], torch.Tensor) else int(f1["ksize"])
        k2 = int(f2["ksize"].item()) if isinstance(f2["ksize"], torch.Tensor) else int(f2["ksize"])
        k3 = int(f3["ksize"].item()) if isinstance(f3["ksize"], torch.Tensor) else int(f3["ksize"])

        if k1 != 7 or k2 != 5 or k3 != 3:
            raise RuntimeError(f"Expected k=(7,5,3) but got ({k1},{k2},{k3}). Regenerate priors or change config.")

        # Record recipe for Stage 3
        self.prior_recipe = {
            "prior_pt_path": prior_pt_path,
            "layer1_prior": layer1_prior,
            "layer2_prior": layer2_prior,
            "layer3_prior": layer3_prior,
            "kernels": [k1, k2, k3],
            "channels": [1, channels, channels, channels],
            "strides": [2, 2, 2],
            "img_size": img_size,
            "final_spatial": final_spatial,
        }

        # Build frozen blocks
        self.block1 = FrozenSeparableBlock(
            in_ch=1, out_ch=channels, ksize=7, stride=2,
            prior_mu=f1["mu"].to(device), prior_sigma=f1["sigma"].to(device), device=device
        )
        self.block2 = FrozenSeparableBlock(
            in_ch=channels, out_ch=channels, ksize=5, stride=2,
            prior_mu=f2["mu"].to(device), prior_sigma=f2["sigma"].to(device), device=device
        )
        self.block3 = FrozenSeparableBlock(
            in_ch=channels, out_ch=channels, ksize=3, stride=2,
            prior_mu=f3["mu"].to(device), prior_sigma=f3["sigma"].to(device), device=device
        )

        # Classifier (trainable)
        feat_dim = channels * final_spatial * final_spatial
        self.fc = nn.Linear(feat_dim, num_classes)

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def trainable_parameters(self):
        # Only FC should train in Stage 2
        return self.fc.parameters()
