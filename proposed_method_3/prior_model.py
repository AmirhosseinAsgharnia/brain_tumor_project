# prior_model.py
# Handcrafted PRIOR network:
#   1ch input -> 60ch
#   Gaussian stage -> Derivative stage -> Gabor stage
#   Downsample: 224 -> 112 -> 56 -> 28
#   Flatten -> deterministic FC
#
# Only the 1x1 pointwise mixing convs + FC are learned.
# The depthwise handcrafted kernels are FIXED buffers (no grads).

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Kernel generators (torch)
# -----------------------------
def _meshgrid(ksize: int, device=None, dtype=None):
    r = ksize // 2
    xs = torch.arange(-r, r + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    return xx, yy

def gaussian2d(ksize: int, sigma: float, device=None, dtype=None):
    xx, yy = _meshgrid(ksize, device=device, dtype=dtype)
    g = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    g = g / (g.sum() + 1e-12)
    return g

def gaussian_derivative_x(ksize: int, sigma: float, device=None, dtype=None):
    xx, yy = _meshgrid(ksize, device=device, dtype=dtype)
    g = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    gx = -(xx / (sigma**2)) * g
    gx = gx - gx.mean()  # zero-mean
    gx = gx / (gx.norm() + 1e-12)
    return gx

def gaussian_derivative_y(ksize: int, sigma: float, device=None, dtype=None):
    return gaussian_derivative_x(ksize, sigma, device=device, dtype=dtype).t()

def gabor2d(
    ksize: int,
    sigma: float,
    theta: float,
    lambd: float,
    gamma: float = 0.5,
    psi: float = 0.0,
    device=None,
    dtype=None,
):
    xx, yy = _meshgrid(ksize, device=device, dtype=dtype)
    # rotate
    x_theta = xx * math.cos(theta) + yy * math.sin(theta)
    y_theta = -xx * math.sin(theta) + yy * math.cos(theta)

    gauss = torch.exp(-(x_theta**2 + (gamma**2) * y_theta**2) / (2.0 * sigma**2))
    wave = torch.cos(2.0 * math.pi * x_theta / lambd + psi)
    g = gauss * wave
    g = g - g.mean()
    g = g / (g.norm() + 1e-12)
    return g


# -----------------------------
# Fixed depthwise conv wrapper
# -----------------------------
class FixedDepthwiseConv2d(nn.Module):
    """
    Depthwise conv with fixed kernels replicated across channels.
    weight buffer shape: [C, 1, k, k]
    """
    def __init__(self, channels: int, kernel: torch.Tensor, stride: int = 1, padding: int = 1):
        super().__init__()
        assert kernel.ndim == 2, "kernel must be [k,k]"
        k = kernel.shape[0]
        assert kernel.shape[0] == kernel.shape[1], "kernel must be square"

        w = kernel.view(1, 1, k, k).repeat(channels, 1, 1, 1)  # [C,1,k,k]
        self.register_buffer("weight", w)
        self.stride = stride
        self.padding = padding
        self.groups = channels

    def forward(self, x):
        # x: [B,C,H,W]
        return F.conv2d(x, self.weight, bias=None, stride=self.stride, padding=self.padding, groups=self.groups) #type: ignore


# -----------------------------
# Prior blocks
# -----------------------------
class PriorBlock(nn.Module):
    """
    Fixed depthwise handcrafted filter -> learned 1x1 mixing -> ReLU
    """
    def __init__(self, channels: int, kernel_2d: torch.Tensor, stride: int = 1, padding: int = 1):
        super().__init__()
        self.dw = FixedDepthwiseConv2d(channels, kernel_2d, stride=stride, padding=padding)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.act(x)
        return x


# -----------------------------
# PriorNet
# -----------------------------
class PriorNet(nn.Module):
    def __init__(self, num_classes: int = 4, channels: int = 60, ksize: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.ksize = ksize

        # Stem: 1 -> 60 (learned)
        self.stem = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

        # --- Create handcrafted kernels (on CPU; buffers move with model.to(device)) ---
        # Gaussian stage kernels (use a mild sigma)
        g1 = gaussian2d(ksize, sigma=1.0)
        g2 = gaussian2d(ksize, sigma=1.4)

        # Derivative stage kernels
        dx = gaussian_derivative_x(ksize, sigma=1.0)
        dy = gaussian_derivative_y(ksize, sigma=1.0)

        # Gabor stage kernels (a couple of orientations)
        gb0 = gabor2d(ksize, sigma=1.6, theta=0.0,        lambd=3.0, gamma=0.7, psi=0.0)
        gb1 = gabor2d(ksize, sigma=1.6, theta=math.pi/4,  lambd=3.0, gamma=0.7, psi=0.0)
        gb2 = gabor2d(ksize, sigma=1.6, theta=math.pi/2,  lambd=3.0, gamma=0.7, psi=0.0)

        pad = ksize // 2

        # 224 -> 112 (stride 2) within Gaussian stage
        self.gaussian_1 = PriorBlock(channels, g1, stride=2, padding=pad)
        self.gaussian_2 = PriorBlock(channels, g2, stride=1, padding=pad)

        # 112 -> 56 (stride 2) within Derivative stage
        self.deriv_1 = PriorBlock(channels, dx, stride=2, padding=pad)
        self.deriv_2 = PriorBlock(channels, dy, stride=1, padding=pad)

        # 56 -> 28 (stride 2) within Gabor stage
        self.gabor_1 = PriorBlock(channels, gb0, stride=2, padding=pad)
        self.gabor_2 = PriorBlock(channels, gb1, stride=1, padding=pad)
        self.gabor_3 = PriorBlock(channels, gb2, stride=1, padding=pad)

        # Head: flatten 60*28*28 -> num_classes (deterministic)
        self.classifier = nn.Sequential(
            nn.Linear(channels * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        # x: [B,1,224,224]
        x = self.stem(x)          # [B,60,224,224]
        x = self.gaussian_1(x)    # [B,60,112,112]
        x = self.gaussian_2(x)    # [B,60,112,112]

        x = self.deriv_1(x)       # [B,60,56,56]
        x = self.deriv_2(x)       # [B,60,56,56]

        x = self.gabor_1(x)       # [B,60,28,28]
        x = self.gabor_2(x)       # [B,60,28,28]
        x = self.gabor_3(x)       # [B,60,28,28]

        x = torch.flatten(x, 1)   # [B, 60*28*28]
        x = self.classifier(x)    # [B, num_classes]
        return x
