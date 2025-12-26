# posterior_model.py
# Bayesian (variational) depthwise-separable CNN for 1-channel input.
# Every Conv/Linear weight + bias is Gaussian: q(theta)=N(mu, sigma^2).
# KL is analytic vs a Gaussian prior p(theta)=N(mu_p, sigma_p^2) with scalar (layerwise) params.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus(x):
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.maximum(x, torch.zeros_like(x))


def kl_diag_gaussian(mu_q, sigma_q, mu_p, sigma_p):
    # KL( N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) ) summed over all elements
    return torch.sum(
        torch.log(sigma_p / sigma_q) +
        (sigma_q**2 + (mu_q - mu_p)**2) / (2.0 * sigma_p**2) - 0.5
    )


class BayesianConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=True,
        prior_mu=0.0,
        prior_sigma=0.1,
        init_rho=-3.0,
    ):
        super().__init__()

        if isinstance(kernel_size, tuple):
            kh, kw = kernel_size
        else:
            kh = kw = kernel_size

        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.use_bias = bias

        # Weight shape matches torch Conv2d:
        # [out_channels, in_channels // groups, kh, kw]
        w_shape = (out_channels, in_channels // groups, kh, kw)
        self.w_mu = nn.Parameter(torch.zeros(w_shape))
        self.w_rho = nn.Parameter(torch.full(w_shape, float(init_rho)))

        if bias:
            self.b_mu = nn.Parameter(torch.zeros(out_channels))
            self.b_rho = nn.Parameter(torch.full((out_channels,), float(init_rho)))
        else:
            self.register_parameter("b_mu", None)
            self.register_parameter("b_rho", None)

        # Scalar Gaussian prior for this layer (simple + stable)
        self.register_buffer("prior_mu", torch.tensor(float(prior_mu)))
        self.register_buffer("prior_sigma", torch.tensor(float(prior_sigma)))

    def sample_params(self):
        w_sigma = softplus(self.w_rho)
        w = self.w_mu + w_sigma * torch.randn_like(w_sigma)

        if self.use_bias:
            b_sigma = softplus(self.b_rho)
            b = self.b_mu + b_sigma * torch.randn_like(b_sigma)
        else:
            b = None

        return w, b

    def forward(self, x, sample=True):
        if sample:
            w, b = self.sample_params()
        else:
            w, b = self.w_mu, (self.b_mu if self.use_bias else None)

        return F.conv2d(x, w, b, stride=self.stride, padding=self.padding, groups=self.groups)

    def kl_loss(self):
        w_sigma = softplus(self.w_rho)
        mu_p = self.prior_mu
        sigma_p = self.prior_sigma
        kl = kl_diag_gaussian(self.w_mu, w_sigma, mu_p, sigma_p)

        if self.use_bias:
            b_sigma = softplus(self.b_rho)
            kl = kl + kl_diag_gaussian(self.b_mu, b_sigma, mu_p, sigma_p)

        return kl


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, prior_mu=0.0, prior_sigma=0.1, init_rho=-3.0):
        super().__init__()
        self.use_bias = bias

        w_shape = (out_features, in_features)
        self.w_mu = nn.Parameter(torch.zeros(w_shape))
        self.w_rho = nn.Parameter(torch.full(w_shape, float(init_rho)))

        if bias:
            self.b_mu = nn.Parameter(torch.zeros(out_features))
            self.b_rho = nn.Parameter(torch.full((out_features,), float(init_rho)))
        else:
            self.register_parameter("b_mu", None)
            self.register_parameter("b_rho", None)

        self.register_buffer("prior_mu", torch.tensor(float(prior_mu)))
        self.register_buffer("prior_sigma", torch.tensor(float(prior_sigma)))

    def sample_params(self):
        w_sigma = softplus(self.w_rho)
        w = self.w_mu + w_sigma * torch.randn_like(w_sigma)

        if self.use_bias:
            b_sigma = softplus(self.b_rho)
            b = self.b_mu + b_sigma * torch.randn_like(b_sigma)
        else:
            b = None

        return w, b

    def forward(self, x, sample=True):
        if sample:
            w, b = self.sample_params()
        else:
            w, b = self.w_mu, (self.b_mu if self.use_bias else None)
        return F.linear(x, w, b)

    def kl_loss(self):
        w_sigma = softplus(self.w_rho)
        mu_p = self.prior_mu
        sigma_p = self.prior_sigma
        kl = kl_diag_gaussian(self.w_mu, w_sigma, mu_p, sigma_p)

        if self.use_bias:
            b_sigma = softplus(self.b_rho)
            kl = kl + kl_diag_gaussian(self.b_mu, b_sigma, mu_p, sigma_p)

        return kl


class PosteriorNet(nn.Module):
    """
    Your depthwise-separable shape, corrected to 1-channel input:

    DW: 1->1 (groups=1, stride=2) then PW 1->32
    ... then standard DW/PW chain up to 1024
    AdaptiveAvgPool -> Linear -> Linear

    All Bayesian.
    """
    def __init__(self, num_classes=4, prior_mu=0.0, prior_sigma=0.1):
        super().__init__()

        # Feature extractor blocks (Bayesian)
        self.feature = nn.ModuleList([
            # 224 -> 112
            BayesianConv2d(1, 1, kernel_size=3, stride=2, padding=1, groups=1, bias=True,
                          prior_mu=prior_mu, prior_sigma=prior_sigma),
            BayesianConv2d(1, 32, kernel_size=1, stride=1, padding=0, groups=1, bias=True,
                          prior_mu=prior_mu, prior_sigma=prior_sigma),
            nn.ReLU(inplace=True),

            # 112 -> 112
            BayesianConv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=True,
                          prior_mu=prior_mu, prior_sigma=prior_sigma),
            BayesianConv2d(32, 32, kernel_size=1, stride=1, padding=0, groups=1, bias=True,
                          prior_mu=prior_mu, prior_sigma=prior_sigma),
            nn.ReLU(inplace=True),

            # 112 -> 112
            BayesianConv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32, bias=True,
                          prior_mu=prior_mu, prior_sigma=prior_sigma),
            BayesianConv2d(32, 64, kernel_size=1, stride=1, padding=0, groups=1, bias=True,
                          prior_mu=prior_mu, prior_sigma=prior_sigma),
            nn.ReLU(inplace=True),

            # 112 -> 56
            BayesianConv2d(64, 64, kernel_size=3, stride=2, padding=1, groups=64, bias=True,
                          prior_mu=prior_mu, prior_sigma=prior_sigma),
            BayesianConv2d(64, 128, kernel_size=1, stride=1, padding=0, groups=1, bias=True,
                          prior_mu=prior_mu, prior_sigma=prior_sigma),
            nn.ReLU(inplace=True),

            # 56 -> 28
            BayesianConv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128, bias=True,
                          prior_mu=prior_mu, prior_sigma=prior_sigma),
            BayesianConv2d(128, 256, kernel_size=1, stride=1, padding=0, groups=1, bias=True,
                          prior_mu=prior_mu, prior_sigma=prior_sigma),
            nn.ReLU(inplace=True),

            # 28 -> 14
            BayesianConv2d(256, 256, kernel_size=3, stride=2, padding=1, groups=256, bias=True,
                          prior_mu=prior_mu, prior_sigma=prior_sigma),
            BayesianConv2d(256, 512, kernel_size=1, stride=1, padding=0, groups=1, bias=True,
                          prior_mu=prior_mu, prior_sigma=prior_sigma),
            nn.ReLU(inplace=True),

            # 14 -> 7
            BayesianConv2d(512, 512, kernel_size=3, stride=2, padding=1, groups=512, bias=True,
                          prior_mu=prior_mu, prior_sigma=prior_sigma),
            BayesianConv2d(512, 1024, kernel_size=1, stride=1, padding=0, groups=1, bias=True,
                          prior_mu=prior_mu, prior_sigma=prior_sigma),
            nn.ReLU(inplace=True),
        ])

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.ModuleList([
            BayesianLinear(1024, 256, bias=True, prior_mu=prior_mu, prior_sigma=prior_sigma),
            nn.ReLU(inplace=True),
            BayesianLinear(256, num_classes, bias=True, prior_mu=prior_mu, prior_sigma=prior_sigma),
        ])

    def forward(self, x, sample=True):
        for layer in self.feature:
            if isinstance(layer, (BayesianConv2d, BayesianLinear)):
                x = layer(x, sample=sample)
            else:
                x = layer(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        for layer in self.classifier:
            if isinstance(layer, (BayesianConv2d, BayesianLinear)):
                x = layer(x, sample=sample)
            else:
                x = layer(x)

        return x

    def kl_loss(self):
        kl = 0.0
        for layer in self.feature:
            if hasattr(layer, "kl_loss"):
                kl = kl + layer.kl_loss()
        for layer in self.classifier:
            if hasattr(layer, "kl_loss"):
                kl = kl + layer.kl_loss()
        return kl
