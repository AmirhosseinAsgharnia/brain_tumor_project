# posterior_model.py
# Stage 3 posterior model: Bayesian depthwise convs + deterministic pointwise convs + Bayesian FC.
# KL(q||p) is closed-form because both prior and posterior are diagonal Gaussians.

import math
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def kl_diag_gaussian(mu_q: torch.Tensor, sigma_q: torch.Tensor,
                     mu_p: torch.Tensor, sigma_p: torch.Tensor) -> torch.Tensor:
    """
    KL( N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) ) for diagonal Gaussians.
    All tensors must be broadcastable to same shape.
    Returns scalar KL (sum over all elements).
    """
    # safety
    eps = 1e-12
    sigma_q = torch.clamp(sigma_q, min=eps)
    sigma_p = torch.clamp(sigma_p, min=eps)

    return torch.sum(
        torch.log(sigma_p / sigma_q) +
        (sigma_q * sigma_q + (mu_q - mu_p) * (mu_q - mu_p)) / (2.0 * sigma_p * sigma_p) - 0.5
    )


class BayesianParam(nn.Module):
    """
    Diagonal Gaussian variational parameterization for a weight tensor:
      q(w) = N(mu, sigma^2), sigma = softplus(rho).
    Prior is fixed diagonal Gaussian with given (mu_p, sigma_p).
    """
    def __init__(self, shape: torch.Size,
                 prior_mu: torch.Tensor,
                 prior_sigma: torch.Tensor,
                 init_mu: Optional[torch.Tensor] = None,
                 init_rho: float = -3.0):
        super().__init__()

        self.mu = nn.Parameter(torch.zeros(shape))
        self.rho = nn.Parameter(torch.full(shape, float(init_rho)))

        # init mu
        with torch.no_grad():
            if init_mu is not None:
                self.mu.copy_(init_mu)
            else:
                self.mu.zero_()

        # Prior buffers (fixed)
        self.register_buffer("prior_mu", prior_mu)
        self.register_buffer("prior_sigma", prior_sigma)

    def sigma(self) -> torch.Tensor:
        return softplus(self.rho)

    def sample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mu)
        return self.mu + self.sigma() * eps

    def kl(self) -> torch.Tensor:
        return kl_diag_gaussian(self.mu, self.sigma(), self.prior_mu, self.prior_sigma)


def _template_to_weight(shape: torch.Size, mu_flat: torch.Tensor, sigma_flat: torch.Tensor,
                        in_ch: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Broadcast prior template (k*k) to a depthwise conv weight shape: [in_ch, 1, k, k]
    """
    k2 = mu_flat.numel()
    k = int(math.isqrt(k2))
    if k * k != k2:
        raise RuntimeError(f"Prior template length {k2} is not a perfect square")

    mu = mu_flat.view(1, 1, k, k).expand(in_ch, 1, k, k).contiguous()
    sigma = sigma_flat.view(1, 1, k, k).expand(in_ch, 1, k, k).contiguous()

    if tuple(mu.shape) != tuple(shape):
        raise RuntimeError(f"Template broadcast mismatch. Want {shape}, got {mu.shape}")

    return mu, sigma


class BayesianDepthwiseConv2d(nn.Module):
    """
    Bayesian depthwise conv: groups=in_ch, weight shape [in_ch, 1, k, k]
    Prior comes from stage_1.5 templates broadcasted to channels.
    """
    def __init__(self, in_ch: int, ksize: int, stride: int,
                 prior_mu_flat: torch.Tensor, prior_sigma_flat: torch.Tensor,
                 init_mu_from: Optional[torch.Tensor] = None):
        super().__init__()
        pad = ksize // 2

        self.in_ch = in_ch
        self.ksize = ksize
        self.stride = stride
        self.padding = pad

        w_shape = torch.Size([in_ch, 1, ksize, ksize])

        prior_mu, prior_sigma = _template_to_weight(w_shape, prior_mu_flat, prior_sigma_flat, in_ch)

        self.w = BayesianParam(
            shape=w_shape,
            prior_mu=prior_mu,
            prior_sigma=prior_sigma,
            init_mu=init_mu_from,
            init_rho=-3.0,
        )

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        w = self.w.sample() if sample else self.w.mu
        return F.conv2d(x, w, bias=None, stride=self.stride, padding=self.padding, groups=self.in_ch)

    def kl(self) -> torch.Tensor:
        return self.w.kl()


class PosteriorSeparableBlock(nn.Module):
    """
    Bayesian depthwise spatial conv + deterministic pointwise mixing conv.
    """
    def __init__(self, in_ch: int, out_ch: int, ksize: int, stride: int,
                 prior_mu_flat: torch.Tensor, prior_sigma_flat: torch.Tensor,
                 init_dw_mu_from: Optional[torch.Tensor] = None):
        super().__init__()

        self.dw = BayesianDepthwiseConv2d(
            in_ch=in_ch,
            ksize=ksize,
            stride=stride,
            prior_mu_flat=prior_mu_flat,
            prior_sigma_flat=prior_sigma_flat,
            init_mu_from=init_dw_mu_from,
        )

        # Deterministic pointwise (trainable)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.kaiming_normal_(self.pw.weight, mode="fan_out", nonlinearity="relu")

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        x = self.dw(x, sample=sample)
        x = self.pw(x)
        x = self.act(x)
        return x

    def kl(self) -> torch.Tensor:
        return self.dw.kl()


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with diagonal Gaussian posterior.
    Prior is fixed diagonal Gaussian.
    """
    def __init__(self, in_features: int, out_features: int,
                 prior_mu: float = 0.0, prior_sigma: float = 1.0,
                 init_mu_from: Optional[torch.Tensor] = None):
        super().__init__()
        w_shape = torch.Size([out_features, in_features])
        b_shape = torch.Size([out_features])

        prior_mu_w = torch.full(w_shape, float(prior_mu))
        prior_sigma_w = torch.full(w_shape, float(prior_sigma))
        prior_mu_b = torch.full(b_shape, float(prior_mu))
        prior_sigma_b = torch.full(b_shape, float(prior_sigma))

        self.w = BayesianParam(w_shape, prior_mu_w, prior_sigma_w, init_mu=init_mu_from, init_rho=-3.0)
        self.b = BayesianParam(b_shape, prior_mu_b, prior_sigma_b, init_mu=None, init_rho=-3.0)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        w = self.w.sample() if sample else self.w.mu
        b = self.b.sample() if sample else self.b.mu
        return F.linear(x, w, b)

    def kl(self) -> torch.Tensor:
        return self.w.kl() + self.b.kl()


class PosteriorCNN(nn.Module):
    """
    Stage 3 posterior CNN with:
      - 3 separable blocks: Bayesian depthwise + deterministic pointwise
      - Bayesian FC
    """
    def __init__(
        self,
        num_classes: int,
        prior_recipe: Dict[str, Any],
        priors_pt_path: str = "./artifacts/priors_diag_gauss.pt",
        init_from_stage2_state: Optional[Dict[str, torch.Tensor]] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load fitted priors
        payload = torch.load(priors_pt_path, map_location=device)
        fits = payload["fits"]

        # Pull recipe
        l1_name = prior_recipe["layer1_prior"]
        l2_name = prior_recipe["layer2_prior"]
        l3_name = prior_recipe["layer3_prior"]

        f1 = fits[l1_name]
        f2 = fits[l2_name]
        f3 = fits[l3_name]

        # Architecture (fixed by recipe)
        channels = int(prior_recipe["channels"][1])
        final_spatial = int(prior_recipe["final_spatial"])

        # Optional init from stage2 sampled weights
        init_dw1 = None
        init_dw2 = None
        init_dw3 = None
        init_fc = None

        if init_from_stage2_state is not None:
            # Stage2 had frozen conv weights at:
            # block1.dw.weight, block2.dw.weight, block3.dw.weight
            # and fc.weight, fc.bias (deterministic)
            if "block1.dw.weight" in init_from_stage2_state:
                init_dw1 = init_from_stage2_state["block1.dw.weight"].to(device)
            if "block2.dw.weight" in init_from_stage2_state:
                init_dw2 = init_from_stage2_state["block2.dw.weight"].to(device)
            if "block3.dw.weight" in init_from_stage2_state:
                init_dw3 = init_from_stage2_state["block3.dw.weight"].to(device)
            if "fc.weight" in init_from_stage2_state:
                init_fc = init_from_stage2_state["fc.weight"].to(device)

        self.block1 = PosteriorSeparableBlock(
            in_ch=1, out_ch=channels, ksize=7, stride=2,
            prior_mu_flat=f1["mu"].to(device), prior_sigma_flat=f1["sigma"].to(device),
            init_dw_mu_from=init_dw1
        )
        self.block2 = PosteriorSeparableBlock(
            in_ch=channels, out_ch=channels, ksize=5, stride=2,
            prior_mu_flat=f2["mu"].to(device), prior_sigma_flat=f2["sigma"].to(device),
            init_dw_mu_from=init_dw2
        )
        self.block3 = PosteriorSeparableBlock(
            in_ch=channels, out_ch=channels, ksize=3, stride=2,
            prior_mu_flat=f3["mu"].to(device), prior_sigma_flat=f3["sigma"].to(device),
            init_dw_mu_from=init_dw3
        )

        feat_dim = channels * final_spatial * final_spatial
        self.fc = BayesianLinear(feat_dim, num_classes, prior_mu=0.0, prior_sigma=1.0, init_mu_from=init_fc)

        self.to(device)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        x = x.to(self.device)
        x = self.block1(x, sample=sample)
        x = self.block2(x, sample=sample)
        x = self.block3(x, sample=sample)
        x = torch.flatten(x, 1)
        x = self.fc(x, sample=sample)
        return x

    def kl_loss(self) -> torch.Tensor:
        return self.block1.kl() + self.block2.kl() + self.block3.kl() + self.fc.kl()
