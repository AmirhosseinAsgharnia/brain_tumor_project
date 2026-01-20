# stage_1_5.py
# Generate handcrafted filter banks and fit diagonal multivariate Gaussian priors.
# Output: ./artifacts/priors_diag_gauss.pt
#
# We keep Sigma diagonal (independent dimensions) but non-isotropic (each element has its own variance).
# We also export a layer_recipe mapping so stage_2 and stage_3 can use the same priors.

import os
import math
from typing import Dict, List, Tuple

import numpy as np
import torch


# -----------------------------
# Basic helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def odd_center_grid(ksize: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Meshgrid (x,y) centered at 0 for odd ksize.
    ksize=3 -> coords [-1,0,1]
    """
    if ksize % 2 != 1:
        raise ValueError("ksize must be odd")
    r = ksize // 2
    coords = np.arange(-r, r + 1, dtype=np.float64)
    x, y = np.meshgrid(coords, coords, indexing="xy")
    return x, y


def zero_mean(k: np.ndarray) -> np.ndarray:
    return k - np.mean(k)


def l2_normalize(k: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.sqrt(np.sum(k * k)))
    return k / (n + eps)


def flatten_k(k: np.ndarray) -> np.ndarray:
    return k.reshape(-1).astype(np.float64)


def safe_var(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Unbiased-ish variance with safety floor to avoid zeros.
    """
    v = np.var(x, axis=axis, ddof=1)
    v[v < 1e-12] = 1e-12
    return v


# -----------------------------
# Filter generators
# -----------------------------
def gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    x, y = odd_center_grid(ksize)
    g = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    # For our prior fitting, we want "shape" more than DC magnitude:
    g = zero_mean(g)
    g = l2_normalize(g)
    return g


def gaussian_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-(x * x) / (2.0 * sigma * sigma))


def d_gaussian_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    # d/dx exp(-x^2/(2s^2)) = -(x/s^2) exp(...)
    return -(x / (sigma * sigma)) * np.exp(-(x * x) / (2.0 * sigma * sigma))


def dd_gaussian_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    # d2/dx2 exp(-x^2/(2s^2)) = ((x^2 - s^2)/s^4) exp(...)
    s2 = sigma * sigma
    return ((x * x - s2) / (s2 * s2)) * np.exp(-(x * x) / (2.0 * s2))


def gaussian_derivative_kernel(ksize: int, sigma: float, kind: str) -> np.ndarray:
    """
    Separable construction:
      dx, dy, dxx, dyy, dxy, lap
    """
    if kind not in {"dx", "dy", "dxx", "dyy", "dxy", "lap"}:
        raise ValueError(f"Unknown derivative kind: {kind}")

    r = ksize // 2
    coords = np.arange(-r, r + 1, dtype=np.float64)

    G = gaussian_1d(coords, sigma)
    dG = d_gaussian_1d(coords, sigma)
    ddG = dd_gaussian_1d(coords, sigma)

    if kind == "dx":
        k = np.outer(G, dG)         # y smooth, x derivative
    elif kind == "dy":
        k = np.outer(dG, G)         # y derivative, x smooth
    elif kind == "dxx":
        k = np.outer(G, ddG)
    elif kind == "dyy":
        k = np.outer(ddG, G)
    elif kind == "dxy":
        k = np.outer(dG, dG)
    else:  # lap
        k = np.outer(G, ddG) + np.outer(ddG, G)

    k = zero_mean(k)
    k = l2_normalize(k)
    return k


def gabor_kernel(
    ksize: int,
    sigma: float,
    lambd: float,
    theta: float,
    psi: float,
    gamma: float,
    is_sin: bool = False,
) -> np.ndarray:
    x, y = odd_center_grid(ksize)

    ct = math.cos(theta)
    st = math.sin(theta)
    x_p = x * ct + y * st
    y_p = -x * st + y * ct

    gauss_env = np.exp(-(x_p * x_p + (gamma * gamma) * (y_p * y_p)) / (2.0 * sigma * sigma))
    phase = (2.0 * math.pi * x_p / max(lambd, 1e-6)) + psi
    carrier = np.sin(phase) if is_sin else np.cos(phase)

    g = gauss_env * carrier
    g = zero_mean(g)
    g = l2_normalize(g)
    return g


# -----------------------------
# Diagonal Gaussian fitting
# -----------------------------
def fit_diag_gaussian(samples: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    samples: list of flattened vectors [d]
    returns: mu [d], var_diag [d]
    """
    if len(samples) < 2:
        raise RuntimeError("Need at least 2 samples to fit variance")

    X = np.stack(samples, axis=0)  # [N,d]
    mu = np.mean(X, axis=0)
    var = safe_var(X, axis=0)
    return mu, var


# -----------------------------
# Build exactly the distributions we want
# -----------------------------
def build_distributions() -> Dict[str, Dict]:
    """
    Returns dict:
      distros[name] = {
         "ksize": int,
         "mu": np.ndarray [d],
         "var": np.ndarray [d],
         "n": int,
      }

      - L1: gauss_k7 + deriv1_k7
      - L2: deriv1_k5 + deriv2_k5
      - L3: gabor_k9
      - L4: gabor_k11
    """
    distros: Dict[str, Dict] = {}

    # Parameter grids
    sigma_gauss = [0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
    sigma_deriv = [0.6, 0.8, 1.0, 1.3, 1.6]
    sigma_gabor = [1.0, 2.0]
    lambd_grid = [2.0, 4.0, 8.0]
    theta_grid = [0.0, math.pi/4, math.pi/2, 3*math.pi/4]
    psi_grid = [0.0, math.pi / 2]
    gamma_grid = [0.5, 1.0]

    # ---- gauss_k7 ----
    samples = []
    for s in sigma_gauss:
        k = gaussian_kernel(7, s)
        samples.append(flatten_k(k))
    mu, var = fit_diag_gaussian(samples)
    distros["gauss_k7"] = {"ksize": 7, "mu": mu, "var": var, "n": len(samples)}

    # ---- deriv1_k7 (dx, dy pooled) ----
    samples = []
    for s in sigma_deriv:
        for kind in ["dx", "dy"]:
            k = gaussian_derivative_kernel(7, s, kind)
            samples.append(flatten_k(k))
    mu, var = fit_diag_gaussian(samples)
    distros["deriv1_k7"] = {"ksize": 7, "mu": mu, "var": var, "n": len(samples)}

    # ---- deriv1_k5 (dx, dy pooled) ----
    samples = []
    for s in sigma_deriv:
        for kind in ["dx", "dy"]:
            k = gaussian_derivative_kernel(5, s, kind)
            samples.append(flatten_k(k))
    mu, var = fit_diag_gaussian(samples)
    distros["deriv1_k5"] = {"ksize": 5, "mu": mu, "var": var, "n": len(samples)}

    # ---- deriv2_k5 (dxx, dyy, dxy, lap pooled) ----
    samples = []
    for s in sigma_deriv:
        for kind in ["dxx", "dyy", "dxy", "lap"]:
            k = gaussian_derivative_kernel(5, s, kind)
            samples.append(flatten_k(k))
    mu, var = fit_diag_gaussian(samples)
    distros["deriv2_k5"] = {"ksize": 5, "mu": mu, "var": var, "n": len(samples)}

    # ---- gabor_k9 (cos + sin pooled) ----
    samples = []
    for s in sigma_gabor:
        for lambd in lambd_grid:
            for theta in theta_grid:
                for psi in psi_grid:
                    for gamma in gamma_grid:
                        kc = gabor_kernel(9, s, lambd, theta, psi, gamma, is_sin=False)
                        ks = gabor_kernel(9, s, lambd, theta, psi, gamma, is_sin=True)
                        samples.append(flatten_k(kc))
                        samples.append(flatten_k(ks))
    mu, var = fit_diag_gaussian(samples)
    distros["gabor_k9"] = {"ksize": 9, "mu": mu, "var": var, "n": len(samples)}

    # ---- gabor_k11 (cos + sin pooled) ----
    samples = []
    for s in sigma_gabor:
        for lambd in lambd_grid:
            for theta in theta_grid:
                for psi in psi_grid:
                    for gamma in gamma_grid:
                        kc = gabor_kernel(11, s, lambd, theta, psi, gamma, is_sin=False)
                        ks = gabor_kernel(11, s, lambd, theta, psi, gamma, is_sin=True)
                        samples.append(flatten_k(kc))
                        samples.append(flatten_k(ks))
    mu, var = fit_diag_gaussian(samples)
    distros["gabor_k11"] = {"ksize": 11, "mu": mu, "var": var, "n": len(samples)}

    return distros

def save_pt(distros: Dict[str, Dict], out_path: str) -> None:
    """
    Save to a torch .pt file with tensors (float32) and a layer_recipe.
    """
    fits: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, d in distros.items():
        ksize = int(d["ksize"])
        mu = torch.tensor(d["mu"], dtype=torch.float32)
        var = torch.tensor(d["var"], dtype=torch.float32)
        n = int(d["n"])
        fits[name] = {
            "ksize": torch.tensor([ksize], dtype=torch.int64),
            "mu": mu,
            "var": var,
            "sigma": torch.sqrt(var),
            "d": torch.tensor([mu.numel()], dtype=torch.int64),
            "n": torch.tensor([n], dtype=torch.int64),
        }

    # This is the mapping stage_2 and stage_3 should share.
    # You can change kernel sizes later, but then you must regenerate priors.
    layer_recipe = {
        "layer1": {"ksize": 7, "prior": ["gauss_k7", "deriv1_k7"]},
        "layer2": {"ksize": 5, "prior": ["deriv1_k5", "deriv2_k5"]},
        "layer3": {"ksize": 9, "prior": ["gabor_k9"]},
        "layer4": {"ksize": 11, "prior": ["gabor_k11"]},
        # How to combine multiple priors for one layer:
        # We'll do a simple mixture later OR pick one per conv block.
        # For now we just record what's available for that layer.
        "notes": "Stage_2/3 should use this mapping so KL(q||p) uses the same p.",
    }

    payload = {
        "fits": fits,
        "layer_recipe": layer_recipe,
        "notes": {
            "covariance": "diagonal",
            "meaning": "Independent per-weight-position variances (non-isotropic).",
            "normalization": "zero-mean then L2",
        },
    }

    torch.save(payload, out_path)
    print(f"[stage_1_5] Saved: {out_path}")
    print(f"[stage_1_5] Distributions: {len(fits)}")
    for k, v in fits.items():
        print(f"  - {k}: k={int(v['ksize'].item())}, d={int(v['d'].item())}, n={int(v['n'].item())}")


def main():
    out_dir = os.path.join(".", "artifacts")
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "priors_diag_gauss.pt")

    distros = build_distributions()
    save_pt(distros, out_path)


if __name__ == "__main__":
    main()