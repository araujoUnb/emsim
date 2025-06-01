# utils_fdtd.py

"""
FDTD‐related helper functions, implemented with PyTorch and plain Python.
Includes:
  - CFL time‐step computation
  - E‐field and H‐field update coefficients (with or without conductivity)
  - First‐order Mur‐ABC coefficient
  - Tensor‐to‐NumPy conversion for plotting
"""

import torch
import math
import numpy as np

from utils_constants import C0  # uses the constant C0 from utils_constants


def compute_cfl_dt(dx: float, dy: float, dz: float, dt_factor: float = 0.99) -> float:
    """
    Compute the time step Δt based on the CFL condition in a 3D grid.

    CFL condition: Δt_cfl = 1 / (c₀ * sqrt(1/dx² + 1/dy² + 1/dz²)).
    We then multiply by dt_factor (<1) for numerical stability.

    Parameters:
    - dx, dy, dz : grid spacings [m]
    - dt_factor  : safety factor (default 0.99)

    Returns:
    - Δt (float): stable time step [s]
    """
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz2 = 1.0 / (dz * dz)
    dt_cfl = 1.0 / (C0 * math.sqrt(inv_dx2 + inv_dy2 + inv_dz2))
    return dt_factor * dt_cfl


def compute_Ce_coeffs(eps_slice: torch.Tensor,
                     sigma_slice: torch.Tensor,
                     dt: float,
                     d: float) -> (torch.Tensor, torch.Tensor):
    """
    Compute electric‐field update coefficients (Ce and Ce_fac) for one E‐component.

    The update formula for Ex, Ey, or Ez with conductivity:
      E_new = Ce * E_old + Ce_fac * (curl of H)

    Where:
      Ce      = (1 - (σ·dt)/(2·ε)) / (1 + (σ·dt)/(2·ε))
      Ce_fac  = (dt / (ε·d)) / (1 + (σ·dt)/(2·ε))

    Parameters:
    - eps_slice   : tensor of ε at that Yee‐location (any shape)
    - sigma_slice : tensor of σ (same shape as eps_slice)
    - dt          : time step [s]
    - d           : grid spacing along the difference direction [m]
                    (e.g., dx for Ez‐update, dy for Ex‐update, dz for Ey‐update)

    Returns:
    - Ce      : tensor with shape of eps_slice (elementwise Ce coefficients)
    - Ce_fac  : tensor with same shape (elementwise Ce_fac coefficients)
    """
    # (σ · dt) / (2 ε) term
    tmp = (sigma_slice * dt) / (2.0 * eps_slice)
    Ce = (1.0 - tmp) / (1.0 + tmp)
    Ce_fac = (dt / (eps_slice * d)) / (1.0 + tmp)
    return Ce, Ce_fac


def compute_Ch_coeffs(mu_slice: torch.Tensor, dt: float, d: float) -> torch.Tensor:
    """
    Compute magnetic‐field update coefficient (Ch_fac) for one H‐component.

    The update formula for Hx, Hy, or Hz (no conductivity):
      H_new = H_old - Ch_fac * (curl of E)

    Where:
      Ch_fac = dt / (μ · d)

    Parameters:
    - mu_slice : tensor of μ at that Yee‐location [any shape]
    - dt       : time step [s]
    - d        : grid spacing along the difference direction [m]

    Returns:
    - Ch_fac   : tensor of same shape as mu_slice, containing dt/(μ·d)
    """
    return dt / (mu_slice * d)


def compute_mur_coeff(c0: float, dt: float, d: float) -> float:
    """
    Compute the first‐order Mur absorbing boundary coefficient α for one axis.

    α = (c₀·Δt - Δx) / (c₀·Δt + Δx)

    Parameters:
    - c0 : speed of light [m/s]
    - dt : time step [s]
    - d  : grid spacing along that axis [m]

    Returns:
    - α (float): Mur coefficient
    """
    return (c0 * dt - d) / (c0 * dt + d)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor (CPU or GPU) to a NumPy array for plotting/analysis.

    Parameters:
    - tensor : any torch.Tensor

    Returns:
    - NumPy array with the same data
    """
    return tensor.detach().cpu().numpy()
