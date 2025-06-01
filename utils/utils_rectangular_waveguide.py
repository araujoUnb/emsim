# utils/utils_rectangular_waveguide.py

import torch
import math
from utils.utils_modes import ModeProfile


class TEModeProfile(ModeProfile):
    """
    TE_mn mode spatial profile for a rectangular waveguide.
    E-field ∝ sin(m π x / a) · sin(n π y / b). Zero if index is zero in that direction.
    """

    def generate(self) -> torch.Tensor:
        # Build a 2D grid: Y has shape [Ny, Nx], X has shape [Ny, Nx]
        Y, X = torch.meshgrid(self.y_lin, self.x_lin, indexing='ij')

        # TE_mn factor = sin(m·π·x / a) * sin(n·π·y / b)
        if self.m > 0:
            term_x = torch.sin(self.m * math.pi * X / self.a)
        else:
            term_x = torch.zeros_like(X)

        if self.n > 0:
            term_y = torch.sin(self.n * math.pi * Y / self.b)
        else:
            term_y = torch.zeros_like(Y)

        return term_x * term_y

    def title_str(self) -> str:
        return f"TE₍{self.m}{self.n}₎ Mode Profile (Rectangular)"


class TMModeProfile(ModeProfile):
    """
    TM_mn mode spatial profile for a rectangular waveguide.
    E_z ∝ sin(m π x / a) · sin(n π y / b).
    """

    def generate(self) -> torch.Tensor:
        Y, X = torch.meshgrid(self.y_lin, self.x_lin, indexing='ij')

        if self.m > 0:
            term_x = torch.sin(self.m * math.pi * X / self.a)
        else:
            term_x = torch.zeros_like(X)

        if self.n > 0:
            term_y = torch.sin(self.n * math.pi * Y / self.b)
        else:
            term_y = torch.zeros_like(Y)

        return term_x * term_y

    def title_str(self) -> str:
        return f"TM₍{self.m}{self.n}₎ Mode Profile (Rectangular)"


def cutoff_frequency_rectangular(a: float,
                                 b: float,
                                 m: int,
                                 n: int) -> float:
    """
    Compute the cutoff frequency f_c for TEₘₙ or TMₘₙ in a rectangular waveguide.

    Formula:
      f_c(m,n) = (c0 / 2) * sqrt( (m/a)² + (n/b)² )

    Parameters:
    - a, b : waveguide dimensions [m]
    - m, n : mode indices (integers)

    Returns:
    - f_c (float) : cutoff frequency [Hz] for TEₘₙ or TMₘₙ mode
    """
    from utils_constants import C0
    return (C0 / 2.0) * math.sqrt((m / a)**2 + (n / b)**2)
