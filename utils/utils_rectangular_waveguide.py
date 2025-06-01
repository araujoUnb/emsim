# utils/utils_rectangular_waveguide.py

import torch
import math
from utils.utils_modes import ModeProfile


class TEModeProfile(ModeProfile):
    """
    TE_mn mode spatial profile for a rectangular waveguide.
    
    In a PEC‐walled rectangular guide of cross‐section a×b:
      - The only nonzero transverse E‐component is E_y(x,y).
      - E_y(x,y) ∝ sin(m π x / a) · cos(n π y / b),  for m ≥ 1 and n ≥ 0.
      - When n = 0, cos(0 · π y / b) = 1, so TE_{m,0}(x,y) = sin(m π x / a).
      - If m = 0, there is no TE_{0,n} mode (E_y ≡ 0).  
    """

    def generate(self) -> torch.Tensor:
        # Build a 2D grid: Y has shape [Ny, Nx], X has shape [Ny, Nx]
        Y, X = torch.meshgrid(self.y_lin, self.x_lin, indexing='ij')
        # We assume self.a, self.b, self.m, self.n were set in ModeProfile.__init__

        # x‐dependence: sin(m π x / a) if m ≥ 1
        if self.m >= 1:
            term_x = torch.sin(self.m * math.pi * X / self.a)
        else:
            # if m=0, TE_{0,n} does not exist → entire field = 0
            return torch.zeros_like(X)

        # y‐dependence: cos(n π y / b) if n ≥ 0
        #   when n=0, cos(0)=1 (uniform in y)
        if self.n >= 0:
            term_y = torch.cos(self.n * math.pi * Y / self.b)
        else:
            # negative n not allowed
            raise ValueError(f"Invalid mode index n={self.n} for TE; must be ≥ 0.")

        return term_x * term_y

    def title_str(self) -> str:
        return f"TE₍{self.m}{self.n}₎ Mode Profile (Rectangular)"


class TMModeProfile(ModeProfile):
    """
    TM_mn mode spatial profile for a rectangular waveguide.
    
    In a PEC‐walled rectangular guide of cross‐section a×b:
      - The only nonzero electric component is E_z(x,y).
      - E_z(x,y) ∝ sin(m π x / a) · sin(n π y / b),  for m ≥ 1 and n ≥ 1.
      - If either m=0 or n=0, TM_{m,n} does not exist (E_z ≡ 0).
    """

    def generate(self) -> torch.Tensor:
        # Build a 2D grid: Y has shape [Ny, Nx], X has shape [Ny, Nx]
        Y, X = torch.meshgrid(self.y_lin, self.x_lin, indexing='ij')
        # We assume self.a, self.b, self.m, self.n were set in ModeProfile.__init__

        # TM modes require both m ≥ 1 and n ≥ 1
        if self.m < 1 or self.n < 1:
            # TM_{m,0} or TM_{0,n} do not exist → entire field = 0
            return torch.zeros_like(X)

        term_x = torch.sin(self.m * math.pi * X / self.a)
        term_y = torch.sin(self.n * math.pi * Y / self.b)

        return term_x * term_y

    def title_str(self) -> str:
        return f"TM₍{self.m}{self.n}₎ Mode Profile (Rectangular)"


def cutoff_frequency_rectangular(a: float,
                                 b: float,
                                 m: int,
                                 n: int) -> float:
    """
    Compute the cutoff frequency f_c for TEₘₙ or TMₘₙ in a rectangular waveguide.

    f_c(m,n) = (c0 / 2) * sqrt((m / a)^2 + (n / b)^2)

    - a, b : waveguide interior dimensions [m]
    - m, n : mode indices (integers). For TE: m ≥ 1, n ≥ 0.  For TM: m ≥ 1, n ≥ 1.

    Returns f_c in Hz.
    """
    from utils.utils_constants import C0
    return (C0 / 2.0) * math.sqrt((m / a)**2 + (n / b)**2)
