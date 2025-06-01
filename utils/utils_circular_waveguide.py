# utils/utils_circular_waveguide.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.utils_modes import ModeProfile

# Para as funções de Bessel, usamos SciPy
from scipy import special as sp

# ------------------------------------------------------------------------------
# Tabela com os primeiros zeros de J0(x) e de J1'(x), para usos comuns.
# ------------------------------------------------------------------------------

# Primeiros 5 zeros de J0(x):
BJ0_ZEROS = [
    2.4048255577,   # J0 zero #1
    5.5200781103,   # J0 zero #2
    8.6537279129,   # J0 zero #3
    11.7915344391,  # J0 zero #4
    14.9309177086   # J0 zero #5
]

# Primeiros 5 zeros de J1'(x):
BJ1PRIME_ZEROS = [
    1.8411837813,   # J1' zero #1  (modo TE11)
    5.3314427735,   # J1' zero #2  (modo TE21)
    8.5363163664,   # J1' zero #3
    11.7061705724,  # J1' zero #4
    14.8635886339   # J1' zero #5
]


class CircularTMModeProfile(ModeProfile):
    """
    TM_0n para guia circular: 
      E_z(r,φ) ∝ J_0(k_{0n} · r), 
    com k_{0n} = (n-ésimo zero de J0) / R. 
    """

    def __init__(self,
                 R: float,
                 n: int,
                 x_lin: torch.Tensor,
                 y_lin: torch.Tensor,
                 device: str = 'cpu'):
        """
        Parâmetros:
        - R      : raio do guia [m]
        - n      : índice radial (n ≥ 1)
        - x_lin  : tensor 1D de coordenadas em x (Nx pontos)
        - y_lin  : tensor 1D de coordenadas em y (Ny pontos)
        - device : 'cpu' ou 'cuda'
        """
        # Passamos a = R, b = R, m = 0, n = n para a base ModeProfile
        super().__init__(a=R, b=R, m=0, n=n, x_lin=x_lin, y_lin=y_lin, device=device)
        self.R = R

    def generate(self) -> torch.Tensor:
        # 1) Obter o n-ésimo zero de J0 e computar k_0n = zero / R
        k_0n = BJ0_ZEROS[self.n - 1] / self.R

        # 2) Construir a grade cartesiana em CPU para usar em SciPy
        Xc, Yc = torch.meshgrid(self.x_lin.cpu(), self.y_lin.cpu(), indexing='ij')  # forma [Nx, Ny]
        Xc_np = Xc.numpy()
        Yc_np = Yc.numpy()
        Rgrid_np = np.sqrt(Xc_np**2 + Yc_np**2)  # [Nx, Ny]

        # 3) Avaliar J0(k_0n * r) em cada ponto radial
        J0_vals_np = sp.j0(k_0n * Rgrid_np)

        # 4) Aplicar máscara: zero fora do círculo (r > R)
        inside_np = (Rgrid_np <= self.R).astype(np.float32)
        profile_np = J0_vals_np * inside_np  # [Nx, Ny]

        # 5) Transpor para [Ny, Nx] e converter para torch.Tensor no device correto
        profile = torch.from_numpy(profile_np).to(self.device).T

        return profile

    def title_str(self) -> str:
        return f"TM₍0{self.n}₎ Profile (Circular)"


class CircularTEModeProfile(ModeProfile):
    """
    TE_1n para guia circular: 
      E_φ(r,φ) ∝ J_1'(k_{1n}·r) · cos(φ), 
    com k_{1n} = (n-ésimo zero de J1') / R.
    """

    def __init__(self,
                 R: float,
                 n: int,
                 x_lin: torch.Tensor,
                 y_lin: torch.Tensor,
                 device: str = 'cpu'):
        """
        Parâmetros:
        - R      : raio do guia [m]
        - n      : índice do zero de J1' (n ≥ 1), aqui m=1 por definição de TE1n
        - x_lin  : tensor 1D de coordenadas em x (Nx pontos)
        - y_lin  : tensor 1D de coordenadas em y (Ny pontos)
        - device : 'cpu' ou 'cuda'
        """
        super().__init__(a=R, b=R, m=1, n=n, x_lin=x_lin, y_lin=y_lin, device=device)
        self.R = R

    def generate(self) -> torch.Tensor:
        # 1) Obter o n-ésimo zero de J1' e computar k_1n = zero / R
        zero = BJ1PRIME_ZEROS[self.n - 1]
        k_1n = zero / self.R

        # 2) Construir a grade cartesiana em CPU para SciPy
        Xc, Yc = torch.meshgrid(self.x_lin.cpu(), self.y_lin.cpu(), indexing='ij')  # [Nx, Ny]
        Xc_np = Xc.numpy()
        Yc_np = Yc.numpy()
        Rgrid_np = np.sqrt(Xc_np**2 + Yc_np**2)   # [Nx, Ny]
        Phi_np   = np.arctan2(Yc_np, Xc_np)       # [Nx, Ny]

        # 3) Calcular J1'(k_1n · r) usando J1'(z) = (J0(z) - J2(z)) / 2
        z_np    = k_1n * Rgrid_np
        J0_np   = sp.j0(z_np)      # J0(k_1n·r)
        J2_np   = sp.jn(2, z_np)    # J2(k_1n·r)
        J1p_np  = 0.5 * (J0_np - J2_np)  # [Nx, Ny]

        # 4) Dependência angular cos(φ) para m=1
        ang_np = np.cos(Phi_np)

        # 5) Zerar fora do raio (r > R)
        inside_np = (Rgrid_np <= self.R).astype(np.float32)

        profile_np = J1p_np * ang_np * inside_np  # [Nx, Ny]

        # 6) Transpor para [Ny, Nx] e converter para torch.Tensor
        profile = torch.from_numpy(profile_np).to(self.device).T

        return profile

    def title_str(self) -> str:
        return f"TE₍1{self.n}₎ Profile (Circular)"
