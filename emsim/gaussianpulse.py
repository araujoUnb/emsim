# emsim/gaussianpulse.py

import torch
import math
from emsim.source import Source

class GaussianPulse(Source):
    """
    Gaussian‐modulated sinusoidal pulse, vetorizado em PyTorch.

      s(t) = exp[-(t - t0)^2 / (2 τ^2)] * cos(ω0 · t),

    where:
      ω0 = 2π f0,
      τ  = 1 / (2π · bandwidth),
      t0 = 5 · τ.
    """

    def __init__(self, f0: float, bandwidth: float, device: str = 'cpu'):
        """
        Parameters:
        - f0        : center frequency (Hz)
        - bandwidth : bandwidth [Hz]
        - device    : torch.device para operações vetorizadas
        """
        self.f0 = f0
        self.omega0 = 2.0 * math.pi * f0
        self.bandwidth = bandwidth
        self.tau = 1.0 / (2.0 * math.pi * bandwidth)
        self.t0 = 5.0 * self.tau
        self.device = device

    def value(self, t: float) -> float:
        """
        Scalar version: computa s(t) para um float t.
        """
        env = math.exp(-((t - self.t0)**2) / (2.0 * self.tau**2))
        return env * math.cos(self.omega0 * t)

    def vector_value(self, t_tensor: torch.Tensor) -> torch.Tensor:
        """
        Vetorizado: recebe um torch.Tensor de tempos (shape [N]) e retorna
        tensor de amplitudes (shape [N]), tudo em PyTorch no self.device.
        """

        # Garantimos float32 no mesmo device
        t = t_tensor.to(self.device).float()

        # Cálculo vetorizado do envelope: exp[-(t - t0)^2 / (2 τ^2)]
        # Note que self.t0 e self.tau são floats; convertemos para tensor no mesmo device
        t0_tensor = torch.tensor(self.t0, device=self.device, dtype=torch.float32)
        tau_tensor = torch.tensor(self.tau, device=self.device, dtype=torch.float32)

        env = torch.exp(-((t - t0_tensor) ** 2) / (2.0 * tau_tensor**2))

        # Cálculo vetorizado da portadora: cos(ω0 * t)
        # ω0 é float; converter para tensor se quisermos, mas cos aceitará um tensor * float
        carrier = torch.cos(self.omega0 * t)

        return env * carrier
