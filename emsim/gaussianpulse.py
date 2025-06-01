# emsim/gaussianpulse.py

import torch
import math
from emsim.source import Source

class GaussianPulse(Source):
    """
    Gaussian‐modulated sinusoidal pulse, vectorized in PyTorch.

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
        - device    : torch.device for all tensor operations
        """
        super().__init__()   # If your Source base class has an __init__, call it
        self.f0 = f0
        self.omega0 = 2.0 * math.pi * f0
        self.bandwidth = bandwidth
        self.tau = 1.0 / (2.0 * math.pi * bandwidth)
        self.t0 = 5.0 * self.tau
        self.device = device

    def value(self, t: float) -> float:
        """
        Scalar version: compute s(t) for a Python float t.
        """
        env = math.exp(-((t - self.t0) ** 2) / (2.0 * self.tau**2))
        return env * math.cos(self.omega0 * t)

    def vector_value(self, t_tensor: torch.Tensor) -> torch.Tensor:
        """
        Vectorized: given a 1D torch.Tensor of times (shape [N]) returns
        a torch.Tensor of amplitudes (shape [N]), all on self.device.
        """
        # Ensure float32 on the correct device
        t = t_tensor.to(self.device).float()

        # Build tensors for t0 and tau
        t0_tensor = torch.tensor(self.t0, device=self.device, dtype=torch.float32)
        tau_tensor = torch.tensor(self.tau, device=self.device, dtype=torch.float32)

        # Envelope: exp[-(t - t0)^2 / (2 τ^2)]
        env = torch.exp(-((t - t0_tensor) ** 2) / (2.0 * tau_tensor**2))

        # Carrier: cos(ω0 · t)
        # (We can multiply a float ω0 by a tensor; PyTorch handles it automatically.)
        carrier = torch.cos(self.omega0 * t)

        return env * carrier
