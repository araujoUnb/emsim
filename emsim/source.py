# emsim/source.py

import abc
import torch
import numpy as np
import matplotlib.pyplot as plt

class Source(abc.ABC):
    """
    Abstract base class for a time‐dependent excitation ("source") in FDTD.
    This version usa PyTorch internamente para gerar vetores de tempo/frequência.
    Subclasses devem implementar value(t) e, opcionalmente, vector_value(tensor_t).
    """

    @abc.abstractmethod
    def value(self, t: float) -> float:
        """
        Return the instantaneous amplitude of the source at a single time t (float, em segundos).
        """
        raise NotImplementedError("Subclasses must implement value(t).")

    def vector_value(self, t_tensor: torch.Tensor) -> torch.Tensor:
        """
        Optional: returns tensor de amplitudes para cada t em t_tensor. Por padrão,
        faz loop sobre .value(t) para cada elemento (menos eficiente), mas subclasses
        podem sobrepor para operações completamente vetorizadas.

        Input:
          t_tensor: 1D torch.Tensor (shape [N]) contendo instantes de tempo (em segundos).
        Output:
          1D torch.Tensor shape [N] com amplitudes correspondentes.
        """
        # Implementação genérica (sem vetorização): itera e chama value()
        vals = [self.value(float(t.cpu().numpy())) for t in t_tensor]
        return torch.tensor(vals, dtype=torch.float32, device=t_tensor.device)

    def sample_torch(self, t_start: float, t_end: float, dt: float, device: str = 'cpu'):
        """
        Gera um tensor de tempos e amplitudes com passo dt usando PyTorch:
          t_vals = torch.arange(t_start, t_end, dt, device=device)
          amp_vals = self.vector_value(t_vals)

        Returns:
          t_vals (torch.Tensor), amp_vals (torch.Tensor)
        """
        # Usamos torch.arange para criar o vetor de tempos
        t_vals = torch.arange(t_start, t_end, dt, device=device, dtype=torch.float32)
        try:
            amp_vals = self.vector_value(t_vals)
        except NotImplementedError:
            # fallback iterativo
            amp_list = []
            for t in t_vals:
                amp_list.append(self.value(float(t.cpu().numpy())))
            amp_vals = torch.tensor(amp_list, dtype=torch.float32, device=device)
        return t_vals, amp_vals

    def plot_time(self, t_start: float, t_end: float, dt: float,
                  device: str = 'cpu',
                  xlabel: str = "Time (s)", ylabel: str = "Amplitude",
                  title: str = "Source Time‐Domain"):
        """
        Plota o sinal no domínio do tempo. Usa torch para amostrar e depois converte para NumPy.
        """
        t_vals, amp_vals = self.sample_torch(t_start, t_end, dt, device=device)
        # Converte para NumPy para plotar
        t_np = t_vals.cpu().numpy()
        amp_np = amp_vals.cpu().numpy()

        plt.figure(figsize=(6,3))
        plt.plot(t_np, amp_np, '-b')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_spectrum(self, t_start: float, t_end: float, dt: float,
                      device: str = 'cpu',
                      xlabel: str = "Frequency (Hz)", ylabel: str = "Magnitude",
                      title: str = "Source Frequency Spectrum"):
        """
        Plota a magnitude do espectro (FFT) do sinal amostrado. Usa torch.fft.rfft.
        """
        t_vals, amp_vals = self.sample_torch(t_start, t_end, dt, device=device)
        N = amp_vals.shape[0]

        # Compute real‐to‐complex FFT
        spectrum = torch.fft.rfft(amp_vals)
        freq_axis = torch.fft.rfftfreq(N, dt, device=device)

        # Converte para NumPy
        freq_np = freq_axis.cpu().numpy()
        spec_np = spectrum.abs().cpu().numpy()

        plt.figure(figsize=(6,3))
        plt.plot(freq_np, spec_np, '-r')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()
