import torch
import numpy as np
import matplotlib.pyplot as plt

from emsim.gaussianpulse import GaussianPulse

# 1) Cria instância de GaussianPulse em PyTorch (device='cpu' ou 'cuda')
device = 'cpu'
f0 = 24e9       # 24 GHz
bandwidth = 2e9 # 2 GHz
pulse = GaussianPulse(f0=f0, bandwidth=bandwidth, device=device)

# 2) Define intervalo de tempo baseado em t0
t0 = pulse.t0         # float
t_max = 2.0 * t0      # cobrindo todo o pulso
dt = pulse.tau / 50.0 # passo de tempo

# 3) Amostragem vetorizada em PyTorch
t_vals, amp_vals = pulse.sample_torch(t_start=0.0, t_end=t_max, dt=dt, device=device)

# Converte para NumPy para plotar
t_np = t_vals.cpu().numpy()
amp_np = amp_vals.cpu().numpy()

# 4) Plot no domínio do tempo
plt.figure(figsize=(6,3))
plt.plot(t_np * 1e9, amp_np, '-b')  # multiplicar tempo por 1e9 para exibir em ns
plt.xlabel('Time (ns)')
plt.ylabel('Amplitude')
plt.title('GaussianPulse Time‐Domain (Torch)')
plt.grid(True)
plt.tight_layout()

# 5) Calcular FFT usando torch.fft.rfft
spectrum = torch.fft.rfft(amp_vals)
freq_axis = torch.fft.rfftfreq(len(amp_vals), dt, device=device)

# Converte para NumPy
freq_np = freq_axis.cpu().numpy()
spec_np = spectrum.abs().cpu().numpy()

# 6) Plot no domínio da frequência
plt.figure(figsize=(6,3))
plt.plot(freq_np / 1e9, spec_np, '-r')  # dividir por 1e9 para exibir em GHz
plt.xlabel('Frequency (GHz)')
plt.ylabel('Magnitude')
plt.title('GaussianPulse Frequency Spectrum (Torch)')
plt.grid(True)
plt.tight_layout()

plt.show()
