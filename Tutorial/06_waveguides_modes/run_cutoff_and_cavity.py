#!/usr/bin/env python3
"""Run cutoff_and_cavity notebook logic and save figures. Execute from project root."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.materials import MaterialGrid
from emsim.fdtd.fields import update_E, update_H
from emsim.boundaries.pec import apply_pec
from emsim.sources.gaussian_pulse import GaussianPulse
from Tutorial.common.theory import cavity_frequency, te10_cutoff, te_impedance

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Cavity
a, b, d = 10e-3, 8e-3, 6e-3
grid = YeeGrid(x_range=(0, a), y_range=(0, b), z_range=(0, d), f0=20e9, resolution=20, courant=0.5)
mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0, mu_r=1.0, sigma=0.0)
mat.compute_coefficients(grid.dt)

Ex = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
Ey = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
Ez = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
Hx = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
Hy = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
Hz = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))

source = GaussianPulse(f0=20e9, bandwidth=15e9)
ic, jc, kc = grid.Nz // 2, grid.Ny // 2, grid.Nx // 2
Ez_record = []
coeffs = grid.get_curl_coefficients()
inv_dx, inv_dy, inv_dz = coeffs["inv_dx"], coeffs["inv_dy"], coeffs["inv_dz"]
n_steps = 3000
for n in range(n_steps):
    update_H(Ex, Ey, Ez, Hx, Hy, Hz, mat.dt_over_mu, inv_dx, inv_dy, inv_dz)
    update_E(Ex, Ey, Ez, Hx, Hy, Hz, mat.Ca, mat.Cb, inv_dx, inv_dy, inv_dz)
    apply_pec(Ex, Ey, Ez, {"x-", "x+", "y-", "y+", "z-", "z+"})
    if n < 100:
        amp = float(source(n * grid.dt).numpy())
        idx = tf.constant([[ic, jc, kc]], dtype=tf.int32)
        new_val = Ez[ic, jc, kc].numpy() + amp
        Ez.assign(tf.tensor_scatter_nd_update(Ez.read_value(), idx, tf.constant([new_val], dtype=Ez.dtype)))
    Ez_record.append(Ez[ic, jc, kc].numpy())

Ez_record = np.array(Ez_record)
N = len(Ez_record)
fft_vals = fft(Ez_record)
freqs = fftfreq(N, grid.dt)
pos_mask = freqs > 0
freqs_pos = freqs[pos_mask]
fft_mag = np.abs(fft_vals[pos_mask])

modes = [(1, 0, 1), (1, 1, 0), (0, 1, 1), (2, 0, 1), (1, 1, 1)]
f_ana = [cavity_frequency(m, n, p, a, b, d) for m, n, p in modes]

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.plot(freqs_pos / 1e9, fft_mag, label="Simulação (FFT Ez centro)")
for (m, n, p), f in zip(modes, f_ana):
    ax.axvline(f / 1e9, color="red", alpha=0.7, linestyle="--")
ax.set_xlabel("Frequência [GHz]")
ax.set_ylabel("|FFT|")
ax.set_title("Cavidade: picos de ressonância vs frequências teóricas f_mnp (linhas vermelhas)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 30)
plt.tight_layout()
plt.savefig(FIG_DIR / "cavity_resonance_vs_theory.png", dpi=120)
plt.close()
print("Guardado:", FIG_DIR / "cavity_resonance_vs_theory.png")

# Waveguide TE10 and Z_TE
a_wr42, b_wr42 = 10.67e-3, 4.32e-3
fc_te10 = te10_cutoff(a_wr42)
f_ghz = np.linspace(12, 26, 200) * 1e9
Z_te = np.array([te_impedance(1, 0, a_wr42, b_wr42, f) for f in f_ghz])
Z_te_real = np.real(Z_te)

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.axvline(fc_te10 / 1e9, color="red", linestyle="--", label=f"fc TE10 = {fc_te10/1e9:.2f} GHz")
ax.plot(f_ghz / 1e9, Z_te_real, label="Z_TE (teoria)")
ax.set_xlabel("Frequência [GHz]")
ax.set_ylabel("Z_TE [Ohm]")
ax.set_title("Impedância TE10 e frequência de corte (guia WR42)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(f_ghz[0]/1e9, f_ghz[-1]/1e9)
plt.tight_layout()
plt.savefig(FIG_DIR / "waveguide_fc_and_ZTE.png", dpi=120)
plt.close()
print("Guardado:", FIG_DIR / "waveguide_fc_and_ZTE.png")
