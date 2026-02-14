#!/usr/bin/env python3
"""Tutorial 04: Reflexão PEC. Teoria: Gamma = -1. Executar da raiz: python Tutorial/04_boundaries/pec_reflection.py"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.materials import MaterialGrid
from emsim.fdtd.fields import update_E, update_H
from emsim.boundaries.pec import apply_pec
from emsim.sources.gaussian_pulse import GaussianPulse
from Tutorial.common.theory import pec_reflection_coefficient

grid = YeeGrid(
    x_range=(0, 3e-3), y_range=(0, 3e-3), z_range=(0, 40e-3),
    f0=10e9, resolution=40, courant=0.5,
)
mat = MaterialGrid(grid.Nz, grid.Ny, grid.Nx, eps_r=1.0, mu_r=1.0, sigma=0.0)
mat.compute_coefficients(grid.dt)

Ex = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
Ey = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
Ez = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
Hx = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
Hy = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))
Hz = tf.Variable(tf.zeros([grid.Nz, grid.Ny, grid.Nx], dtype=tf.float32))

z_src, z_before, z_near = 15, 10, 8
source = GaussianPulse(f0=10e9, bandwidth=4e9)
Ey_before_list, Ey_near_list = [], []

coeffs = grid.get_curl_coefficients()
inv_dx, inv_dy, inv_dz = coeffs["inv_dx"], coeffs["inv_dy"], coeffs["inv_dz"]
for n in range(800):
    update_H(Ex, Ey, Ez, Hx, Hy, Hz, mat.dt_over_mu, inv_dx, inv_dy, inv_dz)
    update_E(Ex, Ey, Ez, Hx, Hy, Hz, mat.Ca, mat.Cb, inv_dx, inv_dy, inv_dz)
    apply_pec(Ex, Ey, Ez, {"z-"})
    amp = float(source(n * grid.dt).numpy())
    idx = tf.constant([[z_src, grid.Ny // 2, grid.Nx // 2]], dtype=tf.int32)
    new_val = Ey[z_src, grid.Ny // 2, grid.Nx // 2].numpy() + amp
    Ey.assign(tf.tensor_scatter_nd_update(Ey.read_value(), idx, tf.constant([new_val], dtype=Ey.dtype)))
    Ey_before_list.append(Ey[z_before, grid.Ny // 2, grid.Nx // 2].numpy())
    Ey_near_list.append(Ey[z_near, grid.Ny // 2, grid.Nx // 2].numpy())

Ey_before = np.array(Ey_before_list)
Ey_near = np.array(Ey_near_list)
t_ns = np.arange(800) * grid.dt * 1e9

fig, ax = plt.subplots(1, 1, figsize=(9, 4))
ax.plot(t_ns, Ey_before, label="Ey antes")
ax.plot(t_ns, Ey_near, label="Ey perto PEC")
ax.set_xlabel("Tempo [ns]")
ax.set_ylabel("Ey [V/m]")
ax.legend()
ax.set_title("Reflexao PEC: onda estacionaria perto do PEC")
ax.grid(True, alpha=0.3)
plt.tight_layout()
out_dir = Path(__file__).resolve().parent / "figures"
out_dir.mkdir(exist_ok=True)
plt.savefig(out_dir / "pec_reflection_ey_t.png", dpi=120)
plt.close()
print("Guardado:", out_dir / "pec_reflection_ey_t.png")

gamma_theory = pec_reflection_coefficient()
print("Teoria: Gamma =", gamma_theory)
print("Simulacao: amplitudes antes/perto PEC =", np.max(np.abs(Ey_before)), np.max(np.abs(Ey_near)))
