#!/usr/bin/env python3
"""Run dielectric notebook logic and save figure. Execute from project root."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.fields import update_E, update_H
from emsim.sources.gaussian_pulse import GaussianPulse
from Tutorial.common.theory import measure_wave_speed

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

def run_and_measure_speed(eps_r, n_steps=2000):
    grid = YeeGrid(
        x_range=(0, 5e-3), y_range=(0, 5e-3), z_range=(0, 60e-3),
        f0=10e9, resolution=40, courant=0.5, eps_r=eps_r, mu_r=1.0, sigma=0.0,
    )
    z_src, z1, z2 = 5, 15, 35
    distance = (z2 - z1) * grid.dz
    source = GaussianPulse(f0=10e9, bandwidth=4e9)
    mat = grid.materials
    coeffs = grid.get_curl_coefficients()
    inv_dx, inv_dy, inv_dz = coeffs["inv_dx"], coeffs["inv_dy"], coeffs["inv_dz"]
    Ey1, Ey2 = [], []
    for n in range(n_steps):
        update_H(grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz,
                 mat.dt_over_mu, inv_dx, inv_dy, inv_dz)
        update_E(grid.Ex, grid.Ey, grid.Ez, grid.Hx, grid.Hy, grid.Hz,
                 mat.Ca, mat.Cb, inv_dx, inv_dy, inv_dz)
        amp = float(source(n * grid.dt).numpy())
        idx = tf.constant([[z_src, grid.Ny//2, grid.Nx//2]], dtype=tf.int32)
        new_val = grid.Ey[z_src, grid.Ny//2, grid.Nx//2].numpy() + amp
        grid.Ey.assign(tf.tensor_scatter_nd_update(grid.Ey.read_value(), idx, tf.constant([new_val], dtype=grid.Ey.dtype)))
        Ey1.append(grid.Ey[z1, grid.Ny//2, grid.Nx//2].numpy())
        Ey2.append(grid.Ey[z2, grid.Ny//2, grid.Nx//2].numpy())
    return measure_wave_speed(Ey1, Ey2, distance, grid.dt)

v_vac = run_and_measure_speed(1.0)
v_diel = run_and_measure_speed(2.0)
eps_r = 2.0
ratio_theory = 1.0 / np.sqrt(eps_r)
ratio_sim = v_diel / v_vac

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
ax.bar(["Teoria (1/sqrt(er))", "Simulacao (v_diel/v_vac)"], [ratio_theory, ratio_sim], color=["C0", "C1"])
ax.set_ylabel("Razao")
ax.set_title("Dielectrico eps_r=2: razao velocidade")
plt.tight_layout()
plt.savefig(FIG_DIR / "dielectric_ratio_comparison.png", dpi=120)
plt.close()
print("Guardado:", FIG_DIR / "dielectric_ratio_comparison.png")
