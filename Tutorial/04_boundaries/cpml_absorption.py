#!/usr/bin/env python3
"""
Tutorial 04 — Absorção CPML (Convolutional PML).

CPML absorve ondas que saem do domínio, reduzindo reflexões parasitas nas bordas.
Com CPML nas faces z- e z+, o campo decai nessas regiões e a simulação permanece estável.

Execução (a partir da raiz do projeto):
  python Tutorial/04_boundaries/cpml_absorption.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.solver import FDTDSolver
from emsim.sources.gaussian_pulse import GaussianPulse
from emsim.ports.lumped_port import LumpedPort


def main():
    grid = YeeGrid(
        x_range=(0, 5e-3),
        y_range=(0, 5e-3),
        z_range=(0, 40e-3),
        f0=10e9,
        resolution=15,
        courant=0.5,
    )
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    kc, jc, ic = grid.Nz // 2, grid.Ny // 2, grid.Nx // 2
    port = LumpedPort("src", i=ic, j=jc, k=kc, direction="z", resistance=50.0)
    solver = FDTDSolver(
        grid=grid,
        source=source,
        ports=[port],
        pml_faces={"z-", "z+"},
        n_pml=8,
        n_steps=500,
    )
    result = solver.run(verbose=False)

    # Plot Ey along z (at center in x,y) to show decay in PML regions
    jc, ic = grid.Ny // 2, grid.Nx // 2
    Ey_slice = grid.Ey[:, jc, ic].numpy()
    z_mm = (np.arange(grid.Nz) + 0.5) * grid.dz * 1e3  # cell centers in mm

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(z_mm, np.abs(Ey_slice), color="C0")
    ax.set_xlabel("z [mm]")
    ax.set_ylabel("|Ey| [V/m]")
    ax.set_title("Campo |Ey| ao longo de z (após 500 passos): decaimento nas regiões CPML (bordas)")
    ax.axvspan(0, 8 * grid.dz * 1e3, alpha=0.2, color="gray", label="PML z-")
    ax.axvspan(z_mm[-1] - 8 * grid.dz * 1e3, z_mm[-1], alpha=0.2, color="gray", label="PML z+")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)
    plt.savefig(out / "cpml_absorption_ey_z.png", dpi=120)
    plt.close()
    print(f"Guardado: {out / 'cpml_absorption_ey_z.png'}")

    print("\nCom CPML, a onda é absorvida nas bordas (sem reflexão forte); campos permanecem finitos.")


if __name__ == "__main__":
    main()
