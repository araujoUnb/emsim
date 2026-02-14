#!/usr/bin/env python3
"""
Benchmark: malha uniforme vs não uniforme.

Compara o mesmo problema físico (propagação em z com refinamento local no centro)
usando (A) malha uniforme com resolução fina global e (B) malha não uniforme
com refinamento só na região central. Regista células totais, tempo e células/segundo.
Escreve results.md, results.csv e opcionalmente efficiency_plot.png.
"""

import sys
import time
from pathlib import Path

# Run from project root so emsim is importable
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.solver import FDTDSolver
from emsim.sources.gaussian_pulse import GaussianPulse
from emsim.ports.lumped_port import LumpedPort


def run_simulation(grid, n_steps: int, verbose: bool = False):
    """Run FDTD for n_steps; return (n_cells, elapsed_sec, cells_per_sec)."""
    source = GaussianPulse(f0=10e9, bandwidth=4e9)
    port = LumpedPort(
        name="port1",
        i=grid.Nx // 2,
        j=grid.Ny // 2,
        k=grid.Nz // 2,
        direction="z",
        resistance=50.0,
    )
    solver = FDTDSolver(grid=grid, source=source, ports=[port], n_steps=n_steps)
    n_cells = grid.Nx * grid.Ny * grid.Nz
    t0 = time.perf_counter()
    solver.run(verbose=verbose)
    elapsed = time.perf_counter() - t0
    cells_per_sec = (n_cells * n_steps) / elapsed if elapsed > 0 else 0.0
    return n_cells, elapsed, cells_per_sec


def main():
    out_dir = Path(__file__).resolve().parent
    n_steps = 300  # short run for quick comparison

    # --- Cenário: domínio alongado em z; refinamento no centro ---
    # Física: ~ 30 mm em z, 5 mm em x,y. Queremos boa resolução no centro (z ~ 15 mm).
    Lz = 30e-3
    Lxy = 5e-3
    # Resolução alvo: ~ lambda/15 @ 10 GHz -> ~ 2 mm; em z queremos ~ 0.5 mm no centro.
    lam = 3e8 / 10e9
    dx_fine = lam / 20   # ~ 1.5 mm
    dz_fine = 0.5e-3    # centro
    dz_coarse = 1.5e-3  # extremos

    # (A) Uniforme: mesma resolução em todo o lado (mais células)
    Nx_u = max(4, int(round(Lxy / dx_fine)))
    Ny_u = Nx_u
    Nz_u = max(10, int(round(Lz / dz_fine)))
    grid_uniform = YeeGrid(
        x_range=(0, Lxy),
        y_range=(0, Lxy),
        z_range=(0, Lz),
        dx=dx_fine,
        dy=dx_fine,
        dz=dz_fine,
    )
    # Ajustar Nz para caber em Lz (YeeGrid pode ter snapado)
    n_cells_u, time_u, cps_u = run_simulation(grid_uniform, n_steps)

    # (B) Não uniforme: menos células em z (fino no centro, grosso nas pontas)
    n_center = 20
    n_outer = 15
    dz_arr = [dz_coarse] * n_outer + [dz_fine] * n_center + [dz_coarse] * n_outer
    total_z = sum(dz_arr)
    if abs(total_z - Lz) > 0.1 * Lz:
        # Scale to match Lz
        scale = Lz / total_z
        dz_arr = [d * scale for d in dz_arr]
    grid_nonuniform = YeeGrid(
        x_range=(0, Lxy),
        y_range=(0, Lxy),
        z_range=(0, Lz),
        dx=dx_fine,
        dy=dx_fine,
        dz=dz_arr,
    )
    n_cells_nu, time_nu, cps_nu = run_simulation(grid_nonuniform, n_steps)

    # --- Tabela e ficheiros ---
    rows = [
        ("Config", "Células", "Tempo (s)", "Células/s"),
        ("Uniforme (fino global)", n_cells_u, f"{time_u:.3f}", f"{cps_u:.0f}"),
        ("Não uniforme (fino no centro)", n_cells_nu, f"{time_nu:.3f}", f"{cps_nu:.0f}"),
    ]
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("config,n_cells,time_s,cells_per_sec\n")
        f.write(f"uniform,{n_cells_u},{time_u:.6f},{cps_u:.2f}\n")
        f.write(f"nonuniform,{n_cells_nu},{time_nu:.6f},{cps_nu:.2f}\n")

    md_lines = [
        "# Resultados: uniforme vs não uniforme",
        "",
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join("---" for _ in rows[0]) + " |",
    ]
    for r in rows[1:]:
        md_lines.append("| " + " | ".join(str(x) for x in r) + " |")
    md_lines.extend([
        "",
        "## Conclusão",
        "",
    ])
    if n_cells_nu < n_cells_u and time_nu < time_u:
        gain_cells = (1 - n_cells_nu / n_cells_u) * 100
        gain_time = (1 - time_nu / time_u) * 100
        md_lines.append(f"- A malha não uniforme reduziu as células em **{gain_cells:.0f}%** e o tempo em **{gain_time:.0f}%** para este cenário (refinamento local no centro).")
    elif n_cells_nu < n_cells_u:
        md_lines.append(f"- A malha não uniforme usa **menos células** ({n_cells_nu} vs {n_cells_u}); o tempo pode variar consoante o hardware.")
    else:
        md_lines.append("- Para este cenário pequeno a diferença pode ser marginal. Ganhos maiores esperados em domínios grandes com refinamento local.")
    md_lines.append("")
    md_path = out_dir / "results.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # Plot opcional
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        configs = ["Uniforme", "Não uniforme"]
        cells = [n_cells_u, n_cells_nu]
        times = [time_u, time_nu]
        axes[0].bar(configs, cells, color=["#1f77b4", "#ff7f0e"])
        axes[0].set_ylabel("Células totais")
        axes[0].set_title("Número de células")
        axes[1].bar(configs, times, color=["#1f77b4", "#ff7f0e"])
        axes[1].set_ylabel("Tempo (s)")
        axes[1].set_title("Tempo de execução")
        plt.tight_layout()
        plt.savefig(out_dir / "efficiency_plot.png", dpi=100)
        plt.close()
    except ImportError:
        pass

    print("Resultados em:", md_path)
    print("CSV em:", csv_path)
    for r in rows:
        print("  ".join(str(x) for x in r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
