"""Performance benchmarks for the FDTD solver.

Measures cells/second and optional memory usage. Run with pytest -v.
Documentation in English.
"""

import time
import pytest

from emsim.fdtd.grid import YeeGrid
from emsim.fdtd.solver import FDTDSolver
from emsim.sources.gaussian_pulse import GaussianPulse


def _run_solver_steps(grid, n_steps: int) -> float:
    """Run solver for n_steps and return elapsed time in seconds."""
    source = GaussianPulse(f0=10e9, bandwidth=5e9)
    solver = FDTDSolver(grid=grid, source=source, pml_faces=set(), n_steps=n_steps)
    start = time.perf_counter()
    solver.run(verbose=False)
    return time.perf_counter() - start


@pytest.mark.benchmark
def test_solver_speed_small_grid():
    """Benchmark FDTD solver on a small grid (~10k cells)."""
    grid = YeeGrid(
        x_range=(0, 10e-3),
        y_range=(0, 10e-3),
        z_range=(0, 10e-3),
        f0=10e9,
        resolution=10,
    )
    n_steps = 500
    elapsed = _run_solver_steps(grid, n_steps)
    total_cells = grid.Nx * grid.Ny * grid.Nz
    cells_per_sec = total_cells * n_steps / elapsed
    # Baseline: should be > 10k cells/s (relaxed for CI/slow machines)
    assert cells_per_sec > 1e4, f"Performance too low: {cells_per_sec/1e6:.2f} Mcells/s"


@pytest.mark.benchmark
def test_solver_scaling_with_grid_size():
    """Check that runtime scales roughly linearly with grid size."""
    base_size = (15e-3, 15e-3, 15e-3)
    n_steps = 100
    times = []
    cells_list = []
    for res in (8, 10, 12):
        grid = YeeGrid(
            x_range=(0, base_size[0]),
            y_range=(0, base_size[1]),
            z_range=(0, base_size[2]),
            f0=10e9,
            resolution=res,
        )
        elapsed = _run_solver_steps(grid, n_steps)
        times.append(elapsed)
        cells_list.append(grid.Nx * grid.Ny * grid.Nz)
    # Each doubling of cells should not increase time by more than ~4x (allow some slack)
    for i in range(1, len(times)):
        ratio = times[i] / times[i - 1]
        cell_ratio = cells_list[i] / cells_list[i - 1]
        assert ratio < cell_ratio * 2, "Runtime scaling worse than expected"


@pytest.mark.benchmark
def test_memory_usage():
    """Measure memory delta for a medium grid (optional: requires psutil)."""
    try:
        import psutil
        import os
    except ImportError:
        pytest.skip("psutil not installed")
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024**2
    grid = YeeGrid(
        x_range=(0, 30e-3),
        y_range=(0, 30e-3),
        z_range=(0, 30e-3),
        f0=10e9,
        resolution=15,
    )
    _run_solver_steps(grid, 100)
    mem_after = process.memory_info().rss / 1024**2
    mem_used = mem_after - mem_before
    # Rough upper bound: 6 fields * 4 bytes * cells * 3 (overhead)
    expected_mb = 6 * 4 * grid.Nx * grid.Ny * grid.Nz / 1024**2 * 3
    assert mem_used < expected_mb, f"Memory use {mem_used:.1f} MB exceeds rough bound"
