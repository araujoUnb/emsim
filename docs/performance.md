# Performance and Benchmarks

This document describes how to run benchmarks, interpret results (cells/second, memory), and optional tuning (XLA, mixed precision).

## Running Benchmarks

Benchmark tests live under `tests/performance/`. Run them with pytest:

```bash
pytest tests/performance/ -v
```

- **test_solver_speed_small_grid**: Runs a small grid for 500 steps and reports cells/second. Baseline: > 10k cells/s (relaxed for CI).
- **test_solver_scaling_with_grid_size**: Checks that runtime scales roughly with grid size.
- **test_memory_usage**: Measures RSS delta for a medium grid (requires `psutil`).

## Interpreting Results

- **Cells per second** = (Nx × Ny × Nz) × n_steps / elapsed_seconds. Higher is better. Typical range: 10k–10M+ depending on hardware and grid size.
- **Memory**: Expect on the order of 6 field arrays × 4 bytes × cell count, plus solver/CPML overhead. The memory test compares used RSS to a rough upper bound.

## Device and Optional Tuning

Use `emsim.device.setup_device()` before creating the grid/solver:

```python
from emsim.device import setup_device

# GPU if available, else CPU
setup_device("auto")

# Optional: XLA JIT (can speed up GPU/CPU)
setup_device("auto", enable_xla=True)

# Optional: mixed precision on GPU (higher throughput, slight numerical difference)
setup_device("auto", enable_xla=True, mixed_precision=True)
```

- **enable_xla**: Turns on TensorFlow XLA JIT; often gives 2–5× speedup when applicable.
- **mixed_precision**: Sets Keras policy to `mixed_float16` on GPU for faster ops; use with care for accuracy-sensitive runs.

## Non-uniform (stretched) grid efficiency

When only part of the domain needs fine resolution, a **non-uniform grid** (variable dx, dy, dz per direction) can reduce total cell count and runtime compared to a globally fine uniform grid. To evaluate gains for your setup:

- Run the benchmark and inspect tables/plots in **Tutorial/02_grid/**:
  ```bash
  python Tutorial/02_grid/run_efficiency_benchmark.py
  ```
- See `Tutorial/02_grid/results.md` and `results.csv` for metrics (cells, time, cells/s) comparing uniform vs non-uniform for a representative scenario (local refinement in the centre). Use the same idea to add your own scenarios and plots.

Expect noticeable gain when a small fraction of the domain requires fine resolution; gain is marginal when most of the domain is already fine.

## Implementation Notes

- **@tf.function**: The core field updates `update_H` and `update_E` in `emsim.fdtd.fields` are compiled with `@tf.function(reduce_retracing=True)` to reduce Python overhead.
- **set_region**: Material grid updates use tensor indices (e.g. `tf.meshgrid` + `tensor_scatter_nd_update`) instead of Python loops for large regions.
- **Port recording**: Modal and lumped ports record into `tf.TensorArray`; conversion to NumPy happens only at the end when building S-parameters or results.
