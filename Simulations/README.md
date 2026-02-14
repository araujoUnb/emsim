# Simulations

This folder contains **simulation cases** that are separate from the core simulator package (`emsim`). Each subfolder corresponds to a specific geometry or study (e.g. WR42 waveguide) and includes:

- **config.yaml** – All parameters (domain, frequency, mode, grid, boundaries, run, postprocess, output).
- **run.py** – Script that loads the config, runs the simulation, and executes post-processing.

Outputs (plots and CSV data) are written to a subfolder (e.g. `WR42/outputs/`) as defined in the YAML. The `s_parameters` post-processing step produces both `s_parameters.png` and `s_parameters.csv` (frequency, S11/S21 real/imag, dB, phase).

## Running a simulation

From the **project root**:

```bash
python Simulations/WR42/run.py
```

From inside the case folder:

```bash
cd Simulations/WR42
python run.py
```

## Adding a new simulation

1. Create a new folder, e.g. `Simulations/MyCase/`.
2. Copy `WR42/config.yaml` and adjust domain, frequency, mode, grid, boundaries, run, and postprocess.
3. Copy `WR42/run.py` and point it to your `config.yaml` (or keep the same pattern: config next to `run.py`).

The simulator API is documented in `emsim.simulation.Simulation`.
