"""Run WR-42 rectangular waveguide simulation from YAML config.

This script loads the configuration from config.yaml (in the same directory),
builds the simulation, and runs it. Results (CSV: result_metadata, s_parameters, optional ez_snapshots) and a run log
(run_info.yaml) are written to the output
directory. To generate figures (structure 3D, S-parameters, field snapshots),
run postprocess.py after the simulation finishes.

Usage
-----
From project root:
    python Simulations/WR42/run.py

From Simulations/WR42:
    python run.py

Then for figures:
    python Simulations/WR42/postprocess.py
"""

from pathlib import Path

from emsim.simulation import Simulation


def main() -> None:
    # Config path: next to this script
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Optional: override parameters (e.g. for quick tests)
    # overrides = {"grid.resolution": 30, "run.n_steps": 4000}
    overrides = None

    sim = Simulation.from_yaml(str(config_path), overrides=overrides)
    print("Configuration loaded. Building grid and solver...")
    sim.build()
    print(f"Grid: {sim.grid}")
    print(f"Total cells: {sim.grid.Nx * sim.grid.Ny * sim.grid.Nz:,}")
    print("Running simulation...")
    sim.run(verbose=True)
    print("Simulation finished. Run postprocess.py for figures.")


if __name__ == "__main__":
    main()
