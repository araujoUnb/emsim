"""Run patch antenna simulation from YAML config.

This script loads the configuration from config.yaml (in the same directory),
builds the simulation, and runs it. Results are written to the output directory.

Usage
-----
From project root:
    python Simulations/Patch_Antenna/run.py

From Simulations/Patch_Antenna:
    python run.py

Then for figures:
    python Simulations/Patch_Antenna/postprocess.py
"""

from pathlib import Path

from emsim.simulation import Simulation


def main() -> None:
    # Config path: next to this script
    config_path = Path(__file__).resolve().parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Optional: override parameters (e.g. for quick tests)
    # overrides = {"grid.resolution": 15, "run.n_steps": 5000}
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
