# WR-42 rectangular waveguide

- **config.yaml** – WR-42 dimensions and FDTD parameters (K-band 18–26.5 GHz, TE10). Optional: `device` (auto|cpu|gpu), `run.conv_tol` for early stop, `output.save` list (e.g. `[s_parameters, E]`) to choose which outputs to write (s_parameters, E; H in future).
- **run.py** – Runs the simulation; then run `postprocess.py` for figures (structure 3D, S-parameters plot, field snapshots).
- **outputs/** – Created by run: `result_metadata.csv`, `s_parameters.csv`, and (if `output.save` includes E) `ez_snapshots.csv`. Postprocess adds: `structure_3d.png`, `s_parameters.png`, `field_snapshots.png`.

Run from project root: `python Simulations/WR42/run.py`.
