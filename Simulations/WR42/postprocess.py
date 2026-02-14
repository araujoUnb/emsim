"""Post-process WR-42 simulation results: load saved result and generate figures.

Run this script after run.py has completed. It reads result CSVs (result_metadata.csv,
s_parameters.csv, ez_snapshots.csv) and config.yaml, then generates
structure_3d.png, s_parameters.png, and field_snapshots.png according to the
'postprocess' list in the config. Uses pandas to load CSVs.

Usage
-----
From project root:
    python Simulations/WR42/postprocess.py [output_dir]

If output_dir is omitted, uses config['output']['dir'] from config.yaml
(next to this script), or Simulations/WR42/outputs.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

# Output dir: from argv, or from config
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"


def load_result_from_csv(output_dir: Path) -> dict:
    """Load simulation result from CSV files (pandas). Returns same structure as legacy result dict."""
    output_dir = Path(output_dir)
    result = {"freqs": [], "S11": [], "S21": [], "Ez_snapshots": [], "n_steps_run": None}

    # Metadata
    meta_path = output_dir / "result_metadata.csv"
    if meta_path.is_file():
        meta = pd.read_csv(meta_path)
        if "n_steps_run" in meta.columns:
            result["n_steps_run"] = int(meta["n_steps_run"].iloc[0])

    # S-parameters
    sp_path = output_dir / "s_parameters.csv"
    if not sp_path.is_file():
        raise FileNotFoundError(f"S-parameters CSV not found: {sp_path}. Run the simulation first.")
    sp = pd.read_csv(sp_path)
    result["freqs"] = sp["frequency_Hz"].values.astype(np.float64)
    result["S11"] = (sp["S11_real"].values + 1j * sp["S11_imag"].values).astype(np.complex128)
    result["S21"] = (sp["S21_real"].values + 1j * sp["S21_imag"].values).astype(np.complex128)

    # Ez snapshots (optional)
    ez_path = output_dir / "ez_snapshots.csv"
    if ez_path.is_file():
        ez_df = pd.read_csv(ez_path)
        snapshots = []
        for idx in sorted(ez_df["snapshot_index"].unique()):
            block = ez_df[ez_df["snapshot_index"] == idx]
            i_vals = block["i"].values
            k_vals = block["k"].values
            ni, nk = int(i_vals.max()) + 1, int(k_vals.max()) + 1
            arr = np.zeros((ni, nk), dtype=np.float64)
            arr[i_vals.astype(int), k_vals.astype(int)] = block["Ez"].values
            snapshots.append(arr)
        result["Ez_snapshots"] = snapshots

    return result


def main() -> None:
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        output_dir = Path(config.get("output", {}).get("dir", SCRIPT_DIR / "outputs"))

    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    result = load_result_from_csv(output_dir)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    freq = config.get("frequency", {})
    f_min = freq.get("f_min")
    f_max = freq.get("f_max")
    if f_min is not None:
        f_min = float(f_min)
    if f_max is not None:
        f_max = float(f_max)

    # Rebuild geometry and grid for structure_3d (from config)
    from emsim.simulation import Simulation
    sim = Simulation()
    sim.load_yaml(str(CONFIG_PATH))
    sim.build()

    postprocess_list = config.get("postprocess", [])

    for name in postprocess_list:
        if name == "structure_3d":
            from emsim.postprocessing.structure_3d import plot_structure_3d
            plot_structure_3d(
                sim.geometry,
                sim.grid,
                save_path=str(output_dir / "structure_3d.png"),
            )
            print(f"  Saved {output_dir / 'structure_3d.png'}")
        elif name == "s_parameters":
            from emsim.postprocessing.s_parameters_plot import plot_s_parameters
            plot_s_parameters(
                result,
                f_min=f_min,
                f_max=f_max,
                save_path=str(output_dir / "s_parameters.png"),
            )
            print(f"  Saved s_parameters.png")
        elif name == "field_snapshots":
            from emsim.postprocessing.field_snapshots import plot_field_snapshots
            snap_int = config.get("run", {}).get("snapshot_interval", 1)
            plot_field_snapshots(
                result,
                save_path=str(output_dir / "field_snapshots.png"),
                snapshot_interval=snap_int,
            )
            print(f"  Saved field_snapshots.png")

    print("Post-processing done.")


if __name__ == "__main__":
    main()
