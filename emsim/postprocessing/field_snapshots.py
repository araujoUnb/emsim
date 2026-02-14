"""Plot 2D field snapshots (e.g. Ez) from FDTD result."""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_field_snapshots(
    result: dict,
    save_path: Optional[str] = None,
    max_snapshots: int = 6,
    snapshot_interval: Optional[int] = None,
) -> None:
    """Plot a sequence of Ez field snapshots (2D slices) from the simulation.

    Parameters
    ----------
    result : dict
        Result dict from FDTDSolver.run() or Simulation.run(), containing
        'Ez_snapshots' (list of 2D arrays). If empty or missing, no figure
        is produced.
    save_path : str, optional
        If provided, save the figure to this path (e.g. PNG).
    max_snapshots : int, optional
        Maximum number of snapshots to show in one row (default 6).
    snapshot_interval : int, optional
        If provided, used in subplot titles to show step index (e.g. "Step 500").
        If None, titles show only the snapshot index.
    """
    snapshots = result.get("Ez_snapshots") or []
    if not snapshots:
        return
    n_snaps = min(len(snapshots), max_snapshots)
    fig, axes = plt.subplots(1, n_snaps, figsize=(3 * n_snaps, 4))
    if n_snaps == 1:
        axes = [axes]
    step = snapshot_interval or 1
    for i, ax in enumerate(axes):
        idx = i * len(snapshots) // n_snaps
        snap = np.asarray(snapshots[idx])
        vmax = max(float(np.abs(snap).max()), 1e-30)
        ax.imshow(snap, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Step {idx * step}")
        ax.set_xlabel("x")
        ax.set_ylabel("z")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
