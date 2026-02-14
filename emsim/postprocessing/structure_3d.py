"""3D visualization of the simulation domain and waveguide structure.

Uses matplotlib's Axes3D to draw the waveguide box (and optionally the
full simulation box) so the user can inspect the geometry before or after
a run.
"""

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # noqa: E402 -- must set backend before pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402

from emsim.geometry.rectangular_waveguide import RectangularWaveguide


def _draw_box_edges(
    ax: Axes3D,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    color: str = "b",
    linewidth: float = 1.5,
    label: Optional[str] = None,
) -> None:
    """Draw the 12 edges of a rectangular box in 3D (wireframe)."""
    v = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min],
        [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max],
        [x_max, y_max, z_max], [x_min, y_max, z_max],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        use_label = label if (i, j) == (0, 1) else None
        ax.plot(
            [v[i, 0], v[j, 0]],
            [v[i, 1], v[j, 1]],
            [v[i, 2], v[j, 2]],
            color=color,
            linewidth=linewidth,
            label=use_label,
        )


def plot_structure_3d(
    geometry: RectangularWaveguide,
    grid: Optional[object] = None,
    save_path: Optional[str] = None,
    show_simulation_box: bool = True,
) -> None:
    """Plot the waveguide and optionally the full simulation domain in 3D.

    Parameters
    ----------
    geometry : RectangularWaveguide
        Waveguide geometry (a, b, length) in metres.
    grid : YeeGrid, optional
        If provided and show_simulation_box is True, the full simulation
        box (grid extent including PML) is drawn in a different color.
    save_path : str, optional
        If provided, the figure is saved to this path (e.g. PNG).
    show_simulation_box : bool, optional
        If True and grid is provided, draw the simulation domain box
        (default True).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Waveguide box (structure): from 0 to a, 0 to b, 0 to length
    x0, x1 = 0.0, geometry.a
    y0, y1 = 0.0, geometry.b
    z0, z1 = 0.0, geometry.length
    _draw_box_edges(ax, x0, x1, y0, y1, z0, z1, color="blue", linewidth=2.0, label="Waveguide")

    if show_simulation_box and grid is not None:
        gx0, gx1 = grid.x_min, grid.x_max
        gy0, gy1 = grid.y_min, grid.y_max
        gz0, gz1 = grid.z_min, grid.z_max
        _draw_box_edges(
            ax, gx0, gx1, gy0, gy1, gz0, gz1,
            color="gray", linewidth=1.0, label="Simulation box",
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D structure: waveguide and simulation domain")
    ax.set_xlim([0, geometry.a])
    ax.set_ylim([0, geometry.b])
    ax.set_zlim([0, geometry.length])
    if hasattr(ax, "legend"):
        ax.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
