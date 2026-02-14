"""Unified 3D visualization with PyVista (Jupyter and export)."""

from typing import Any, Optional

from emsim.geometry.waveguide import RectangularWaveguide
from emsim.geometry.antennas import PatchAntenna
from emsim.geometry.primitives import Box, Cylinder, Sphere


def _get_mesh(geom, **kwargs: Any):
    """Return a PyVista DataSet or MultiBlock for the geometry."""
    if hasattr(geom, "to_pyvista"):
        return geom.to_pyvista(**kwargs)
    raise TypeError(f"Geometry {type(geom).__name__} has no to_pyvista() and is not registered.")


def plot_geometry(
    geom,
    grid=None,
    save_path: Optional[str] = None,
    notebook: bool = True,
    backend: str = "pyvista",
    **kwargs: Any,
):
    """Plot geometry (and optionally simulation box) in 3D using PyVista.

    Parameters
    ----------
    geom : RectangularWaveguide | PatchAntenna | Box | Cylinder | Sphere
        Geometry to display (must implement to_pyvista()).
    grid : YeeGrid, optional
        If provided, the simulation domain (grid extent) can be drawn as a wireframe.
    save_path : str, optional
        If set, save the figure to this path (e.g. PNG).
    notebook : bool
        If True, use PyVista's notebook backend for Jupyter (default True).
    backend : str
        "pyvista" (default) or "matplotlib" for fallback.
    **kwargs
        Passed to PyVista Plotter or to geom.to_pyvista().

    Returns
    -------
    pyvista.Plotter or None
        The plotter if backend is pyvista and not showing/saving; None otherwise.
    """
    if backend == "matplotlib":
        return _plot_geometry_matplotlib(geom, grid=grid, save_path=save_path, **kwargs)

    try:
        import pyvista as pv
    except ImportError as e:
        raise ImportError("PyVista is required. Install with: pip install pyvista") from e

    if notebook:
        pv.set_jupyter_backend("trame")  # or "panel", "ipyvtk"
    pl = pv.Plotter(notebook=notebook, **kwargs)

    # Add geometry mesh(es). Box and RectangularWaveguide: semi-transparent to show volume (no wall thickness in model).
    mesh = _get_mesh(geom)
    opacity = 0.45 if isinstance(geom, (RectangularWaveguide, Box)) else 1.0
    if hasattr(mesh, "n_blocks") and mesh.n_blocks > 1:
        for i in range(mesh.n_blocks):
            blk = mesh.get_block(i)
            if blk is not None:
                pl.add_mesh(blk, show_edges=True, opacity=opacity)
    else:
        pl.add_mesh(mesh, show_edges=True, opacity=opacity)

    # Optionally add simulation box from grid
    if grid is not None:
        try:
            bx = (grid.x_min, grid.x_max, grid.y_min, grid.y_max, grid.z_min, grid.z_max)
            box = pv.Box(bounds=bx)
            pl.add_mesh(box, style="wireframe", color="gray", line_width=1)
        except Exception:
            pass

    if save_path:
        pl.screenshot(save_path)
        pl.close()
        return None
    pl.show()
    return None


def _plot_geometry_matplotlib(geom, grid=None, save_path=None, **kwargs):
    """Fallback: plot using matplotlib. Draws the actual shape (box, cylinder, sphere), not just AABB."""
    from pathlib import Path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    def draw_box(ax, x0, x1, y0, y1, z0, z1, color="b"):
        v = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ])
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for i, j in edges:
            ax.plot([v[i, 0], v[j, 0]], [v[i, 1], v[j, 1]], [v[i, 2], v[j, 2]], color=color)

    def draw_cylinder(ax, cyl: Cylinder, color="b", n=24):
        """Wireframe cylinder (axis along z): two circles + vertical segments."""
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cx, cy, r, z0, z1 = cyl.center_x, cyl.center_y, cyl.radius, cyl.z_min, cyl.z_max
        x_bottom = cx + r * np.cos(t)
        y_bottom = cy + r * np.sin(t)
        ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), np.full_like(t, z0), color=color)
        ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), np.full_like(t, z1), color=color)
        for i in range(0, n, max(1, n // 8)):
            ax.plot([x_bottom[i], x_bottom[i]], [y_bottom[i], y_bottom[i]], [z0, z1], color=color)

    def draw_sphere_simple(ax, sph: Sphere, color="b", n=24):
        """Sphere as circles in 3 planes (equator + two meridians)."""
        cx, cy, cz, r = sph.center_x, sph.center_y, sph.center_z, sph.radius
        t = np.linspace(0, 2 * np.pi, n)
        # equator (z = cz)
        ax.plot(cx + r * np.cos(t), cy + r * np.sin(t), np.full_like(t, cz), color=color)
        # circle in xz-plane (y = cy)
        ax.plot(cx + r * np.cos(t), np.full_like(t, cy), cz + r * np.sin(t), color=color)
        # circle in yz-plane (x = cx)
        ax.plot(np.full_like(t, cx), cy + r * np.cos(t), cz + r * np.sin(t), color=color)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    color = "blue"

    if isinstance(geom, Cylinder):
        draw_cylinder(ax, geom, color=color)
        b = geom.bounds()
    elif isinstance(geom, Sphere):
        draw_sphere_simple(ax, geom, color=color)
        b = geom.bounds()
    else:
        # Box, RectangularWaveguide, PatchAntenna (or any with bounds)
        b = geom.bounds()
        draw_box(ax, b[0], b[1], b[2], b[3], b[4], b[5], color=color)

    if grid is not None:
        draw_box(ax, grid.x_min, grid.x_max, grid.y_min, grid.y_max, grid.z_min, grid.z_max, color="gray")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_xlim(b[0], b[1])
    ax.set_ylim(b[2], b[3])
    ax.set_zlim(b[4], b[5])
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
    return None
