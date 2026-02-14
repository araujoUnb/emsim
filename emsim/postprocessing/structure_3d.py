"""3D visualization of the simulation domain and geometry.

Delegates to emsim.geometry.viz.plot_geometry() which supports PyVista
and any geometry with to_pyvista() (RectangularWaveguide, PatchAntenna, Box, etc.).
Falls back to matplotlib if backend="matplotlib".
"""

from typing import Optional

from emsim.geometry.viz import plot_geometry


def plot_structure_3d(
    geometry,
    grid: Optional[object] = None,
    save_path: Optional[str] = None,
    show_simulation_box: bool = True,
    backend: str = "pyvista",
) -> None:
    """Plot the geometry and optionally the full simulation domain in 3D.

    Parameters
    ----------
    geometry : RectangularWaveguide | PatchAntenna | Box | Cylinder | Sphere
        Geometry to display (must implement to_pyvista() or bounds()).
    grid : YeeGrid, optional
        If provided and show_simulation_box is True, the simulation box is drawn.
    save_path : str, optional
        If provided, the figure is saved to this path (e.g. PNG).
    show_simulation_box : bool, optional
        If True and grid is provided, draw the simulation domain box (default True).
    backend : str, optional
        "pyvista" (default) or "matplotlib".
    """
    plot_geometry(
        geometry,
        grid=grid if show_simulation_box else None,
        save_path=save_path,
        notebook=False,
        backend=backend,
    )
