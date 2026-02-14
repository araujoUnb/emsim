"""Post-processing utilities for FDTD simulation results.

This package provides plotting and analysis of simulation outputs:

- structure_3d : 3D visualization of the simulation domain and geometry.
- s_parameters_plot : S11/S21 magnitude vs frequency.
- field_snapshots : 2D snapshots of field components (e.g. Ez) over time.
"""

__all__ = [
    "plot_structure_3d",
    "plot_s_parameters",
    "plot_field_snapshots",
]


def __getattr__(name: str):
    """Lazy imports to avoid loading matplotlib on package import."""
    if name == "plot_structure_3d":
        from emsim.postprocessing.structure_3d import plot_structure_3d
        return plot_structure_3d
    if name == "plot_s_parameters":
        from emsim.postprocessing.s_parameters_plot import plot_s_parameters
        return plot_s_parameters
    if name == "plot_field_snapshots":
        from emsim.postprocessing.field_snapshots import plot_field_snapshots
        return plot_field_snapshots
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
