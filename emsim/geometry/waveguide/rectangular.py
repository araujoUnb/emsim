"""Rectangular waveguide geometry definition."""

from dataclasses import dataclass
from typing import Any


@dataclass
class RectangularWaveguide:
    """A hollow rectangular waveguide with PEC walls.

    Parameters
    ----------
    a      : float  width  (x-direction) [m]
    b      : float  height (y-direction) [m]
    length : float  length (z-direction) [m]
    """
    a: float
    b: float
    length: float

    @property
    def x_range(self):
        return (0.0, self.a)

    @property
    def y_range(self):
        return (0.0, self.b)

    @property
    def z_range(self):
        return (0.0, self.length)

    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max, z_min, z_max) [m]."""
        return (0.0, self.a, 0.0, self.b, 0.0, self.length)

    def to_pyvista(self, **kwargs: Any):
        """Return a PyVista box mesh for the waveguide volume."""
        try:
            import pyvista as pv
        except ImportError as e:
            raise ImportError("PyVista is required for to_pyvista(). Install with: pip install pyvista") from e
        return pv.Box(
            bounds=(0.0, self.a, 0.0, self.b, 0.0, self.length),
            **kwargs,
        )
