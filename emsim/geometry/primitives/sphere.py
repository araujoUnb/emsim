"""Sphere primitive."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Sphere:
    """Sphere.

    Parameters
    ----------
    center_x, center_y, center_z : float
        Center [m].
    radius : float
        Radius [m].
    """
    center_x: float
    center_y: float
    center_z: float
    radius: float

    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max, z_min, z_max) [m] (AABB)."""
        r = self.radius
        return (
            self.center_x - r, self.center_x + r,
            self.center_y - r, self.center_y + r,
            self.center_z - r, self.center_z + r,
        )

    def to_pyvista(self, **kwargs: Any):
        """Return a PyVista sphere mesh."""
        try:
            import pyvista as pv
        except ImportError as e:
            raise ImportError("PyVista is required for to_pyvista(). Install with: pip install pyvista") from e
        return pv.Sphere(
            center=(self.center_x, self.center_y, self.center_z),
            radius=self.radius,
            **kwargs,
        )
