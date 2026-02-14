"""Cylinder primitive (axis along z)."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Cylinder:
    """Cylinder with axis along z.

    Parameters
    ----------
    center_x, center_y : float
        Center of the cylinder in the xy-plane [m].
    radius : float
        Radius [m].
    z_min, z_max : float
        Extent along z [m].
    """
    center_x: float
    center_y: float
    radius: float
    z_min: float
    z_max: float

    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max, z_min, z_max) [m] (AABB)."""
        r = self.radius
        return (
            self.center_x - r, self.center_x + r,
            self.center_y - r, self.center_y + r,
            self.z_min, self.z_max,
        )

    def to_pyvista(self, **kwargs: Any):
        """Return a PyVista cylinder mesh (axis along z)."""
        try:
            import pyvista as pv
        except ImportError as e:
            raise ImportError("PyVista is required for to_pyvista(). Install with: pip install pyvista") from e
        height = self.z_max - self.z_min
        center = (self.center_x, self.center_y, (self.z_min + self.z_max) / 2)
        cyl = pv.Cylinder(center=center, radius=self.radius, height=height, direction=(0, 0, 1), **kwargs)
        return cyl
