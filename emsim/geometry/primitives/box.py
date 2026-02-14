"""Box (cuboid) primitive."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Box:
    """Axis-aligned box. Bounds in m: x_min, x_max, y_min, y_max, z_min, z_max."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def bounds(self):
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

    def to_pyvista(self, **kwargs: Any):
        try:
            import pyvista as pv
        except ImportError as e:
            raise ImportError("PyVista required: pip install pyvista") from e
        return pv.Box(bounds=self.bounds(), **kwargs)
