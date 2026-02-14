"""Rectangular waveguide geometry definition."""

from dataclasses import dataclass


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
