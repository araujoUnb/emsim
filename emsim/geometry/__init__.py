"""Geometry package: waveguide, antennas, primitives, and visualization."""

from emsim.geometry.waveguide import RectangularWaveguide
from emsim.geometry.antennas import PatchAntenna
from emsim.geometry.primitives import Box, Cylinder, Sphere

from emsim.geometry.viz import plot_geometry

__all__ = [
    "RectangularWaveguide",
    "PatchAntenna",
    "Box",
    "Cylinder",
    "Sphere",
    "plot_geometry",
]
