"""Dispersive material models for FDTD (Drude, Lorentz, Debye).

Factory functions return DispersiveMaterial instances configured for
the corresponding frequency-dependent permittivity model. Used by the
catalog and by add_dispersive_region in MaterialGrid.
"""

from .drude import DrudeMaterial
from .lorentz import LorentzMaterial
from .debye import DebyeMaterial

__all__ = ["DrudeMaterial", "LorentzMaterial", "DebyeMaterial"]
