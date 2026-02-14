"""Materials package for the EMSIM FDTD electromagnetic simulator.

Provides material definitions (isotropic, dispersive, anisotropic),
a built-in catalog, CSV/JSON loading, and a central MaterialManager for
applying materials to FDTD grids. All public API is documented in English.

Public API
----------
- Material: isotropic, non-dispersive material (eps_r, mu_r, sigma).
- DispersiveMaterial: frequency-dispersive (Drude, Lorentz, Debye).
- AnisotropicMaterial: 3x3 permittivity/permeability tensors.
- MATERIAL_CATALOG: built-in dict of common materials.
- MaterialManager: central manager; use get_material_manager().
- load_materials_from_csv, load_materials_from_json: load user libraries.
"""

from .base import Material, DispersiveMaterial, AnisotropicMaterial
from .catalog import MATERIAL_CATALOG
from .loader import load_materials_from_csv, load_materials_from_json
from .manager import MaterialManager, get_material_manager

__all__ = [
    "Material",
    "DispersiveMaterial",
    "AnisotropicMaterial",
    "MATERIAL_CATALOG",
    "MaterialManager",
    "get_material_manager",
    "load_materials_from_csv",
    "load_materials_from_json",
]
