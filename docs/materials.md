# Materials System

The EMSIM FDTD simulator provides a unified materials system: built-in catalog, user libraries (CSV/JSON), and runtime custom materials. You can assign isotropic, dispersive (Drude, Lorentz, Debye), and anisotropic materials to regions of the grid.

## Overview

- **Material**: Isotropic, non-dispersive (ε_r, μ_r, σ).
- **DispersiveMaterial**: Frequency-dependent permittivity (Drude for metals/plasmas, Lorentz for resonances, Debye for relaxation). The solver uses auxiliary differential equations (ADE) with polarization **P** and current **J**.
- **AnisotropicMaterial**: Diagonal (or full) 3×3 permittivity tensor for uniaxial crystals, liquid crystals, etc.

## Catalog and Manager

Use the global material manager to resolve materials by name and apply them to the grid:

```python
from emsim.materials import get_material_manager

mgr = get_material_manager()
mat = mgr.get("rogers_ro4003c")   # by key (normalized name)
mat = mgr.get("Copper Drude")     # name is normalized to key

mgr.list_all()                    # all keys
mgr.list_by_category("dielectric")
mgr.search("rogers")
```

Built-in catalog keys include: `vacuum`, `air`, `fr4`, `rogers_ro4003c`, `rt5880`, `alumina`, `ptfe`, `silicon`, `gaas`, `copper_dc`, `copper_drude`, `gold`, `aluminum`, `silver`, `water`, `muscle`, `skin_dry`, `lossy_test`.

## Loading Custom Libraries

CSV (header: name, eps_r, mu_r, sigma, category, description; optional columns for dispersive/anisotropic):

```python
from emsim.materials import load_materials_from_csv, get_material_manager

materials = load_materials_from_csv("materials_library.csv")
mgr = get_material_manager()
mgr.load_csv("materials_library.csv")  # registers into manager
```

JSON: same keys as catalog entries; use `load_materials_from_json(path)` or `mgr.load_json(path)`.

## Applying Materials to the Grid

**Isotropic (and fallback for dispersive/anisotropic if not yet implemented in grid):**

```python
mgr.apply_to_grid(
    grid,
    region={"i": (10, 40), "j": (10, 40), "k": (0, 5)},
    material_name="rogers_ro4003c",
)
grid.materials.compute_coefficients(grid.dt)
```

**Dispersive (Drude):** `apply_to_grid` with a material name that resolves to a `DispersiveMaterial` with `model="drude"` will call `grid.materials.add_dispersive_region(...)`. The solver then uses the Drude ADE update in those cells.

**Anisotropic:** Pass an `AnisotropicMaterial` (or a name that resolves to one). The grid’s `add_anisotropic_region` is used; the solver applies a diagonal ε⁻¹ update in those cells.

## YAML Configuration

In your simulation config you can define:

```yaml
materials:
  library: "materials_library.csv"   # optional
  regions:
    - name: substrate
      material: rogers_ro4003c
      geometry:
        i_range: [10, 40]
        j_range: [10, 40]
        k_range: [0, 5]
    - name: ground
      material: copper_drude
      geometry:
        i_range: [10, 40]
        j_range: [10, 40]
        k_range: [5, 6]
    - name: anisotropic_layer
      material:
        type: anisotropic
        name: "Liquid Crystal"
        eps_r_xx: 3.0
        eps_r_yy: 3.0
        eps_r_zz: 2.0
      geometry:
        k_range: [10, 12]
```

After building the grid, the simulation loads the optional library and applies each region via the material manager; `compute_coefficients(dt)` is called after material changes.

## Dispersive Models

- **Drude**: ε(ω) = ε_∞ − ω_p²/(ω² + i γ ω). Parameters: `eps_inf`, `omega_p`, `gamma` (rad/s). Used for metals and plasmas.
- **Lorentz**: Multi-pole resonance. Parameters: `eps_inf`, `delta_eps`, `omega_0`, `delta` (lists for multiple poles).
- **Debye**: Relaxation. Parameters: `eps_s`, `eps_inf`, `tau`.

References: Taflove & Hagness, *Computational Electrodynamics*, 3rd ed., Ch. 9.
