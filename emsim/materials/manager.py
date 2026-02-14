"""Central manager for electromagnetic materials used in FDTD simulations.

MaterialManager provides a single entry point to built-in catalog, user-loaded
libraries (CSV/JSON), and custom materials defined at runtime. Use
get_material_manager() to obtain the global singleton instance. All
documentation is in English per project standards.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path

from .base import Material, DispersiveMaterial, AnisotropicMaterial
from .catalog import MATERIAL_CATALOG
from .loader import load_materials_from_csv, load_materials_from_json


class MaterialManager:
    """Manages material definitions and application to FDTD grids.

    Combines built-in catalog, user-loaded files, and runtime-added custom
    materials. Materials are accessed by key (normalized name: lowercase,
    spaces/dashes replaced by underscores). Use get_material_manager() to
    obtain the global instance.

    Parameters
    ----------
    None
        Constructor takes no arguments; built-in catalog is loaded automatically.

    Attributes
    ----------
    _materials : dict
        Internal mapping from key to Material (or subclass) instance.

    Examples
    --------
    >>> mgr = get_material_manager()
    >>> mat = mgr.get("rogers_ro4003c")
    >>> mat.eps_r
    3.38
    >>> mgr.add_custom("my_substrate", eps_r=4.2, sigma=0.01)
    >>> mgr.list_by_category("dielectric")
    ['air', 'fr4', 'rogers_ro4003c', ...]
    """

    def __init__(self) -> None:
        self._materials: Dict[str, Material] = dict(MATERIAL_CATALOG)

    def get(self, name: str) -> Material:
        """Return a material by name (or normalized key).

        Parameters
        ----------
        name : str
            Material name or key (e.g. "Copper (Drude)", "copper_drude").

        Returns
        -------
        Material
            The material instance (Material, DispersiveMaterial, or
            AnisotropicMaterial).

        Raises
        ------
        KeyError
            If no material is registered under the normalized key.

        Examples
        --------
        >>> mgr.get("vacuum")
        Vacuum (eps_r=1.0, mu_r=1.0, sigma=0.0 S/m)
        >>> mgr.get("copper_drude").model
        'drude'
        """
        key = name.lower().replace(" ", "_").replace("-", "_").strip()
        if key not in self._materials:
            available = ", ".join(sorted(self._materials.keys())[:10])
            raise KeyError(
                f"Material '{name}' not found. "
                f"Available (sample): {available} ..."
            )
        return self._materials[key]

    def add_custom(
        self,
        name: str,
        eps_r: float,
        mu_r: float = 1.0,
        sigma: float = 0.0,
        category: str = "custom",
        description: str = "",
        source: str = "",
        **kwargs: Any,
    ) -> Material:
        """Register a custom isotropic material at runtime.

        Parameters
        ----------
        name : str
            Display name; also used as key (normalized).
        eps_r : float
            Relative permittivity.
        mu_r : float, optional
            Relative permeability. Default 1.0.
        sigma : float, optional
            Conductivity [S/m]. Default 0.0.
        category : str, optional
            Category for filtering. Default "custom".
        description : str, optional
            Optional description.
        source : str, optional
            Optional reference.
        **kwargs : any
            Ignored (for compatibility with future dispersive/anisotropic
            add_custom overloads).

        Returns
        -------
        Material
            The newly registered Material instance.

        Examples
        --------
        >>> mgr.add_custom("Lab substrate", eps_r=4.2, sigma=0.015)
        """
        mat = Material(
            name=name,
            eps_r=eps_r,
            mu_r=mu_r,
            sigma=sigma,
            category=category,
            description=description,
            source=source,
        )
        key = name.lower().replace(" ", "_").replace("-", "_").strip()
        self._materials[key] = mat
        return mat

    def load_csv(self, filepath: str) -> None:
        """Load materials from a CSV file and merge into the catalog.

        New materials are added; existing keys are overwritten by the file
        entries. See loader.load_materials_from_csv for CSV format.

        Parameters
        ----------
        filepath : str
            Path to the CSV file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the CSV format is invalid.
        """
        new_materials = load_materials_from_csv(filepath)
        self._materials.update(new_materials)

    def load_json(self, filepath: str) -> None:
        """Load materials from a JSON file and merge into the catalog.

        See loader.load_materials_from_json for expected JSON structure.

        Parameters
        ----------
        filepath : str
            Path to the JSON file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the JSON structure is invalid.
        """
        new_materials = load_materials_from_json(filepath)
        self._materials.update(new_materials)

    def list_all(self) -> List[str]:
        """Return sorted list of all material keys."""
        return sorted(self._materials.keys())

    def list_by_category(self, category: str) -> List[str]:
        """Return material keys in the given category.

        Parameters
        ----------
        category : str
            One of the Material.category values (e.g. "dielectric", "conductor").

        Returns
        -------
        list of str
            Sorted list of keys.
        """
        return sorted(
            k for k, m in self._materials.items()
            if m.category == category
        )

    def search(self, query: str) -> List[str]:
        """Search materials by key or description (case-insensitive).

        Parameters
        ----------
        query : str
            Substring to search for.

        Returns
        -------
        list of str
            Sorted list of matching keys.
        """
        q = query.lower()
        return sorted(
            k for k, m in self._materials.items()
            if q in k or q in m.description.lower()
        )

    def apply_to_grid(
        self,
        grid,
        region: Dict[str, tuple],
        material_name: str,
    ) -> None:
        """Apply a catalog material to a region of an FDTD MaterialGrid.

        The grid's materials must have set_region (and optionally
        add_dispersive_region / add_anisotropic_region when those features
        exist). This method resolves the material by name and calls the
        appropriate grid method.

        Parameters
        ----------
        grid : object
            Must have attribute materials (MaterialGrid) with at least
            set_region(i_range, j_range, k_range, eps_r=, mu_r=, sigma=).
        region : dict
            Keys "i", "j", "k", each value (i_min, i_max) style index range.
        material_name : str
            Key or name of the material (e.g. "rogers_ro4003c", "copper_drude").

        Examples
        --------
        >>> mgr.apply_to_grid(
        ...     grid,
        ...     region={"i": (10, 40), "j": (10, 40), "k": (0, 5)},
        ...     material_name="rogers_ro4003c",
        ... )
        """
        mat = self.get(material_name)
        i_range = region["i"]
        j_range = region["j"]
        k_range = region["k"]

        if isinstance(mat, DispersiveMaterial):
            if hasattr(grid.materials, "add_dispersive_region"):
                grid.materials.add_dispersive_region(
                    i_range=i_range,
                    j_range=j_range,
                    k_range=k_range,
                    material=mat,
                )
            else:
                # Fallback: use DC equivalent if dispersive not yet implemented
                grid.materials.set_region(
                    i_range, j_range, k_range,
                    eps_r=mat.eps_inf if mat.eps_inf is not None else mat.eps_r,
                    mu_r=mat.mu_r,
                    sigma=mat.sigma,
                )
        elif isinstance(mat, AnisotropicMaterial):
            if hasattr(grid.materials, "add_anisotropic_region"):
                grid.materials.add_anisotropic_region(
                    i_range=i_range,
                    j_range=j_range,
                    k_range=k_range,
                    material=mat,
                )
            else:
                grid.materials.set_region(
                    i_range, j_range, k_range,
                    eps_r=mat.eps_r_xx,  # simplistic fallback
                    mu_r=mat.mu_r_xx,
                    sigma=mat.sigma,
                )
        else:
            grid.materials.set_region(
                i_range, j_range, k_range,
                eps_r=mat.eps_r,
                mu_r=mat.mu_r,
                sigma=mat.sigma,
            )

    def export_to_csv(self, filepath: str) -> None:
        """Export current material catalog to a CSV file.

        Only Material (non-dispersive, non-anisotropic) fields are written;
        DispersiveMaterial and AnisotropicMaterial are exported with their
        scalar/base fields plus model-specific columns when applicable.

        Parameters
        ----------
        filepath : str
            Output path for the CSV file.
        """
        import csv
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "name", "eps_r", "mu_r", "sigma",
            "category", "description", "source",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for mat in self._materials.values():
                row = mat.to_dict()
                writer.writerow(row)


_global_manager: Optional[MaterialManager] = None


def get_material_manager() -> MaterialManager:
    """Return the global MaterialManager singleton.

    Creates the instance on first call; subsequent calls return the same
    instance. Use this to share one catalog across the application.

    Returns
    -------
    MaterialManager
        The global material manager.

    Examples
    --------
    >>> mgr = get_material_manager()
    >>> mgr.list_all()
    ['air', 'alumina', 'copper_dc', ...]
    """
    global _global_manager
    if _global_manager is None:
        _global_manager = MaterialManager()
    return _global_manager
