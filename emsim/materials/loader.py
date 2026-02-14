"""Load material definitions from CSV and JSON files.

Enables user-defined material libraries without modifying code. CSV format
expects columns: name, eps_r, mu_r, sigma, category, description, source.
JSON format expects a dict of material keys with property dicts (same fields).
All loaded materials are returned as Material instances; dispersive and
anisotropic entries require the appropriate extra fields and are instantiated
as DispersiveMaterial or AnisotropicMaterial when detected.
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any

from .base import Material, DispersiveMaterial, AnisotropicMaterial


def _normalize_key(name: str) -> str:
    """Convert material name to catalog key (lowercase, spaces/underscores)."""
    return name.lower().replace(" ", "_").replace("-", "_").strip()


def load_materials_from_csv(filepath: str) -> Dict[str, Material]:
    """Load materials from a CSV file.

    The CSV must have a header row. Required columns: name, eps_r.
    Optional columns: mu_r (default 1.0), sigma (default 0.0), category,
    description, source. For dispersive materials, include columns: model,
    eps_inf, omega_p, gamma (Drude) or model, eps_s, tau (Debye), etc.
    For anisotropic materials, include eps_r_xx, eps_r_yy, eps_r_zz and
    optionally off-diagonal terms.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (absolute or relative to cwd).

    Returns
    -------
    dict
        Mapping from normalized material key (str) to Material (or
        DispersiveMaterial / AnisotropicMaterial) instance.

    Raises
    ------
    FileNotFoundError
        If filepath does not exist.
    ValueError
        If required columns are missing or values are invalid.

    Examples
    --------
    >>> materials = load_materials_from_csv("my_materials.csv")
    >>> materials["custom_substrate"].eps_r
    4.2
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Material library not found: {filepath}")

    materials = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "name" not in reader.fieldnames:
            raise ValueError("CSV must have a header with at least 'name' and 'eps_r'")

        for row in reader:
            name = row.get("name", "").strip()
            if not name:
                continue

            try:
                eps_r = float(row.get("eps_r", 1.0))
            except (TypeError, ValueError):
                raise ValueError(f"Invalid eps_r for material '{name}'")

            mu_r = float(row.get("mu_r", 1.0))
            sigma = float(row.get("sigma", 0.0))
            category = row.get("category", "custom").strip() or "custom"
            description = row.get("description", "").strip()
            source = row.get("source", "").strip()

            model = row.get("model", "").strip().lower()
            if model == "drude":
                eps_inf = float(row.get("eps_inf", 1.0))
                omega_p = float(row.get("omega_p", 0.0))
                gamma = float(row.get("gamma", 0.0))
                mat = DispersiveMaterial(
                    name=name,
                    eps_r=eps_r,
                    mu_r=mu_r,
                    sigma=sigma,
                    model="drude",
                    eps_inf=eps_inf,
                    omega_p=omega_p,
                    gamma=gamma,
                    category=category,
                    description=description,
                    source=source,
                )
            elif model == "debye":
                eps_s = float(row.get("eps_s", eps_r))
                tau = float(row.get("tau", 0.0))
                mat = DispersiveMaterial(
                    name=name,
                    eps_r=eps_r,
                    mu_r=mu_r,
                    sigma=sigma,
                    model="debye",
                    eps_s=eps_s,
                    tau=tau,
                    category=category,
                    description=description,
                    source=source,
                )
            elif "eps_r_xx" in row:
                mat = AnisotropicMaterial(
                    name=name,
                    eps_r=eps_r,
                    mu_r=mu_r,
                    sigma=sigma,
                    eps_r_xx=float(row.get("eps_r_xx", 1.0)),
                    eps_r_yy=float(row.get("eps_r_yy", 1.0)),
                    eps_r_zz=float(row.get("eps_r_zz", 1.0)),
                    eps_r_xy=float(row.get("eps_r_xy", 0.0)),
                    eps_r_xz=float(row.get("eps_r_xz", 0.0)),
                    eps_r_yz=float(row.get("eps_r_yz", 0.0)),
                    category=category,
                    description=description,
                    source=source,
                )
            else:
                mat = Material(
                    name=name,
                    eps_r=eps_r,
                    mu_r=mu_r,
                    sigma=sigma,
                    category=category,
                    description=description,
                    source=source,
                )

            key = _normalize_key(name)
            materials[key] = mat

    return materials


def load_materials_from_json(filepath: str) -> Dict[str, Material]:
    """Load materials from a JSON file.

    Expected format: a JSON object whose keys are material identifiers and
    whose values are objects with at least "name" and "eps_r". Optional
    keys: mu_r, sigma, category, description, source. For dispersive
    materials include "model" ("drude", "lorentz", "debye") and the
    corresponding parameters. For anisotropic include eps_r_xx, eps_r_yy,
    eps_r_zz, etc.

    Parameters
    ----------
    filepath : str
        Path to the JSON file.

    Returns
    -------
    dict
        Mapping from material key to Material (or DispersiveMaterial /
        AnisotropicMaterial) instance.

    Raises
    ------
    FileNotFoundError
        If filepath does not exist.
    ValueError
        If structure is invalid.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Material library not found: {filepath}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object (dict) of materials")

    materials = {}
    for key, props in data.items():
        if not isinstance(props, dict):
            raise ValueError(f"Material '{key}' must be an object")

        name = props.get("name", key)
        eps_r = float(props.get("eps_r", 1.0))
        mu_r = float(props.get("mu_r", 1.0))
        sigma = float(props.get("sigma", 0.0))
        category = props.get("category", "custom")
        description = props.get("description", "")
        source = props.get("source", "")

        model = props.get("model", "").lower()
        if model == "drude":
            mat = DispersiveMaterial(
                name=name,
                eps_r=eps_r,
                mu_r=mu_r,
                sigma=sigma,
                model="drude",
                eps_inf=float(props.get("eps_inf", 1.0)),
                omega_p=float(props.get("omega_p", 0.0)),
                gamma=float(props.get("gamma", 0.0)),
                category=category,
                description=description,
                source=source,
            )
        elif model == "lorentz":
            delta_eps = props.get("delta_eps")
            omega_0 = props.get("omega_0")
            delta = props.get("delta")
            if delta_eps is not None:
                delta_eps = tuple(delta_eps)
            if omega_0 is not None:
                omega_0 = tuple(omega_0)
            if delta is not None:
                delta = tuple(delta)
            mat = DispersiveMaterial(
                name=name,
                eps_r=eps_r,
                mu_r=mu_r,
                sigma=sigma,
                model="lorentz",
                delta_eps=delta_eps,
                omega_0=omega_0,
                delta=delta,
                category=category,
                description=description,
                source=source,
            )
        elif model == "debye":
            mat = DispersiveMaterial(
                name=name,
                eps_r=eps_r,
                mu_r=mu_r,
                sigma=sigma,
                model="debye",
                eps_s=float(props.get("eps_s", eps_r)),
                tau=float(props.get("tau", 0.0)),
                category=category,
                description=description,
                source=source,
            )
        elif props.get("anisotropic") or "eps_r_xx" in props:
            mat = AnisotropicMaterial(
                name=name,
                eps_r=eps_r,
                mu_r=mu_r,
                sigma=sigma,
                eps_r_xx=float(props.get("eps_r_xx", 1.0)),
                eps_r_yy=float(props.get("eps_r_yy", 1.0)),
                eps_r_zz=float(props.get("eps_r_zz", 1.0)),
                eps_r_xy=float(props.get("eps_r_xy", 0.0)),
                eps_r_xz=float(props.get("eps_r_xz", 0.0)),
                eps_r_yz=float(props.get("eps_r_yz", 0.0)),
                mu_r_xx=float(props.get("mu_r_xx", 1.0)),
                mu_r_yy=float(props.get("mu_r_yy", 1.0)),
                mu_r_zz=float(props.get("mu_r_zz", 1.0)),
                mu_r_xy=float(props.get("mu_r_xy", 0.0)),
                mu_r_xz=float(props.get("mu_r_xz", 0.0)),
                mu_r_yz=float(props.get("mu_r_yz", 0.0)),
                category=category,
                description=description,
                source=source,
            )
        else:
            mat = Material(
                name=name,
                eps_r=eps_r,
                mu_r=mu_r,
                sigma=sigma,
                category=category,
                description=description,
                source=source,
            )

        materials[_normalize_key(key)] = mat

    return materials
