"""Unit tests for the materials package: catalog, manager, and loader.

All tests use the public API. Documentation and assertions are in English.
"""

import csv
import json
import tempfile
from pathlib import Path

import pytest

from emsim.materials import (
    Material,
    DispersiveMaterial,
    AnisotropicMaterial,
    MATERIAL_CATALOG,
    get_material_manager,
    load_materials_from_csv,
    load_materials_from_json,
)


def test_catalog_has_expected_materials():
    """Built-in catalog must contain vacuum, air, and common RF materials."""
    assert "vacuum" in MATERIAL_CATALOG
    assert "air" in MATERIAL_CATALOG
    assert "fr4" in MATERIAL_CATALOG
    assert "rogers_ro4003c" in MATERIAL_CATALOG
    assert "copper_dc" in MATERIAL_CATALOG
    assert "copper_drude" in MATERIAL_CATALOG


def test_catalog_vacuum_properties():
    """Vacuum must have eps_r=1, mu_r=1, sigma=0."""
    mat = MATERIAL_CATALOG["vacuum"]
    assert mat.eps_r == 1.0
    assert mat.mu_r == 1.0
    assert mat.sigma == 0.0
    assert mat.category == "reference"


def test_catalog_dispersive_copper():
    """Copper Drude entry must be DispersiveMaterial with drude model."""
    mat = MATERIAL_CATALOG["copper_drude"]
    assert isinstance(mat, DispersiveMaterial)
    assert mat.model == "drude"
    assert mat.eps_inf == 1.0
    assert mat.omega_p is not None
    assert mat.gamma is not None


def test_manager_singleton():
    """get_material_manager must return the same instance."""
    a = get_material_manager()
    b = get_material_manager()
    assert a is b


def test_manager_get_by_name():
    """Manager get() must resolve name and normalized key."""
    mgr = get_material_manager()
    m1 = mgr.get("vacuum")
    m2 = mgr.get("Vacuum")
    assert m1 is m2
    assert m1.eps_r == 1.0


def test_manager_get_raises_unknown():
    """Manager get() must raise KeyError for unknown material."""
    mgr = get_material_manager()
    with pytest.raises(KeyError, match="not found"):
        mgr.get("nonexistent_material_xyz")


def test_manager_add_custom():
    """add_custom must register a new material and allow get()."""
    mgr = get_material_manager()
    mat = mgr.add_custom(
        "TestSubstrate",
        eps_r=4.2,
        sigma=0.01,
        description="Unit test custom material",
    )
    assert mat.eps_r == 4.2
    assert mat.sigma == 0.01
    key = "testsubstrate"
    assert mgr.get(key).name == "TestSubstrate"


def test_manager_list_all():
    """list_all must return sorted keys including built-in and custom."""
    mgr = get_material_manager()
    keys = mgr.list_all()
    assert isinstance(keys, list)
    assert keys == sorted(keys)
    assert "vacuum" in keys
    assert "air" in keys


def test_manager_list_by_category():
    """list_by_category must return only materials in that category."""
    mgr = get_material_manager()
    dielectrics = mgr.list_by_category("dielectric")
    assert "vacuum" in dielectrics or "air" in dielectrics
    assert "fr4" in dielectrics
    conductors = mgr.list_by_category("conductor")
    assert "copper_dc" in conductors or "gold" in conductors


def test_manager_search():
    """search must find materials by key or description substring."""
    mgr = get_material_manager()
    results = mgr.search("rogers")
    assert len(results) >= 1
    assert any("rogers" in k for k in results)


def test_apply_to_grid_integration():
    """apply_to_grid must set region using catalog material."""
    from emsim.fdtd.grid import YeeGrid

    grid = YeeGrid(
        x_range=(0, 20e-3),
        y_range=(0, 20e-3),
        z_range=(0, 20e-3),
        f0=10e9,
        resolution=15,
    )
    mgr = get_material_manager()
    # Use indices within grid bounds (e.g. center region)
    ni, nj, nk = grid.Nx, grid.Ny, grid.Nz
    i_range = (ni // 4, ni // 2)
    j_range = (nj // 4, nj // 2)
    k_range = (nk // 4, nk // 2)
    mgr.apply_to_grid(
        grid,
        region={"i": i_range, "j": j_range, "k": k_range},
        material_name="rogers_ro4003c",
    )
    grid.materials.compute_coefficients(grid.dt)
    from emsim.constants import EPS0
    expected_eps = 3.38 * EPS0
    ki, kj, kk = (k_range[0] + k_range[1]) // 2, (j_range[0] + j_range[1]) // 2, (i_range[0] + i_range[1]) // 2
    val = grid.materials.eps[ki, kj, kk].numpy()
    assert abs(val - expected_eps) < 1e-10


def test_load_materials_from_csv():
    """load_materials_from_csv must parse CSV and return Material dict."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("name,eps_r,mu_r,sigma,category,description\n")
        f.write("CustomDielectric,5.0,1.0,0.0,dielectric,Test\n")
        path = f.name
    try:
        materials = load_materials_from_csv(path)
        assert "customdielectric" in materials
        mat = materials["customdielectric"]
        assert mat.eps_r == 5.0
        assert mat.category == "dielectric"
    finally:
        Path(path).unlink()


def test_load_materials_from_json():
    """load_materials_from_json must parse JSON and return Material dict."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "my_material": {
                "name": "My Material",
                "eps_r": 6.0,
                "mu_r": 1.0,
                "sigma": 0.02,
                "category": "custom",
            }
        }, f)
        path = f.name
    try:
        materials = load_materials_from_json(path)
        assert "my_material" in materials
        mat = materials["my_material"]
        assert mat.eps_r == 6.0
        assert mat.sigma == 0.02
    finally:
        Path(path).unlink()


def test_material_to_dict():
    """Material.to_dict() must return a dict of attributes."""
    mat = MATERIAL_CATALOG["vacuum"]
    d = mat.to_dict()
    assert "name" in d
    assert "eps_r" in d
    assert d["eps_r"] == 1.0


def test_material_grid_set_material_by_name():
    """MaterialGrid.set_material must accept catalog name string."""
    from emsim.fdtd.grid import YeeGrid

    grid = YeeGrid(
        x_range=(0, 5e-3),
        y_range=(0, 5e-3),
        z_range=(0, 5e-3),
        f0=10e9,
        resolution=10,
    )
    grid.materials.set_material((0, 2), (0, 2), (0, 2), "fr4")
    grid.materials.compute_coefficients(grid.dt)
    from emsim.constants import EPS0
    assert abs(grid.materials.eps[1, 1, 1].numpy() - 4.4 * EPS0) < 1e-9
