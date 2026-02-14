"""Unit tests for geometry primitives (Box, Cylinder, Sphere)."""

import pytest

from emsim.geometry import Box, Cylinder, Sphere, RectangularWaveguide, PatchAntenna


def test_box_bounds():
    b = Box(0.0, 1.0, 0.0, 2.0, 0.0, 3.0)
    assert b.bounds() == (0.0, 1.0, 0.0, 2.0, 0.0, 3.0)


def test_cylinder_bounds():
    c = Cylinder(center_x=0.5, center_y=0.5, radius=0.2, z_min=0.0, z_max=1.0)
    x0, x1, y0, y1, z0, z1 = c.bounds()
    assert x0 == pytest.approx(0.3)
    assert x1 == pytest.approx(0.7)
    assert z0 == 0.0
    assert z1 == 1.0


def test_sphere_bounds():
    s = Sphere(center_x=0.0, center_y=0.0, center_z=0.0, radius=1.0)
    assert s.bounds() == (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)


def test_box_to_pyvista():
    b = Box(0, 1, 0, 1, 0, 1)
    mesh = b.to_pyvista()
    assert mesh.n_cells > 0
    assert mesh.bounds == (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)


def test_cylinder_to_pyvista():
    c = Cylinder(0, 0, 0.5, 0.0, 1.0)
    mesh = c.to_pyvista()
    assert mesh.n_cells > 0


def test_sphere_to_pyvista():
    s = Sphere(0, 0, 0, 1.0)
    mesh = s.to_pyvista()
    assert mesh.n_cells > 0


def test_waveguide_bounds_and_to_pyvista():
    g = RectangularWaveguide(a=0.01, b=0.005, length=0.02)
    assert g.bounds() == (0.0, 0.01, 0.0, 0.005, 0.0, 0.02)
    mesh = g.to_pyvista()
    assert mesh.n_cells > 0


def test_patch_antenna_bounds_and_to_pyvista():
    p = PatchAntenna(
        patch_width=0.032,
        patch_length=0.04,
        substrate_width=0.06,
        substrate_length=0.06,
        substrate_thickness=0.001524,
        substrate_eps_r=3.38,
        substrate_kappa=1e-3,
        feed_x=-0.006,
        sim_box=(0.2, 0.2, 0.15),
    )
    b = p.bounds()
    assert len(b) == 6
    mesh = p.to_pyvista()
    assert hasattr(mesh, "n_blocks") and mesh.n_blocks >= 1


def test_plot_geometry_matplotlib_backend(tmp_path):
    """plot_geometry with matplotlib backend runs without display."""
    from emsim.geometry import plot_geometry
    b = Box(0, 0.01, 0, 0.01, 0, 0.02)
    plot_geometry(b, backend="matplotlib", save_path=str(tmp_path / "test_geom_plot.png"))
