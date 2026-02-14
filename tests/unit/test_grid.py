"""Tests for the YeeGrid class."""

import math
import pytest
import tensorflow as tf
from emsim.constants import C0
from emsim.fdtd.grid import YeeGrid


def test_grid_dimensions():
    g = YeeGrid(
        x_range=(0, 0.1), y_range=(0, 0.05), z_range=(0, 0.2),
        dx=0.01, dy=0.01, dz=0.01,
    )
    assert g.Nx == 10
    assert g.Ny == 5
    assert g.Nz == 20


def test_grid_auto_spacing():
    f0 = 10e9
    g = YeeGrid(
        x_range=(0, 0.01), y_range=(0, 0.005), z_range=(0, 0.05),
        f0=f0, resolution=20,
    )
    lam = C0 / f0
    expected_dx = lam / 20
    assert abs(g.dx - expected_dx) / expected_dx < 0.2  # within 20% due to snapping


def test_field_shapes():
    g = YeeGrid(
        x_range=(0, 0.1), y_range=(0, 0.05), z_range=(0, 0.2),
        dx=0.01, dy=0.01, dz=0.01,
    )
    shape = (g.Nz, g.Ny, g.Nx)
    assert g.Ex.shape == shape
    assert g.Ey.shape == shape
    assert g.Ez.shape == shape
    assert g.Hx.shape == shape
    assert g.Hy.shape == shape
    assert g.Hz.shape == shape


def test_cfl_condition():
    g = YeeGrid(
        x_range=(0, 0.1), y_range=(0, 0.05), z_range=(0, 0.2),
        dx=0.01, dy=0.01, dz=0.01,
        courant=0.99,
    )
    inv2 = (1 / g.dx) ** 2 + (1 / g.dy) ** 2 + (1 / g.dz) ** 2
    dt_max = 1.0 / (C0 * math.sqrt(inv2))
    assert g.dt <= dt_max
    assert g.dt == pytest.approx(0.99 * dt_max, rel=1e-10)


def test_reset_fields():
    g = YeeGrid(
        x_range=(0, 0.1), y_range=(0, 0.05), z_range=(0, 0.2),
        dx=0.01,
    )
    g.Ex.assign(tf.ones_like(g.Ex))
    g.reset_fields()
    assert float(tf.reduce_max(tf.abs(g.Ex)).numpy()) == 0.0


def test_material_coefficients():
    g = YeeGrid(
        x_range=(0, 0.01), y_range=(0, 0.01), z_range=(0, 0.01),
        dx=0.001,
    )
    # For vacuum with sigma=0: Ca=1, Cb=dt/eps0
    ca = float(g.materials.Ca[0, 0, 0].numpy())
    assert ca == pytest.approx(1.0, rel=1e-6)


# --- Non-uniform grid (stretched mesh) ---


def test_nonuniform_grid_dx_array():
    """Non-uniform grid: dx 1D array defines Nx; dx_array has correct length."""
    # 20 cells in x with finer spacing in the middle
    dx_fine = 0.0005
    dx_coarse = 0.001
    dx_arr = [dx_coarse] * 5 + [dx_fine] * 10 + [dx_coarse] * 5
    g = YeeGrid(
        x_range=(0, 0.1), y_range=(0, 0.05), z_range=(0, 0.05),
        dx=dx_arr, dy=0.001, dz=0.001,
    )
    assert g.Nx == 20
    assert len(g.dx_array) == 20
    assert g.dx_at(0) == pytest.approx(dx_coarse, rel=1e-6)
    assert g.dx_at(10) == pytest.approx(dx_fine, rel=1e-6)
    assert g.dy_at(0) == pytest.approx(0.001, rel=1e-6)
    assert g.dz_at(0) == pytest.approx(0.001, rel=1e-6)


def test_nonuniform_grid_curl_coefficients_shapes():
    """get_curl_coefficients() returns tensors with shapes for update_H/update_E."""
    dx_arr = [0.001] * 8 + [0.0005] * 4 + [0.001] * 8  # Nx=20
    g = YeeGrid(
        x_range=(0, 0.1), y_range=(0, 0.05), z_range=(0, 0.05),
        dx=dx_arr, dy=0.001, dz=0.001,
    )
    coeffs = g.get_curl_coefficients()
    inv_dx, inv_dy, inv_dz = coeffs["inv_dx"], coeffs["inv_dy"], coeffs["inv_dz"]
    assert inv_dx.shape == (1, 1, g.Nx - 1)
    assert inv_dy.shape == (1, g.Ny - 1, 1)
    assert inv_dz.shape == (g.Nz - 1, 1, 1)


def test_nonuniform_grid_cfl_uses_smallest_cell():
    """With non-uniform grid, dt is limited by the smallest cell (strictest CFL)."""
    # One very small cell in z
    dz_arr = [0.001] * 49 + [0.0002] + [0.001] * 50  # Nz=100, one fine cell
    g = YeeGrid(
        x_range=(0, 0.05), y_range=(0, 0.05), z_range=(0, 0.1),
        dx=0.001, dy=0.001, dz=dz_arr, courant=0.99,
    )
    # dt_max from smallest cell ~ 1/(c*sqrt(1/0.0002^2 + ...)) << 1/(c*sqrt(3/0.001^2))
    dt_uniform = 0.99 / (C0 * math.sqrt(3.0 / 0.001**2))
    assert g.dt < dt_uniform * 0.5  # non-uniform dt should be significantly smaller


def test_uniform_grid_equivalent_to_scalar():
    """Uniform grid created with scalar dx produces same curl coeffs as constant array."""
    g1 = YeeGrid(
        x_range=(0, 0.02), y_range=(0, 0.02), z_range=(0, 0.02),
        dx=0.001, dy=0.001, dz=0.001,
    )
    g2 = YeeGrid(
        x_range=(0, 0.02), y_range=(0, 0.02), z_range=(0, 0.02),
        dx=[0.001] * 20, dy=[0.001] * 20, dz=[0.001] * 20,
    )
    assert g1.Nx == g2.Nx and g1.Ny == g2.Ny and g1.Nz == g2.Nz
    assert g1.dt == pytest.approx(g2.dt, rel=1e-9)
    c1, c2 = g1.get_curl_coefficients(), g2.get_curl_coefficients()
    assert float(c1["inv_dx"][0, 0, 0].numpy()) == pytest.approx(
        float(c2["inv_dx"][0, 0, 0].numpy()), rel=1e-6
    )
