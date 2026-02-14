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
