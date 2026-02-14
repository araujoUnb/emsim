"""Tests for vectorised field update functions."""

import math
import pytest
import tensorflow as tf
import numpy as np
from emsim.constants import EPS0, MU0
from emsim.fdtd.fields import update_H, update_E


def _make_fields(Nz, Ny, Nx):
    """Create zero-initialised field Variables."""
    shape = [Nz, Ny, Nx]
    return tuple(tf.Variable(tf.zeros(shape, dtype=tf.float32)) for _ in range(6))


def test_curl_of_uniform_field_is_zero():
    """A spatially uniform E-field has zero curl → H should not change."""
    Nz, Ny, Nx = 10, 10, 10
    Ex, Ey, Ez, Hx, Hy, Hz = _make_fields(Nz, Ny, Nx)
    Ex.assign(tf.ones_like(Ex))
    Ey.assign(tf.ones_like(Ey) * 2.0)
    Ez.assign(tf.ones_like(Ez) * 3.0)

    dt_over_mu = tf.ones([Nz, Ny, Nx], dtype=tf.float32) * 1e-12
    dx = dy = dz = 0.001
    inv_dx = tf.constant(1.0 / dx, shape=(1, 1, Nx - 1), dtype=tf.float32)
    inv_dy = tf.constant(1.0 / dy, shape=(1, Ny - 1, 1), dtype=tf.float32)
    inv_dz = tf.constant(1.0 / dz, shape=(Nz - 1, 1, 1), dtype=tf.float32)

    update_H(Ex, Ey, Ez, Hx, Hy, Hz, dt_over_mu, inv_dx, inv_dy, inv_dz)

    assert float(tf.reduce_max(tf.abs(Hx)).numpy()) == pytest.approx(0.0, abs=1e-15)
    assert float(tf.reduce_max(tf.abs(Hy)).numpy()) == pytest.approx(0.0, abs=1e-15)
    assert float(tf.reduce_max(tf.abs(Hz)).numpy()) == pytest.approx(0.0, abs=1e-15)


def test_energy_conservation_pec_cavity():
    """In a PEC cavity with no losses, total EM energy must be conserved.

    We initialise a sinusoidal E-field, run several steps with update_H and
    update_E, and check that total energy stays constant.
    """
    Nz, Ny, Nx = 20, 20, 20
    dx = dy = dz = 0.001
    dt = 0.5 / (3e8 * math.sqrt(3.0 / dx ** 2))  # CFL with lower courant

    Ex, Ey, Ez, Hx, Hy, Hz = _make_fields(Nz, Ny, Nx)

    dt_over_mu = tf.constant(dt / MU0, dtype=tf.float32) * tf.ones([Nz, Ny, Nx], tf.float32)
    Ca = tf.ones([Nz, Ny, Nx], dtype=tf.float32)  # no loss
    Cb = tf.constant(dt / EPS0, dtype=tf.float32) * tf.ones([Nz, Ny, Nx], tf.float32)
    inv_dx = tf.constant(1.0 / dx, shape=(1, 1, Nx - 1), dtype=tf.float32)
    inv_dy = tf.constant(1.0 / dy, shape=(1, Ny - 1, 1), dtype=tf.float32)
    inv_dz = tf.constant(1.0 / dz, shape=(Nz - 1, 1, 1), dtype=tf.float32)

    # Initialise Ez with a sinusoidal pattern (satisfies PEC BCs: zero at boundaries)
    k = np.arange(Nz); j = np.arange(Ny); i = np.arange(Nx)
    K, J, I = np.meshgrid(k, j, i, indexing='ij')
    ez_init = np.sin(np.pi * I / Nx) * np.sin(np.pi * J / Ny) * np.sin(np.pi * K / Nz)
    Ez.assign(tf.constant(ez_init, dtype=tf.float32))

    def compute_energy():
        e2 = tf.reduce_sum(Ex ** 2 + Ey ** 2 + Ez ** 2) * EPS0
        h2 = tf.reduce_sum(Hx ** 2 + Hy ** 2 + Hz ** 2) * MU0
        return float((0.5 * (e2 + h2)).numpy())

    E0 = compute_energy()
    assert E0 > 0  # sanity: we put energy in

    # Run 100 steps
    for _ in range(100):
        update_H(Ex, Ey, Ez, Hx, Hy, Hz, dt_over_mu, inv_dx, inv_dy, inv_dz)
        update_E(Ex, Ey, Ez, Hx, Hy, Hz, Ca, Cb, inv_dx, inv_dy, inv_dz)
        # PEC: zero tangential E at boundaries (implicit since the slicing
        # in update_E starts at index 1 and our init is zero at boundaries)

    E_final = compute_energy()
    rel_change = abs(E_final - E0) / E0
    assert rel_change < 0.02, f"Energy changed by {rel_change * 100:.2f}%"
