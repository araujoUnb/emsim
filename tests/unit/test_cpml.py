"""Tests for the CPML boundary condition."""

import numpy as np
import tensorflow as tf
from emsim.constants import EPS0, MU0
from emsim.boundaries.cpml import CPML, _compute_cpml_coeffs


def test_sigma_profile_monotonic():
    """The CPML sigma coefficients should create a graded absorption."""
    b_e, c_e, k_e, b_h, c_h, k_h = _compute_cpml_coeffs(N_pml=8, d_cell=0.001, dt=1e-12)
    # b should be <= 1 (exponential decay factor)
    assert np.all(b_e <= 1.0 + 1e-7)
    assert np.all(b_h <= 1.0 + 1e-7)
    # c should be <= 0 (negative because b-1 < 0)
    assert np.all(c_e <= 1e-7)
    assert np.all(c_h <= 1e-7)
    # kappa should be >= 1
    assert np.all(k_e >= 1.0 - 1e-7)
    assert np.all(k_h >= 1.0 - 1e-7)


def test_cpml_allocation():
    """CPML should allocate all 24 psi arrays with correct shapes."""
    Nz, Ny, Nx = 30, 10, 20
    N_pml = 6
    dt = 1e-12
    dx = dy = dz = 0.001
    dt_over_mu = tf.ones([Nz, Ny, Nx], dtype=tf.float32) * (dt / MU0)
    Cb = tf.ones([Nz, Ny, Nx], dtype=tf.float32) * (dt / EPS0)

    cpml = CPML(Nz, Ny, Nx, N_pml, dx, dy, dz, dt, dt_over_mu, Cb)

    Np = N_pml - 1  # plus faces have N-1 cells
    # Check x- psi shapes
    assert cpml.psi_Hyx_xm.shape == (Nz - 1, Ny, N_pml)
    assert cpml.psi_Hzx_xm.shape == (Nz, Ny - 1, N_pml)
    # Check x+ psi shapes
    assert cpml.psi_Hyx_xp.shape == (Nz - 1, Ny, Np)
    # Check y- psi shapes
    assert cpml.psi_Hxy_ym.shape == (Nz - 1, N_pml, Nx)
    # Check z- psi shapes
    assert cpml.psi_Hxz_zm.shape == (N_pml, Ny - 1, Nx)


def test_cpml_does_not_crash():
    """Smoke test: run a few CPML update steps without errors."""
    Nz, Ny, Nx = 20, 10, 15
    N_pml = 4
    dt = 1e-12
    dx = dy = dz = 0.001
    dt_over_mu = tf.ones([Nz, Ny, Nx], dtype=tf.float32) * (dt / MU0)
    Cb = tf.ones([Nz, Ny, Nx], dtype=tf.float32) * (dt / EPS0)

    cpml = CPML(Nz, Ny, Nx, N_pml, dx, dy, dz, dt, dt_over_mu, Cb)

    shape = [Nz, Ny, Nx]
    Ex = tf.Variable(tf.random.normal(shape))
    Ey = tf.Variable(tf.random.normal(shape))
    Ez = tf.Variable(tf.random.normal(shape))
    Hx = tf.Variable(tf.random.normal(shape))
    Hy = tf.Variable(tf.random.normal(shape))
    Hz = tf.Variable(tf.random.normal(shape))

    # Should not raise
    cpml.update_H(Ex, Ey, Ez, Hx, Hy, Hz)
    cpml.update_E(Ex, Ey, Ez, Hx, Hy, Hz)
