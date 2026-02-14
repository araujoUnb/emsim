"""Unit tests for MaterialGrid class."""

import pytest
import numpy as np
import tensorflow as tf

from emsim.fdtd.materials import MaterialGrid
from emsim.constants import EPS0, MU0


def test_material_grid_creation():
    """Test basic MaterialGrid instantiation."""
    mat = MaterialGrid(Nz=10, Ny=8, Nx=6)
    
    assert mat.Nz == 10
    assert mat.Ny == 8
    assert mat.Nx == 6
    assert mat.eps.shape == (10, 8, 6)
    assert mat.mu.shape == (10, 8, 6)
    assert mat.sigma.shape == (10, 8, 6)


def test_material_grid_default_values():
    """Test default material values (vacuum)."""
    mat = MaterialGrid(Nz=5, Ny=5, Nx=5, eps_r=1.0, mu_r=1.0, sigma=0.0)
    
    # Check vacuum properties
    assert tf.reduce_all(mat.eps == EPS0).numpy()
    assert tf.reduce_all(mat.mu == MU0).numpy()
    assert tf.reduce_all(mat.sigma == 0.0).numpy()


def test_material_grid_custom_values():
    """Test custom uniform material."""
    mat = MaterialGrid(Nz=5, Ny=5, Nx=5, eps_r=2.0, mu_r=1.5, sigma=0.01)
    
    expected_eps = EPS0 * 2.0
    expected_mu = MU0 * 1.5
    
    assert tf.reduce_all(tf.abs(mat.eps - expected_eps) < 1e-20).numpy()
    assert tf.reduce_all(tf.abs(mat.mu - expected_mu) < 1e-30).numpy()
    assert tf.reduce_all(mat.sigma == 0.01).numpy()


def test_compute_coefficients():
    """Test FDTD coefficient computation."""
    mat = MaterialGrid(Nz=3, Ny=3, Nx=3, eps_r=1.0, mu_r=1.0, sigma=0.0)
    dt = 1e-12
    
    mat.compute_coefficients(dt)
    
    # For lossless case: Ca = 1, Cb = dt/eps, dt_over_mu = dt/mu
    assert hasattr(mat, 'Ca')
    assert hasattr(mat, 'Cb')
    assert hasattr(mat, 'dt_over_mu')
    
    # Lossless: Ca should be 1
    assert tf.reduce_all(tf.abs(mat.Ca - 1.0) < 1e-6).numpy()
    
    # Cb should be dt/eps
    expected_Cb = dt / EPS0
    assert tf.reduce_all(tf.abs(mat.Cb - expected_Cb) < 1e-20).numpy()


def test_set_region_permittivity():
    """Test setting permittivity in a region."""
    mat = MaterialGrid(Nz=20, Ny=20, Nx=20, eps_r=1.0)
    dt = 1e-12
    mat.compute_coefficients(dt)
    
    # Set a substrate region with eps_r = 3.38
    mat.set_region(
        i_range=(5, 15),
        j_range=(5, 15),
        k_range=(0, 5),
        eps_r=3.38
    )
    
    # Check that region was updated
    eps_in_region = mat.eps[2, 10, 10].numpy()
    expected = EPS0 * 3.38
    assert np.isclose(eps_in_region, expected, rtol=1e-6)
    
    # Check that outside region is still vacuum
    eps_outside = mat.eps[10, 10, 10].numpy()
    assert np.isclose(eps_outside, EPS0, rtol=1e-6)


def test_set_region_conductivity():
    """Test setting conductivity in a region."""
    mat = MaterialGrid(Nz=15, Ny=15, Nx=15, sigma=0.0)
    
    mat.set_region(
        i_range=(3, 10),
        j_range=(3, 10),
        k_range=(3, 10),
        sigma=0.05
    )
    
    # Check conductivity
    sigma_in = mat.sigma[5, 5, 5].numpy()
    assert np.isclose(sigma_in, 0.05)
    
    sigma_out = mat.sigma[12, 12, 12].numpy()
    assert np.isclose(sigma_out, 0.0)


def test_set_region_invalid_indices():
    """Test that invalid indices raise ValueError."""
    mat = MaterialGrid(Nz=10, Ny=10, Nx=10)
    
    with pytest.raises(ValueError, match="Invalid i_range"):
        mat.set_region(i_range=(-1, 5), j_range=(0, 5), k_range=(0, 5), eps_r=2.0)
    
    with pytest.raises(ValueError, match="Invalid i_range"):
        mat.set_region(i_range=(5, 15), j_range=(0, 5), k_range=(0, 5), eps_r=2.0)
    
    with pytest.raises(ValueError, match="Invalid k_range"):
        mat.set_region(i_range=(0, 5), j_range=(0, 5), k_range=(8, 12), eps_r=2.0)


def test_coefficients_after_region_change():
    """Test that coefficients must be recomputed after set_region."""
    mat = MaterialGrid(Nz=10, Ny=10, Nx=10, eps_r=1.0)
    dt = 1e-12
    mat.compute_coefficients(dt)
    
    # Store original Cb
    Cb_original = mat.Cb[5, 5, 5].numpy()
    
    # Change material
    mat.set_region(i_range=(4, 7), j_range=(4, 7), k_range=(4, 7), eps_r=4.0)
    mat.compute_coefficients(dt)  # Must recompute!
    
    # Cb in region should be different: Cb = dt/eps
    Cb_new = mat.Cb[5, 5, 5].numpy()
    expected_Cb_new = dt / (EPS0 * 4.0)
    
    assert not np.isclose(Cb_new, Cb_original)
    assert np.isclose(Cb_new, expected_Cb_new, rtol=1e-6)
