# tests/test_rectangular_waveguide.py

import torch
import numpy as np
import pytest

from objects.rectangular_waveguide import RectangularWaveguide
from utils.utils_constants import EPS0, MU0


@pytest.fixture
def small_waveguide():
    """
    Fixture: a small rectangular waveguide with known dimensions and resolution.

    - Width a = 0.1 m
    - Height b = 0.05 m
    - Length = 0.2 m
    - Resolution: Nx = 4, Ny = 3, Nz = 2
    - Material properties (chosen just for testing):
       ε_r = 2.5, μ_r = 1.2, σ = 1e-3 S/m
    """
    a = 0.1
    b = 0.05
    length = 0.2
    resolution = (4, 3, 2)  # (Nx, Ny, Nz)
    material = {'epsilon_r': 2.5, 'mu_r': 1.2, 'conductivity': 1e-3}
    device = 'cpu'

    wg = RectangularWaveguide(
        a=a,
        b=b,
        length=length,
        resolution=resolution,
        material=material,
        device=device
    )
    # If your implementation requires a separate grid generation call:
    try:
        wg.generate_grid()  # build X, Y, Z, if your class needs it
    except AttributeError:
        # If generate_grid() is already called inside __init__, ignore
        pass

    return wg


def test_grid_shape_and_values(small_waveguide):
    wg = small_waveguide
    Nx, Ny, Nz = wg.Nx, wg.Ny, wg.Nz

    # --- 1) Verify that X, Y, Z tensors exist and have shape [Nz, Ny, Nx] ---

    # If your class stores X, Y, Z as attributes:
    X = wg.X
    Y = wg.Y
    Z = wg.Z

    assert isinstance(X, torch.Tensor)
    assert isinstance(Y, torch.Tensor)
    assert isinstance(Z, torch.Tensor)

    assert X.shape == (Nz, Ny, Nx)
    assert Y.shape == (Nz, Ny, Nx)
    assert Z.shape == (Nz, Ny, Nx)

    # Check that X varies from x_min to x_max along the last (Nx) dimension
    x_vals = X[0, 0, :].cpu().numpy()
    assert np.isclose(x_vals[0], wg.x_min)
    assert np.isclose(x_vals[-1], wg.x_max)

    # Check that Y varies from y_min to y_max along the middle (Ny) dimension
    y_vals = Y[0, :, 0].cpu().numpy()
    assert np.isclose(y_vals[0], wg.y_min)
    assert np.isclose(y_vals[-1], wg.y_max)

    # Check that Z varies from z_min to z_max along the first (Nz) dimension
    z_vals = Z[:, 0, 0].cpu().numpy()
    assert np.isclose(z_vals[0], wg.z_min)
    assert np.isclose(z_vals[-1], wg.z_max)


def test_material_maps(small_waveguide):
    wg = small_waveguide
    Nx, Ny, Nz = wg.Nx, wg.Ny, wg.Nz

    # --- 2A) Get the permittivity, permeability, and conductivity maps ---
    eps_map = wg.get_permittivity_map()    # shape [Nz, Ny, Nx]
    mu_map = wg.get_permeability_map()     # shape [Nz, Ny, Nx]
    sigma_map = wg.get_conductivity_map()  # shape [Nz, Ny, Nx]

    # All should be torch tensors of correct shape
    assert isinstance(eps_map, torch.Tensor)
    assert isinstance(mu_map, torch.Tensor)
    assert isinstance(sigma_map, torch.Tensor)

    assert eps_map.shape == (Nz, Ny, Nx)
    assert mu_map.shape == (Nz, Ny, Nx)
    assert sigma_map.shape == (Nz, Ny, Nx)

    # --- 2B) Infer ε_r and μ_r from these maps ---
    # We know the user provided ε_r = 2.5, μ_r = 1.2, and σ = 1e-3

    # At an arbitrary index, say (0,0,0):
    eps00 = eps_map[0, 0, 0].item()
    mu00 = mu_map[0, 0, 0].item()
    sigma00 = sigma_map[0, 0, 0].item()

    # Compute the relative permittivity and permeability:
    inferred_eps_r = eps00 / EPS0
    inferred_mu_r = mu00 / MU0

    # Check that they match the values we passed (within numerical tolerance)
    assert np.isclose(inferred_eps_r, 2.5, rtol=1e-6, atol=1e-8)
    assert np.isclose(inferred_mu_r, 1.2, rtol=1e-6, atol=1e-8)

    # Check that conductivity matches exactly 1e-3
    assert np.isclose(sigma00, 1e-3, rtol=1e-7, atol=1e-9)

    # Finally, check that *every* element of each map matches those constants
    assert torch.allclose(eps_map, torch.full((Nz, Ny, Nx), EPS0 * 2.5))
    assert torch.allclose(mu_map, torch.full((Nz, Ny, Nx), MU0 * 1.2))
    assert torch.allclose(sigma_map, torch.full((Nz, Ny, Nx), 1e-3))


def test_side_surfaces(small_waveguide):
    wg = small_waveguide
    Nx, Ny, Nz = wg.Nx, wg.Ny, wg.Nz

    # --- 3) Verify generate_side_surfaces_numpy() produces 4 patches ---
    surfaces = wg.generate_side_surfaces_numpy()
    assert isinstance(surfaces, list)
    assert len(surfaces) == 4

    # Unpack: each entry is (X_patch, Y_patch, Z_patch)
    X0, Y0, Z0 = surfaces[0]  # at x = x_min, shape [Ny, Nz]
    Xa, Ya, Za = surfaces[1]  # at x = x_max, shape [Ny, Nz]
    Xb, Yb, Zb = surfaces[2]  # at y = y_min, shape [Nx, Nz]
    Xc, Yc, Zc = surfaces[3]  # at y = y_max, shape [Nx, Nz]

    # 3A) Check shapes explicitly
    assert X0.shape == (Ny, Nz)
    assert Y0.shape == (Ny, Nz)
    assert Z0.shape == (Ny, Nz)

    assert Xa.shape == (Ny, Nz)
    assert Ya.shape == (Ny, Nz)
    assert Za.shape == (Ny, Nz)

    assert Xb.shape == (Nx, Nz)
    assert Yb.shape == (Nx, Nz)
    assert Zb.shape == (Nx, Nz)

    assert Xc.shape == (Nx, Nz)
    assert Yc.shape == (Nx, Nz)
    assert Zc.shape == (Nx, Nz)

    # 3B) Check that X0 is constant = x_min and Xa = x_max
    assert np.allclose(X0, wg.x_min)
    assert np.allclose(Xa, wg.x_max)

    # 3C) Check that Yb is constant = y_min and Yc = y_max
    assert np.allclose(Yb, wg.y_min)
    assert np.allclose(Yc, wg.y_max)


def test_default_material_if_none():
    """
    When material=None is passed in, the waveguide should default to vacuum:
    ε_r = 1, μ_r = 1, σ = 0.
    """
    a = 0.05
    b = 0.02
    length = 0.1
    resolution = (3, 3, 3)

    wg = RectangularWaveguide(
        a=a,
        b=b,
        length=length,
        resolution=resolution,
        material=None,
        device='cpu'
    )

    # If generate_grid() needs to be called, do it here; otherwise ignore.
    try:
        wg.generate_grid()
    except AttributeError:
        pass

    # Retrieve the maps
    eps_map = wg.get_permittivity_map()
    mu_map = wg.get_permeability_map()
    sigma_map = wg.get_conductivity_map()

    # At index (0,0,0), permittivity = EPS0 → ε_r = eps_map/EPS0 = 1
    eps00 = eps_map[0, 0, 0].item()
    mu00 = mu_map[0, 0, 0].item()
    sigma00 = sigma_map[0, 0, 0].item()

    assert np.isclose(eps00 / EPS0, 1.0, rtol=1e-6, atol=1e-8)
    assert np.isclose(mu00 / MU0, 1.0, rtol=1e-6, atol=1e-8)
    assert np.isclose(sigma00, 0.0, rtol=1e-9, atol=1e-12)

    # And verify the entire arrays are uniform
    Nz, Ny, Nx = resolution[2], resolution[1], resolution[0]
    assert torch.allclose(eps_map, torch.full((Nz, Ny, Nx), EPS0))
    assert torch.allclose(mu_map, torch.full((Nz, Ny, Nx), MU0))
    assert torch.allclose(sigma_map, torch.zeros((Nz, Ny, Nx)))


if __name__ == "__main__":
    pytest.main()
