# tests/test_mode_profiles_rectangular.py

import torch
import numpy as np
import pytest

from utils.utils_rectangular_waveguide import TEModeProfile, TMModeProfile


@pytest.fixture
def rect_coords():
    """
    Provide a simple rectangular grid of x and y coordinates for testing.
    Let a = 1.0, b = 0.5, Nx = 50, Ny = 40.
    """
    a = 1.0
    b = 0.5
    Nx = 50
    Ny = 40
    x_lin = torch.linspace(0.0, a, Nx)
    y_lin = torch.linspace(0.0, b, Ny)
    return a, b, x_lin, y_lin, Nx, Ny


def test_te11_rectangular(rect_coords):
    a, b, x_lin, y_lin, Nx, Ny = rect_coords

    # TE11: m=1, n=1
    mode = TEModeProfile(a=a, b=b, m=1, n=1, x_lin=x_lin, y_lin=y_lin, device='cpu')
    profile = mode.generate()  # shape [Ny, Nx]
    assert isinstance(profile, torch.Tensor)
    assert profile.shape == (Ny, Nx)

    prof_np = profile.numpy()

    # Check zeros at all four PEC walls: x=0, x=a, y=0, y=b
    assert np.allclose(prof_np[:, 0], 0.0, atol=1e-6)
    assert np.allclose(prof_np[:, -1], 0.0, atol=1e-6)
    assert np.allclose(prof_np[0, :], 0.0, atol=1e-6)
    assert np.allclose(prof_np[-1, :], 0.0, atol=1e-6)

    # Check center value: x=a/2, y=b/2
    ix = (np.abs(x_lin.numpy() - a / 2)).argmin()
    iy = (np.abs(y_lin.numpy() - b / 2)).argmin()
    center_val = prof_np[iy, ix]
    # sin(pi * (a/2) / a) = sin(pi/2) = 1; same for y: so product = 1
    assert np.isclose(center_val, 1.0, atol=1e-2)


def test_tm11_rectangular(rect_coords):
    a, b, x_lin, y_lin, Nx, Ny = rect_coords

    # TM11: m=1, n=1
    mode = TMModeProfile(a=a, b=b, m=1, n=1, x_lin=x_lin, y_lin=y_lin, device='cpu')
    profile = mode.generate()
    assert isinstance(profile, torch.Tensor)
    assert profile.shape == (Ny, Nx)

    prof_np = profile.numpy()

    # TM11 also zero at all four PEC walls
    assert np.allclose(prof_np[:, 0], 0.0, atol=1e-6)
    assert np.allclose(prof_np[:, -1], 0.0, atol=1e-6)
    assert np.allclose(prof_np[0, :], 0.0, atol=1e-6)
    assert np.allclose(prof_np[-1, :], 0.0, atol=1e-6)

    # Center value check: sin(pi/2)*sin(pi/2) = 1
    ix = (np.abs(x_lin.numpy() - a / 2)).argmin()
    iy = (np.abs(y_lin.numpy() - b / 2)).argmin()
    center_val = prof_np[iy, ix]
    assert np.isclose(center_val, 1.0, atol=1e-2)


if __name__ == "__main__":
    pytest.main()
