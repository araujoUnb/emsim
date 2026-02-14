"""Vectorised Yee-grid field update functions for 3D FDTD.

All operations use tensor slicing — zero Python loops.
The Sullivan convention is used: every field array has shape [Nz, Ny, Nx]
and the half-cell stagger is implicit in the difference formulas.

Update order per time step:
  1. update_H  (H at n-1/2 → n+1/2 using E at n)
  2. update_E  (E at n   → n+1   using H at n+1/2)
"""

import tensorflow as tf


# ---------------------------------------------------------------------------
# H-field update:  H^{n+1/2} = H^{n-1/2} - (dt/mu) * curl(E^n)
# ---------------------------------------------------------------------------

def update_H(Ex, Ey, Ez, Hx, Hy, Hz, dt_over_mu, dx, dy, dz):
    """Advance H fields by half a time step using the curl of E.

    Parameters
    ----------
    Ex, Ey, Ez : tf.Variable, shape [Nz, Ny, Nx]
    Hx, Hy, Hz : tf.Variable, shape [Nz, Ny, Nx]
    dt_over_mu : tf.Tensor, shape [Nz, Ny, Nx]  (precomputed dt/mu)
    dx, dy, dz : float  grid spacings
    """
    # curl_E_x = dEz/dy - dEy/dz
    # Valid indices: k=0..Nz-2, j=0..Ny-2, i=0..Nx-1
    dEz_dy = (Ez[:-1, 1:, :] - Ez[:-1, :-1, :]) / dy
    dEy_dz = (Ey[1:, :-1, :] - Ey[:-1, :-1, :]) / dz
    Hx[:-1, :-1, :].assign(
        Hx[:-1, :-1, :] - dt_over_mu[:-1, :-1, :] * (dEz_dy - dEy_dz)
    )

    # curl_E_y = dEx/dz - dEz/dx
    # Valid indices: k=0..Nz-2, j=0..Ny-1, i=0..Nx-2
    dEx_dz = (Ex[1:, :, :-1] - Ex[:-1, :, :-1]) / dz
    dEz_dx = (Ez[:-1, :, 1:] - Ez[:-1, :, :-1]) / dx
    Hy[:-1, :, :-1].assign(
        Hy[:-1, :, :-1] - dt_over_mu[:-1, :, :-1] * (dEx_dz - dEz_dx)
    )

    # curl_E_z = dEy/dx - dEx/dy
    # Valid indices: k=0..Nz-1, j=0..Ny-2, i=0..Nx-2
    dEy_dx = (Ey[:, :-1, 1:] - Ey[:, :-1, :-1]) / dx
    dEx_dy = (Ex[:, 1:, :-1] - Ex[:, :-1, :-1]) / dy
    Hz[:, :-1, :-1].assign(
        Hz[:, :-1, :-1] - dt_over_mu[:, :-1, :-1] * (dEy_dx - dEx_dy)
    )


# ---------------------------------------------------------------------------
# E-field update:  E^{n+1} = Ca * E^n + Cb * curl(H^{n+1/2})
# ---------------------------------------------------------------------------

def update_E(Ex, Ey, Ez, Hx, Hy, Hz, Ca, Cb, dx, dy, dz):
    """Advance E fields by a full time step using the curl of H.

    Parameters
    ----------
    Ex, Ey, Ez : tf.Variable, shape [Nz, Ny, Nx]
    Hx, Hy, Hz : tf.Variable, shape [Nz, Ny, Nx]
    Ca, Cb     : tf.Tensor, shape [Nz, Ny, Nx]  (precomputed coefficients)
    dx, dy, dz : float  grid spacings
    """
    # curl_H_x = dHz/dy - dHy/dz
    # Valid indices: k=1..Nz-1, j=1..Ny-1, i=0..Nx-1
    dHz_dy = (Hz[1:, 1:, :] - Hz[1:, :-1, :]) / dy
    dHy_dz = (Hy[1:, 1:, :] - Hy[:-1, 1:, :]) / dz
    Ex[1:, 1:, :].assign(
        Ca[1:, 1:, :] * Ex[1:, 1:, :] + Cb[1:, 1:, :] * (dHz_dy - dHy_dz)
    )

    # curl_H_y = dHx/dz - dHz/dx
    # Valid indices: k=1..Nz-1, j=0..Ny-1, i=1..Nx-1
    dHx_dz = (Hx[1:, :, 1:] - Hx[:-1, :, 1:]) / dz
    dHz_dx = (Hz[1:, :, 1:] - Hz[1:, :, :-1]) / dx
    Ey[1:, :, 1:].assign(
        Ca[1:, :, 1:] * Ey[1:, :, 1:] + Cb[1:, :, 1:] * (dHx_dz - dHz_dx)
    )

    # curl_H_z = dHy/dx - dHx/dy
    # Valid indices: k=0..Nz-1, j=1..Ny-1, i=1..Nx-1
    dHy_dx = (Hy[:, 1:, 1:] - Hy[:, 1:, :-1]) / dx
    dHx_dy = (Hx[:, 1:, 1:] - Hx[:, :-1, 1:]) / dy
    Ez[:, 1:, 1:].assign(
        Ca[:, 1:, 1:] * Ez[:, 1:, 1:] + Cb[:, 1:, 1:] * (dHy_dx - dHx_dy)
    )
