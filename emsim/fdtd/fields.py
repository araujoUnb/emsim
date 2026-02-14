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

@tf.function(reduce_retracing=True)
def update_H(Ex, Ey, Ez, Hx, Hy, Hz, dt_over_mu, inv_dx, inv_dy, inv_dz):
    """Advance H fields by half a time step using the curl of E.

    Parameters
    ----------
    Ex, Ey, Ez : tf.Variable, shape [Nz, Ny, Nx]
    Hx, Hy, Hz : tf.Variable, shape [Nz, Ny, Nx]
    dt_over_mu : tf.Tensor, shape [Nz, Ny, Nx]  (precomputed dt/mu)
    inv_dx, inv_dy, inv_dz : tf.Tensor, shapes (1,1,Nx-1), (1,Ny-1,1), (Nz-1,1,1)
    """
    # curl_E_x = dEz/dy - dEy/dz
    dEz_dy = (Ez[:-1, 1:, :] - Ez[:-1, :-1, :]) * inv_dy
    dEy_dz = (Ey[1:, :-1, :] - Ey[:-1, :-1, :]) * inv_dz
    Hx[:-1, :-1, :].assign(
        Hx[:-1, :-1, :] - dt_over_mu[:-1, :-1, :] * (dEz_dy - dEy_dz)
    )

    # curl_E_y = dEx/dz - dEz/dx
    dEx_dz = (Ex[1:, :, :-1] - Ex[:-1, :, :-1]) * inv_dz
    dEz_dx = (Ez[:-1, :, 1:] - Ez[:-1, :, :-1]) * inv_dx
    Hy[:-1, :, :-1].assign(
        Hy[:-1, :, :-1] - dt_over_mu[:-1, :, :-1] * (dEx_dz - dEz_dx)
    )

    # curl_E_z = dEy/dx - dEx/dy
    dEy_dx = (Ey[:, :-1, 1:] - Ey[:, :-1, :-1]) * inv_dx
    dEx_dy = (Ex[:, 1:, :-1] - Ex[:, :-1, :-1]) * inv_dy
    Hz[:, :-1, :-1].assign(
        Hz[:, :-1, :-1] - dt_over_mu[:, :-1, :-1] * (dEy_dx - dEx_dy)
    )


# ---------------------------------------------------------------------------
# E-field update:  E^{n+1} = Ca * E^n + Cb * curl(H^{n+1/2})
# ---------------------------------------------------------------------------

@tf.function(reduce_retracing=True)
def update_E(Ex, Ey, Ez, Hx, Hy, Hz, Ca, Cb, inv_dx, inv_dy, inv_dz):
    """Advance E fields by a full time step using the curl of H.

    Parameters
    ----------
    Ex, Ey, Ez : tf.Variable, shape [Nz, Ny, Nx]
    Hx, Hy, Hz : tf.Variable, shape [Nz, Ny, Nx]
    Ca, Cb     : tf.Tensor, shape [Nz, Ny, Nx]  (precomputed coefficients)
    inv_dx, inv_dy, inv_dz : tf.Tensor, shapes (1,1,Nx-1), (1,Ny-1,1), (Nz-1,1,1)
    """
    # curl_H_x = dHz/dy - dHy/dz
    dHz_dy = (Hz[1:, 1:, :] - Hz[1:, :-1, :]) * inv_dy
    dHy_dz = (Hy[1:, 1:, :] - Hy[:-1, 1:, :]) * inv_dz
    Ex[1:, 1:, :].assign(
        Ca[1:, 1:, :] * Ex[1:, 1:, :] + Cb[1:, 1:, :] * (dHz_dy - dHy_dz)
    )

    # curl_H_y = dHx/dz - dHz/dx
    dHx_dz = (Hx[1:, :, 1:] - Hx[:-1, :, 1:]) * inv_dz
    dHz_dx = (Hz[1:, :, 1:] - Hz[1:, :, :-1]) * inv_dx
    Ey[1:, :, 1:].assign(
        Ca[1:, :, 1:] * Ey[1:, :, 1:] + Cb[1:, :, 1:] * (dHx_dz - dHz_dx)
    )

    # curl_H_z = dHy/dx - dHx/dy
    dHy_dx = (Hy[:, 1:, 1:] - Hy[:, 1:, :-1]) * inv_dx
    dHx_dy = (Hx[:, 1:, 1:] - Hx[:, :-1, 1:]) * inv_dy
    Ez[:, 1:, 1:].assign(
        Ca[:, 1:, 1:] * Ez[:, 1:, 1:] + Cb[:, 1:, 1:] * (dHy_dx - dHx_dy)
    )
