"""Dispersive E-field updates for 3D FDTD (Drude and Lorentz ADE).

Implements auxiliary differential equation (ADE) method for frequency-dependent
permittivity. Drude: dJ/dt = eps0*omega_p^2*E - gamma*J, dP/dt = J,
D = eps0*eps_inf*E + P. Lorentz: second-order in P. All docstrings in English.
References: Taflove & Hagness, Computational Electrodynamics, 3rd ed., Ch. 9.
"""

import tensorflow as tf
from emsim.constants import EPS0


def update_E_drude(
    Ex, Ey, Ez,
    Hx, Hy, Hz,
    Px, Py, Pz,
    Jx, Jy, Jz,
    mask,
    eps_inf: float,
    omega_p: float,
    gamma: float,
    dt: float,
    inv_dx, inv_dy, inv_dz,
    E_old_x=None,
    E_old_y=None,
    E_old_z=None,
):
    """Advance E and auxiliary J, P in Drude cells by one time step.

    Uses E at time n (E_old_* if provided, else current Ex,Ey,Ez) for J and D:
    J^{n+1} = a*J^n + b*E^n; P^{n+1} = P^n + dt*J^{n+1};
    D^{n+1} = eps0*eps_inf*E^n + P^n + dt*curl(H); E^{n+1} = (D^{n+1} - P^{n+1})/(eps0*eps_inf).
    When called after standard update_E, pass E_old so D uses pre-update E.

    Parameters
    ----------
    inv_dx, inv_dy, inv_dz : tf.Tensor
        Curl coefficients (shapes (1,1,Nx-1), (1,Ny-1,1), (Nz-1,1,1)).
    """
    if E_old_x is None:
        E_old_x = Ex[1:, 1:, :]
    if E_old_y is None:
        E_old_y = Ey[1:, :, 1:]
    if E_old_z is None:
        E_old_z = Ez[:, 1:, 1:]

    dHz_dy = (Hz[1:, 1:, :] - Hz[1:, :-1, :]) * inv_dy
    dHy_dz = (Hy[1:, 1:, :] - Hy[:-1, 1:, :]) * inv_dz
    curl_Hx = dHz_dy - dHy_dz
    dHx_dz = (Hx[1:, :, 1:] - Hx[:-1, :, 1:]) * inv_dz
    dHz_dx = (Hz[1:, :, 1:] - Hz[1:, :, :-1]) * inv_dx
    curl_Hy = dHx_dz - dHz_dx
    dHy_dx = (Hy[:, 1:, 1:] - Hy[:, 1:, :-1]) * inv_dx
    dHx_dy = (Hx[:, 1:, 1:] - Hx[:, :-1, 1:]) * inv_dy
    curl_Hz = dHy_dx - dHx_dy

    dt_f = tf.constant(dt, dtype=tf.float32)
    a_j = (2.0 - gamma * dt) / (2.0 + gamma * dt)
    b_j = (2.0 * dt * EPS0 * omega_p * omega_p) / (2.0 + gamma * dt)
    eps0_eps_inf = EPS0 * eps_inf

    Jx_s = Jx[1:, 1:, :]
    Jy_s = Jy[1:, :, 1:]
    Jz_s = Jz[:, 1:, 1:]
    Px_s = Px[1:, 1:, :]
    Py_s = Py[1:, :, 1:]
    Pz_s = Pz[:, 1:, 1:]
    m_x = mask[1:, 1:, :]
    m_y = mask[1:, :, 1:]
    m_z = mask[:, 1:, 1:]

    Jx_new = a_j * Jx_s + b_j * E_old_x
    Jy_new = a_j * Jy_s + b_j * E_old_y
    Jz_new = a_j * Jz_s + b_j * E_old_z
    Px_new = Px_s + dt_f * Jx_new
    Py_new = Py_s + dt_f * Jy_new
    Pz_new = Pz_s + dt_f * Jz_new
    Dx = eps0_eps_inf * E_old_x + Px_s + dt_f * curl_Hx
    Dy = eps0_eps_inf * E_old_y + Py_s + dt_f * curl_Hy
    Dz = eps0_eps_inf * E_old_z + Pz_s + dt_f * curl_Hz
    Ex_new = (Dx - Px_new) / eps0_eps_inf
    Ey_new = (Dy - Py_new) / eps0_eps_inf
    Ez_new = (Dz - Pz_new) / eps0_eps_inf

    Ex[1:, 1:, :].assign(m_x * Ex_new + (1.0 - m_x) * Ex[1:, 1:, :])
    Ey[1:, :, 1:].assign(m_y * Ey_new + (1.0 - m_y) * Ey[1:, :, 1:])
    Ez[:, 1:, 1:].assign(m_z * Ez_new + (1.0 - m_z) * Ez[:, 1:, 1:])
    Jx[1:, 1:, :].assign(m_x * Jx_new + (1.0 - m_x) * Jx_s)
    Jy[1:, :, 1:].assign(m_y * Jy_new + (1.0 - m_y) * Jy_s)
    Jz[:, 1:, 1:].assign(m_z * Jz_new + (1.0 - m_z) * Jz_s)
    Px[1:, 1:, :].assign(m_x * Px_new + (1.0 - m_x) * Px_s)
    Py[1:, :, 1:].assign(m_y * Py_new + (1.0 - m_y) * Py_s)
    Pz[:, 1:, 1:].assign(m_z * Pz_new + (1.0 - m_z) * Pz_s)


def update_E_anisotropic(
    Ex, Ey, Ez,
    Hx, Hy, Hz,
    mask,
    eps_r_xx: float,
    eps_r_yy: float,
    eps_r_zz: float,
    dt: float,
    inv_dx, inv_dy, inv_dz,
    E_old_x, E_old_y, E_old_z,
):
    """Overwrite E in anisotropic cells with diagonal tensor update.

    inv_dx, inv_dy, inv_dz : tf.Tensor, curl coefficients.
    """
    dHz_dy = (Hz[1:, 1:, :] - Hz[1:, :-1, :]) * inv_dy
    dHy_dz = (Hy[1:, 1:, :] - Hy[:-1, 1:, :]) * inv_dz
    curl_Hx = dHz_dy - dHy_dz
    dHx_dz = (Hx[1:, :, 1:] - Hx[:-1, :, 1:]) * inv_dz
    dHz_dx = (Hz[1:, :, 1:] - Hz[1:, :, :-1]) * inv_dx
    curl_Hy = dHx_dz - dHz_dx
    dHy_dx = (Hy[:, 1:, 1:] - Hy[:, 1:, :-1]) * inv_dx
    dHx_dy = (Hx[:, 1:, 1:] - Hx[:, :-1, 1:]) * inv_dy
    curl_Hz = dHy_dx - dHx_dy

    dt_f = tf.constant(dt, dtype=tf.float32)
    eps0_xx = EPS0 * eps_r_xx
    eps0_yy = EPS0 * eps_r_yy
    eps0_zz = EPS0 * eps_r_zz
    # D_new = eps*E_old + dt*curl_H; E_new = D_new/eps
    Dx = eps0_xx * E_old_x + dt_f * curl_Hx
    Dy = eps0_yy * E_old_y + dt_f * curl_Hy
    Dz = eps0_zz * E_old_z + dt_f * curl_Hz
    Ex_new = Dx / eps0_xx
    Ey_new = Dy / eps0_yy
    Ez_new = Dz / eps0_zz

    m_x = mask[1:, 1:, :]
    m_y = mask[1:, :, 1:]
    m_z = mask[:, 1:, 1:]
    Ex[1:, 1:, :].assign(m_x * Ex_new + (1.0 - m_x) * Ex[1:, 1:, :])
    Ey[1:, :, 1:].assign(m_y * Ey_new + (1.0 - m_y) * Ey[1:, :, 1:])
    Ez[:, 1:, 1:].assign(m_z * Ez_new + (1.0 - m_z) * Ez[:, 1:, 1:])


def update_E_with_dispersion(
    Ex, Ey, Ez, Hx, Hy, Hz,
    materials,
    inv_dx, inv_dy, inv_dz, dt: float,
):
    """Perform E-field update with dispersive (Drude) regions.

    Saves E at time n, runs standard E update everywhere, then overwrites E
    (and updates P, J) in Drude cells using saved E and the material grid's
    drude_mask and parameters. Call instead of update_E when
    materials.dispersive_regions is non-empty.

    Parameters
    ----------
    Ex, Ey, Ez, Hx, Hy, Hz : tf.Variable
        Field components.
    materials : MaterialGrid
        Must have Ca, Cb, and if has_dispersive: Px–Jz, _drude_mask, _drude_*.
    dx, dy, dz, dt : float
        Grid and time step.
    """
    from emsim.fdtd.fields import update_E

    # Save E at time n for Drude D = eps0*eps_inf*E + P + dt*curl_H
    E_old_x = tf.identity(Ex[1:, 1:, :])
    E_old_y = tf.identity(Ey[1:, :, 1:])
    E_old_z = tf.identity(Ez[:, 1:, 1:])

    # Standard update everywhere
    update_E(Ex, Ey, Ez, Hx, Hy, Hz, materials.Ca, materials.Cb, inv_dx, inv_dy, inv_dz)

    if materials.dispersive_regions and getattr(materials, "_drude_mask", None) is not None:
        update_E_drude(
            Ex, Ey, Ez, Hx, Hy, Hz,
            materials.Px, materials.Py, materials.Pz,
            materials.Jx, materials.Jy, materials.Jz,
            materials._drude_mask,
            materials._drude_eps_inf,
            materials._drude_omega_p,
            materials._drude_gamma,
            dt, inv_dx, inv_dy, inv_dz,
            E_old_x, E_old_y, E_old_z,
        )

    if materials.anisotropic_regions and getattr(materials, "_aniso_mask", None) is not None:
        update_E_anisotropic(
            Ex, Ey, Ez, Hx, Hy, Hz,
            materials._aniso_mask,
            materials._aniso_eps_xx,
            materials._aniso_eps_yy,
            materials._aniso_eps_zz,
            dt, inv_dx, inv_dy, inv_dz,
            E_old_x, E_old_y, E_old_z,
        )
