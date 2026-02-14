"""Convolutional Perfectly Matched Layer (CFS-CPML) for all 6 faces.

Implements the full Complex Frequency-Shifted PML with three grading
profiles: sigma (loss), kappa (coordinate stretching) and alpha
(CFS attenuation for evanescent waves).

The standard field update is performed over the full domain first,
then the CPML psi-corrections are applied in the PML slab regions.

Each psi update and field correction uses slices that exactly match
the valid update range of the corresponding field component in fields.py:
  Hx: [:-1, :-1, :]   Hy: [:-1, :, :-1]   Hz: [:, :-1, :-1]
  Ex: [1:, 1:, :]      Ey: [1:, :, 1:]      Ez: [:, 1:, 1:]

For "-" faces the PML has N cells; for "+" faces it has N-1 cells
because the outermost cell has no forward-difference partner.

Reference:
  Roden & Gedney, "Convolution PML (CPML): An efficient FDTD
  implementation of the CFS-PML for arbitrary media", Microwave
  and Optical Technology Letters, 27(5), 2000.
  Taflove & Hagness, "Computational Electrodynamics", 3rd ed., Ch. 7.
"""

import numpy as np
import tensorflow as tf

from emsim.constants import EPS0, ETA0


def _compute_cpml_coeffs(N_pml, d_cell, dt, m=3, m_a=1,
                          sigma_ratio=0.8, kappa_max=7.0, alpha_max=0.05):
    """Compute CFS-CPML recursive coefficients for one axis direction.

    Parameters
    ----------
    N_pml      : int    Number of PML cells.
    d_cell     : float  Grid spacing for this axis [m].
    dt         : float  Time step [s].
    m          : int    Polynomial grading order for sigma and kappa (default 3).
    m_a        : int    Polynomial grading order for alpha (default 1).
    sigma_ratio: float  sigma_max = sigma_ratio * (m+1) / (eta * d_cell).
    kappa_max  : float  Maximum coordinate-stretching factor (>=1, 1=off).
    alpha_max  : float  Maximum CFS alpha [S/m] (attenuates evanescent modes).

    Returns
    -------
    (b_e, c_e, kappa_e, b_h, c_h, kappa_h) — arrays of length N_pml.
    Index 0 = outermost PML cell (highest sigma / kappa).
    """
    sigma_max = sigma_ratio * (m + 1) / (ETA0 * d_cell)

    # --- E-field positions (integer grid points) ---
    sigma_e = np.zeros(N_pml, dtype=np.float64)
    kappa_e = np.ones(N_pml, dtype=np.float64)
    alpha_e = np.zeros(N_pml, dtype=np.float64)
    for i in range(N_pml):
        rho = (N_pml - i) / N_pml          # 1 at outer edge, 0 at inner
        sigma_e[i] = sigma_max * rho ** m
        kappa_e[i] = 1.0 + (kappa_max - 1.0) * rho ** m
        alpha_e[i] = alpha_max * (1.0 - rho) ** m_a   # 0 at outer, max at inner

    denom_e = kappa_e * EPS0 + dt * (sigma_e + kappa_e * alpha_e)
    b_e = np.exp(-(sigma_e / kappa_e + alpha_e) * dt / EPS0)
    c_e = np.where(
        np.abs(sigma_e) + np.abs(alpha_e) > 1e-30,
        sigma_e * (b_e - 1.0) / (kappa_e * (sigma_e + kappa_e * alpha_e + 1e-30)),
        0.0,
    )

    # --- H-field positions (half-integer grid points, shifted by 0.5) ---
    sigma_h = np.zeros(N_pml, dtype=np.float64)
    kappa_h = np.ones(N_pml, dtype=np.float64)
    alpha_h = np.zeros(N_pml, dtype=np.float64)
    for i in range(N_pml):
        rho = (N_pml - i - 0.5) / N_pml
        if rho > 0:
            sigma_h[i] = sigma_max * rho ** m
            kappa_h[i] = 1.0 + (kappa_max - 1.0) * rho ** m
            alpha_h[i] = alpha_max * (1.0 - rho) ** m_a

    b_h = np.exp(-(sigma_h / kappa_h + alpha_h) * dt / EPS0)
    c_h = np.where(
        np.abs(sigma_h) + np.abs(alpha_h) > 1e-30,
        sigma_h * (b_h - 1.0) / (kappa_h * (sigma_h + kappa_h * alpha_h + 1e-30)),
        0.0,
    )

    return (b_e.astype(np.float32), c_e.astype(np.float32), kappa_e.astype(np.float32),
            b_h.astype(np.float32), c_h.astype(np.float32), kappa_h.astype(np.float32))


def _bx(c):
    return tf.reshape(c, [1, 1, -1])


def _by(c):
    return tf.reshape(c, [1, -1, 1])


def _bz(c):
    return tf.reshape(c, [-1, 1, 1])


class CPML:
    """CFS-CPML applied to selected boundary faces.

    Includes sigma (loss), kappa (coordinate stretching) and alpha
    (CFS evanescent-wave attenuation) grading profiles.
    """

    def __init__(self, Nz, Ny, Nx, N_pml, dx, dy, dz, dt,
                 dt_over_mu, Cb, pml_faces=None,
                 kappa_max=7.0, alpha_max=0.05):
        self.N = N_pml
        self.Nz, self.Ny, self.Nx = Nz, Ny, Nx
        self.dx, self.dy, self.dz = dx, dy, dz
        self._dom = dt_over_mu
        self._Cb = Cb
        self.pml_faces = pml_faces or set()  # Store which faces have PML
        N = N_pml
        Nm = N      # cells on minus faces
        Np = N - 1  # cells on plus faces (no fwd diff at boundary)

        # Compute coefficients for each axis (now includes kappa)
        bxe, cxe, kxe, bxh, cxh, kxh = _compute_cpml_coeffs(N, dx, dt, kappa_max=kappa_max, alpha_max=alpha_max)
        bye, cye, kye, byh, cyh, kyh = _compute_cpml_coeffs(N, dy, dt, kappa_max=kappa_max, alpha_max=alpha_max)
        bze, cze, kze, bzh, czh, kzh = _compute_cpml_coeffs(N, dz, dt, kappa_max=kappa_max, alpha_max=alpha_max)

        # Store 1/kappa for the field-update denominator correction.
        # In standard FDTD the finite-difference is divided by dx; with
        # coordinate stretching it becomes dx*kappa, so the correction
        # factor applied to the *difference* from the normal update is
        # (1/kappa - 1), i.e. we multiply the existing finite-difference
        # term by 1/kappa instead of 1.  The psi term handles the rest.
        # For simplicity we store 1/kappa and apply it as a multiplier
        # in the psi correction formula.

        # --- Minus-face coeffs (full N, index 0 = outermost) ---
        self._bxh_m = tf.constant(bxh);  self._cxh_m = tf.constant(cxh)
        self._byh_m = tf.constant(byh);  self._cyh_m = tf.constant(cyh)
        self._bzh_m = tf.constant(bzh);  self._czh_m = tf.constant(czh)
        self._bxe_m = tf.constant(bxe);  self._cxe_m = tf.constant(cxe)
        self._bye_m = tf.constant(bye);  self._cye_m = tf.constant(cye)
        self._bze_m = tf.constant(bze);  self._cze_m = tf.constant(cze)

        # --- Plus-face coeffs (N-1 innermost cells, reversed) ---
        self._bxh_p = tf.constant(bxh[::-1][:Np].copy())
        self._cxh_p = tf.constant(cxh[::-1][:Np].copy())
        self._byh_p = tf.constant(byh[::-1][:Np].copy())
        self._cyh_p = tf.constant(cyh[::-1][:Np].copy())
        self._bzh_p = tf.constant(bzh[::-1][:Np].copy())
        self._czh_p = tf.constant(czh[::-1][:Np].copy())
        self._bxe_p = tf.constant(bxe[::-1][:Np].copy())
        self._cxe_p = tf.constant(cxe[::-1][:Np].copy())
        self._bye_p = tf.constant(bye[::-1][:Np].copy())
        self._cye_p = tf.constant(cye[::-1][:Np].copy())
        self._bze_p = tf.constant(bze[::-1][:Np].copy())
        self._cze_p = tf.constant(cze[::-1][:Np].copy())

        # --- Kappa correction: (1/kappa - 1) for each axis/field/face ---
        # The PML stretches the derivative: dF/dx -> (1/kappa)*dF/dx + psi.
        # The normal FDTD already applied dF/dx.  The correction is:
        #   (1/kappa - 1)*dF/dx + psi
        # Minus faces (N cells)
        self._ikm1_xh_m = tf.constant((1.0/kxh - 1.0).astype(np.float32))
        self._ikm1_yh_m = tf.constant((1.0/kyh - 1.0).astype(np.float32))
        self._ikm1_zh_m = tf.constant((1.0/kzh - 1.0).astype(np.float32))
        self._ikm1_xe_m = tf.constant((1.0/kxe - 1.0).astype(np.float32))
        self._ikm1_ye_m = tf.constant((1.0/kye - 1.0).astype(np.float32))
        self._ikm1_ze_m = tf.constant((1.0/kze - 1.0).astype(np.float32))
        # Plus faces (N-1 cells, reversed)
        self._ikm1_xh_p = tf.constant(((1.0/kxh - 1.0)[::-1][:Np]).copy().astype(np.float32))
        self._ikm1_yh_p = tf.constant(((1.0/kyh - 1.0)[::-1][:Np]).copy().astype(np.float32))
        self._ikm1_zh_p = tf.constant(((1.0/kzh - 1.0)[::-1][:Np]).copy().astype(np.float32))
        self._ikm1_xe_p = tf.constant(((1.0/kxe - 1.0)[::-1][:Np]).copy().astype(np.float32))
        self._ikm1_ye_p = tf.constant(((1.0/kye - 1.0)[::-1][:Np]).copy().astype(np.float32))
        self._ikm1_ze_p = tf.constant(((1.0/kze - 1.0)[::-1][:Np]).copy().astype(np.float32))

        # --- H-update psi arrays ---
        # Field update regions:
        #   Hx: [:-1, :-1, :]  -> Nz-1, Ny-1, Nx
        #   Hy: [:-1, :, :-1]  -> Nz-1, Ny,   Nx-1
        #   Hz: [:, :-1, :-1]  -> Nz,   Ny-1, Nx-1

        # x-
        self.psi_Hyx_xm = tf.Variable(tf.zeros([Nz-1, Ny, Nm]))
        self.psi_Hzx_xm = tf.Variable(tf.zeros([Nz, Ny-1, Nm]))
        # x+
        self.psi_Hyx_xp = tf.Variable(tf.zeros([Nz-1, Ny, Np]))
        self.psi_Hzx_xp = tf.Variable(tf.zeros([Nz, Ny-1, Np]))

        # y-
        self.psi_Hxy_ym = tf.Variable(tf.zeros([Nz-1, Nm, Nx]))
        self.psi_Hzy_ym = tf.Variable(tf.zeros([Nz, Nm, Nx-1]))
        # y+
        self.psi_Hxy_yp = tf.Variable(tf.zeros([Nz-1, Np, Nx]))
        self.psi_Hzy_yp = tf.Variable(tf.zeros([Nz, Np, Nx-1]))

        # z-
        self.psi_Hxz_zm = tf.Variable(tf.zeros([Nm, Ny-1, Nx]))
        self.psi_Hyz_zm = tf.Variable(tf.zeros([Nm, Ny, Nx-1]))
        # z+
        self.psi_Hxz_zp = tf.Variable(tf.zeros([Np, Ny-1, Nx]))
        self.psi_Hyz_zp = tf.Variable(tf.zeros([Np, Ny, Nx-1]))

        # --- E-update psi arrays ---
        # Field update regions:
        #   Ex: [1:, 1:, :]  -> Nz-1, Ny-1, Nx
        #   Ey: [1:, :, 1:]  -> Nz-1, Ny,   Nx-1
        #   Ez: [:, 1:, 1:]  -> Nz,   Ny-1, Nx-1

        # x-
        self.psi_Eyx_xm = tf.Variable(tf.zeros([Nz-1, Ny, Nm]))
        self.psi_Ezx_xm = tf.Variable(tf.zeros([Nz, Ny-1, Nm]))
        # x+
        self.psi_Eyx_xp = tf.Variable(tf.zeros([Nz-1, Ny, Np]))
        self.psi_Ezx_xp = tf.Variable(tf.zeros([Nz, Ny-1, Np]))

        # y-
        self.psi_Exy_ym = tf.Variable(tf.zeros([Nz-1, Nm, Nx]))
        self.psi_Ezy_ym = tf.Variable(tf.zeros([Nz, Nm, Nx-1]))
        # y+
        self.psi_Exy_yp = tf.Variable(tf.zeros([Nz-1, Np, Nx]))
        self.psi_Ezy_yp = tf.Variable(tf.zeros([Nz, Np, Nx-1]))

        # z-
        self.psi_Exz_zm = tf.Variable(tf.zeros([Nm, Ny-1, Nx]))
        self.psi_Eyz_zm = tf.Variable(tf.zeros([Nm, Ny, Nx-1]))
        # z+
        self.psi_Exz_zp = tf.Variable(tf.zeros([Np, Ny-1, Nx]))
        self.psi_Eyz_zp = tf.Variable(tf.zeros([Np, Ny, Nx-1]))

    # ------------------------------------------------------------------
    def update_H(self, Ex, Ey, Ez, Hx, Hy, Hz):
        N = self.N
        dx, dy, dz = self.dx, self.dy, self.dz
        dom = self._dom

        # ---- x- face (i = 0..N-1) ----
        if 'x-' in self.pml_faces:
            d = (Ez[:-1, :, 1:N+1] - Ez[:-1, :, :N]) / dx
            self.psi_Hyx_xm.assign(
                _bx(self._bxh_m) * self.psi_Hyx_xm
                + _bx(self._cxh_m) * d)
            Hy[:-1, :, :N].assign(
                Hy[:-1, :, :N] + dom[:-1, :, :N] * (self.psi_Hyx_xm + _bx(self._ikm1_xh_m) * d))

            d = (Ey[:, :-1, 1:N+1] - Ey[:, :-1, :N]) / dx
            self.psi_Hzx_xm.assign(
                _bx(self._bxh_m) * self.psi_Hzx_xm
                + _bx(self._cxh_m) * d)
            Hz[:, :-1, :N].assign(
                Hz[:, :-1, :N] - dom[:, :-1, :N] * (self.psi_Hzx_xm + _bx(self._ikm1_xh_m) * d))

        # ---- x+ face (i = Nx-N .. Nx-2, giving N-1 cells) ----
        if 'x+' in self.pml_faces:
            s = self.Nx - N
            e = self.Nx
            d = (Ez[:-1, :, s+1:e] - Ez[:-1, :, s:e-1]) / dx
            self.psi_Hyx_xp.assign(
                _bx(self._bxh_p) * self.psi_Hyx_xp
                + _bx(self._cxh_p) * d)
            Hy[:-1, :, s:e-1].assign(
                Hy[:-1, :, s:e-1] + dom[:-1, :, s:e-1] * (self.psi_Hyx_xp + _bx(self._ikm1_xh_p) * d))

            d = (Ey[:, :-1, s+1:e] - Ey[:, :-1, s:e-1]) / dx
            self.psi_Hzx_xp.assign(
                _bx(self._bxh_p) * self.psi_Hzx_xp
                + _bx(self._cxh_p) * d)
            Hz[:, :-1, s:e-1].assign(
                Hz[:, :-1, s:e-1] - dom[:, :-1, s:e-1] * (self.psi_Hzx_xp + _bx(self._ikm1_xh_p) * d))

        # ---- y- face (j = 0..N-1) ----
        if 'y-' in self.pml_faces:
            d = (Ez[:-1, 1:N+1, :] - Ez[:-1, :N, :]) / dy
            self.psi_Hxy_ym.assign(
                _by(self._byh_m) * self.psi_Hxy_ym
                + _by(self._cyh_m) * d)
            Hx[:-1, :N, :].assign(
                Hx[:-1, :N, :] - dom[:-1, :N, :] * (self.psi_Hxy_ym + _by(self._ikm1_yh_m) * d))

            d = (Ex[:, 1:N+1, :-1] - Ex[:, :N, :-1]) / dy
            self.psi_Hzy_ym.assign(
                _by(self._byh_m) * self.psi_Hzy_ym
                + _by(self._cyh_m) * d)
            Hz[:, :N, :-1].assign(
                Hz[:, :N, :-1] + dom[:, :N, :-1] * (self.psi_Hzy_ym + _by(self._ikm1_yh_m) * d))

        # ---- y+ face ----
        if 'y+' in self.pml_faces:
            s = self.Ny - N
            e = self.Ny
            d = (Ez[:-1, s+1:e, :] - Ez[:-1, s:e-1, :]) / dy
            self.psi_Hxy_yp.assign(
                _by(self._byh_p) * self.psi_Hxy_yp
                + _by(self._cyh_p) * d)
            Hx[:-1, s:e-1, :].assign(
                Hx[:-1, s:e-1, :] - dom[:-1, s:e-1, :] * (self.psi_Hxy_yp + _by(self._ikm1_yh_p) * d))

            d = (Ex[:, s+1:e, :-1] - Ex[:, s:e-1, :-1]) / dy
            self.psi_Hzy_yp.assign(
                _by(self._byh_p) * self.psi_Hzy_yp
                + _by(self._cyh_p) * d)
            Hz[:, s:e-1, :-1].assign(
                Hz[:, s:e-1, :-1] + dom[:, s:e-1, :-1] * (self.psi_Hzy_yp + _by(self._ikm1_yh_p) * d))

        # ---- z- face (k = 0..N-1) ----
        if 'z-' in self.pml_faces:
            d = (Ey[1:N+1, :-1, :] - Ey[:N, :-1, :]) / dz
            self.psi_Hxz_zm.assign(
                _bz(self._bzh_m) * self.psi_Hxz_zm
                + _bz(self._czh_m) * d)
            Hx[:N, :-1, :].assign(
                Hx[:N, :-1, :] + dom[:N, :-1, :] * (self.psi_Hxz_zm + _bz(self._ikm1_zh_m) * d))

            d = (Ex[1:N+1, :, :-1] - Ex[:N, :, :-1]) / dz
            self.psi_Hyz_zm.assign(
                _bz(self._bzh_m) * self.psi_Hyz_zm
                + _bz(self._czh_m) * d)
            Hy[:N, :, :-1].assign(
                Hy[:N, :, :-1] - dom[:N, :, :-1] * (self.psi_Hyz_zm + _bz(self._ikm1_zh_m) * d))

        # ---- z+ face ----
        if 'z+' in self.pml_faces:
            s = self.Nz - N
            e = self.Nz
            d = (Ey[s+1:e, :-1, :] - Ey[s:e-1, :-1, :]) / dz
            self.psi_Hxz_zp.assign(
                _bz(self._bzh_p) * self.psi_Hxz_zp
                + _bz(self._czh_p) * d)
            Hx[s:e-1, :-1, :].assign(
                Hx[s:e-1, :-1, :] + dom[s:e-1, :-1, :] * (self.psi_Hxz_zp + _bz(self._ikm1_zh_p) * d))

            d = (Ex[s+1:e, :, :-1] - Ex[s:e-1, :, :-1]) / dz
            self.psi_Hyz_zp.assign(
                _bz(self._bzh_p) * self.psi_Hyz_zp
                + _bz(self._czh_p) * d)
            Hy[s:e-1, :, :-1].assign(
                Hy[s:e-1, :, :-1] - dom[s:e-1, :, :-1] * (self.psi_Hyz_zp + _bz(self._ikm1_zh_p) * d))

    # ------------------------------------------------------------------
    def update_E(self, Ex, Ey, Ez, Hx, Hy, Hz):
        N = self.N
        dx, dy, dz = self.dx, self.dy, self.dz
        Cb = self._Cb

        # ---- x- face ----
        if 'x-' in self.pml_faces:
            d = (Hz[1:, :, 1:N+1] - Hz[1:, :, :N]) / dx
            self.psi_Eyx_xm.assign(
                _bx(self._bxe_m) * self.psi_Eyx_xm
                + _bx(self._cxe_m) * d)
            Ey[1:, :, :N].assign(
                Ey[1:, :, :N] - Cb[1:, :, :N] * (self.psi_Eyx_xm + _bx(self._ikm1_xe_m) * d))

            d = (Hy[:, 1:, 1:N+1] - Hy[:, 1:, :N]) / dx
            self.psi_Ezx_xm.assign(
                _bx(self._bxe_m) * self.psi_Ezx_xm
                + _bx(self._cxe_m) * d)
            Ez[:, 1:, :N].assign(
                Ez[:, 1:, :N] + Cb[:, 1:, :N] * (self.psi_Ezx_xm + _bx(self._ikm1_xe_m) * d))

        # ---- x+ face ----
        if 'x+' in self.pml_faces:
            s = self.Nx - N
            e = self.Nx
            d = (Hz[1:, :, s+1:e] - Hz[1:, :, s:e-1]) / dx
            self.psi_Eyx_xp.assign(
                _bx(self._bxe_p) * self.psi_Eyx_xp
                + _bx(self._cxe_p) * d)
            Ey[1:, :, s:e-1].assign(
                Ey[1:, :, s:e-1] - Cb[1:, :, s:e-1] * (self.psi_Eyx_xp + _bx(self._ikm1_xe_p) * d))

            d = (Hy[:, 1:, s+1:e] - Hy[:, 1:, s:e-1]) / dx
            self.psi_Ezx_xp.assign(
                _bx(self._bxe_p) * self.psi_Ezx_xp
                + _bx(self._cxe_p) * d)
            Ez[:, 1:, s:e-1].assign(
                Ez[:, 1:, s:e-1] + Cb[:, 1:, s:e-1] * (self.psi_Ezx_xp + _bx(self._ikm1_xe_p) * d))

        # ---- y- face ----
        if 'y-' in self.pml_faces:
            d = (Hz[1:, 1:N+1, :] - Hz[1:, :N, :]) / dy
            self.psi_Exy_ym.assign(
                _by(self._bye_m) * self.psi_Exy_ym
                + _by(self._cye_m) * d)
            Ex[1:, :N, :].assign(
                Ex[1:, :N, :] + Cb[1:, :N, :] * (self.psi_Exy_ym + _by(self._ikm1_ye_m) * d))

            d = (Hx[:, 1:N+1, 1:] - Hx[:, :N, 1:]) / dy
            self.psi_Ezy_ym.assign(
                _by(self._bye_m) * self.psi_Ezy_ym
                + _by(self._cye_m) * d)
            Ez[:, :N, 1:].assign(
                Ez[:, :N, 1:] - Cb[:, :N, 1:] * (self.psi_Ezy_ym + _by(self._ikm1_ye_m) * d))

        # ---- y+ face ----
        if 'y+' in self.pml_faces:
            s = self.Ny - N
            e = self.Ny
            d = (Hz[1:, s+1:e, :] - Hz[1:, s:e-1, :]) / dy
            self.psi_Exy_yp.assign(
                _by(self._bye_p) * self.psi_Exy_yp
                + _by(self._cye_p) * d)
            Ex[1:, s:e-1, :].assign(
                Ex[1:, s:e-1, :] + Cb[1:, s:e-1, :] * (self.psi_Exy_yp + _by(self._ikm1_ye_p) * d))

            d = (Hx[:, s+1:e, 1:] - Hx[:, s:e-1, 1:]) / dy
            self.psi_Ezy_yp.assign(
                _by(self._bye_p) * self.psi_Ezy_yp
                + _by(self._cye_p) * d)
            Ez[:, s:e-1, 1:].assign(
                Ez[:, s:e-1, 1:] - Cb[:, s:e-1, 1:] * (self.psi_Ezy_yp + _by(self._ikm1_ye_p) * d))

        # ---- z- face ----
        if 'z-' in self.pml_faces:
            d = (Hy[1:N+1, 1:, :] - Hy[:N, 1:, :]) / dz
            self.psi_Exz_zm.assign(
                _bz(self._bze_m) * self.psi_Exz_zm
                + _bz(self._cze_m) * d)
            Ex[:N, 1:, :].assign(
                Ex[:N, 1:, :] - Cb[:N, 1:, :] * (self.psi_Exz_zm + _bz(self._ikm1_ze_m) * d))

            d = (Hx[1:N+1, :, 1:] - Hx[:N, :, 1:]) / dz
            self.psi_Eyz_zm.assign(
                _bz(self._bze_m) * self.psi_Eyz_zm
                + _bz(self._cze_m) * d)
            Ey[:N, :, 1:].assign(
                Ey[:N, :, 1:] + Cb[:N, :, 1:] * (self.psi_Eyz_zm + _bz(self._ikm1_ze_m) * d))

        # ---- z+ face ----
        if 'z+' in self.pml_faces:
            s = self.Nz - N
            e = self.Nz
            d = (Hy[s+1:e, 1:, :] - Hy[s:e-1, 1:, :]) / dz
            self.psi_Exz_zp.assign(
                _bz(self._bze_p) * self.psi_Exz_zp
                + _bz(self._cze_p) * d)
            Ex[s:e-1, 1:, :].assign(
                Ex[s:e-1, 1:, :] - Cb[s:e-1, 1:, :] * (self.psi_Exz_zp + _bz(self._ikm1_ze_p) * d))

            d = (Hx[s+1:e, :, 1:] - Hx[s:e-1, :, 1:]) / dz
            self.psi_Eyz_zp.assign(
                _bz(self._bze_p) * self.psi_Eyz_zp
                + _bz(self._cze_p) * d)
            Ey[s:e-1, :, 1:].assign(
                Ey[s:e-1, :, 1:] + Cb[s:e-1, :, 1:] * (self.psi_Eyz_zp + _bz(self._ikm1_ze_p) * d))
