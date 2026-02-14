"""Analytical mode profiles and cutoff frequencies for rectangular waveguides.

Mode profiles account for Yee grid staggering: Ey lives at half-integer x
positions and integer y positions, while Hx lives at integer x positions
and half-integer y positions.
"""

import math

import tensorflow as tf

from emsim.constants import C0


def te_mode_profile(m: int, n: int, a: float, b: float, Ny: int, Nx: int,
                    dx: float = None, dy: float = None):
    """Generate the transverse E-field (Ey) profile for the TE_mn mode.

    On the Yee grid Ey is located at (i, j+1/2, k) — i.e. it is staggered
    by half a cell in the x-direction relative to the cell corner.  We
    sample the analytical profile at these staggered positions so that
    the discrete profile matches the physical field locations.

    Normalized such that sum(|profile|^2) * dx * dy = 1.

    Parameters
    ----------
    m, n : int   mode indices (m >= 1 for TE)
    a    : float waveguide width [m]
    b    : float waveguide height [m]
    Ny   : int   number of grid points in y
    Nx   : int   number of grid points in x
    dx   : float grid spacing in x (if None, computed as a/(Nx-1))
    dy   : float grid spacing in y (if None, computed as b/(Ny-1))

    Returns
    -------
    tf.Tensor of shape [Ny, Nx], power normalized.
    """
    if dx is None:
        dx = a / max(Nx - 1, 1)
    if dy is None:
        dy = b / max(Ny - 1, 1)

    # Ey lives at x = (i+0.5)*dx, y = j*dy  on the Yee grid.
    # For injection at a z-plane all Nx x-positions are valid.
    x = tf.constant([(i + 0.5) * dx for i in range(Nx)], dtype=tf.float32)
    y = tf.constant([j * dy for j in range(Ny)], dtype=tf.float32)
    X, Y = tf.meshgrid(x, y)  # [Ny, Nx]

    profile = tf.sin(float(m) * math.pi * X / a) * tf.cos(float(n) * math.pi * Y / b)

    power = tf.reduce_sum(profile ** 2) * dx * dy
    if power > 0:
        profile = profile / tf.sqrt(power)

    return profile


def hx_mode_profile(m: int, n: int, a: float, b: float, Ny: int, Nx: int,
                    dx: float = None, dy: float = None):
    """Generate the transverse H-field (Hx) profile for the TE_mn mode.

    On the Yee grid Hx is located at (i+1/2, j, k+1/2) — staggered by
    half a cell in y relative to Ey.  We sample at these positions.

    The spatial pattern is the same analytical function as Ey (for TE_mn)
    but evaluated at the Hx stagger points.  Power normalized so that
    sum(|profile|^2) * dx * dy = 1.

    Parameters
    ----------
    m, n : int   mode indices (m >= 1 for TE)
    a    : float waveguide width [m]
    b    : float waveguide height [m]
    Ny   : int   number of grid points in y
    Nx   : int   number of grid points in x
    dx   : float grid spacing in x (if None, computed as a/(Nx-1))
    dy   : float grid spacing in y (if None, computed as b/(Ny-1))

    Returns
    -------
    tf.Tensor of shape [Ny, Nx], power normalized.
    """
    if dx is None:
        dx = a / max(Nx - 1, 1)
    if dy is None:
        dy = b / max(Ny - 1, 1)

    # Hx lives at x = i*dx, y = (j+0.5)*dy  on the Yee grid.
    x = tf.constant([i * dx for i in range(Nx)], dtype=tf.float32)
    y = tf.constant([(j + 0.5) * dy for j in range(Ny)], dtype=tf.float32)
    X, Y = tf.meshgrid(x, y)  # [Ny, Nx]

    profile = tf.sin(float(m) * math.pi * X / a) * tf.cos(float(n) * math.pi * Y / b)

    power = tf.reduce_sum(profile ** 2) * dx * dy
    if power > 0:
        profile = profile / tf.sqrt(power)

    return profile


def tm_mode_profile(m: int, n: int, a: float, b: float, Ny: int, Nx: int):
    """Generate the Ez profile for the TM_mn mode.

    Ez(x, y) proportional to sin(m*pi*x/a) * sin(n*pi*y/b)
    """
    x = tf.linspace(0.0, a, Nx)
    y = tf.linspace(0.0, b, Ny)
    X, Y = tf.meshgrid(x, y)

    profile = tf.sin(float(m) * math.pi * X / a) * tf.sin(float(n) * math.pi * Y / b)
    peak = tf.reduce_max(tf.abs(profile))
    if peak > 0:
        profile = profile / peak
    return profile


def cutoff_frequency(m: int, n: int, a: float, b: float) -> float:
    """Cutoff frequency for TE_mn or TM_mn mode [Hz].

    f_c = (C0 / 2) * sqrt((m/a)^2 + (n/b)^2)
    """
    return (C0 / 2.0) * math.sqrt((m / a) ** 2 + (n / b) ** 2)


def mode_impedance(m: int, n: int, a: float, b: float, f: float) -> complex:
    """Wave impedance for TE_mn mode at frequency f [Ohm].

    For TE modes in a rectangular waveguide:
        Z_TE(f) = eta / sqrt(1 - (fc/f)^2)

    For f > fc the result is real (propagating mode).
    For f < fc the result is positive imaginary (evanescent mode):
        Z_TE = j * eta / sqrt((fc/f)^2 - 1)

    Parameters
    ----------
    m, n : int   mode indices
    a    : float waveguide width [m]
    b    : float waveguide height [m]
    f    : float operating frequency [Hz]

    Returns
    -------
    complex : mode impedance [Ohm]
    """
    fc = cutoff_frequency(m, n, a, b)
    eta = 376.73031346177  # mu0 * C0 [Ohm]

    if f <= 0:
        f = 1.0  # guard against zero/negative frequency

    ratio_sq = (fc / f) ** 2
    if ratio_sq < 1.0:
        # Propagating: real impedance
        return complex(eta / math.sqrt(1.0 - ratio_sq))
    else:
        # Evanescent: positive imaginary impedance  Z = +j*eta/sqrt((fc/f)^2-1)
        return 1j * eta / math.sqrt(ratio_sq - 1.0)
