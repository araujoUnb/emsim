"""Soft-source injection for FDTD excitation.

Adds the source field to the existing field (soft source) rather than
overwriting it (hard source), which avoids artificial reflections at the
source plane.
"""

import tensorflow as tf


def inject_soft_source(field, k_plane, mode_profile_2d, amplitude):
    """Add a source contribution at a z-plane.

    Parameters
    ----------
    field : tf.Variable, shape [Nz, Ny, Nx]
        The E-field component to inject into (typically Ey).
    k_plane : int
        The z-index of the injection plane.
    mode_profile_2d : tf.Tensor, shape [Ny, Nx]
        Spatial mode profile (e.g. TE10).
    amplitude : float or tf.Tensor (scalar)
        Time-dependent source amplitude at the current time step.
    """
    current = field[k_plane, :, :]
    field[k_plane, :, :].assign(current + amplitude * mode_profile_2d)
