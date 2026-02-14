"""Perfect Electric Conductor (PEC) boundary condition.

Sets tangential E-field components to zero on the specified faces and regions.
"""

import tensorflow as tf


def apply_pec(Ex, Ey, Ez, faces):
    """Zero the tangential E-field components on the given boundary faces.

    Parameters
    ----------
    Ex, Ey, Ez : tf.Variable, shape [Nz, Ny, Nx]
    faces : set of str
        Which faces to enforce PEC. Subset of
        {'x-', 'x+', 'y-', 'y+', 'z-', 'z+'}.
    """
    if 'x-' in faces:
        # Tangential to x-face: Ey, Ez
        Ey[:, :, 0].assign(tf.zeros_like(Ey[:, :, 0]))
        Ez[:, :, 0].assign(tf.zeros_like(Ez[:, :, 0]))

    if 'x+' in faces:
        Ey[:, :, -1].assign(tf.zeros_like(Ey[:, :, -1]))
        Ez[:, :, -1].assign(tf.zeros_like(Ez[:, :, -1]))

    if 'y-' in faces:
        # Tangential to y-face: Ex, Ez
        Ex[:, 0, :].assign(tf.zeros_like(Ex[:, 0, :]))
        Ez[:, 0, :].assign(tf.zeros_like(Ez[:, 0, :]))

    if 'y+' in faces:
        Ex[:, -1, :].assign(tf.zeros_like(Ex[:, -1, :]))
        Ez[:, -1, :].assign(tf.zeros_like(Ez[:, -1, :]))

    if 'z-' in faces:
        # Tangential to z-face: Ex, Ey
        Ex[0, :, :].assign(tf.zeros_like(Ex[0, :, :]))
        Ey[0, :, :].assign(tf.zeros_like(Ey[0, :, :]))

    if 'z+' in faces:
        Ex[-1, :, :].assign(tf.zeros_like(Ex[-1, :, :]))
        Ey[-1, :, :].assign(tf.zeros_like(Ey[-1, :, :]))


def apply_pec_patch(Ex, Ey, Ez, i_range: tuple, j_range: tuple, k: int,
                    normal: str = 'z'):
    """Apply PEC to a 2D internal region (patch or ground plane).
    
    This zeroes tangential E-field components on an internal conductor
    surface, such as a microstrip patch or ground plane.
    
    Parameters
    ----------
    Ex, Ey, Ez : tf.Variable, shape [Nz, Ny, Nx]
        Electric field components.
    i_range : (i_min, i_max)
        Index range in x direction (inclusive).
    j_range : (j_min, j_max)
        Index range in y direction (inclusive).
    k : int
        Index in z direction where the patch is located.
    normal : str, optional
        Direction normal to the patch surface ('x', 'y', or 'z').
        Default is 'z' for horizontal patches.
    
    Examples
    --------
    Ground plane at z=0:
    >>> apply_pec_patch(Ex, Ey, Ez, i_range=(10, 50), j_range=(10, 50),
    ...                 k=0, normal='z')
    
    Patch at z=k_patch:
    >>> apply_pec_patch(Ex, Ey, Ez, i_range=(20, 40), j_range=(25, 45),
    ...                 k=k_patch, normal='z')
    """
    i_min, i_max = i_range
    j_min, j_max = j_range
    
    if normal == 'z':
        # Patch in xy-plane: zero tangential components Ex and Ey
        Ex[k, j_min:j_max, i_min:i_max].assign(
            tf.zeros_like(Ex[k, j_min:j_max, i_min:i_max])
        )
        Ey[k, j_min:j_max, i_min:i_max].assign(
            tf.zeros_like(Ey[k, j_min:j_max, i_min:i_max])
        )
    
    elif normal == 'y':
        # Patch in xz-plane: zero tangential components Ex and Ez
        Ex[k:k+1, j_min, i_min:i_max].assign(
            tf.zeros_like(Ex[k:k+1, j_min, i_min:i_max])
        )
        Ez[k:k+1, j_min, i_min:i_max].assign(
            tf.zeros_like(Ez[k:k+1, j_min, i_min:i_max])
        )
    
    elif normal == 'x':
        # Patch in yz-plane: zero tangential components Ey and Ez
        Ey[k:k+1, j_min:j_max, i_min].assign(
            tf.zeros_like(Ey[k:k+1, j_min:j_max, i_min])
        )
        Ez[k:k+1, j_min:j_max, i_min].assign(
            tf.zeros_like(Ez[k:k+1, j_min:j_max, i_min])
        )
    
    else:
        raise ValueError(f"normal must be 'x', 'y', or 'z', got {normal!r}")
