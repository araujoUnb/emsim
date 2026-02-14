"""Port definition for field recording and source injection in FDTD."""

from typing import Dict, Any

import tensorflow as tf


class Port:
    """A waveguide port at a fixed z-plane for source injection and field recording.
    
    This class implements the PortBase protocol for modal waveguide ports.

    Parameters
    ----------
    name           : str   identifier (e.g. "port1")
    k_plane        : int   z-index of the port plane
    mode_profile_E : tf.Tensor [Ny, Nx]  transverse E-field mode profile (Ey for TE)
    mode_profile_H : tf.Tensor [Ny, Nx]  transverse H-field mode profile (Hx for TE)
    direction      : int   +1 for forward (z+), -1 for backward (z-)
    """

    def __init__(self, name: str, k_plane: int, mode_profile_E, mode_profile_H, direction: int = 1):
        self.name = name
        self.k_plane = k_plane
        self.mode_profile_E = tf.cast(mode_profile_E, tf.float32)
        self.mode_profile_H = tf.cast(mode_profile_H, tf.float32)
        self.direction = direction
        self._E_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        self._H_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        self._record_index = 0

    def record(self, Ey, Hx, dy, dx):
        """Compute and store the modal overlap integrals at this port.

        Computes:
            overlap_E = sum(Ey[k, :, :] * mode_profile_E) * dy * dx
            overlap_H = sum(Hx[k, :, :] * mode_profile_H) * dy * dx
        """
        E_slice = Ey[self.k_plane, :, :]
        overlap_E = tf.reduce_sum(E_slice * self.mode_profile_E) * dy * dx
        H_slice = Hx[self.k_plane, :, :]
        overlap_H = tf.reduce_sum(H_slice * self.mode_profile_H) * dy * dx
        self._E_array = self._E_array.write(self._record_index, overlap_E)
        self._H_array = self._H_array.write(self._record_index, overlap_H)
        self._record_index += 1

    def reset(self):
        """Clear all temporal records."""
        self._E_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        self._H_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        self._record_index = 0

    @property
    def E_record(self) -> list:
        """Time series of E-field overlap (list; from TensorArray when read)."""
        return self._E_array.stack().numpy().tolist() if self._record_index > 0 else []

    @property
    def H_record(self) -> list:
        """Time series of H-field overlap (list; from TensorArray when read)."""
        return self._H_array.stack().numpy().tolist() if self._record_index > 0 else []

    def compute_result(self, dt: float, **kwargs) -> Dict[str, Any]:
        """Return recorded data for S-parameter calculation.
        
        Parameters
        ----------
        dt : float
            Time step [s] (not used here, needed for interface compliance).
        **kwargs
            Additional parameters (not used for modal ports).
        
        Returns
        -------
        dict
            Results dictionary containing:
            - 'type': 'modal'
            - 'E_record': list of E-field overlap integrals
            - 'H_record': list of H-field overlap integrals
        """
        E_record = self._E_array.stack().numpy().tolist() if self._record_index > 0 else []
        H_record = self._H_array.stack().numpy().tolist() if self._record_index > 0 else []
        return {
            'type': 'modal',
            'E_record': E_record,
            'H_record': H_record,
        }