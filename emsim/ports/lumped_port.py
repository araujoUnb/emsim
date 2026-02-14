"""Lumped port for point-source excitation and impedance measurement.

A lumped port represents a localized voltage source with internal resistance,
suitable for feeding antennas, microstrip lines, and other circuit-like structures.
"""

from typing import Dict, Any

import numpy as np
import tensorflow as tf
from scipy.fft import fft, fftfreq


class LumpedPort:
    """Lumped (point) port with internal resistance.
    
    This port injects voltage at a single grid cell and measures the resulting
    current to compute input impedance Z_in = V/I in the frequency domain.
    
    Parameters
    ----------
    name : str
        Unique identifier for this port.
    i, j, k : int
        Grid indices where the port is located.
    direction : str
        Field component direction ('x', 'y', or 'z').
    resistance : float, optional
        Internal resistance [Ohm] for impedance normalization (default: 50.0).
    
    Attributes
    ----------
    V_record : list of float
        Time series of voltage values.
    I_record : list of float
        Time series of current values.
    """
    
    def __init__(self, name: str, i: int, j: int, k: int,
                 direction: str, resistance: float = 50.0):
        if direction not in ('x', 'y', 'z'):
            raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}")
        
        self.name = name
        self.position = (i, j, k)
        self.direction = direction
        self.resistance = resistance
        self.V_record = []
        self.I_record = []
    
    def inject(self, field: tf.Variable, amplitude: float, dl: float):
        """Inject voltage as a soft source: E += V/dl.
        
        Parameters
        ----------
        field : tf.Variable
            Electric field component (Ex, Ey, or Ez) to inject into.
        amplitude : float
            Voltage amplitude at this time step [V].
        dl : float
            Length over which voltage is applied [m].
        """
        i, j, k = self.position
        # Soft-source injection: add to existing field
        field[k, j, i].assign_add(amplitude / dl)
    
    def record(self, E_field: tf.Variable, H_tangential_1: tf.Variable,
               H_tangential_2: tf.Variable, dl: float, ds: float):
        """Record voltage V and compute current I via Ampère's law.
        
        Parameters
        ----------
        E_field : tf.Variable
            Electric field component at the port location.
        H_tangential_1, H_tangential_2 : tf.Variable
            Magnetic field components tangential to the port direction.
        dl : float
            Length element for voltage [m].
        ds : float
            Surface element for current [m²].
        
        Notes
        -----
        Voltage: V = E * dl
        Current: I ≈ ∮ H·dl ≈ (H1 - H2) * ds (finite difference approximation)
        """
        i, j, k = self.position
        
        # Voltage at port
        V = float(E_field[k, j, i].numpy()) * dl
        self.V_record.append(V)
        
        # Current from circulation of H around the port
        # Simplified: use difference of adjacent H components
        if self.direction == 'z':
            # For Ez port: use Hx and Hy
            I = float((H_tangential_1[k, j, i] - H_tangential_2[k, j, i]).numpy()) * ds
        elif self.direction == 'y':
            # For Ey port: use Hx and Hz
            I = float((H_tangential_1[k, j, i] - H_tangential_2[k, j, i]).numpy()) * ds
        elif self.direction == 'x':
            # For Ex port: use Hy and Hz
            I = float((H_tangential_1[k, j, i] - H_tangential_2[k, j, i]).numpy()) * ds
        else:
            I = 0.0
        
        self.I_record.append(I)
    
    def reset(self):
        """Clear all temporal records."""
        self.V_record = []
        self.I_record = []
    
    def compute_result(self, dt: float, **kwargs) -> Dict[str, Any]:
        """Compute input impedance Z_in = V/I in the frequency domain.
        
        Parameters
        ----------
        dt : float
            Time step between recordings [s].
        **kwargs
            Additional parameters (currently unused).
        
        Returns
        -------
        dict
            Results dictionary containing:
            - 'type': 'lumped'
            - 'freqs': frequency array [Hz] (positive frequencies only)
            - 'Z_in': complex impedance array [Ohm]
            - 'S11': complex reflection coefficient
            - 'V_record': time-domain voltage
            - 'I_record': time-domain current
        """
        if not self.V_record or not self.I_record:
            raise ValueError(f"Port {self.name} has no recorded data. Call record() during simulation.")
        
        # FFT to frequency domain
        V_fft = fft(self.V_record)
        I_fft = fft(self.I_record)
        freq_axis = fftfreq(len(self.V_record), dt)
        
        # Impedance Z_in = V/I (avoid division by zero)
        Z_in = np.where(np.abs(I_fft) > 1e-12, V_fft / I_fft, 0.0 + 0.0j)
        
        # Reflection coefficient S11 = (Z_in - Z0) / (Z_in + Z0)
        Z0 = self.resistance
        S11 = (Z_in - Z0) / (Z_in + Z0)
        
        # Return only positive frequencies
        pos_freq_mask = freq_axis >= 0
        
        return {
            'type': 'lumped',
            'freqs': freq_axis[pos_freq_mask],
            'Z_in': Z_in[pos_freq_mask],
            'S11': S11[pos_freq_mask],
            'V_record': self.V_record,
            'I_record': self.I_record,
        }
