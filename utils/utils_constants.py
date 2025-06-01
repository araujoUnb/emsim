# utils_constants.py

"""
Defines fundamental physical constants used throughout the FDTD solver
and any related EM modules (e.g. antenna routines).
"""

import math

# Vacuum permittivity [F/m]
EPS0: float = 8.8541878128e-12

# Vacuum permeability [H/m]
MU0: float = 4.0e-7 * math.pi

# Speed of light in vacuum [m/s]
C0: float = 1.0 / math.sqrt(EPS0 * MU0)
