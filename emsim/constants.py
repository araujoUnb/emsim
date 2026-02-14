"""Physical constants for electromagnetic simulation."""

import math

EPS0 = 8.8541878128e-12           # Vacuum permittivity [F/m]
MU0 = 4.0e-7 * math.pi           # Vacuum permeability [H/m]
C0 = 1.0 / math.sqrt(EPS0 * MU0) # Speed of light [m/s]
ETA0 = math.sqrt(MU0 / EPS0)     # Free-space impedance [ohm] ~376.73
