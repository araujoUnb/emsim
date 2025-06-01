# plot_pml_sigma.py

import numpy as np
import matplotlib.pyplot as plt
from emsim.boundary_conditions import PMLBoundary

# ------------------------------------------------------------------
# 1) Set up and extract the PML conductivity profile
# ------------------------------------------------------------------

N_pml = 8          # number of PML cells
dt    = 1e-12      # time step [s]
dx    = 1e-3       # grid spacing in x [m] (for a PML along the X face)
m     = 3          # polynomial order for σ grading
R_inf = 1e-6       # desired reflection coefficient at outer PML boundary

# Instantiate a PMLBoundary for the "X-" face
pml = PMLBoundary(face="X-", dt=dt, dx=dx, N_pml=N_pml, m=m, R_inf=R_inf)

# The attribute sigma_profile is a 1D tensor of length N_pml
sigma_profile = pml.sigma_profile.numpy()

# ------------------------------------------------------------------
# 2) Plot σ vs. PML cell index
# ------------------------------------------------------------------

plt.figure(figsize=(6, 4))
plt.plot(np.arange(1, N_pml + 1), sigma_profile, marker="o", linestyle="-")
plt.title("PML Conductivity Profile (σ) vs. Cell Index")
plt.xlabel("PML Cell Index (1 = interface → 8 = outermost)")
plt.ylabel("Conductivity σ [S/m]")
plt.grid(True)
plt.tight_layout()
plt.show()
