# run_wr42.py

"""
WR-42 Rectangular Waveguide S‐Parameter Simulation using TorchFDTD.

This script:
  1) Defines the WR-42 dimensions (a, b, length) in meters.
  2) Chooses f0=24 GHz, bandwidth=6 GHz.
  3) Computes Δx ≈ λ₀/30, then derives integer Nx, Ny, Nz, and “snaps” dx, dy, dz.
  4) Builds RectangularWaveguide with resolution=(Nx,Ny,Nz).
  5) Defines TE₁₀ mode (via TEModeProfile).
  6) Creates a GaussianPulse at 24 GHz.
  7) Defines two ports (input & output) with mode_profile=TE₁₀.
  8) Instantiates TorchFDTD with PEC sidewalls and PML ends.
  9) Checks the CFL ratio.
 10) Runs the FDTD loop for N_steps and record_interval, injecting automatically.
 11) Computes and plots S₁₁ and S₂₁ between 20–26 GHz.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from objects.rectangular_waveguide import RectangularWaveguide
from utils.utils_rectangular_waveguide import TEModeProfile
from utils.utils_constants import C0
from emsim.gaussianpulse import GaussianPulse
from emsim.torchFDTD import TorchFDTD

#------------------------------------------------------------------------------
# 1) WR-42 waveguide dimensions in meters
#------------------------------------------------------------------------------
a_m = 10700e-6   # 0.0107 m
b_m =  4300e-6   # 0.0043 m
L_m = 50000e-6   # 0.050 m

#------------------------------------------------------------------------------
# 2) Center and bandwidth (Hz)
#------------------------------------------------------------------------------
f_start   = 20e9
f_0       = 24e9
f_stop    = 26e9
bandwidth = f_stop - f_start  # 6 GHz

#------------------------------------------------------------------------------
# 3) Compute λ₀, desired Δx = λ₀/30, derive Nx,Ny,Nz → “snap” dx,dy,dz
#------------------------------------------------------------------------------
lambda0 = C0 / f_0  # free-space wavelength

dx_desired = lambda0 / 30.0
dy_desired = dx_desired
dz_desired = dx_desired

from math import ceil

Nx = int(ceil(a_m / dx_desired))
Ny = int(ceil(b_m / dy_desired))
Nz = int(ceil(L_m / dz_desired))

dx = a_m / Nx
dy = b_m / Ny
dz = L_m / Nz

print(f"Fixed resolution: Nx={Nx}, Ny={Ny}, Nz={Nz}")
print(f"  => Δx = {dx:.4e} m, Δy = {dy:.4e} m, Δz = {dz:.4e} m")

#------------------------------------------------------------------------------
# 4) Build RectangularWaveguide with exactly (Nx, Ny, Nz) points
#------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

geom = RectangularWaveguide(
    a=a_m,
    b=b_m,
    length=L_m,
    resolution=(Nx, Ny, Nz),
    material=None,  # vacuum (εr=1, μr=1, σ=0)
    device=device
)



#------------------------------------------------------------------------------
# 5) Create TE₁₀ mode profile on [Ny, Nx]
#------------------------------------------------------------------------------
mode_te10 = TEModeProfile(
    a=a_m,
    b=b_m,
    m=1,
    n=0,
    x_lin=torch.linspace(0.0, a_m, Nx, device=device),
    y_lin=torch.linspace(0.0, b_m, Ny, device=device),
    device=device
)

mode2d = mode_te10.generate().cpu().numpy()
assert mode2d.shape == (Ny, Nx)
print(f"TE₁₀ mode2d   max = {mode2d.max():.3f},   min = {mode2d.min():.3f}")

#------------------------------------------------------------------------------
# 6) Create a GaussianPulse source (float32) at f0=24 GHz, bandwidth=6 GHz
#------------------------------------------------------------------------------
pulse = GaussianPulse(
    f0=f_0,
    bandwidth=bandwidth,
    device=device
)

#------------------------------------------------------------------------------
# 7) Define input_ports and output_ports lists
#------------------------------------------------------------------------------
pml_cells = 8
k_src = pml_cells
k_rec = geom.Nz - 1 - pml_cells

input_ports = [
    {
        'name'        : 'Port1_TE10',
        'location'    : ('z', k_src),
        'mode_profile': mode_te10,
        'source'      : pulse
    }
]

output_ports = [
    {
        'name'        : 'Port2_TE10',
        'location'    : ('z', k_rec),
        'mode_profile': mode_te10,
        'source'      : None
    }
]

print(f"Using k_src = {k_src}, k_rec = {k_rec}")
print("Input ports:", input_ports)
print("Output ports:", output_ports)

#------------------------------------------------------------------------------
# 8) Define boundary conditions: PEC on X±, Y±; PML on Z±
#------------------------------------------------------------------------------
bc_list = ["PEC", "PEC", "PEC", "PEC", "PML", "PML"]

#------------------------------------------------------------------------------
# 9) Instantiate TorchFDTD solver
#------------------------------------------------------------------------------
solver = TorchFDTD(
    geometry=geom,
    input_ports=input_ports,
    output_ports=output_ports,
    f0=f_0,
    dt_factor=0.99,
    bc_list=bc_list,
    sim_box=(0.0, a_m, 0.0, b_m, 0.0, L_m),
    sim_resolution=None,
    verbose=True
)

#------------------------------------------------------------------------------
# 10) Check CFL number
#------------------------------------------------------------------------------
S = solver.check_cfl()
if S >= 1.0:
    raise RuntimeError(f"CFL number S = {S:.5f} is ≥ 1.0 → Unstable! Adjust dt_factor or grid.")

#------------------------------------------------------------------------------
# 11) Run FDTD loop for N_steps time steps
#------------------------------------------------------------------------------
N_steps = 12000
record_interval = 300

results = solver.run(
    N_steps=N_steps,
    record_interval=record_interval
)

#------------------------------------------------------------------------------
# 12) Extract & plot S-parameters (20–26 GHz band)
#------------------------------------------------------------------------------
freqs    = results['freqs']         # 1D numpy array of frequencies
S_params = results['S_params']      # dict: (in_name,out_name) → complex array

# Reflection: S(Port1_TE10→Port1_TE10)
S11 = S_params[('Port1_TE10','Port1_TE10')]
# Transmission: S(Port1_TE10→Port2_TE10)
S21 = S_params[('Port1_TE10','Port2_TE10')]

# Restrict to [20 GHz, 26 GHz]
mask = np.logical_and(freqs >= f_start, freqs <= f_stop)

plt.figure(figsize=(6,4))
plt.plot(freqs[mask]*1e-9, 20*np.log10(np.abs(S11[mask])), 'k-',  label='S11 (dB)')
plt.plot(freqs[mask]*1e-9, 20*np.log10(np.abs(S21[mask])), 'r--', label='S21 (dB)')
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.title("WR-42 TE₁₀ S‐Parameters (20–26 GHz)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# 13) (Optional) Plot the first Ez snapshot at mid‐plane
#------------------------------------------------------------------------------
Ez_list = results['Ez_snapshots']
if Ez_list:
    Ez0 = Ez_list[0].cpu().numpy()  # [Ny+1, Nx]
    plt.figure(figsize=(5,4))
    plt.imshow(
        Ez0,
        origin='lower',
        extent=(0, a_m*1e3, 0, b_m*1e3),  # convert to mm for axis ticks
        cmap='RdBu',
        aspect='auto'
    )
    plt.colorbar(label='Ez (V/m)')
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.title("Ez at mid‐plane (first snapshot)")
    plt.tight_layout()
    plt.show()
