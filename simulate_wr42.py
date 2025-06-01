import torch
import numpy as np
import matplotlib.pyplot as plt

from objects.rectangular_waveguide import RectangularWaveguide
from utils.utils_rectangular_waveguide import TEModeProfile
from emsim.gaussianpulse import GaussianPulse
from emsim.torchFDTD import TorchFDTD

#------------------------------------------------------------------------------
# 1) WR-42 waveguide dimensions in meters (converted from micrometers)
#------------------------------------------------------------------------------
a_m = 10700e-6   # 0.0107 m (waveguide width)
b_m =  4300e-6   # 0.0043 m (waveguide height)
L_m = 50000e-6   # 0.050  m (waveguide length)

#------------------------------------------------------------------------------
# 2) Choose center frequency and bandwidth
#------------------------------------------------------------------------------
f_start   = 20e9   # 20 GHz
f_0       = 24e9   # 24 GHz
f_stop    = 26e9   # 26 GHz
bandwidth = f_stop - f_start  # 6 GHz

#------------------------------------------------------------------------------
# 3) Compute desired Δx = λ₀/30 and derive integer Nx, Ny, Nz → “snap” dx, dy, dz
#------------------------------------------------------------------------------
c0 = 2.99792458e8  # Speed of light [m/s]
lambda0 = c0 / f_0  # Free-space wavelength ≈ 0.01249 m

dx_desired = lambda0 / 30.0  # ≈ 0.0004163 m
dy_desired = dx_desired
dz_desired = dx_desired

from math import ceil

Nx = int(ceil(a_m / dx_desired))  # e.g. 0.0107/0.0004163 ≈ 25.7 → 26
Ny = int(ceil(b_m / dy_desired))  # e.g. 0.0043/0.0004163 ≈ 10.3 → 11
Nz = int(ceil(L_m / dz_desired))  # e.g. 0.050/0.0004163 ≈ 120.1 → 121

# Snap grid spacings so Nx·dx = a_m, etc.
dx = a_m / Nx
dy = b_m / Ny
dz = L_m / Nz

print(f"Fixed resolution: Nx={Nx}, Ny={Ny}, Nz={Nz}")
print(f"  => Δx = {dx:.4e} m, Δy = {dy:.4e} m, Δz = {dz:.4e} m")

#------------------------------------------------------------------------------
# 4) Build the RectangularWaveguide with exactly (Nx,Ny,Nz) points
#------------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

geom = RectangularWaveguide(
    a=a_m,
    b=b_m,
    length=L_m,
    resolution=(Nx, Ny, Nz),
    material=None,   # vacuum (εᵣ=1, μᵣ=1, σ=0)
    device=device
)

# Print actual grid spacing inside geometry (linspace uses Nx points → Δx = a_m/(Nx-1))
actual_dx = geom.x_lin[1].item() - geom.x_lin[0].item()
actual_dy = geom.y_lin[1].item() - geom.y_lin[0].item()
actual_dz = geom.z_lin[1].item() - geom.z_lin[0].item()

print("Geometry’s grid spacing (from torch.linspace):")
print(f"  actual Δx = {actual_dx:.4e} m  (should be close to {dx:.4e} m)")
print(f"  actual Δy = {actual_dy:.4e} m  (should be close to {dy:.4e} m)")
print(f"  actual Δz = {actual_dz:.4e} m  (should be close to {dz:.4e} m)")

#------------------------------------------------------------------------------
# 5) Build TE₁₀ mode profile
#------------------------------------------------------------------------------
mode_te10 = TEModeProfile(
    a=a_m,
    b=b_m,
    m=1,
    n=0,
    x_lin=geom.x_lin,
    y_lin=geom.y_lin,
    device=device
)

# Quick sanity-check:
mode2d = mode_te10.generate().cpu().numpy()
assert mode2d.shape == (Ny, Nx), "Mode profile must be [Ny, Nx]"
print(f"TE₁₀ mode2d   max = {mode2d.max():.3f},   min = {mode2d.min():.3f}")

#------------------------------------------------------------------------------
# 6) Create GaussianPulse
#------------------------------------------------------------------------------
pulse = GaussianPulse(
    f0=f_0,
    bandwidth=bandwidth,
    device=device
)

#------------------------------------------------------------------------------
# 7) Boundary conditions: PEC on X±, Y±; PML on Z±
#------------------------------------------------------------------------------
bc_list = ["PEC", "PEC", "PEC", "PEC", "PML", "PML"]

#------------------------------------------------------------------------------
# 8) Instantiate TorchFDTD solver
#------------------------------------------------------------------------------
solver = TorchFDTD(
    geometry=geom,
    input_ports=None,
    output_ports=None,
    f0=f_0,
    dt_factor=0.99,
    bc_list=bc_list,
    sim_box=None,
    sim_resolution=None
)

#------------------------------------------------------------------------------
# 9) Check the CFL number before running
#------------------------------------------------------------------------------
S = solver.check_cfl()
if S >= 1.0:
    raise RuntimeError(f"CFL number S = {S:.5f} is ≥ 1.0 → Unstable. Reduce dt_factor or increase dx/dy/dz.")

#------------------------------------------------------------------------------
# 10) Choose source & receiver planes
#------------------------------------------------------------------------------
pml_cells = 8
k_src = pml_cells
k_rec = geom.Nz - 1 - pml_cells
print(f"Using k_src = {k_src}, k_rec = {k_rec}")

#------------------------------------------------------------------------------
# 11) Run FDTD loop
#------------------------------------------------------------------------------
N_steps = 12000
record_interval = 300

results = solver.run(
    N_steps=N_steps,
    record_interval=record_interval,
    source=pulse,
    mode_profile=mode_te10,
    k_src=k_src,
    k_rec=k_rec
)

#------------------------------------------------------------------------------
# 12) Extract & plot S-parameters
#------------------------------------------------------------------------------
freqs = results["freqs"]  # length ~ N_steps//2 + 1
S11   = results["S11"]
S21   = results["S21"]

mask = np.logical_and(freqs >= f_start, freqs <= f_stop)

plt.figure(figsize=(6,4))
plt.plot(freqs[mask]*1e-9, 20*np.log10(np.abs(S11[mask])), 'k-', label='S11 (dB)')
plt.plot(freqs[mask]*1e-9, 20*np.log10(np.abs(S21[mask])), 'r--', label='S21 (dB)')
plt.xlabel("Frequency (GHz)")
plt.ylabel("Magnitude (dB)")
plt.title("WR-42 TE₁₀ S‐Parameters (TorchFDTD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# 13) (Optional) Plot first Ez snapshot at mid-plane
#------------------------------------------------------------------------------
Ez_list = results["Ez_snapshots"]
if Ez_list:
    Ez0 = Ez_list[0].cpu().numpy()  # [Ny, Nx]
    plt.figure(figsize=(5,4))
    plt.imshow(
        Ez0,
        origin='lower',
        extent=(0, a_m*1e3, 0, b_m*1e3),  # mm axis
        cmap='RdBu',
        aspect='auto'
    )
    plt.colorbar(label='Ez (V/m)')
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.title("Ez at mid‐plane (first snapshot)")
    plt.tight_layout()
    plt.show()
