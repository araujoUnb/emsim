# torchFDTD.py

"""
TorchFDTD: A 3D Yee‐grid FDTD solver in PyTorch that automatically iterates over
multiple input_ports and output_ports.  

Port definition:
    Each port is a dictionary with keys:
      'name'        : string identifier
      'location'    : ('z', k_index)      # currently only Z‐ports are supported
      'mode_profile': ModeProfile instance (e.g. TEModeProfile or TMModeProfile)
      'source'      : Source instance or None

Solver tasks:
  - Build a single time‐vector of length N_steps.
  - For each input_port, precompute its source waveform (e.g. GaussianPulse).
  - At each time step, inject every source onto its specified plane.
  - Update all H fields, apply H‐boundary conditions.
  - Update all E fields, apply E‐boundary conditions.
  - At each time step, compute an overlap integral on each port and store it.
  - Optionally record Ez snapshots every record_interval steps.
  - After the time loop, perform FFT for each port’s time‐series to compute S‐parameters.

A convergence criterion in the time domain is implemented: if the reflected signal
for every input port falls below a specified tolerance, the simulation stops early.
A stability (CFL) check is provided via check_cfl().

Everything is stored in float32 by default. The default grid spacing is λ₀/30 unless
overridden by sim_resolution. The verbose flag controls progress printing.
"""

import math
import torch
import numpy as np

from emsim.simulator import Simulator
from utils.utils_constants import C0
from emsim.boundary_conditions import PECBoundary, PMLBoundary
from emsim.gaussianpulse import GaussianPulse


class TorchFDTD(Simulator):
    """
    PyTorch‐based 3D FDTD solver using the Yee algorithm, with automatic port iteration,
    a convergence test, and a verbose flag for detailed printouts.

    Parameters
    ----------
    geometry       : Object3D
                     Provides x_min, x_max, y_min, y_max, z_min, z_max,
                     Nx, Ny, Nz, device, and methods get_permittivity_map(),
                     get_permeability_map(), get_conductivity_map().

    input_ports    : list of dict
                     Each dict must contain:
                       - 'name'        : unique string identifier
                       - 'location'    : ('z', k_index)  # only Z‐ports supported
                       - 'mode_profile': ModeProfile instance (e.g. TEModeProfile)
                       - 'source'      : Source instance (e.g. GaussianPulse)

    output_ports   : list of dict
                     Each dict must contain:
                       - 'name'        : unique string identifier
                       - 'location'    : ('z', k_index)
                       - 'mode_profile': ModeProfile instance used for overlap
                       - 'source'      : None  (no source at an output port)

    f0             : float
                     Center frequency [Hz] used to compute λ0 = c0 / f0.

    dt_factor      : float
                     Safety factor for the CFL condition (< 1.0).

    bc_list        : list of str, length 6
                     Order: ["X-", "X+", "Y-", "Y+", "Z-", "Z+"].
                     Each entry is "PEC" or "PML".

    sim_box        : tuple or None
                     If not None, overrides geometry’s domain bounds:
                     (x_min, x_max, y_min, y_max, z_min, z_max).

    sim_resolution : float or None
                     If not None, sets Δx = Δy = Δz = sim_resolution (meters),
                     overriding λ0/30. Otherwise Δ = λ0/30.

    verbose        : bool
                     If True, print detailed progress messages during run().

    conv_tol       : float
                     Convergence tolerance: if |reflected| falls below conv_tol for
                     all input ports, time loop stops early.
    """

    def __init__(self,
                 geometry,
                 input_ports: list,
                 output_ports: list,
                 f0: float,
                 dt_factor: float,
                 bc_list: list[str],
                 sim_box: tuple | None = None,
                 sim_resolution: float | None = None,
                 verbose: bool = False,
                 conv_tol: float = 1e-4):
        super().__init__(geometry=geometry)
        self.verbose = verbose
        self.conv_tol = conv_tol

        # 1) Determine simulation domain either from sim_box or geometry bounds
        if sim_box is None:
            self.x_min = geometry.x_min
            self.x_max = geometry.x_max
            self.y_min = geometry.y_min
            self.y_max = geometry.y_max
            self.z_min = geometry.z_min
            self.z_max = geometry.z_max
        else:
            (self.x_min, self.x_max,
             self.y_min, self.y_max,
             self.z_min, self.z_max) = sim_box

        # Compute physical lengths
        Lx = self.x_max - self.x_min
        Ly = self.y_max - self.y_min
        Lz = self.z_max - self.z_min

        # 2) Compute λ₀ and default grid spacing Δ = λ₀ / 30 (unless overridden)
        c0 = C0  # Speed of light in free space [m/s]
        self.f0 = f0
        self.lambda0 = c0 / f0  # Free-space wavelength [m]

        if sim_resolution is None:
            self.dx = self.lambda0 / 30.0
            self.dy = self.lambda0 / 30.0
            self.dz = self.lambda0 / 30.0
        else:
            self.dx = sim_resolution
            self.dy = sim_resolution
            self.dz = sim_resolution

        # 3) Determine integer grid counts Nx, Ny, Nz so that Nx·dx ≥ Lx, etc.
        from math import ceil
        self.Nx = int(ceil(Lx / self.dx))
        self.Ny = int(ceil(Ly / self.dy))
        self.Nz = int(ceil(Lz / self.dz))

        # Snap grid spacing so that Nx·dx = Lx exactly, etc.
        self.dx = Lx / self.Nx
        self.dy = Ly / self.Ny
        self.dz = Lz / self.Nz

        # 4) Compute time‐step Δt from 3D CFL condition
        inv_dx2 = (1.0 / self.dx) ** 2
        inv_dy2 = (1.0 / self.dy) ** 2
        inv_dz2 = (1.0 / self.dz) ** 2
        self.dt = dt_factor * (1.0 / c0) / math.sqrt(inv_dx2 + inv_dy2 + inv_dz2)

        # If verbose, test CFL at initialization
        if self.verbose:
            self.check_cfl()

        # 5) Build grid vectors (1D linspace) on the same device as geometry
        self.device = geometry.device
        self.x_lin = torch.linspace(self.x_min, self.x_max, self.Nx, device=self.device)
        self.y_lin = torch.linspace(self.y_min, self.y_max, self.Ny, device=self.device)
        self.z_lin = torch.linspace(self.z_min, self.z_max, self.Nz, device=self.device)

        # 6) Build a full 3D mesh (Z, Y, X) on the Yee‐grid
        Z, Y, X = torch.meshgrid(self.z_lin, self.y_lin, self.x_lin, indexing='ij')
        self.X = X  # shape [Nz, Ny, Nx]
        self.Y = Y
        self.Z = Z

        # 7) Fetch material maps (ε, μ, σ) from geometry, all shape [Nz, Ny, Nx]
        self.eps_map   = geometry.get_permittivity_map().to(self.device)   # ε₀·εᵣ
        self.mu_map    = geometry.get_permeability_map().to(self.device)   # μ₀·μᵣ
        self.sigma_map = geometry.get_conductivity_map().to(self.device)   # σ

        # 8) Allocate Yee‐fields (staggered in space) on the chosen device
        self.Hx = torch.zeros((self.Nz + 1, self.Ny,   self.Nx),   device=self.device)
        self.Hy = torch.zeros((self.Nz,   self.Ny + 1, self.Nx),   device=self.device)
        self.Hz = torch.zeros((self.Nz,   self.Ny,   self.Nx + 1), device=self.device)

        self.Ex = torch.zeros((self.Nz,   self.Ny + 1, self.Nx + 1), device=self.device)
        self.Ey = torch.zeros((self.Nz + 1, self.Ny,   self.Nx + 1), device=self.device)
        self.Ez = torch.zeros((self.Nz + 1, self.Ny + 1, self.Nx   ), device=self.device)

        # 9) Set up boundary‐condition objects (one per face)
        #    bc_list must be length 6: ["X-", "X+", "Y-", "Y+", "Z-", "Z+"]
        assert len(bc_list) == 6, "bc_list must have 6 entries [X-,X+,Y-,Y+,Z-,Z+]."
        faces = ["X-", "X+", "Y-", "Y+", "Z-", "Z+"]
        self.bc_list_objs = []

        for face_label, bc_type in zip(faces, bc_list):
            if bc_type.upper() == "PEC":
                self.bc_list_objs.append(PECBoundary(face_label))
                if self.verbose:
                    print(f"[Init] Assigned PEC boundary on face {face_label}")
            elif bc_type.upper() == "PML":
                # Decide which grid spacing to use for this face
                if face_label in {"X-", "X+"}:
                    d_loc = self.dx
                elif face_label in {"Y-", "Y+"}:
                    d_loc = self.dy
                else:  # "Z-" or "Z+"
                    d_loc = self.dz

                pml = PMLBoundary(
                    face=face_label,
                    dt=self.dt,
                    dx=d_loc,
                    N_pml=8,      # 8 cells thick by convention
                    m=3,          # cubic grading
                    R_inf=1e-6    # target reflection coefficient
                )
                pml.device = self.device
                # Allocate PML’s split‐field tensors now that Nz,Ny,Nx are known
                pml.allocate_tensors(Nz=self.Nz, Ny=self.Ny, Nx=self.Nx, device=self.device)
                self.bc_list_objs.append(pml)
                if self.verbose:
                    print(f"[Init] Assigned PML boundary on face {face_label} (8 cells)")
            else:
                raise ValueError(f"Unknown BC type '{bc_type}'. Use 'PEC' or 'PML'.")

        # 10) Store the input_ports and output_ports lists
        # Each entry in input_ports must be a dict with keys:
        #   'name', 'location'=('z',k_index), 'mode_profile', 'source'
        #
        # Each entry in output_ports must be a dict with keys:
        #   'name', 'location'=('z',k_index), 'mode_profile', 'source'=None
        self.input_ports  = input_ports
        self.output_ports = output_ports

        if self.verbose:
            print(f"[Init] Geometry resolution: Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}")
            print(f"[Init] Grid spacing: Δx={self.dx:.3e} m, Δy={self.dy:.3e} m, Δz={self.dz:.3e} m")

        # 11) Prepare a container for Ez snapshots (optional)
        self.Ez_records = []

    def check_cfl(self) -> float:
        """
        Compute the 3D CFL number:

            S = c0·Δt·sqrt((1/Δx)^2 + (1/Δy)^2 + (1/Δz)^2)

        Returns
        -------
        S : float
            The Courant number. If verbose=True, prints status.
        """
        c0 = C0
        dx, dy, dz = self.dx, self.dy, self.dz
        dt = self.dt

        inv_dx2 = (1.0 / dx) ** 2
        inv_dy2 = (1.0 / dy) ** 2
        inv_dz2 = (1.0 / dz) ** 2

        S = c0 * dt * math.sqrt(inv_dx2 + inv_dy2 + inv_dz2)
        if self.verbose:
            if S < 1.0:
                print(f"[CFL check] Courant number S = {S:.5f} → OK (below 1.0)")
            else:
                print(f"[CFL check] Courant number S = {S:.5f} → WARNING: ≥ 1.0 (unstable!)")
        return S

    def run(self, 
            N_steps: int, 
            record_interval: int = 1):
        """
        Run the FDTD time loop for N_steps steps, automatically iterating over
        every port in self.input_ports and self.output_ports.

        Implements a convergence test in the time domain: if the magnitude of the
        reflected overlap integral for every input port falls below self.conv_tol,
        the loop breaks early.

        If verbose=True, prints concise progress messages (one per sub‐step).

        Parameters
        ----------
        N_steps        : int
                         Number of time steps to run.
        record_interval: int
                         Every `record_interval` steps, save one Ez snapshot at mid‐plane.

        Returns
        -------
        results : dict
            {
              'freqs'        : 1D numpy array of positive frequencies,
              'S_params'     : dict mapping (in_name, out_name) → Sij complex array,
              'Ez_snapshots' : list of 2D torch.Tensor Ez planes [Ny+1, Nx]
            }
        """
        device = self.device
        dt = self.dt

        if self.verbose:
            print(f"[Run] Starting simulation for N_steps = {N_steps}")

        # 1) Build the 1D time tensor of length N_steps
        time_tensor = torch.arange(N_steps, dtype=torch.float32, device=device) * dt

        # 2) Precompute each input port’s source waveform s_t_dict[port_name] → numpy array
        s_t_dict = {}
        for port in self.input_ports:
            port_name = port['name']
            source_obj = port['source']
            if source_obj is None:
                raise ValueError(f"Input port '{port_name}' has no 'source' defined.")
            if self.verbose:
                print(f"[Run] Precomputing waveform for input port '{port_name}'")
            # Compute a torch.Tensor of shape [N_steps], then move to CPU numpy
            s_t = source_obj.vector_value(time_tensor).detach().cpu().numpy()
            s_t_dict[port_name] = s_t

        # 3) Precompute each port’s spatial mode‐profile (2D) on the correct device
        mode2d_dict = {}
        for port in (self.input_ports + self.output_ports):
            port_name = port['name']
            mode_profile = port['mode_profile']
            if self.verbose:
                print(f"[Run] Generating 2D mode profile for port '{port_name}'")
            mode2d = mode_profile.generate().to(self.device).float()  # shape [Ny, Nx]
            mode2d_dict[port_name] = mode2d

        # 4) Prepare per‐port time‐series arrays (NumPy) for FFT & S‐params
        E_inc_times = {}
        E_ref_times = {}
        E_trn_times = {}
        for in_port in self.input_ports:
            in_name = in_port['name']
            E_inc_times[in_name] = np.zeros(N_steps, dtype=np.float32)
            E_ref_times[in_name] = np.zeros(N_steps, dtype=np.float32)
            E_trn_times[in_name] = {}
            for out_port in self.output_ports:
                out_name = out_port['name']
                E_trn_times[in_name][out_name] = np.zeros(N_steps, dtype=np.float32)

        # 5) Aliases & constants for the time loop
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz

        # Precompute reciprocals and maps as torch.Tensors
        recip_mu  = self.dt / self.mu_map      # shape [Nz,Ny,Nx]
        recip_eps = self.dt / self.eps_map     # shape [Nz,Ny,Nx]
        sigma_map = self.sigma_map             # shape [Nz,Ny,Nx]

        if self.verbose:
            print(f"[Run] Entering main time loop...")

        Ez_snapshots = []
        # Keep track of convergence: if all |E_ref_times[in][n]| < conv_tol for enough steps, break
        converged = False

        for n in range(N_steps):
            # Print progress every 10% if verbose
            if self.verbose and (n % max(1, N_steps // 10) == 0):
                print(f"[Run] Time step {n+1}/{N_steps}")

            # 6.1) Update H‐fields (Yee algorithm)
            if self.verbose:
                print("  [Step] Updating H‐fields")
            for k in range(Nz):
                for j in range(Ny - 1):
                    for i in range(Nx - 1):
                        curlE = ((self.Ez[k, j + 1, i] - self.Ez[k, j, i]) / dy
                                 - (self.Ey[k + 1, j, i] - self.Ey[k, j, i]) / dz)
                        self.Hx[k, j, i] -= recip_mu[k, j, i] * curlE
            for k in range(Nz - 1):
                for j in range(Ny):
                    for i in range(Nx - 1):
                        curlE = ((self.Ex[k + 1, j, i] - self.Ex[k, j, i]) / dz
                                 - (self.Ez[k, j, i + 1] - self.Ez[k, j, i]) / dx)
                        self.Hy[k, j, i] -= recip_mu[k, j, i] * curlE
            for k in range(Nz - 1):
                for j in range(Ny - 1):
                    for i in range(Nx):
                        curlE = ((self.Ey[k, j, i + 1] - self.Ey[k, j, i]) / dx
                                 - (self.Ex[k, j + 1, i] - self.Ex[k, j, i]) / dy)
                        self.Hz[k, j, i] -= recip_mu[k, j, i] * curlE

            # 6.2) Apply H‐boundary conditions
            if self.verbose:
                print("  [Step] Applying boundary conditions on H‐fields")
            for bc in self.bc_list_objs:
                if isinstance(bc, PMLBoundary):
                    bc.apply_on_H(self.Hx, self.Hy, self.Hz,
                                  self.Ey, self.Ez,
                                  self.mu_map,
                                  dx, dy, dz)
                else:  # PECBoundary
                    bc.apply_on_H(self.Hx, self.Hy, self.Hz)

            # 6.3) Inject every input_port’s source on its specified plane
            if self.verbose:
                print("  [Step] Injecting sources")
            for in_port in self.input_ports:
                in_name = in_port['name']
                axis, k_idx = in_port['location']
                mode2d = mode2d_dict[in_name]     # shape [Ny, Nx]
                s_val = s_t_dict[in_name][n]      # scalar value at time step n

                if axis != 'z':
                    raise NotImplementedError("Only 'z' ports are supported at this time.")

                # Inject into Ey at z = k_idx
                # Ey[k_idx, j, i] = mode2d[j, i] * s_val
                for j in range(Ny):
                    for i in range(Nx):
                        self.Ey[k_idx, j, i] = mode2d[j, i] * s_val

            # 6.4) Update E‐fields (Yee algorithm + σ·E loss)
            if self.verbose:
                print("  [Step] Updating E‐fields")
            for k in range(Nz):
                for j in range(1, Ny):
                    for i in range(1, Nx):
                        curlH = ((self.Hz[k, j, i] - self.Hz[k, j, i - 1]) / dx
                                 - (self.Hy[k - 1, j, i] - self.Hy[k, j, i]) / dy)
                        eold = self.Ex[k, j, i]
                        self.Ex[k, j, i] = (
                            eold
                            + recip_eps[k, j, i] * curlH
                            - (self.dt / self.eps_map[k, j, i]) * (sigma_map[k, j, i] * eold)
                        )
            for k in range(1, Nz):
                for j in range(Ny):
                    for i in range(1, Nx):
                        curlH = ((self.Hx[k, j, i] - self.Hx[k - 1, j, i]) / dz
                                 - (self.Hz[k, j, i - 1] - self.Hz[k, j, i]) / dx)
                        eold = self.Ey[k, j, i]
                        self.Ey[k, j, i] = (
                            eold
                            + recip_eps[k, j, i] * curlH
                            - (self.dt / self.eps_map[k, j, i]) * (sigma_map[k, j, i] * eold)
                        )
            for k in range(1, Nz):
                for j in range(1, Ny):
                    for i in range(Nx):
                        curlH = ((self.Hy[k, j, i] - self.Hy[k, j, i - 1]) / dx
                                 - (self.Hx[k, j - 1, i] - self.Hx[k, j, i]) / dy)
                        eold = self.Ez[k, j, i]
                        self.Ez[k, j, i] = (
                            eold
                            + recip_eps[k, j, i] * curlH
                            - (self.dt / self.eps_map[k, j, i]) * (sigma_map[k, j, i] * eold)
                        )

            # 6.5) Apply E‐boundary conditions
            if self.verbose:
                print("  [Step] Applying boundary conditions on E‐fields")
            for bc in self.bc_list_objs:
                if isinstance(bc, PMLBoundary):
                    bc.apply_on_E(self.Ex, self.Ey, self.Ez,
                                  self.Hx, self.Hy, self.Hz,
                                  self.eps_map,
                                  dx, dy, dz)
                else:
                    bc.apply_on_E(self.Ex, self.Ey, self.Ez)

            # 6.6) Compute overlap integrals for each port at this time step
            if self.verbose:
                print("  [Step] Computing overlap integrals")
            # For each input port: compute reflected on same plane,
            # and transmitted to each output port.
            # E_inc_times[in_name][n] = s_t (incident)
            # E_ref_times[in_name][n] = ∑ Ey[k_in,j,i]·mode2d_in[j,i] · dy
            # E_trn_times[in_name][out_name][n] = ∑ Ey[k_out,j,i]·mode2d_out[j,i] · dy

            for in_port in self.input_ports:
                in_name = in_port['name']
                _, k_idx = in_port['location']
                mode2d_in = mode2d_dict[in_name]  # [Ny, Nx]

                # Compute reflection overlap
                sum_ref = 0.0
                for j in range(Ny):
                    for i in range(Nx):
                        Ey_val   = float(self.Ey[k_idx, j, i].item())
                        mode_val = float(mode2d_in[j, i].item())
                        sum_ref += Ey_val * mode_val

                E_inc_times[in_name][n] = s_t_dict[in_name][n]
                E_ref_times[in_name][n] = sum_ref * dy

                # Compute transmission to each output port
                for out_port in self.output_ports:
                    out_name = out_port['name']
                    _, k_rec_idx = out_port['location']
                    mode2d_out = mode2d_dict[out_name]
                    sum_trn = 0.0
                    for j in range(Ny):
                        for i in range(Nx):
                            Ey_val_out  = float(self.Ey[k_rec_idx, j, i].item())
                            mode_val_out = float(mode2d_out[j, i].item())
                            sum_trn += Ey_val_out * mode_val_out
                    E_trn_times[in_name][out_name][n] = sum_trn * dy

            # 6.7) Optionally record one Ez‐slice at midplane for visualization
            if (n % record_interval) == 0:
                mid_k = Nz // 2
                Ez_plane = self.Ez[mid_k, :, :].detach().cpu().clone()
                Ez_snapshots.append(Ez_plane)

            # 6.8) Convergence test: if all |E_ref_times[in][n]| < conv_tol, break
            if (n > 0) and (self.conv_tol is not None):
                all_below_tol = True
                for in_port in self.input_ports:
                    in_name = in_port['name']
                    if abs(E_ref_times[in_name][n]) >= self.conv_tol:
                        all_below_tol = False
                        break
                if all_below_tol:
                    if self.verbose:
                        print(f"[Run] Converged at time step {n+1} (all reflections < {self.conv_tol}).")
                    converged = True
                    break

        # End of time loop

        if self.verbose and not converged:
            print("[Run] Reached N_steps without full convergence.")

        # 7) After time loop, perform FFT on each port’s time‐series to get S‐parameters
        if self.verbose:
            print("[Run] Performing FFT for S‐parameter calculation.")

        freqs = np.fft.rfftfreq(N_steps, d=dt)
        S_params = {}

        for in_port in self.input_ports:
            in_name = in_port['name']
            E_inc_f = np.fft.rfft(E_inc_times[in_name])
            E_ref_f = np.fft.rfft(E_ref_times[in_name])
            # Reflection coefficient S_{ii}
            S_params[(in_name, in_name)] = E_ref_f / E_inc_f
            # Transmission to each output port
            for out_port in self.output_ports:
                out_name = out_port['name']
                E_trn_f = np.fft.rfft(E_trn_times[in_name][out_name])
                S_params[(in_name, out_name)] = E_trn_f / E_inc_f

        if self.verbose:
            print("[Run] FFT and S‐parameter calculation complete.")

        return {
            'freqs'        : freqs,         # 1D numpy array of positive frequencies
            'S_params'     : S_params,      # dict {(in_name, out_name): complex Sij array}
            'Ez_snapshots' : Ez_snapshots   # list of 2D torch.Tensor Ez planes [Ny+1, Nx]
        }
