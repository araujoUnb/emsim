# torchFDTD.py

"""
TorchFDTD: A 3D Yee‐grid FDTD solver in PyTorch for arbitrarily shaped objects (e.g., rectangular waveguides).
This implementation includes:
  - Automatic grid‐spacing = λ₀/30 (unless overridden).
  - Courant‐Friedrichs‐Lewy (CFL) time‐step calculation.
  - Yee‐cell update loops for E and H fields (3D).
  - Material maps (ε, μ, σ) obtained from a Geometry (subclass of Object3D).
  - Boundary conditions: PEC (Perfect Electric Conductor) and PML (Perfectly Matched Layer, split‐field Berenger).
  - Injection of an arbitrary “mode profile” provided by the user (e.g., TE₁₀, TM₁₁, circular modes, etc.).
  - A generic time‐domain Source (e.g. GaussianPulse) that can be swapped out for other waveforms.
  - Recording of mode‐overlap at source & receiver planes to compute S‐parameters (S11, S21).
  - Option to record Ez snapshots at specified intervals for visualization.
  
Everything is thoroughly documented in English so you can follow the theory behind each step.
"""

import math
import torch
import numpy as np

from simulator import Simulator
from utils.utils_constants import C0
from emsim.boundary_conditions import PECBoundary, PMLBoundary
from emsim.source import Source  # Base class for time‐domain sources


class TorchFDTD(Simulator):
    """
    A PyTorch‐based 3D FDTD solver using the Yee algorithm.
    
    This class inherits from the abstract `Simulator` base class. It requires:
      - geometry: an Object3D instance (e.g., RectangularWaveguide) providing:
          * x_min, x_max, y_min, y_max, z_min, z_max
          * Nx, Ny, Nz (grid resolution in each dimension) – these come from geometry if you pass explicit resolution,
            but TorchFDTD will recompute Nx,Ny,Nz from λ₀/30 if `sim_resolution` is None.
          * device (either 'cpu' or 'cuda')
          * Methods get_permittivity_map(), get_permeability_map(), get_conductivity_map().
      - input_ports, output_ports: placeholders for future port definitions (we do not use them internally here).
      - f0: center frequency (Hz) of excitation → used to set spatial grid resolution Δ = λ₀/30 unless overridden.
      - dt_factor: a CFL safety factor (e.g., 0.99) to compute Δt from Δx,Δy,Δz.
      - bc_list: list of six strings (["PEC","PEC","PEC","PEC","PML","PML"], for example)
                 corresponding to faces ["X-","X+","Y-","Y+","Z-","Z+"].
      - sim_box: optional override for simulation region (ignored if None).
      - sim_resolution: user override for Δx=Δy=Δz (in meters). If None → defaults to λ₀/30.
    
    After initialization, call:
        run(
          N_steps: int,
          record_interval: int,
          source: Source,
          mode_profile: ModeProfile,
          k_src: int,
          k_rec: int
        )
    to execute the time‐stepping.  `mode_profile` must be an object whose `.generate()` method returns a 2D torch.Tensor
    of shape [Ny, Nx], describing the transverse spatial dependence of the injected mode.
    """

    def __init__(self,
                 geometry,
                 input_ports: list,
                 output_ports: list,
                 f0: float,
                 dt_factor: float,
                 bc_list: list[str],
                 sim_box: tuple | None = None,
                 sim_resolution: float | None = None):
        """
        Initialize all solver parameters, allocate fields & material maps, and set boundary conditions.
        
        Parameters:
        -----------
        geometry       : Object3D
                         Provides x_min, x_max, y_min, y_max, z_min, z_max, Nx, Ny, Nz, device,
                         as well as get_permittivity_map(), get_permeability_map(), get_conductivity_map().
                         Note: TorchFDTD will recompute Nx,Ny,Nz from λ₀/30 if sim_resolution is None.
        
        input_ports    : list
                         Placeholder for input‐port definitions (e.g., location indices, source waveforms).
        
        output_ports   : list
                         Placeholder for output‐port definitions (e.g., where to record Ez, H).
        
        f0             : float
                         Center frequency [Hz] used to compute spatial grid resolution if sim_resolution is None.
        
        dt_factor      : float
                         Safety factor for the CFL condition, typically < 1 (e.g. 0.99).
        
        bc_list        : list of str, length 6
                         Each entry is either "PEC" or "PML". The order is:
                         ["X-", "X+", "Y-", "Y+", "Z-", "Z+"]. 
                         Use "PEC" for a Perfect Electric Conductor boundary; "PML" for an 8‐cell PML.
        
        sim_box        : tuple, optional
                         If provided, overrides geometry bounds. Format:
                         (x_min, x_max, y_min, y_max, z_min, z_max). If None, use geometry’s bounds.
        
        sim_resolution : float, optional
                         If provided, sets Δx=Δy=Δz = sim_resolution [m], overriding λ₀/30.
                         Otherwise, Δx=Δy=Δz = λ₀ / 30.0.
        """
        super().__init__(geometry=geometry)
        
        # -------------------------------------------------------------
        # 1) Determine simulation domain extents (x_min,x_max, etc.)
        # -------------------------------------------------------------
        if sim_box is None:
            # Use the geometry's bounds directly
            self.x_min = geometry.x_min
            self.x_max = geometry.x_max
            self.y_min = geometry.y_min
            self.y_max = geometry.y_max
            self.z_min = geometry.z_min
            self.z_max = geometry.z_max
        else:
            # Unpack user‐provided simulation box
            (self.x_min, self.x_max,
             self.y_min, self.y_max,
             self.z_min, self.z_max) = sim_box

        # Compute physical lengths of the box
        Lx = self.x_max - self.x_min
        Ly = self.y_max - self.y_min
        Lz = self.z_max - self.z_min

        # -------------------------------------------------------------
        # 2) Compute free‐space wavelength λ₀ and set grid spacing Δx,Δy,Δz
        # -------------------------------------------------------------
        c0 = C0  # Speed of light in free space [m/s]
        self.f0 = f0
        self.lambda0 = c0 / self.f0  # λ₀ = c₀ / f₀

        if sim_resolution is None:
            # Default: Δx = Δy = Δz = λ₀ / 30
            self.dx = self.lambda0 / 30.0
            self.dy = self.lambda0 / 30.0
            self.dz = self.lambda0 / 30.0
        else:
            # User explicitly overrides grid spacing
            self.dx = sim_resolution
            self.dy = sim_resolution
            self.dz = sim_resolution

        # -------------------------------------------------------------
        # 3) Determine integer grid counts Nx, Ny, Nz so that Nx*dx ≥ Lx, etc.
        # -------------------------------------------------------------
        from math import ceil

        # Round up so that Nx*dx ≥ Lx, Ny*dy ≥ Ly, Nz*dz ≥ Lz
        self.Nx = int(ceil(Lx / self.dx))
        self.Ny = int(ceil(Ly / self.dy))
        self.Nz = int(ceil(Lz / self.dz))

        # “Snap” dx, dy, dz so that Nx·dx = Lx exactly, etc.
        self.dx = Lx / self.Nx
        self.dy = Ly / self.Ny
        self.dz = Lz / self.Nz

        # -------------------------------------------------------------
        # 4) Compute time‐step Δt from 3D CFL condition:
        #    Δt ≤ (1 / c₀) / sqrt( (1/Δx)^2 + (1/Δy)^2 + (1/Δz)^2 )
        # -------------------------------------------------------------
        inv_dx2 = (1.0 / self.dx) ** 2
        inv_dy2 = (1.0 / self.dy) ** 2
        inv_dz2 = (1.0 / self.dz) ** 2

        # dt_factor < 1 ensures we are under the CFL limit
        self.dt = dt_factor * (1.0 / c0) / math.sqrt(inv_dx2 + inv_dy2 + inv_dz2)

        # -------------------------------------------------------------
        # 5) Build 1D grid vectors as torch.Tensors on the same device as geometry
        # -------------------------------------------------------------
        self.device = geometry.device
        self.x_lin = torch.linspace(self.x_min, self.x_max, self.Nx, device=self.device)
        self.y_lin = torch.linspace(self.y_min, self.y_max, self.Ny, device=self.device)
        self.z_lin = torch.linspace(self.z_min, self.z_max, self.Nz, device=self.device)

        # -------------------------------------------------------------
        # 6) Build the 3D Yee‐grid (X, Y, Z) each of shape [Nz, Ny, Nx]
        # -------------------------------------------------------------
        # We choose indexing='ij' so that:
        #   Z[k,j,i] = z_lin[k], Y[k,j,i] = y_lin[j], X[k,j,i] = x_lin[i]
        Z, Y, X = torch.meshgrid(self.z_lin, self.y_lin, self.x_lin, indexing='ij')
        self.X = X  # [Nz, Ny, Nx]
        self.Y = Y
        self.Z = Z

        # -------------------------------------------------------------
        # 7) Query material maps ε(r), μ(r), σ(r) from geometry
        #    Each has shape [Nz, Ny, Nx].
        # -------------------------------------------------------------
        # geometry.get_permittivity_map() returns ε0 * εr as a torch.Tensor [Nz,Ny,Nx].
        self.eps_map    = geometry.get_permittivity_map().to(self.device)    # [Nz,Ny,Nx]
        self.mu_map     = geometry.get_permeability_map().to(self.device)    # [Nz,Ny,Nx]
        self.sigma_map  = geometry.get_conductivity_map().to(self.device)    # [Nz,Ny,Nx]

        # -------------------------------------------------------------
        # 8) Allocate Yee‐fields (E and H) as zero‐filled torch.Tensors.
        #    Following the standard Yee‐grid staggering:
        #
        #      Hx: shape [Nz+1, Ny,   Nx]
        #      Hy: shape [Nz,   Ny+1, Nx]
        #      Hz: shape [Nz,   Ny,   Nx+1]
        #
        #      Ex: shape [Nz,   Ny+1, Nx+1]
        #      Ey: shape [Nz+1, Ny,   Nx+1]
        #      Ez: shape [Nz+1, Ny+1, Nx]
        # -------------------------------------------------------------
        self.Hx = torch.zeros((self.Nz + 1, self.Ny, self.Nx), device=self.device)
        self.Hy = torch.zeros((self.Nz, self.Ny + 1, self.Nx), device=self.device)
        self.Hz = torch.zeros((self.Nz, self.Ny, self.Nx + 1), device=self.device)

        self.Ex = torch.zeros((self.Nz,   self.Ny + 1, self.Nx + 1), device=self.device)
        self.Ey = torch.zeros((self.Nz + 1, self.Ny,   self.Nx + 1), device=self.device)
        self.Ez = torch.zeros((self.Nz + 1, self.Ny + 1, self.Nx),   device=self.device)

        # -------------------------------------------------------------
        # 9) Set up boundary condition objects, one per face in bc_list.
        #    The list must be length 6, in the order:
        #      ["X-", "X+", "Y-", "Y+", "Z-", "Z+"], each "PEC" or "PML".
        # -------------------------------------------------------------
        assert len(bc_list) == 6, "bc_list must have 6 entries: [X-,X+,Y-,Y+,Z-,Z+]."
        faces = ["X-", "X+", "Y-", "Y+", "Z-", "Z+"]
        self.bc_list_objs = []

        for face_label, bc_type in zip(faces, bc_list):
            if bc_type.upper() == "PEC":
                # Perfect Electric Conductor on this face
                self.bc_list_objs.append(PECBoundary(face_label))
            elif bc_type.upper() == "PML":
                # 8‐cell Berenger PML on this face
                # Determine the grid spacing normal to that face:
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
                    N_pml=8,      # Use 8‐cell thickness by default
                    m=3,          # cubic polynomial grading
                    R_inf=1e-6    # target reflection coefficient
                )
                pml.device = self.device
                # Now that Nx,Ny,Nz are known, allocate the PML split‐field arrays
                pml.allocate_tensors(
                    Nz=self.Nz,
                    Ny=self.Ny,
                    Nx=self.Nx,
                    device=self.device
                )
                self.bc_list_objs.append(pml)
            else:
                raise ValueError(f"Unknown BC type '{bc_type}'. Use 'PEC' or 'PML'.")

        # -------------------------------------------------------------
        # 10) Store placeholders for input_ports & output_ports
        #     (We do not use them directly in this version.)
        # -------------------------------------------------------------
        self.input_ports = input_ports
        self.output_ports = output_ports

        # Initialize container for any Ez snapshots
        self.Ez_records = []
        
    def check_cfl(self) -> float:
        """
        Compute and return the 3D Courant–Friedrichs–Lewy (CFL) number:
        
            S = c0 * Δt * sqrt( (1/Δx)^2 + (1/Δy)^2 + (1/Δz)^2 ).
        
        In a standard 3D Yee‐grid, stability requires S ≤ 1. If this method returns
        a value ≥ 1, the current Δt is too large for the given Δx, Δy, Δz.

        Returns:
            The Courant number S (a float).
        """
        # Speed of light in free space (m/s)
        c0 = C0

        # Retrieve our grid spacings (m) and time‐step (s)
        dx = self.dx
        dy = self.dy
        dz = self.dz
        dt = self.dt

        # Compute (1/Δx)^2 + (1/Δy)^2 + (1/Δz)^2
        inv_dx2 = (1.0 / dx) ** 2
        inv_dy2 = (1.0 / dy) ** 2
        inv_dz2 = (1.0 / dz) ** 2

        # 3D CFL number:
        S = c0 * dt * math.sqrt(inv_dx2 + inv_dy2 + inv_dz2)

        # Print a friendly message
        if S < 1.0:
            print(f"[CFL check] Courant number S = {S:.5f}  →  OK (below 1.0)")
        else:
            print(f"[CFL check] Courant number S = {S:.5f}  →  WARNING: ≥ 1.0 (unstable!)")

        return S

    def run(self,
            N_steps: int,
            record_interval: int = 1,
            source: Source = None,
            mode_profile = None,
            k_src: int = 0,
            k_rec: int = None):
        """
        Run the FDTD time loop for N_steps time‐steps.
        Optionally record Ez or Ey snapshots every record_interval steps.

        Required keyword arguments:
        - source       : an instance of Source (e.g. GaussianPulse). Must implement .vector_value(time_tensor).
        - mode_profile : an object with a .generate() method returning a 2D torch.Tensor [Ny, Nx].
                         This tensor describes the transverse mode shape (e.g., TE₁₀, TM₁₁, circular modes, etc.).
        - k_src        : index along z where we force Ey = mode_profile(x,y) * s(t). 
                         Must be in [0 .. Nz].
        - k_rec        : index along z where we sample Ey to compute transmitted overlap. 
                         If None, defaults to Nz-2 (two cells before top PML). 

        Returns a dictionary:
          {
            "freqs"       : 1D NumPy array of frequencies (Hz, length N_steps//2+1),
            "S11"         : 1D NumPy array of S11(f),
            "S21"         : 1D NumPy array of S21(f),
            "Ez_snapshots": list of 2D torch.Tensor snapshots of Ez at midplane (optional),
          }
        """
        # 1) Validate inputs
        if source is None:
            raise ValueError("You must pass a `source` object (e.g. GaussianPulse) to run().")
        if mode_profile is None:
            raise ValueError("You must pass a `mode_profile` object to run().")

        # Extract basic parameters and device
        device = self.device
        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz
        dx, dy, dz = self.dx, self.dy, self.dz
        dt = self.dt

        # Determine default receiver plane if not provided
        if k_rec is None:
            # We assume PML on +Z is 8 cells. The last “interior” plane is Nz-1,
            # so two cells in would be Nz-2. Adjust as desired.
            k_rec = Nz - 2

        # --------------------------------------------------------
        # 2) Build a 1D time‐tensor of length N_steps:
        #    time_tensor[n] = n * dt, all on the correct device
        # --------------------------------------------------------
        time_tensor = torch.arange(N_steps, device=device, dtype=torch.float32) * dt

        # --------------------------------------------------------
        # 3) Use the Source to compute s_t for all time steps in one shot:
        #    s_t is a 1D tensor of shape [N_steps], on `device`.
        # --------------------------------------------------------
        s_t = source.vector_value(time_tensor)  # → shape [N_steps]

        # --------------------------------------------------------
        # 4) Precompute the 2D mode‐profile for injection + projection:
        #    mode_profile.generate() must return a 2D torch.Tensor of shape [Ny, Nx].
        # --------------------------------------------------------
        mode_2d = mode_profile.generate().to(device)  # [Ny, Nx]

        # --------------------------------------------------------
        # 5) Pre‐allocate CPU NumPy arrays to store the mode‐overlap time‐series:
        #    We store 3 arrays of length N_steps: E_inc_time, E_ref_time, E_trn_time.
        # --------------------------------------------------------
        E_inc_time = np.zeros(N_steps, dtype=np.float32)
        E_ref_time = np.zeros(N_steps, dtype=np.float32)
        E_trn_time = np.zeros(N_steps, dtype=np.float32)

        # --------------------------------------------------------
        # 6) Precompute normalization constants (if desired):
        #    We will multiply the sum by dx*dy when projecting. In practice, the ratio S11 = E_ref/E_inc
        #    is invariant under a constant scale factor. We keep dx*dy for clarity.
        # --------------------------------------------------------
        normalization_factor = dx * dy

        # --------------------------------------------------------
        # 7) Prepare to record Ez snapshots if requested
        # --------------------------------------------------------
        Ez_snapshots = []

        # --------------------------------------------------------
        # 8) Main FDTD time‐stepping loop
        # --------------------------------------------------------
        for n in range(N_steps):
            # --------------------------------------------
            # 8.1) H‐field updates (Yee algorithm)
            # --------------------------------------------
            recip_mu = dt / self.mu_map  # shape [Nz, Ny, Nx]

            # Hx‐update: Hx[k,j,i] -= (dt/μ) * [ (Ez[k,j+1,i] - Ez[k,j,i]) / dy  -  (Ey[k+1,j,i] - Ey[k,j,i]) / dz ]
            for k in range(0, Nz):
                for j in range(0, Ny - 1):
                    for i in range(0, Nx - 1):
                        curlE = ((self.Ez[k, j + 1, i] - self.Ez[k, j, i]) / dy
                                  - (self.Ey[k + 1, j, i] - self.Ey[k, j, i]) / dz)
                        self.Hx[k, j, i] -= recip_mu[k, j, i] * curlE

            # Hy‐update: Hy[k,j,i] -= (dt/μ) * [ (Ex[k+1,j,i] - Ex[k,j,i]) / dz  -  (Ez[k,j,i+1] - Ez[k,j,i]) / dx ]
            for k in range(0, Nz - 1):
                for j in range(0, Ny):
                    for i in range(0, Nx - 1):
                        curlE = ((self.Ex[k + 1, j, i] - self.Ex[k, j, i]) / dz
                                  - (self.Ez[k, j, i + 1] - self.Ez[k, j, i]) / dx)
                        self.Hy[k, j, i] -= recip_mu[k, j, i] * curlE

            # Hz‐update: Hz[k,j,i] -= (dt/μ) * [ (Ey[k,j,i+1] - Ey[k,j,i]) / dx  -  (Ex[k,j+1,i] - Ex[k,j,i]) / dy ]
            for k in range(0, Nz - 1):
                for j in range(0, Ny - 1):
                    for i in range(0, Nx):
                        curlE = ((self.Ey[k, j, i + 1] - self.Ey[k, j, i]) / dx
                                  - (self.Ex[k, j + 1, i] - self.Ex[k, j, i]) / dy)
                        self.Hz[k, j, i] -= recip_mu[k, j, i] * curlE

            # --------------------------------------------
            # 8.2) Apply boundary conditions on H‐fields:
            #      - If PECBoundary: bc.apply_on_H(Hx,Hy,Hz)
            #      - If PMLBoundary: bc.apply_on_H(Hx,Hy,Hz, Ey,Ez, mu_map, dx,dy,dz)
            # --------------------------------------------
            for bc in self.bc_list_objs:
                if isinstance(bc, PMLBoundary):
                    bc.apply_on_H(self.Hx, self.Hy, self.Hz,
                                  self.Ey, self.Ez,
                                  self.mu_map, dx, dy, dz)
                else:  # PECBoundary
                    bc.apply_on_H(self.Hx, self.Hy, self.Hz)

            # --------------------------------------------
            # 8.3) Inject the chosen mode into Ey at plane k_src:
            #      Ey[k_src, j, i] = mode_2d[j,i] * s_t[n]
            #      i runs 0..Nx-1, j runs 0..Ny-1.  We leave Ey[* , *, Nx] and Ey[* , Ny, *] as boundary ghost cells.
            # --------------------------------------------
            for j in range(Ny):
                for i in range(Nx):
                    self.Ey[k_src, j, i] = mode_2d[j, i] * s_t[n]

            # --------------------------------------------
            # 8.4) E‐field updates (Yee algorithm + σ·E loss term)
            # --------------------------------------------
            recip_eps = dt / self.eps_map  # shape [Nz, Ny, Nx]
            sigma_map = self.sigma_map     # shape [Nz, Ny, Nx]

            # Ex‐update: Ex[k,j,i] += (dt/ε) * [ (Hz[k,j,i] - Hz[k,j,i-1]) / dx  -  (Hy[k,j,i] - Hy[k-1,j,i]) / dy ]
            #           then subtract (σ·Ex·dt/ε) for conductive loss
            for k in range(0, Nz):
                for j in range(1, Ny):
                    for i in range(1, Nx):
                        curlH = ((self.Hz[k, j, i] - self.Hz[k, j, i - 1]) / dx
                                  - (self.Hy[k - 1, j, i] - self.Hy[k, j, i]) / dy)
                        eold = self.Ex[k, j, i]
                        self.Ex[k, j, i] = (
                            eold
                            + recip_eps[k, j, i] * curlH
                            - (dt / self.eps_map[k, j, i]) * (sigma_map[k, j, i] * eold)
                        )

            # Ey‐update: Ey[k,j,i] += (dt/ε) * [ (Hx[k,j,i] - Hx[k-1,j,i]) / dz  -  (Hz[k,j,i] - Hz[k,j-1,i]) / dx ]
            #           minus σ·Ey·dt/ε
            for k in range(1, Nz):
                for j in range(0, Ny):
                    for i in range(1, Nx):
                        curlH = ((self.Hx[k, j, i] - self.Hx[k - 1, j, i]) / dz
                                  - (self.Hz[k, j, i] - self.Hz[k, j - 1, i]) / dx)
                        eold = self.Ey[k, j, i]
                        self.Ey[k, j, i] = (
                            eold
                            + recip_eps[k, j, i] * curlH
                            - (dt / self.eps_map[k, j, i]) * (sigma_map[k, j, i] * eold)
                        )

            # Ez‐update: Ez[k,j,i] += (dt/ε) * [ (Hy[k,j,i] - Hy[k,j,i-1]) / dx  -  (Hx[k,j,i] - Hx[k,j-1,i]) / dy ]
            #           minus σ·Ez·dt/ε
            for k in range(1, Nz):
                for j in range(1, Ny):
                    for i in range(0, Nx):
                        curlH = ((self.Hy[k, j, i] - self.Hy[k, j, i - 1]) / dx
                                  - (self.Hx[k, j - 1, i] - self.Hx[k, j, i]) / dy)
                        eold = self.Ez[k, j, i]
                        self.Ez[k, j, i] = (
                            eold
                            + recip_eps[k, j, i] * curlH
                            - (dt / self.eps_map[k, j, i]) * (sigma_map[k, j, i] * eold)
                        )

            # --------------------------------------------
            # 8.5) Apply boundary conditions on E‐fields:
            #      - If PECBoundary: bc.apply_on_E(Ex,Ey,Ez)
            #      - If PMLBoundary: bc.apply_on_E(Ex,Ey,Ez, Hx,Hy,Hz, eps_map, dx,dy,dz)
            # --------------------------------------------
            for bc in self.bc_list_objs:
                if isinstance(bc, PMLBoundary):
                    bc.apply_on_E(self.Ex, self.Ey, self.Ez,
                                  self.Hx, self.Hy, self.Hz,
                                  self.eps_map, dx, dy, dz)
                else:
                    bc.apply_on_E(self.Ex, self.Ey, self.Ez)

            # --------------------------------------------
            # 8.6) Sample the overlap integral at source & receiver to build time‐series
            #       We compute:
            #         E_inc_time[n] = ∑_{j,i} [ mode_2d[j,i] * (mode_2d[j,i] * s_t[n]) ] Δx Δy
            #                       = (∑_{j,i} mode_2d[j,i]^2 ) · s_t[n] · (dx·dy)
            #                       → we choose to store simply s_t[n], since the constant ∑ mode^2·dx·dy
            #                         cancels out in S11/S21 = (E_ref/E_inc).
            #
            #         E_ref_time[n] = ∑_{j,i} [ mode_2d[j,i] * Ey(k_src,j,i) ] Δx Δy
            #         E_trn_time[n] = ∑_{j,i} [ mode_2d[j,i] * Ey(k_rec,j,i) ] Δx Δy
            # --------------------------------------------
            sum_ref = 0.0
            sum_trn = 0.0
            # We call mode_2d[j,i] and Ey[...] as scalars (CPU floats) in Python loops.
            # This could be optimized further, but clarity is preferred here.
            for j in range(Ny):
                for i in range(Nx):
                    # Ey at source plane * mode_shape at that point
                    sum_ref += float(self.Ey[k_src, j, i].item()) * float(mode_2d[j, i].item())
                    # Ey at receiver plane * mode_shape at that point
                    sum_trn += float(self.Ey[k_rec, j, i].item()) * float(mode_2d[j, i].item())

            # Incident time history is exactly s_t[n] if mode_2d has been “unit‐normalized”
            # (so that ∑ mode_2d[j,i]^2·dx·dy = 1).  In general, one could divide each sum by the norm.
            E_inc_time[n] = s_t[n].item()
            E_ref_time[n] = sum_ref * normalization_factor
            E_trn_time[n] = sum_trn * normalization_factor

            # --------------------------------------------
            # 8.7) Optionally record an Ez slice at the mid‐plane for visualization
            # --------------------------------------------
            if (n % record_interval) == 0:
                mid_k = Nz // 2
                Ez_plane = self.Ez[mid_k, :, :].detach().cpu().clone()  # shape [Ny, Nx]
                Ez_snapshots.append(Ez_plane)

        # End of time‐loop

        # --------------------------------------------------------
        # 9) FFT postprocessing to obtain S‐parameters in the 20–26 GHz band
        # --------------------------------------------------------
        freqs = np.fft.rfftfreq(N_steps, d=dt)  # positive frequencies only

        E_inc_f = np.fft.rfft(E_inc_time)
        E_ref_f = np.fft.rfft(E_ref_time)
        E_trn_f = np.fft.rfft(E_trn_time)

        # Compute S11 & S21 as complex‐valued arrays
        S11 = E_ref_f / E_inc_f
        S21 = E_trn_f / E_inc_f

        # Package results in a dictionary and return
        return {
            "freqs": freqs,            # 1D NumPy array of length N_steps//2+1
            "S11": S11,                # 1D NumPy array (complex) of length N_steps//2+1
            "S21": S21,                # 1D NumPy array (complex) of length N_steps//2+1
            "Ez_snapshots": Ez_snapshots  # list of 2D torch.Tensor [Ny, Nx] Ez planes
        }
