# emsim/boundary_conditions.py

import torch
import math
from abc import ABC, abstractmethod


class BoundaryCondition(ABC):
    """
    Abstract base class for any boundary condition (BC) in an FDTD solver.
    Subclasses must implement apply_on_H() and apply_on_E() to modify
    the magnetic (H) and electric (E) field tensors, respectively.
    """

    @abstractmethod
    def apply_on_H(self, Hx: torch.Tensor, Hy: torch.Tensor, Hz: torch.Tensor, *args, **kwargs):
        """
        Apply this boundary condition to the magnetic field components Hx, Hy, Hz.
        Any additional arguments (e.g., split‐field tensors, material maps) can be passed via *args/**kwargs.
        """
        pass

    @abstractmethod
    def apply_on_E(self, Ex: torch.Tensor, Ey: torch.Tensor, Ez: torch.Tensor, *args, **kwargs):
        """
        Apply this boundary condition to the electric field components Ex, Ey, Ez.
        Any additional arguments (e.g., split‐field tensors, material maps) can be passed via *args/**kwargs.
        """
        pass


class PECBoundary(BoundaryCondition):
    """
    Perfect Electric Conductor (PEC) boundary condition.

    For a “PEC” face, the tangential electric fields must be zero on that face,
    and the tangential magnetic fields must also be zero on the corresponding adjacent H‐grid.

    face: string in {"X-", "X+", "Y-", "Y+", "Z-", "Z+"} indicating which boundary face this object represents.
    """

    def __init__(self, face: str):
        """
        Initialize a PEC boundary for the specified face.

        Parameters:
        - face: one of "X-", "X+", "Y-", "Y+", "Z-", "Z+".
        """
        assert face in {"X-", "X+", "Y-", "Y+", "Z-", "Z+"}, "Invalid face for PECBoundary"
        self.face = face

    def apply_on_H(self, Hx: torch.Tensor, Hy: torch.Tensor, Hz: torch.Tensor, *args, **kwargs):
        """
        Zero out the tangential magnetic field components on the specified face.

        For each face:
          - "X-": i = 0   → Hx[:,:,0] and Hz[:,:,0] must be zero.
          - "X+": i = -1  → Hx[:,:, -1] and Hz[:,:, -1] must be zero.
          - "Y-": j = 0   → Hx[:,0,:] and Hy[:,0,:] must be zero.
          - "Y+": j = -1  → Hx[:,-1,:] and Hy[:,-1,:] must be zero.
          - "Z-": k = 0   → Hy[0,:,:] and Hz[0,:,:] must be zero.
          - "Z+": k = -1  → Hy[-1,:,:] and Hz[-1,:,:] must be zero.
        """
        if self.face == "X-":
            Hx[:, :, 0] = 0.0
            Hz[:, :, 0] = 0.0
        elif self.face == "X+":
            Hx[:, :, -1] = 0.0
            Hz[:, :, -1] = 0.0

        elif self.face == "Y-":
            Hx[:, 0, :] = 0.0
            Hy[:, 0, :] = 0.0
        elif self.face == "Y+":
            Hx[:, -1, :] = 0.0
            Hy[:, -1, :] = 0.0

        elif self.face == "Z-":
            Hy[0, :, :] = 0.0
            Hz[0, :, :] = 0.0
        elif self.face == "Z+":
            Hy[-1, :, :] = 0.0
            Hz[-1, :, :] = 0.0

    def apply_on_E(self, Ex: torch.Tensor, Ey: torch.Tensor, Ez: torch.Tensor, *args, **kwargs):
        """
        Zero out the tangential electric field components on the specified face.

        For each face:
          - "X-": i = 0   → Ez[:,:,0] and Ey[:,:,0] must be zero.
          - "X+": i = -1  → Ez[:,:, -1] and Ey[:,:,-1] must be zero.
          - "Y-": j = 0   → Ez[:,0,:] and Ex[:,0,:] must be zero.
          - "Y+": j = -1  → Ez[:,-1,:] and Ex[:,-1,:] must be zero.
          - "Z-": k = 0   → Ex[0,:,:] and Ey[0,:,:] must be zero.
          - "Z+": k = -1  → Ex[-1,:,:] and Ey[-1,:,:] must be zero.
        """
        if self.face == "X-":
            Ez[:, :, 0] = 0.0
            Ey[:, :, 0] = 0.0
        elif self.face == "X+":
            Ez[:, :, -1] = 0.0
            Ey[:, :, -1] = 0.0

        elif self.face == "Y-":
            Ez[:, 0, :] = 0.0
            Ex[:, 0, :] = 0.0
        elif self.face == "Y+":
            Ez[:, -1, :] = 0.0
            Ex[:, -1, :] = 0.0

        elif self.face == "Z-":
            Ex[0, :, :] = 0.0
            Ey[0, :, :] = 0.0
        elif self.face == "Z+":
            Ex[-1, :, :] = 0.0
            Ey[-1, :, :] = 0.0


class PMLBoundary(BoundaryCondition):
    """
    Perfectly Matched Layer (PML) boundary condition using Berenger split‐field formulation.

    We implement an 8‐cell‐thick PML along one face of the domain. In a Berenger PML, each
    field component is split into two sub‐fields which evolve with an added conductivity σ.
    This class currently demonstrates the procedure for the "X-" face—I.e., the left face in x.
    For other faces ("X+", "Y-", "Y+", "Z-", "Z+"), the loops below must be adapted similarly.

    Parameters:
    - face: which boundary face ("X-", "X+", "Y-", "Y+", "Z-", "Z+").
    - dt:  time step of the FDTD simulation [s].
    - dx:  grid spacing normal to that face [m] (dx if face in x, dy if face in y, dz if face in z).
    - N_pml: number of PML cells (e.g., 8).
    - m: polynomial grading order for σ profile (commonly 2 or 3).
    - R_inf: desired reflection coefficient at the PML outer edge (e.g., 1e-6).
    """

    def __init__(
        self,
        face: str,
        dt: float,
        dx: float,
        N_pml: int = 8,
        m: int = 3,
        R_inf: float = 1e-6,
    ):
        assert face in {"X-", "X+", "Y-", "Y+", "Z-", "Z+"}, "Invalid face for PMLBoundary"
        self.face = face
        self.dt = dt
        self.dx = dx
        self.N_pml = N_pml
        self.m = m
        self.R_inf = R_inf

        # Characteristic impedance of free space (Ω)
        eta0 = 377.0

        # Compute σ_max according to classical formula:
        # σ_max = - (m+1) * ln(R_inf) / (2 * η0 * N_pml * dx)
        self.sigma_max = - (self.m + 1) * math.log(self.R_inf) / (2.0 * eta0 * self.N_pml)

        # Build the 1D conductivity profile (length N_pml), using midpoint of each cell:
        sigma_profile = torch.zeros(self.N_pml, dtype=torch.float32)
        for i in range(self.N_pml):
            # normalized position in [0, 1], using cell center (i + 0.5)/N_pml
            normalized = (i + 0.5) / self.N_pml
            sigma_profile[i] = self.sigma_max * (normalized ** self.m)
        self.sigma_profile = sigma_profile  # 1D tensor of length N_pml

        # We will allocate split‐field tensors once we know the full domain size (Nz, Ny, Nx).
        self.device = None
        self.H_xy = None  # Will hold H_xy in PML region
        self.H_xz = None  # Will hold H_xz in PML region
        self.E_yx = None  # Will hold E_yx in PML region
        self.E_zx = None  # Will hold E_zx in PML region

    def allocate_tensors(self, Nz: int, Ny: int, Nx: int, device: str):
        """
        Allocate zero‐initialized split‐field tensors for the PML subfields,
        using the same shapes as the Yee‐grid fields in the main simulation.

        For the face "X-":
          - Hx has shape [Nz+1, Ny,   Nx]. We split into H_xy and H_xz with identical shape.
          - Ex has shape [Nz,   Ny+1, Nx+1]. We split into E_yx and E_zx with that same shape.

        The device argument is for putting these tensors on CPU vs. GPU.
        """
        self.device = device

        # Split components for Hx in the PML; same shape as Hx in the main solver:
        self.H_xy = torch.zeros((Nz + 1, Ny, Nx), device=device)
        self.H_xz = torch.zeros((Nz + 1, Ny, Nx), device=device)

        # Split components for Ex in the PML; same shape as Ex in the main solver:
        self.E_yx = torch.zeros((Nz, Ny + 1, Nx + 1), device=device)
        self.E_zx = torch.zeros((Nz, Ny + 1, Nx + 1), device=device)

    def apply_on_H(
        self,
        Hx: torch.Tensor,
        Hy: torch.Tensor,
        Hz: torch.Tensor,
        Ey: torch.Tensor,
        Ez: torch.Tensor,
        mu_map: torch.Tensor,
        dx: float,
        dy: float,
        dz: float,
    ):
        """
        Update the split‐field Hx subfields inside the PML for the "X-" face.
        Outside the PML (i >= N_pml), Hx is not modified here and is updated by the main FDTD loop.

        Parameters:
        - Hx, Hy, Hz: full-domain magnetic field tensors (Hx shape [Nz+1,Ny,Nx]).
        - Ey, Ez: full-domain electric field tensors (Ez shape [Nz+1,Ny+1,Nx], Ey shape [Nz+1,Ny,Nx+1]).
        - mu_map: 3D tensor of μ values (shape [Nz,Ny,Nx]).
        - dx, dy, dz: grid spacing in x, y, z directions.
        """
        Np = self.N_pml
        dt = self.dt
        sig = self.sigma_profile  # 1D tensor [Np]

        Nzp1, Nyp, Nxp = Hx.shape  # Nz+1, Ny, Nx

        # Iterate over i = 0..Np-1 (the PML cells on the X- face)
        for i in range(Np):
            sigma_i = sig[i]  # conductivity at layer i

            # Precompute coefficients for this layer:
            # alpha_H = (1 - σ dt/2) / (1 + σ dt/2)
            # beta_H  = dt / (μ * (1 + σ dt/2))
            alpha_H = torch.zeros((Nzp1, Nyp), device=self.device)
            beta_H = torch.zeros((Nzp1, Nyp), device=self.device)

            # For Hx subfields, μ_map is defined on [Nz,Ny,Nx], so we replicate one row for k = Nz:
            mu_full = torch.zeros((Nzp1, Nyp, Nxp), device=self.device)
            mu_full[:Nzp1 - 1, :, :] = mu_map  # copy the [Nz,Ny,Nx] portion
            mu_full[-1, :, :] = mu_map[-1, :, :]  # replicate last μ for Hx at k = Nz

            # Now alpha_H and beta_H at each k,j (i fixed):
            # We will index mu_full[k,j,i] for the μ at that (k,j,i).
            for k in range(Nzp1):
                for j in range(Nyp):
                    mu_val = mu_full[k, j, i]
                    denom = 1.0 + sigma_i * dt / 2.0
                    alpha_H[k, j] = (1.0 - sigma_i * dt / 2.0) / denom
                    beta_H[k, j] = dt / (mu_val * denom)

            # 1) Update H_xy subfield in PML layer i:
            #    H_xy[k,j,i] = alpha_H * H_xy[k,j,i]
            #                  - beta_H * (Ez[k,j,i] - Ez[k,j-1,i]) / dy
            for k in range(0, Nzp1 - 1):       # k = 0..Nz-1
                for j in range(1, Nyp):        # j = 1..Ny-1
                    old_val = self.H_xy[k, j, i]
                    dEz_dy = (Ez[k, j, i] - Ez[k, j - 1, i]) / dy
                    new_val = alpha_H[k, j] * old_val - beta_H[k, j] * dEz_dy
                    self.H_xy[k, j, i] = new_val

            # 2) Update H_xz subfield in PML layer i:
            #    H_xz[k,j,i] = alpha_H * H_xz[k,j,i]
            #                  + beta_H * (Ey[k, j, i] - Ey[k-1, j, i]) / dz
            for k in range(1, Nzp1):         # k = 1..Nz
                for j in range(0, Nyp):      # j = 0..Ny-1
                    old_val = self.H_xz[k, j, i]
                    dEy_dz = (Ey[k, j, i] - Ey[k - 1, j, i]) / dz
                    new_val = alpha_H[k, j] * old_val + beta_H[k, j] * dEy_dz
                    self.H_xz[k, j, i] = new_val

            # 3) Reconstruct Hx in that PML layer: Hx = H_xy + H_xz
            for k in range(Nzp1):
                for j in range(Nyp):
                    Hx[k, j, i] = self.H_xy[k, j, i] + self.H_xz[k, j, i]

        # Cells with i >= Np remain untouched by PML; they will be updated by the main FDTD update.


    def apply_on_E(
        self,
        Ex: torch.Tensor,
        Ey: torch.Tensor,
        Ez: torch.Tensor,
        Hx: torch.Tensor,
        Hy: torch.Tensor,
        Hz: torch.Tensor,
        eps_map: torch.Tensor,
        dx: float,
        dy: float,
        dz: float,
    ):
        """
        Update the split‐field Ex subfields inside the PML for the "X-" face.
        Outside the PML (i >= N_pml), Ex is not modified here and is updated by the main FDTD loop.

        Parameters:
        - Ex, Ey, Ez: full‐domain electric field tensors (Ex shape [Nz, Ny+1, Nx+1]).
        - Hx, Hy, Hz: full‐domain magnetic field tensors.
        - eps_map: 3D tensor of ε values (shape [Nz,Ny,Nx]).
        - dx, dy, dz: grid spacing in x, y, z directions.
        """
        Np = self.N_pml
        dt = self.dt
        sig = self.sigma_profile

        Nze, Nye_p, Nxe_p = Ex.shape  # Ex shape = [Nz, Ny+1, Nx+1]

        # Iterate over i = 0..Np-1 (the PML cells on the X- face)
        for i in range(Np):
            sigma_i = sig[i]

            # Precompute coefficients:
            # alpha_E = (1 - σ dt/2) / (1 + σ dt/2)
            # beta_E  = dt / (ε * (1 + σ dt/2))
            alpha_E = torch.zeros((Nze, Nye_p, Nxe_p), device=self.device)
            beta_E = torch.zeros((Nze, Nye_p, Nxe_p), device=self.device)

            # ε_map is defined on [Nz,Ny,Nx]; to align with Ex[k,j,i], we replicate last row/col as needed
            eps_full = torch.zeros((Nze, Nye_p, Nxe_p), device=self.device)
            eps_full[:, :Nye_p - 1, :Nxe_p - 1] = eps_map  # copy interior [Nz,Ny,Nx]
            # No need for strict replication outside interior because we won't read those indices

            for k in range(Nze):
                for j in range(Nye_p):
                    eps_val = eps_full[k, j, i]
                    denom = 1.0 + sigma_i * dt / 2.0
                    alpha_E[k, j, i] = (1.0 - sigma_i * dt / 2.0) / denom
                    beta_E[k, j, i] = dt / (eps_val * denom)

            # 1) Update E_yx subfield in PML layer i:
            #    E_yx[k,j,i] = alpha_E * E_yx[k,j,i]
            #                  + beta_E * (Hz[k,j,i] - Hz[k,j,i-1]) / dx
            for k in range(0, Nze):
                for j in range(1, Nye_p - 1):
                    old_val = self.E_yx[k, j, i]
                    # Interpret Hz[k,j,i] and Hz[k,j,i-1]; if i-1 < 0, treat as 0 (outside domain)
                    Hz_i = Hz[k, j, i] if i < Hz.shape[2] else 0.0
                    Hz_im1 = Hz[k, j, i - 1] if i - 1 >= 0 else 0.0
                    dHz_dx = (Hz_i - Hz_im1) / dx
                    new_val = alpha_E[k, j, i] * old_val + beta_E[k, j, i] * dHz_dx
                    self.E_yx[k, j, i] = new_val

            # 2) Update E_zx subfield in PML layer i:
            #    E_zx[k,j,i] = alpha_E * E_zx[k,j,i]
            #                  - beta_E * (Hy[k,j,i] - Hy[k-1,j,i]) / dx
            for k in range(1, Nze):
                for j in range(0, Nye_p - 1):
                    old_val = self.E_zx[k, j, i]
                    Hy_i = Hy[k, j, i] if i < Hy.shape[2] else 0.0
                    Hy_km1 = Hy[k - 1, j, i] if k - 1 >= 0 else 0.0
                    dHy_dx = (Hy_i - Hy_km1) / dx
                    new_val = alpha_E[k, j, i] * old_val - beta_E[k, j, i] * dHy_dx
                    self.E_zx[k, j, i] = new_val

            # 3) Reconstruct Ex in that PML layer: Ex = E_yx + E_zx
            for k in range(Nze):
                for j in range(Nye_p):
                    Ex[k, j, i] = self.E_yx[k, j, i] + self.E_zx[k, j, i]

        # Cells with i >= Np remain untouched by PML; they will be updated by the main FDTD update.
