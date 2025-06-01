# rectangular_waveguide.py

import torch
import numpy as np
from .object3d import Object3D

class RectangularWaveguide(Object3D):
    def __init__(self, a, b, length, resolution=(50, 30, 20), material=None, device='cpu'):
        """
        Rectangular waveguide (hollow ends) with width a, height b, and length in z.

        Parameters:
        - a: width along X [meters]
        - b: height along Y [meters]
        - length: length along Z [meters]
        - resolution: tuple (Nx, Ny, Nz) # points in (X, Y, Z)
        - material: dict { 'epsilon_r':..., 'mu_r':..., 'conductivity':... }
        - device: 'cpu' or 'cuda'
        """
        x_bounds = (0.0, a)
        y_bounds = (0.0, b)
        z_bounds = (0.0, length)
        super().__init__(x_bounds, y_bounds, z_bounds, resolution, material, device)
        self.generate_grid()

    def generate_grid(self):
        """
        Build 3D tensors X, Y, Z of shape [Nz, Ny, Nx] using torch.linspace + torch.meshgrid.
        """
        x_lin = torch.linspace(self.x_min, self.x_max, self.Nx, device=self.device)
        y_lin = torch.linspace(self.y_min, self.y_max, self.Ny, device=self.device)
        z_lin = torch.linspace(self.z_min, self.z_max, self.Nz, device=self.device)

        # indexing='ij' so that dimensions line up as Z,Y,X
        Z, Y, X = torch.meshgrid(z_lin, y_lin, x_lin, indexing='ij')

        self.X = X
        self.Y = Y
        self.Z = Z

    def get_permittivity_map(self):
        """
        Return the 3D permittivity map ε = ε0 * εr (constant here).
        """
        eps = self.epsilon0 * self.epsilon_r
        return eps * torch.ones_like(self.X, device=self.device)

    def get_permeability_map(self):
        """
        Return the 3D permeability map μ = μ0 * μr (constant here).
        """
        mu = self.mu0 * self.mu_r
        return mu * torch.ones_like(self.X, device=self.device)

    def get_conductivity_map(self):
        """
        Return the 3D conductivity map σ (constant here).
        """
        sigma = self.sigma
        return sigma * torch.ones_like(self.X, device=self.device)

    def generate_side_surfaces_numpy(self):
        """
        Build four continuous surface patches (NumPy arrays) corresponding to the
        side walls only (x=0, x=a, y=0, y=b). Front (z=0) and back (z=length) remain open.
        Returns a list of (X_patch, Y_patch, Z_patch), each of shape [Ny, Nz] or [Nx, Nz].
        """
        # Convert bounds to plain floats
        x_min, x_max = float(self.x_min), float(self.x_max)
        y_min, y_max = float(self.y_min), float(self.y_max)
        z_min, z_max = float(self.z_min), float(self.z_max)

        Nx, Ny, Nz = self.Nx, self.Ny, self.Nz

        # Build 1D linspaces in NumPy for each axis
        xs = np.linspace(x_min, x_max, Nx)
        ys = np.linspace(y_min, y_max, Ny)
        zs = np.linspace(z_min, z_max, Nz)

        surfaces = []

        # 1) Wall at x = x_min  (shape [Ny, Nz])
        Y1, Z1 = np.meshgrid(ys, zs, indexing='ij')  # [Ny, Nz]
        X1 = np.full_like(Y1, x_min)                  # [Ny, Nz]
        surfaces.append((X1, Y1, Z1))

        # 2) Wall at x = x_max  (shape [Ny, Nz])
        Y2, Z2 = np.meshgrid(ys, zs, indexing='ij')
        X2 = np.full_like(Y2, x_max)
        surfaces.append((X2, Y2, Z2))

        # 3) Wall at y = y_min  (shape [Nx, Nz])
        X3, Z3 = np.meshgrid(xs, zs, indexing='ij')  # [Nx, Nz]
        Y3 = np.full_like(X3, y_min)                 # [Nx, Nz]
        surfaces.append((X3, Y3, Z3))

        # 4) Wall at y = y_max  (shape [Nx, Nz])
        X4, Z4 = np.meshgrid(xs, zs, indexing='ij')
        Y4 = np.full_like(X4, y_max)
        surfaces.append((X4, Y4, Z4))

        return surfaces
