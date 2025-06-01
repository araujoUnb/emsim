# object3d.py

import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class Object3D:
    def __init__(self, x_bounds, y_bounds, z_bounds, resolution=(10,10,10), material=None, device='cpu'):
        """
        Base class for any 3D object, using PyTorch to generate grids and store material.

        Parameters:
        - x_bounds: tuple (x_min, x_max) in meters
        - y_bounds: tuple (y_min, y_max) in meters
        - z_bounds: tuple (z_min, z_max) in meters
        - resolution: tuple (Nx, Ny, Nz) # number of points in each axis
        - material: dict or None, keys:
            * 'epsilon_r'    (relative permittivity)
            * 'mu_r'         (relative permeability)
            * 'conductivity' (bulk conductivity [S/m]), optional
        - device: 'cpu' or 'cuda'
        """
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.z_min, self.z_max = z_bounds

        if isinstance(resolution, int):
            self.Nx = self.Ny = self.Nz = resolution
        else:
            self.Nx, self.Ny, self.Nz = resolution

        self.device = device

        # Default material = vacuum
        default_mat = {'epsilon_r': 1.0, 'mu_r': 1.0, 'conductivity': 0.0}
        if material is None:
            material = default_mat.copy()
        else:
            # fill in any missing keys with defaults
            for key, val in default_mat.items():
                material.setdefault(key, val)

        self.epsilon_r = torch.tensor(material['epsilon_r'], dtype=torch.float32, device=self.device)
        self.mu_r      = torch.tensor(material['mu_r'],      dtype=torch.float32, device=self.device)
        self.sigma     = torch.tensor(material['conductivity'], dtype=torch.float32, device=self.device)

        self.epsilon0 = 8.8541878128e-12
        self.mu0      = 4e-7 * np.pi

        # placeholders for 3D grid
        self.X = None
        self.Y = None
        self.Z = None

    def generate_grid(self):
        """
        Abstract. Subclasses must generate self.X, self.Y, self.Z (shape [Nz, Ny, Nx]).
        """
        raise NotImplementedError("generate_grid() must be implemented in subclasses.")

    def get_permittivity_map(self):
        eps = self.epsilon0 * self.epsilon_r
        return eps * torch.ones_like(self.X, device=self.device)

    def get_permeability_map(self):
        mu = self.mu0 * self.mu_r
        return mu * torch.ones_like(self.X, device=self.device)

    def get_conductivity_map(self):
        sigma = self.sigma
        return sigma * torch.ones_like(self.X, device=self.device)

    def generate_side_surfaces_numpy(self):
        """
        Abstract. Subclasses return (Xs, Ys, Zs) as lists of NumPy arrays,
        each array representing one continuous surface patch (for plotting).
        """
        raise NotImplementedError("generate_side_surfaces_numpy() must be implemented in subclasses.")

    def plot_boundary(self):
        """
        Plot ONLY the four side-walls as continuous surfaces in 3D using Matplotlib,
        leaving front and back (z=0, z=length) open (hollow ends).
        """
        # Get lists of NumPy arrays: each list entry is (X_patch, Y_patch, Z_patch)
        # for one wall of the object.
        surfaces = self.generate_side_surfaces_numpy()

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each wall as a surface
        for (Xp, Yp, Zp) in surfaces:
            ax.plot_surface(Xp, Yp, Zp, rstride=1, cstride=1, alpha=0.6, color='cyan', edgecolor='k', linewidth=0.2)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')

        # Equal aspect ratio
        max_range = max(self.x_max - self.x_min, self.y_max - self.y_min, self.z_max - self.z_min) / 2.0
        mid_x = (self.x_max + self.x_min) / 2.0
        mid_y = (self.y_max + self.y_min) / 2.0
        mid_z = (self.z_max + self.z_min) / 2.0
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # Annotate material in the title
        eps_r = self.epsilon_r.item()
        mu_r  = self.mu_r.item()
        sigma = self.sigma.item()
        ax.set_title(f"Rectangular Tube Boundary\nεᵣ={eps_r:.2f}, μᵣ={mu_r:.2f}, σ={sigma:.4g} S/m")

        plt.tight_layout()
        plt.show()
