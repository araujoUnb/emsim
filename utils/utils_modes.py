# utils/utils_modes.py

import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plotting)

class ModeProfile:
    """
    Abstract base class for waveguide mode spatial profiles.
    Subclasses must implement `generate()`, which returns a 2D torch.Tensor [Ny, Nx]
    containing the field amplitude in the transverse plane.

    Provides three visualization methods:
      - plot_heatmap(): 2D color map (imshow).
      - plot_line(direction): 1D central cut (line plot).
      - plot_3d(): 3D surface plot.
    """

    def __init__(self,
                 a: float,
                 b: float,
                 m: int,
                 n: int,
                 x_lin: torch.Tensor,
                 y_lin: torch.Tensor,
                 device: str = 'cpu'):
        """
        Parameters:
        - a, b    : physical dimensions of the waveguide cross section (x-width, y-height) in meters
        - m, n    : integer mode indices (m, n ≥ 0, but not both zero)
        - x_lin   : 1D tensor of x-coordinates (length Nx)
        - y_lin   : 1D tensor of y-coordinates (length Ny)
        - device  : 'cpu' or 'cuda'
        """
        self.a = a
        self.b = b
        self.m = m
        self.n = n
        self.x_lin = x_lin.to(device=device, dtype=torch.float32)  # [Nx]
        self.y_lin = y_lin.to(device=device, dtype=torch.float32)  # [Ny]
        self.device = device

    def generate(self) -> torch.Tensor:
        """
        Abstract method. Must return a 2D torch.Tensor of shape [Ny, Nx]
        with the mode amplitude at each (y, x). Subclasses override this.
        """
        raise NotImplementedError("ModeProfile.generate() must be implemented in subclasses.")

    def plot_heatmap(self):
        """
        Plot the full 2D mode profile as a heatmap (imshow).
        """
        profile = self.generate().detach().cpu().numpy()  # shape [Ny, Nx]

        extent = [
            self.x_lin.min().item(),
            self.x_lin.max().item(),
            self.y_lin.min().item(),
            self.y_lin.max().item()
        ]

        plt.figure(figsize=(5, 4))
        plt.imshow(
            profile,
            origin='lower',
            extent=extent,
            aspect='auto',
            cmap='viridis'
        )
        plt.colorbar(label='Amplitude')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title(self.title_str() + " (Heatmap)")
        plt.tight_layout()
        plt.show()

    def plot_line(self, direction: str = 'x'):
        """
        Plot a 1D central cut of the mode profile.
        - direction='x': extract the row at y ≈ b/2, plot amplitude vs. x.
        - direction='y': extract the column at x ≈ a/2, plot amplitude vs. y.
        """
        profile = self.generate().detach().cpu().numpy()  # [Ny, Nx]
        x_np = self.x_lin.cpu().numpy()  # [Nx]
        y_np = self.y_lin.cpu().numpy()  # [Ny]

        if direction == 'x':
            y_center = self.b / 2.0
            iy = np.abs(y_np - y_center).argmin()
            data_line = profile[iy, :]       # length Nx
            coord = x_np
            xlabel = 'x [m]'
            title = f"{self.title_str()} – Cut at y ≈ {y_np[iy]:.3f} m"
        elif direction == 'y':
            x_center = self.a / 2.0
            ix = np.abs(x_np - x_center).argmin()
            data_line = profile[:, ix]       # length Ny
            coord = y_np
            xlabel = 'y [m]'
            title = f"{self.title_str()} – Cut at x ≈ {x_np[ix]:.3f} m"
        else:
            raise ValueError("direction must be 'x' or 'y'")

        plt.figure(figsize=(5, 3))
        plt.plot(coord, data_line, '-o', markersize=3)
        plt.xlabel(xlabel)
        plt.ylabel('Amplitude')
        plt.title(title + " (Line)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_3d(self):
        """
        Plot the 2D mode profile as a 3D surface (x, y, amplitude).
        """
        profile = self.generate().detach().cpu().numpy()  # [Ny, Nx]
        x_np = self.x_lin.cpu().numpy()  # [Nx]
        y_np = self.y_lin.cpu().numpy()  # [Ny]

        # Create 2D meshgrid for plotting
        Xg, Yg = np.meshgrid(x_np, y_np, indexing='xy')  # both [Ny, Nx]
        Zg = profile  # [Ny, Nx]

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')

        # Plot surface
        ax.plot_surface(
            Xg, Yg, Zg,
            rstride=1, cstride=1,
            cmap='viridis',
            edgecolor='k',
            linewidth=0.2,
            alpha=0.8
        )

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('Amplitude')
        ax.set_title(self.title_str() + " (3D Surface)")

        # Equal aspect ratio
        max_range = np.array([self.a, self.b, np.nanmax(np.abs(Zg))]).max() / 2.0
        mid_x = (self.x_lin.max().item() + self.x_lin.min().item()) / 2.0
        mid_y = (self.y_lin.max().item() + self.y_lin.min().item()) / 2.0
        mid_z = 0.0  # center the z-axis on zero
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        plt.tight_layout()
        plt.show()

    def title_str(self) -> str:
        """
        Default title string. Subclasses may override as needed.
        """
        return f"Mode m={self.m}, n={self.n}"
