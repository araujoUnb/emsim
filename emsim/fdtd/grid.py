"""Yee grid allocation and CFL time step computation for 3D FDTD."""

import math

import tensorflow as tf

from emsim.constants import C0
from emsim.fdtd.materials import MaterialGrid


class YeeGrid:
    """3D Yee staggered grid with all field components as tf.Variable.

    Uses the Sullivan convention: all field arrays have shape [Nz, Ny, Nx].
    The half-cell staggering is implicit in the finite-difference formulas.

    Parameters
    ----------
    x_range : tuple (x_min, x_max) in metres
    y_range : tuple (y_min, y_max) in metres
    z_range : tuple (z_min, z_max) in metres
    dx, dy, dz : grid spacing in metres (if None, computed from f0 and resolution)
    f0 : centre frequency [Hz] (used to auto-compute dx if dx is None)
    resolution : cells per wavelength (default 20)
    courant : Courant factor < 1 (default 0.99)
    eps_r, mu_r, sigma : uniform material properties (default vacuum)
    """

    def __init__(self, x_range, y_range, z_range, *,
                 dx=None, dy=None, dz=None,
                 f0=None, resolution=20, courant=0.99,
                 eps_r=1.0, mu_r=1.0, sigma=0.0):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range

        # Compute grid spacing
        if dx is None:
            if f0 is None:
                raise ValueError("Provide either dx/dy/dz or f0 for auto spacing.")
            lam0 = C0 / f0
            dx = lam0 / resolution
        if dy is None:
            dy = dx
        if dz is None:
            dz = dx

        # Snap grid counts so that N*d exactly covers the domain
        Lx = self.x_max - self.x_min
        Ly = self.y_max - self.y_min
        Lz = self.z_max - self.z_min

        self.Nx = max(int(round(Lx / dx)), 1)
        self.Ny = max(int(round(Ly / dy)), 1)
        self.Nz = max(int(round(Lz / dz)), 1)

        self.dx = Lx / self.Nx
        self.dy = Ly / self.Ny
        self.dz = Lz / self.Nz

        # CFL time step
        inv2 = (1.0 / self.dx) ** 2 + (1.0 / self.dy) ** 2 + (1.0 / self.dz) ** 2
        dt_max = 1.0 / (C0 * math.sqrt(inv2))
        self.dt = courant * dt_max
        self.courant = courant

        # Material grid and coefficients
        self.materials = MaterialGrid(self.Nz, self.Ny, self.Nx,
                                      eps_r=eps_r, mu_r=mu_r, sigma=sigma)
        self.materials.compute_coefficients(self.dt)

        # Allocate field components as mutable tf.Variable
        shape = [self.Nz, self.Ny, self.Nx]
        self.Ex = tf.Variable(tf.zeros(shape, dtype=tf.float32), name="Ex")
        self.Ey = tf.Variable(tf.zeros(shape, dtype=tf.float32), name="Ey")
        self.Ez = tf.Variable(tf.zeros(shape, dtype=tf.float32), name="Ez")
        self.Hx = tf.Variable(tf.zeros(shape, dtype=tf.float32), name="Hx")
        self.Hy = tf.Variable(tf.zeros(shape, dtype=tf.float32), name="Hy")
        self.Hz = tf.Variable(tf.zeros(shape, dtype=tf.float32), name="Hz")

    def reset_fields(self):
        """Zero all field components."""
        zero = tf.zeros([self.Nz, self.Ny, self.Nx], dtype=tf.float32)
        for f in (self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz):
            f.assign(zero)

    @property
    def shape(self):
        return (self.Nz, self.Ny, self.Nx)

    def __repr__(self):
        return (f"YeeGrid(Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}, "
                f"dx={self.dx:.4e}, dy={self.dy:.4e}, dz={self.dz:.4e}, "
                f"dt={self.dt:.4e})")
