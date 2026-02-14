"""Yee grid allocation and CFL time step computation for 3D FDTD."""

import math

import numpy as np
import tensorflow as tf

from emsim.constants import C0
from emsim.fdtd.materials import MaterialGrid


def _to_1d(value, size: int, name: str):
    """Convert value to 1D array of length size. If scalar, broadcast to size."""
    if np.isscalar(value) or (isinstance(value, (int, float))):
        return np.full(size, float(value), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1 or arr.size != size:
        raise ValueError(f"{name} must be scalar or 1D array of size {size}, got shape {getattr(arr, 'shape', len(arr))}")
    return arr


def _face_inv_spacing(d: np.ndarray):
    """Inverse spacing at faces between cells: inv[j] = 2/(d[j]+d[j+1]) for j=0..len(d)-2."""
    n = len(d)
    if n < 2:
        return np.array([1.0 / d[0]] if n == 1 else [], dtype=np.float32)
    inv = 2.0 / (d[:-1] + d[1:])
    return inv.astype(np.float32)


class YeeGrid:
    """3D Yee staggered grid with all field components as tf.Variable.

    Uses the Sullivan convention: all field arrays have shape [Nz, Ny, Nx].
    The half-cell staggering is implicit in the finite-difference formulas.

    Supports uniform and non-uniform (stretched) grids:
    - If dx, dy, dz are scalars or omitted: uniform grid; Nx, Ny, Nz derived from
      domain size and spacing (or f0/resolution).
    - If dx (and optionally dy, dz) are 1D arrays: non-uniform grid; Nx = len(dx),
      and domain length in x is sum(dx). CFL uses the minimum cell size in the grid.

    Parameters
    ----------
    x_range : tuple (x_min, x_max) in metres
    y_range : tuple (y_min, y_max) in metres
    z_range : tuple (z_min, z_max) in metres
    dx, dy, dz : grid spacing in metres (float or 1D array). If None, computed from f0 and resolution.
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
        Lx = self.x_max - self.x_min
        Ly = self.y_max - self.y_min
        Lz = self.z_max - self.z_min

        # Determine uniform vs non-uniform and set Nx, Ny, Nz and spacing arrays
        uniform = True
        dx_arr = dy_arr = dz_arr = None

        if dx is None:
            if f0 is None:
                raise ValueError("Provide either dx/dy/dz or f0 for auto spacing.")
            lam0 = C0 / f0
            dx = lam0 / resolution
        if dy is None:
            dy = dx
        if dz is None:
            dz = dx

        # Check if any spacing is array-like
        def is_array_like(v):
            if v is None or np.isscalar(v):
                return False
            try:
                a = np.asarray(v)
                return a.ndim == 1 and a.size > 1
            except Exception:
                return False

        if is_array_like(dx):
            dx_arr = np.asarray(dx, dtype=np.float32).ravel()
            self.Nx = len(dx_arr)
            self._dx_scalar = float(np.mean(dx_arr))  # for default dy/dz when they are None
            uniform = False
        else:
            dx_scalar = float(dx)
            self.Nx = max(int(round(Lx / dx_scalar)), 1)
            self._dx_scalar = Lx / self.Nx
            dx_arr = np.full(self.Nx, self._dx_scalar, dtype=np.float32)

        if is_array_like(dy):
            dy_arr = np.asarray(dy, dtype=np.float32).ravel()
            self.Ny = len(dy_arr)
            uniform = False
        else:
            dy_scalar = float(dy) if dy is not None else self._dx_scalar
            self.Ny = max(int(round(Ly / dy_scalar)), 1)
            self._dy_scalar = Ly / self.Ny
            dy_arr = np.full(self.Ny, self._dy_scalar, dtype=np.float32)

        if is_array_like(dz):
            dz_arr = np.asarray(dz, dtype=np.float32).ravel()
            self.Nz = len(dz_arr)
            uniform = False
        else:
            dz_scalar = float(dz) if dz is not None else self._dx_scalar
            self.Nz = max(int(round(Lz / dz_scalar)), 1)
            self._dz_scalar = Lz / self.Nz
            dz_arr = np.full(self.Nz, self._dz_scalar, dtype=np.float32)

        # Ensure all three arrays exist (uniform path may have set only some)
        if dx_arr is None:
            dx_arr = np.full(self.Nx, self._dx_scalar, dtype=np.float32)
        if dy_arr is None:
            dy_arr = np.full(self.Ny, self._dy_scalar, dtype=np.float32)
        if dz_arr is None:
            dz_arr = np.full(self.Nz, self._dz_scalar, dtype=np.float32)

        self._uniform = uniform
        self._dx_arr = dx_arr
        self._dy_arr = dy_arr
        self._dz_arr = dz_arr

        # Scalar spacing for backward compatibility: mean for non-uniform
        self.dx = float(np.mean(dx_arr))
        self.dy = float(np.mean(dy_arr))
        self.dz = float(np.mean(dz_arr))

        # CFL: dt_max = 1/(c * sqrt(1/dx_i^2 + 1/dy_j^2 + 1/dz_k^2)) minimized over (i,j,k)
        inv_dx2 = 1.0 / (dx_arr ** 2)
        inv_dy2 = 1.0 / (dy_arr ** 2)
        inv_dz2 = 1.0 / (dz_arr ** 2)
        # Maximum inv2 at any cell (worst case)
        inv2_max = np.max(inv_dx2) + np.max(inv_dy2) + np.max(inv_dz2)
        dt_max = 1.0 / (C0 * math.sqrt(inv2_max))
        self.dt = courant * dt_max
        self.courant = courant

        # Face-centered inverse spacings for curl (length Nx-1, Ny-1, Nz-1)
        inv_dx = _face_inv_spacing(dx_arr)   # length Nx-1
        inv_dy = _face_inv_spacing(dy_arr)   # length Ny-1
        inv_dz = _face_inv_spacing(dz_arr)   # length Nz-1

        # Broadcast to 3D for use in field updates (to match slice shapes)
        # update_H / update_E use: (1, 1, Nx-1), (1, Ny-1, 1), (Nz-1, 1, 1)
        self._inv_dx = tf.constant(
            np.reshape(inv_dx, (1, 1, -1)),
            dtype=tf.float32,
            name="inv_dx",
        )
        self._inv_dy = tf.constant(
            np.reshape(inv_dy, (1, -1, 1)),
            dtype=tf.float32,
            name="inv_dy",
        )
        self._inv_dz = tf.constant(
            np.reshape(inv_dz, (-1, 1, 1)),
            dtype=tf.float32,
            name="inv_dz",
        )

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

    @property
    def dx_array(self):
        """1D array of cell spacings in x (length Nx). For uniform grid, constant array."""
        return self._dx_arr

    @property
    def dy_array(self):
        """1D array of cell spacings in y (length Ny)."""
        return self._dy_arr

    @property
    def dz_array(self):
        """1D array of cell spacings in z (length Nz)."""
        return self._dz_arr

    def dx_at(self, i: int) -> float:
        """Spacing in x at cell index i (for port dl/ds)."""
        return float(self._dx_arr[i])

    def dy_at(self, j: int) -> float:
        """Spacing in y at cell index j."""
        return float(self._dy_arr[j])

    def dz_at(self, k: int) -> float:
        """Spacing in z at cell index k."""
        return float(self._dz_arr[k])

    def get_curl_coefficients(self):
        """Return tensors for curl finite-difference coefficients.

        Returns a dict with keys: inv_dx, inv_dy, inv_dz.
        Shapes: inv_dx (1, 1, Nx-1), inv_dy (1, Ny-1, 1), inv_dz (Nz-1, 1, 1).
        Use in update_H and update_E as: (E[j+1]-E[j]) * inv_dy etc.
        """
        return {
            "inv_dx": self._inv_dx,
            "inv_dy": self._inv_dy,
            "inv_dz": self._inv_dz,
        }

    def reset_fields(self):
        """Zero all field components and dispersive auxiliary fields (P, J) if present."""
        zero = tf.zeros([self.Nz, self.Ny, self.Nx], dtype=tf.float32)
        for f in (self.Ex, self.Ey, self.Ez, self.Hx, self.Hy, self.Hz):
            f.assign(zero)
        m = self.materials
        if getattr(m, "Px", None) is not None:
            for f in (m.Px, m.Py, m.Pz, m.Jx, m.Jy, m.Jz):
                f.assign(zero)

    @property
    def shape(self):
        return (self.Nz, self.Ny, self.Nx)

    def __repr__(self):
        dxr = f"dx={self.dx:.4e}" if self._uniform else f"dx=mean={self.dx:.4e}"
        dyr = f"dy={self.dy:.4e}" if self._uniform else f"dy=mean={self.dy:.4e}"
        dzr = f"dz={self.dz:.4e}" if self._uniform else f"dz=mean={self.dz:.4e}"
        return (f"YeeGrid(Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}, "
                f"{dxr}, {dyr}, {dzr}, dt={self.dt:.4e})")
