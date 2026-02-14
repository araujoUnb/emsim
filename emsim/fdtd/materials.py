"""Material property grids and precomputed update coefficients for FDTD."""

import tensorflow as tf

from emsim.constants import EPS0, MU0


class MaterialGrid:
    """Stores 3D material property maps and precomputes FDTD update coefficients.

    All arrays have shape [Nz, Ny, Nx] matching the Yee grid.
    """

    def __init__(self, Nz: int, Ny: int, Nx: int,
                 eps_r: float = 1.0, mu_r: float = 1.0, sigma: float = 0.0):
        self.Nz = Nz
        self.Ny = Ny
        self.Nx = Nx

        shape = [Nz, Ny, Nx]
        self.eps = tf.constant(EPS0 * eps_r, dtype=tf.float32) * tf.ones(shape, dtype=tf.float32)
        self.mu = tf.constant(MU0 * mu_r, dtype=tf.float32) * tf.ones(shape, dtype=tf.float32)
        self.sigma = tf.constant(sigma, dtype=tf.float32) * tf.ones(shape, dtype=tf.float32)

    def compute_coefficients(self, dt: float):
        """Precompute Ca, Cb (E-field) and dt_over_mu (H-field) coefficients.

        E-field update: E_new = Ca * E_old + Cb * curl_H
        H-field update: H_new = H_old - dt_over_mu * curl_E
        """
        dt_t = tf.constant(dt, dtype=tf.float32)
        half_sigma_dt_over_eps = self.sigma * dt_t / (2.0 * self.eps)

        # E-field coefficients (lossy media)
        self.Ca = (1.0 - half_sigma_dt_over_eps) / (1.0 + half_sigma_dt_over_eps)
        self.Cb = (dt_t / self.eps) / (1.0 + half_sigma_dt_over_eps)

        # H-field coefficient
        self.dt_over_mu = dt_t / self.mu
    
    def set_region(self, i_range: tuple, j_range: tuple, k_range: tuple,
                   eps_r: float = None, mu_r: float = None, sigma: float = None):
        """Define material properties in a specific region of the grid.
        
        This method allows heterogeneous materials by setting different
        permittivity, permeability, and conductivity in spatial regions.
        
        Parameters
        ----------
        i_range : (i_min, i_max)
            Index range in x direction (inclusive).
        j_range : (j_min, j_max)
            Index range in y direction (inclusive).
        k_range : (k_min, k_max)
            Index range in z direction (inclusive).
        eps_r : float, optional
            Relative permittivity. If None, keep current value.
        mu_r : float, optional
            Relative permeability. If None, keep current value.
        sigma : float, optional
            Conductivity [S/m]. If None, keep current value.
        
        Notes
        -----
        After calling this method, you must call compute_coefficients(dt) again
        to update the FDTD update coefficients.
        """
        i_min, i_max = i_range
        j_min, j_max = j_range
        k_min, k_max = k_range
        
        # Validate indices
        if not (0 <= i_min < i_max <= self.Nx):
            raise ValueError(f"Invalid i_range {i_range}, must be within [0, {self.Nx})")
        if not (0 <= j_min < j_max <= self.Ny):
            raise ValueError(f"Invalid j_range {j_range}, must be within [0, {self.Ny})")
        if not (0 <= k_min < k_max <= self.Nz):
            raise ValueError(f"Invalid k_range {k_range}, must be within [0, {self.Nz})")
        
        # Update permittivity
        if eps_r is not None:
            eps_value = EPS0 * eps_r
            # Create indices for the region
            indices = []
            for k in range(k_min, k_max):
                for j in range(j_min, j_max):
                    for i in range(i_min, i_max):
                        indices.append([k, j, i])
            
            indices_tensor = tf.constant(indices, dtype=tf.int32)
            updates = tf.fill([len(indices)], eps_value)
            self.eps = tf.tensor_scatter_nd_update(self.eps, indices_tensor, updates)
        
        # Update permeability
        if mu_r is not None:
            mu_value = MU0 * mu_r
            indices = []
            for k in range(k_min, k_max):
                for j in range(j_min, j_max):
                    for i in range(i_min, i_max):
                        indices.append([k, j, i])
            
            indices_tensor = tf.constant(indices, dtype=tf.int32)
            updates = tf.fill([len(indices)], mu_value)
            self.mu = tf.tensor_scatter_nd_update(self.mu, indices_tensor, updates)
        
        # Update conductivity
        if sigma is not None:
            indices = []
            for k in range(k_min, k_max):
                for j in range(j_min, j_max):
                    for i in range(i_min, i_max):
                        indices.append([k, j, i])
            
            indices_tensor = tf.constant(indices, dtype=tf.int32)
            updates = tf.fill([len(indices)], sigma)
            self.sigma = tf.tensor_scatter_nd_update(self.sigma, indices_tensor, updates)
