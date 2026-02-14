"""Material property grids and precomputed update coefficients for FDTD.

Supports isotropic materials via set_region and set_material (catalog or
Material instance). Dispersive regions (Drude, Lorentz) use auxiliary
fields P and J; add_dispersive_region() registers them for the solver.
"""

import tensorflow as tf
from typing import Union, TYPE_CHECKING, List, Dict, Any, Optional

from emsim.constants import EPS0, MU0

if TYPE_CHECKING:
    from emsim.materials.base import Material


class MaterialGrid:
    """Stores 3D material property maps and precomputes FDTD update coefficients.

    All arrays have shape [Nz, Ny, Nx] matching the Yee grid.
    Dispersive regions use auxiliary polarization P and current J (Drude model).
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

        # Dispersive (Drude/Lorentz) regions and auxiliary fields
        self.dispersive_regions: List[Dict[str, Any]] = []
        self._drude_mask: Optional[tf.Tensor] = None
        self._drude_eps_inf: Optional[float] = None
        self._drude_omega_p: Optional[float] = None
        self._drude_gamma: Optional[float] = None
        self.Px: Optional[tf.Variable] = None
        self.Py: Optional[tf.Variable] = None
        self.Pz: Optional[tf.Variable] = None
        self.Jx: Optional[tf.Variable] = None
        self.Jy: Optional[tf.Variable] = None
        self.Jz: Optional[tf.Variable] = None

        # Anisotropic regions (diagonal permittivity tensor)
        self.anisotropic_regions: List[Dict[str, Any]] = []
        self._aniso_mask: Optional[tf.Tensor] = None
        self._aniso_eps_xx: Optional[float] = None
        self._aniso_eps_yy: Optional[float] = None
        self._aniso_eps_zz: Optional[float] = None

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
        
        # Build indices as tensors (no Python loop over region)
        k_idx = tf.range(k_min, k_max, dtype=tf.int32)
        j_idx = tf.range(j_min, j_max, dtype=tf.int32)
        i_idx = tf.range(i_min, i_max, dtype=tf.int32)
        kk, jj, ii = tf.meshgrid(k_idx, j_idx, i_idx, indexing="ij")
        indices_t = tf.reshape(tf.stack([kk, jj, ii], axis=-1), [-1, 3])
        n_cells = tf.shape(indices_t)[0]

        if eps_r is not None:
            eps_value = EPS0 * eps_r
            updates = tf.fill([n_cells], eps_value)
            self.eps = tf.tensor_scatter_nd_update(self.eps, indices_t, updates)

        if mu_r is not None:
            mu_value = MU0 * mu_r
            updates = tf.fill([n_cells], mu_value)
            self.mu = tf.tensor_scatter_nd_update(self.mu, indices_t, updates)

        if sigma is not None:
            updates = tf.fill([n_cells], sigma)
            self.sigma = tf.tensor_scatter_nd_update(self.sigma, indices_t, updates)

    def set_material(
        self,
        i_range: tuple,
        j_range: tuple,
        k_range: tuple,
        material: Union[str, "Material"],
    ) -> None:
        """Set material in a region by catalog name or Material instance.

        If material is a string, it is resolved via get_material_manager().get().
        For isotropic Material instances, eps_r, mu_r, sigma are applied.
        DispersiveMaterial and AnisotropicMaterial are handled via fallback
        to equivalent isotropic parameters until full support is implemented.

        Parameters
        ----------
        i_range : tuple (i_min, i_max)
            Index range in x direction.
        j_range : tuple (j_min, j_max)
            Index range in y direction.
        k_range : tuple (k_min, k_max)
            Index range in z direction.
        material : str or Material
            Catalog key (e.g. "rogers_ro4003c") or Material instance from
            emsim.materials.

        Notes
        -----
        After calling this method, call compute_coefficients(dt) again to
        update FDTD coefficients.
        """
        from emsim.materials import get_material_manager
        from emsim.materials.base import Material, DispersiveMaterial, AnisotropicMaterial

        if isinstance(material, str):
            mgr = get_material_manager()
            mat = mgr.get(material)
        else:
            mat = material

        if isinstance(mat, AnisotropicMaterial):
            self.set_region(
                i_range, j_range, k_range,
                eps_r=mat.eps_r_xx,
                mu_r=mat.mu_r_xx,
                sigma=mat.sigma,
            )
        elif isinstance(mat, DispersiveMaterial):
            eps_r = mat.eps_inf if mat.eps_inf is not None else mat.eps_r
            self.set_region(
                i_range, j_range, k_range,
                eps_r=eps_r,
                mu_r=mat.mu_r,
                sigma=mat.sigma,
            )
        else:
            self.set_region(
                i_range, j_range, k_range,
                eps_r=mat.eps_r,
                mu_r=mat.mu_r,
                sigma=mat.sigma,
            )

    def add_dispersive_region(
        self,
        i_range: tuple,
        j_range: tuple,
        k_range: tuple,
        material: "Material",
    ) -> None:
        """Register a dispersive (Drude/Lorentz) material region.

        Auxiliary fields P and J are allocated on first call. Only one
        Drude parameter set is supported (last added region wins in overlap).
        Lorentz is stored for future use; currently only Drude is used in the solver.

        Parameters
        ----------
        i_range, j_range, k_range : tuple (min, max)
            Index ranges for the region.
        material : DispersiveMaterial
            Must have model "drude" or "lorentz" and corresponding parameters.
        """
        from emsim.materials.base import DispersiveMaterial

        if not isinstance(material, DispersiveMaterial):
            raise TypeError("add_dispersive_region requires a DispersiveMaterial")

        region = {
            "bounds": (i_range, j_range, k_range),
            "material": material,
            "model": material.model,
        }
        self.dispersive_regions.append(region)

        shape = [self.Nz, self.Ny, self.Nx]
        if self.Px is None:
            self.Px = tf.Variable(tf.zeros(shape, dtype=tf.float32))
            self.Py = tf.Variable(tf.zeros(shape, dtype=tf.float32))
            self.Pz = tf.Variable(tf.zeros(shape, dtype=tf.float32))
            self.Jx = tf.Variable(tf.zeros(shape, dtype=tf.float32))
            self.Jy = tf.Variable(tf.zeros(shape, dtype=tf.float32))
            self.Jz = tf.Variable(tf.zeros(shape, dtype=tf.float32))

        if material.model == "drude" and material.omega_p is not None and material.gamma is not None:
            eps_inf = material.eps_inf if material.eps_inf is not None else 1.0
            self._drude_eps_inf = eps_inf
            self._drude_omega_p = material.omega_p
            self._drude_gamma = material.gamma
            k_min, k_max = k_range[0], k_range[1]
            j_min, j_max = j_range[0], j_range[1]
            i_min, i_max = i_range[0], i_range[1]
            if not (0 <= k_min < k_max <= self.Nz and 0 <= j_min < j_max <= self.Ny and 0 <= i_min < i_max <= self.Nx):
                raise ValueError(
                    f"Dispersive region bounds (i={i_range}, j={j_range}, k={k_range}) "
                    f"must be within grid [0,{self.Nx}) [0,{self.Ny}) [0,{self.Nz})"
                )
            indices = _region_indices(k_min, k_max, j_min, j_max, i_min, i_max)
            indices_t = tf.constant(indices, dtype=tf.int32)
            n = len(indices)
            mask = tf.zeros(shape, dtype=tf.float32)
            mask = tf.tensor_scatter_nd_update(mask, indices_t, tf.ones(n, dtype=tf.float32))
            if self._drude_mask is None:
                self._drude_mask = mask
            else:
                self._drude_mask = tf.maximum(self._drude_mask, mask)

    def add_anisotropic_region(
        self,
        i_range: tuple,
        j_range: tuple,
        k_range: tuple,
        material: "Material",
    ) -> None:
        """Register an anisotropic (diagonal permittivity tensor) region.

        Only diagonal eps_r_xx, eps_r_yy, eps_r_zz are used. One parameter set
        per grid (last added wins in overlap). E-update in these cells uses
        E = inv(eps)*D with D = eps*E_old + dt*curl(H).

        Parameters
        ----------
        i_range, j_range, k_range : tuple (min, max)
            Index ranges for the region.
        material : AnisotropicMaterial
            Must have eps_r_xx, eps_r_yy, eps_r_zz (and optionally off-diagonal for future).
        """
        from emsim.materials.base import AnisotropicMaterial

        if not isinstance(material, AnisotropicMaterial):
            raise TypeError("add_anisotropic_region requires an AnisotropicMaterial")

        region = {
            "bounds": (i_range, j_range, k_range),
            "material": material,
        }
        self.anisotropic_regions.append(region)

        k_min, k_max = k_range[0], k_range[1]
        j_min, j_max = j_range[0], j_range[1]
        i_min, i_max = i_range[0], i_range[1]
        if not (0 <= k_min < k_max <= self.Nz and 0 <= j_min < j_max <= self.Ny and 0 <= i_min < i_max <= self.Nx):
            raise ValueError(
                f"Anisotropic region bounds (i={i_range}, j={j_range}, k={k_range}) "
                f"must be within grid [0,{self.Nx}) [0,{self.Ny}) [0,{self.Nz})"
            )
        indices = _region_indices(k_min, k_max, j_min, j_max, i_min, i_max)
        indices_t = tf.constant(indices, dtype=tf.int32)
        n = len(indices)
        mask = tf.zeros([self.Nz, self.Ny, self.Nx], dtype=tf.float32)
        mask = tf.tensor_scatter_nd_update(mask, indices_t, tf.ones(n, dtype=tf.float32))
        if self._aniso_mask is None:
            self._aniso_mask = mask
        else:
            self._aniso_mask = tf.maximum(self._aniso_mask, mask)
        self._aniso_eps_xx = material.eps_r_xx
        self._aniso_eps_yy = material.eps_r_yy
        self._aniso_eps_zz = material.eps_r_zz


def _region_indices(k_min: int, k_max: int, j_min: int, j_max: int, i_min: int, i_max: int):
    """List of [k,j,i] indices for a rectangular region (for scatter_nd_update)."""
    indices = []
    for k in range(k_min, k_max):
        for j in range(j_min, j_max):
            for i in range(i_min, i_max):
                indices.append([k, j, i])
    return indices
